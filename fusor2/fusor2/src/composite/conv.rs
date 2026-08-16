//! `conv` and `grouped_conv` as macro ops over `Window` + `Contract`.
//!
//! An `EinSpec` contracts over **several** labels directly, so the windowed view
//! is contracted in place: `c` and every `k` label are contracted, `p` labels
//! are free, and the operand's non-affine index map is the Launch lowering's
//! business.

use fusor2_autograd::tape::{GraphTape, TapeExt};
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::ir::logical::{EinSpec, Logical, Label};
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::{Dim, SlidingWindow, StrideSpec};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::{MacroAttr, MacroOp, const_dim, index_run, macro_op};
use crate::tensor::Tensor;

/// Zero-pad one axis by `left` before and `right` after.
///
/// A `Scatter{Set}` into a `Const` zero leaf, with `unique: true` because the
/// index run is strictly increasing by construction. `cat`, `stack`, `repeat`
/// and `slice_assign` are the same node.
pub fn pad_with_zeros(x: &Tensor, axis: u32, left: u64, right: u64) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let facts = x.graph.facts(x.id);
    let len = const_dim(
        *facts
            .shape
            .get(axis as usize)
            .ok_or_else(|| Error::Shape(format!("pad axis {axis} out of range")))?,
        "pad_with_zeros",
    )?;
    let idx = index_run(&x.graph, left, len)?;
    let mut padded = facts.shape.clone();
    padded[axis as usize] = Dim::Const(left + len + right);
    let (xid, iid) = (x.id, idx);
    let dtype = facts.dtype;
    let id = x.graph.build(|t| {
        let base = t.zeros_shaped(dtype, &padded)?;
        t.scatter_set(axis, base, iid, xid, true)
    })?;
    Ok(x.graph.tensor(id))
}

/// Symmetric padding of one axis.
/// Split axis `axis` of `v` into `(outer, inner)`.
///
/// Always legal, at any strides: `Restride` composes relative to the current
/// strides, so the outer axis is just the inner one's stride times the inner
/// extent.
fn split_axis(t: &mut GraphTape<'_>, v: Val, axis: usize, outer: u64, inner: u64) -> Result<Val> {
    let shape = t.shape_of(v);
    let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
    for (i, d) in shape.iter().copied().enumerate() {
        if i == axis {
            specs.push(StrideSpec::dim_with(
                axis as u32,
                Dim::Const(outer),
                inner as u32,
            ));
            specs.push(StrideSpec::dim(axis as u32, Dim::Const(inner)));
        } else {
            specs.push(StrideSpec::dim(i as u32, d));
        }
    }
    t.restride(&specs, v)
}

/// Merge axes `axis` and `axis + 1`, both of decidable extent.
///
/// Only legal when the pair is contiguous with respect to each other, which is
/// true of a freshly produced contraction output — the only place this is used.
fn merge_axes(t: &mut GraphTape<'_>, v: Val, axis: usize) -> Result<Val> {
    let shape = t.shape_of(v);
    let a = const_dim(shape[axis], "merge_axes")?;
    let b = const_dim(shape[axis + 1], "merge_axes")?;
    let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
    for (i, d) in shape.iter().copied().enumerate() {
        match i {
            _ if i == axis => specs.push(StrideSpec::dim(
                (axis + 1) as u32,
                Dim::Const(a.saturating_mul(b)),
            )),
            _ if i == axis + 1 => {}
            _ => specs.push(StrideSpec::dim(i as u32, d)),
        }
    }
    t.restride(&specs, v)
}

/// Every spatial axis's `(window, step)`.
fn windows(kernel: &[u64], stride: &[u32], first_spatial: u32) -> Result<SmallVec<[SlidingWindow; 3]>> {
    kernel
        .iter()
        .zip(stride)
        .enumerate()
        .map(|(i, (k, s))| {
            let k = u32::try_from(*k)
                .map_err(|_| Error::Shape(format!("kernel extent {k} exceeds a u32")))?;
            Ok(SlidingWindow::new(first_spatial + i as u32, k, (*s).max(1)))
        })
        .collect()
}

/// N-dimensional convolution. `stride`, `padding` and `dilation` are per-axis.
///
/// `x` is `[batch, in_ch, ...spatial]`, `weight` is
/// `[out_ch, in_ch, ...kernel]` and `bias` is `[out_ch]`.
pub fn conv(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
) -> Result<Tensor> {
    grouped_conv(x, weight, bias, stride, padding, dilation, 1)
}

/// Grouped convolution, PyTorch layout: `weight` is
/// `[out_ch, in_ch / groups, ...kernel]`.
pub fn grouped_conv(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
    groups: u32,
) -> Result<Tensor> {
    let graph = &x.graph;
    let xf = graph.facts(x.id);
    let wf = graph.facts(weight.id);
    let spatial = xf.rank().checked_sub(2).ok_or_else(|| {
        Error::Shape("conv input is [batch, in_ch, ...spatial] and needs rank >= 2".into())
    })?;
    if wf.rank() != spatial + 2 {
        return Err(Error::Shape(format!(
            "conv weight rank {} does not match a {spatial}-d convolution",
            wf.rank()
        )));
    }
    if stride.len() != spatial || padding.len() != spatial || dilation.len() != spatial {
        return Err(Error::Shape(format!(
            "conv expects {spatial} entries of stride/padding/dilation"
        )));
    }
    if dilation.iter().any(|d| *d != 1) {
        // `SlidingWindow` carries `(window, step)` only, which is what keeps
        // the adjoint decidable.
        return Err(Error::Legality(
            "dilated convolution is not expressible as a SlidingWindow".into(),
        ));
    }
    let groups = groups.max(1) as u64;
    let in_ch = const_dim(xf.shape[1], "conv in_channels")?;
    let out_ch = const_dim(wf.shape[0], "conv out_channels")?;
    let per_group_in = const_dim(wf.shape[1], "conv weight in_channels")?;
    if in_ch % groups != 0 || out_ch % groups != 0 {
        return Err(Error::Shape(format!(
            "grouped conv needs groups {groups} to divide both {in_ch} and {out_ch}"
        )));
    }
    if per_group_in != in_ch / groups {
        return Err(Error::Shape(format!(
            "grouped conv weight declares {per_group_in} input channels per group, expected {}",
            in_ch / groups
        )));
    }
    let kernel: Vec<u64> = (0..spatial)
        .map(|i| const_dim(wf.shape[2 + i], "conv kernel"))
        .collect::<Result<_>>()?;

    // Padding happens before the macro node so the scatter's index leaf is an
    // ordinary operand.
    let mut padded = x.clone();
    for (i, p) in padding.iter().enumerate() {
        padded = pad_with_zeros(&padded, (2 + i) as u32, *p as u64, *p as u64)?;
    }

    let ops = {
        let mut v = vec![padded.id, weight.id];
        if let Some(b) = bias {
            v.push(b.id);
        }
        v
    };
    let attrs = MacroAttr::Conv {
        padding: padding.iter().copied().collect(),
        stride: stride.iter().copied().collect(),
        groups: groups as u32,
        spatial: spatial as u32,
    };
    let (xid, wid) = (padded.id, weight.id);
    let bid = bias.map(|b| b.id);
    let stride: SmallVec<[u32; 3]> = stride.iter().copied().collect();

    macro_op(graph, MacroOp::Conv, attrs, &ops, move |t| {
        conv_defn(t, xid, wid, bid, &kernel, &stride, spatial, groups, out_ch)
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    weight: Val,
    bias: Option<Val>,
    kernel: &[u64],
    stride: &[u32],
    spatial: usize,
    groups: u64,
    out_ch: u64,
) -> Result<Val> {
    let grouped = groups > 1;

    // x: [batch, in_ch, ...spatial] -> optionally [batch, g, cpg, ...spatial]
    let x = if grouped {
        let in_ch = const_dim(t.shape_of(x)[1], "conv in_channels")?;
        split_axis(t, x, 1, groups, in_ch / groups)?
    } else {
        x
    };
    let channel_axis = if grouped { 2 } else { 1 };
    let first_spatial = channel_axis as u32 + 1;
    let x = t.add(Logical::Window {
        specs: windows(kernel, stride, first_spatial)?,
        x,
    })?;

    // weight: [out_ch, cpg, ...kernel] -> optionally [g, opg, cpg, ...kernel]
    let weight = if grouped {
        split_axis(t, weight, 0, groups, out_ch / groups)?
    } else {
        weight
    };

    // `b` and every `p` are free axes of the left operand, `o` is the free
    // axis of the right, and `c` plus every `k` are contracted. A group label
    // is a batch label — it is the only difference grouping makes.
    let mut next = 0u8;
    let mut fresh = || {
        let l = Label(next);
        next += 1;
        l
    };
    let batch = fresh();
    let group = fresh();
    let channel = fresh();
    let out_label = fresh();
    let pos: Vec<Label> = (0..spatial).map(|_| fresh()).collect();
    let ks: Vec<Label> = (0..spatial).map(|_| fresh()).collect();

    // a's axis order is [batch, (g), c, p..., k...] — `Window` appends the
    // window axes after every positional axis, in ascending axis order.
    let mut a: SmallVec<[Label; 6]> = SmallVec::new();
    a.push(batch);
    if grouped {
        a.push(group);
    }
    a.push(channel);
    a.extend(pos.iter().copied());
    a.extend(ks.iter().copied());

    let mut b: SmallVec<[Label; 6]> = SmallVec::new();
    if grouped {
        b.push(group);
    }
    b.push(out_label);
    b.push(channel);
    b.extend(ks.iter().copied());

    let mut out: SmallVec<[Label; 6]> = SmallVec::new();
    out.push(batch);
    if grouped {
        out.push(group);
    }
    out.push(out_label);
    out.extend(pos.iter().copied());

    let acc = fusor2_autograd::tape::accum_dtype(t.dtype_of(x));
    let dtype = t.dtype_of(x);
    let y = t.contract(x, weight, EinSpec { a, b, out }, acc)?;
    let y = t.cast(dtype, y)?;

    // back to [batch, out_ch, ...spatial]
    let y = if grouped { merge_axes(t, y, 1)? } else { y };

    match bias {
        Some(bias) => {
            // `[out_ch]` broadcasts over the channel axis, which is axis 1 —
            // not the last one, so the right-aligned rule needs an explicit
            // reshape first.
            let shape = t.shape_of(y);
            let mut spec: SmallVec<[StrideSpec; 6]> = SmallVec::new();
            spec.push(StrideSpec::broadcast(shape[0]));
            spec.push(StrideSpec::dim(0, shape[1]));
            for d in shape.iter().skip(2).copied() {
                spec.push(StrideSpec::broadcast(d));
            }
            let bias = t.restride(&spec, bias)?;
            t.binary(BinOp::Add, y, bias)
        }
        None => Ok(y),
    }
}
