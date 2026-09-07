//! Nearest and bilinear upsampling, macro ops over `Gather` and `Map`.
//!
//! This value is one `Gather` per axis with a repeated index run — a real
//! node, whose adjoint is the `ScatterAdd` that sums the duplicated
//! positions, with four lowerings underneath it.

use fusor_autograd::tape::{GraphTape, TapeExt};
use fusor_ir::autograd::{Tape, Val};
use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::Id;
use fusor_ir::scalar::BinOp;
use fusor_ir::shape::Dim;
use fusor_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::{MacroAttr, MacroOp, const_dim, index_leaf, macro_op};
use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// A rank-1 `f32` weight leaf.
fn weight_leaf(graph: &GraphRef, dtype: Dtype, values: &[f32]) -> Result<Id> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        match dtype {
            Dtype::F16 => bytes.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes()),
            Dtype::BF16 => {
                bytes.extend_from_slice(&half::bf16::from_f32(*v).to_bits().to_le_bytes())
            }
            _ => bytes.extend_from_slice(&v.to_le_bytes()),
        }
    }
    graph.constant_leaf(dtype, &[Dim::Const(values.len() as u64)], bytes)
}

/// Nearest-neighbour source index for each output position.
fn nearest_indices(input: u64, output: u64) -> Vec<u32> {
    (0..output)
        .map(|o| {
            let src = (o * input)
                .checked_div(output)
                .map_or(0, |src| src.min(input.saturating_sub(1)));
            src as u32
        })
        .collect()
}

/// `[b, c, h, w] -> [b, c, h * scale_h, w * scale_w]`, each pixel repeated
/// exactly. Both scales must be at least 1.
pub fn upsample_nearest2d(x: &Tensor, scale_h: u32, scale_w: u32) -> Result<Tensor> {
    if scale_h < 1 || scale_w < 1 {
        return Err(Error::Shape("upsample scales must be at least 1".into()));
    }
    let facts = x.graph.facts(x.id);
    if facts.rank() != 4 {
        return Err(Error::Shape(format!(
            "upsample_nearest2d needs [b, c, h, w], got rank {}",
            facts.rank()
        )));
    }
    let h = const_dim(facts.shape[2], "upsample height")?;
    let w = const_dim(facts.shape[3], "upsample width")?;
    let size = [
        Dim::Const(h * scale_h as u64),
        Dim::Const(w * scale_w as u64),
    ];
    upsample_nearest_axes(x, &size, 2, &[scale_h, scale_w])
}

/// Nearest upsampling of the trailing `size.len()` axes to the given extents.
pub fn upsample_nearest(x: &Tensor, size: &[Dim]) -> Result<Tensor> {
    let rank = x.graph.facts(x.id).rank();
    let first = rank
        .checked_sub(size.len())
        .ok_or_else(|| Error::Shape("upsample names more axes than the value has".into()))?;
    let scales: SmallVec<[u32; 3]> = size
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let src = const_dim(x.graph.facts(x.id).shape[first + i], "upsample source")?;
            let dst = const_dim(*d, "upsample target")?;
            Ok((dst / src.max(1)) as u32)
        })
        .collect::<Result<_>>()?;
    upsample_nearest_axes(x, size, first as u32, &scales)
}

fn upsample_nearest_axes(x: &Tensor, size: &[Dim], first: u32, scales: &[u32]) -> Result<Tensor> {
    let graph = &x.graph;
    let facts = graph.facts(x.id);
    let mut index_ids = Vec::with_capacity(size.len());
    for (i, target) in size.iter().enumerate() {
        let axis = first as usize + i;
        let src = const_dim(facts.shape[axis], "upsample source")?;
        let dst = const_dim(*target, "upsample target")?;
        if dst < src {
            return Err(Error::Shape(
                "upsample only grows an axis; use a pool to shrink one".into(),
            ));
        }
        index_ids.push((axis as u32, index_leaf(graph, &nearest_indices(src, dst))?));
    }

    let mut ops = vec![x.id];
    ops.extend(index_ids.iter().map(|(_, id)| *id));
    let xid = x.id;
    let attrs = MacroAttr::Upsample {
        scales: scales.iter().copied().collect(),
    };
    macro_op(graph, MacroOp::Upsample, attrs, &ops, move |t| {
        let mut v = xid;
        for (axis, idx) in index_ids {
            v = t.gather(axis, v, idx)?;
        }
        Ok(v)
    })
}

/// Bilinear upsampling of the last two axes.
///
/// Four taps, each a pair of gathers, combined by two `Map`s against
/// per-position weight vectors uploaded once. `align_corners` picks between
/// the two standard coordinate conventions.
pub fn upsample_bilinear(x: &Tensor, size: &[Dim], align_corners: bool) -> Result<Tensor> {
    let graph = &x.graph;
    let facts = graph.facts(x.id);
    if size.len() != 2 || facts.rank() < 2 {
        return Err(Error::Shape(
            "upsample_bilinear resamples exactly the last two axes".into(),
        ));
    }
    let dtype = facts.dtype;
    let rank = facts.rank();
    let (ay, ax) = ((rank - 2) as u32, (rank - 1) as u32);
    let (sh, sw) = (
        const_dim(facts.shape[ay as usize], "bilinear source height")?,
        const_dim(facts.shape[ax as usize], "bilinear source width")?,
    );
    let (dh, dw) = (
        const_dim(size[0], "bilinear target height")?,
        const_dim(size[1], "bilinear target width")?,
    );

    let (y0, y1, wy) = bilinear_taps(sh, dh, align_corners);
    let (x0, x1, wx) = bilinear_taps(sw, dw, align_corners);

    let y0 = index_leaf(graph, &y0)?;
    let y1 = index_leaf(graph, &y1)?;
    let x0 = index_leaf(graph, &x0)?;
    let x1 = index_leaf(graph, &x1)?;
    let wy = weight_leaf(graph, dtype, &wy)?;
    let wx = weight_leaf(graph, dtype, &wx)?;

    let xid = x.id;
    let ops = vec![xid, y0, y1, x0, x1, wy, wx];
    let attrs = MacroAttr::Upsample {
        scales: smallvec::smallvec![(dh / sh.max(1)) as u32, (dw / sw.max(1)) as u32],
    };
    macro_op(graph, MacroOp::Upsample, attrs, &ops, move |t| {
        // Interpolate rows first, then columns: two 2-tap blends rather than
        // one 4-tap, which is the same value and half the gathers.
        let top = t.gather(ay, xid, y0)?;
        let bottom = t.gather(ay, xid, y1)?;
        let rows = blend(t, top, bottom, wy, ay)?;
        let left = t.gather(ax, rows, x0)?;
        let right = t.gather(ax, rows, x1)?;
        blend(t, left, right, wx, ax)
    })
}

/// `a + w * (b - a)`, with `w` a per-position vector broadcast over `axis`.
fn blend(t: &mut GraphTape<'_>, a: Val, b: Val, w: Val, axis: u32) -> Result<Val> {
    let shape = t.shape_of(a);
    let mut specs: SmallVec<[fusor_ir::shape::StrideSpec; 6]> = SmallVec::new();
    for (i, d) in shape.iter().copied().enumerate() {
        if i == axis as usize {
            specs.push(fusor_ir::shape::StrideSpec::dim(0, d));
        } else {
            specs.push(fusor_ir::shape::StrideSpec::broadcast(d));
        }
    }
    let w = t.restride(&specs, w)?;
    let delta = t.binary(BinOp::Sub, b, a)?;
    let scaled = t.binary(BinOp::Mul, w, delta)?;
    t.binary(BinOp::Add, a, scaled)
}

/// `(lower, upper, fraction)` per output position.
fn bilinear_taps(src: u64, dst: u64, align_corners: bool) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut lo = Vec::with_capacity(dst as usize);
    let mut hi = Vec::with_capacity(dst as usize);
    let mut frac = Vec::with_capacity(dst as usize);
    let last = src.saturating_sub(1);
    for o in 0..dst {
        let pos = if align_corners {
            if dst <= 1 {
                0.0
            } else {
                o as f64 * last as f64 / (dst - 1) as f64
            }
        } else {
            (((o as f64 + 0.5) * src as f64 / dst as f64) - 0.5).max(0.0)
        };
        let l = pos.floor().min(last as f64) as u64;
        let h = (l + 1).min(last);
        lo.push(l as u32);
        hi.push(h as u32);
        frac.push((pos - l as f64) as f32);
    }
    (lo, hi, frac)
}
