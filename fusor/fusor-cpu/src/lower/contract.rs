//! Direct platform-GEMM lowering for CPU contractions.

use std::sync::Arc;

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, BufferDecl, ElementType, KernelIr, ScalarElement, Stmt, TileExpr,
};
use fusor_ir::ir::launch::{ContractSide, Launch};
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{ScalarExpr, ScalarKind};
use fusor_ir::shape::{Dim, Layout};
use fusor_ir::target::LowerCtx;
use fusor_tile::build::Kernel;

use super::{Binds, DEFAULT_BLOCK, OperandSrc, Translate, global_lane, grid_for, view};

pub(crate) fn lower(node: &Node, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::Launch(Launch::Contract {
        m,
        n,
        k,
        batch,
        post,
        acc,
        a,
        b,
        ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a contraction launch".into()));
    };
    let [m, n, k, batch] = [
        concrete(cx, *m, "m")?.max(1),
        concrete(cx, *n, "n")?.max(1),
        concrete(cx, *k, "k")?.max(1),
        concrete(cx, *batch, "batch")?.max(1),
    ];
    let binds = Binds::build(cx)?;
    let out = binds.of(cx.launch.root)?;
    // The platform GEMM is Accelerate, so it exists on macOS only; everywhere
    // else every contraction takes the Cranelift SIMD path below.
    let platform = if cfg!(target_os = "macos") {
        side(cx, &binds, a, [batch, m, k])
            .zip(side(cx, &binds, b, [batch, k, n]))
            .and_then(|(bound_a, bound_b)| {
                gemm_name(
                    m, n, k, batch, &out, &bound_a, &bound_b, &a.pre, &b.pre, post,
                )
                .ok()
            })
    } else {
        None
    };
    if let Some(name) = platform {
        return Ok(KernelIr {
            buffers: binds.buffers,
            grid: [1, 1, 1],
            block: 1,
            body: Vec::new(),
            byte_arena: None,
            name,
        });
    }
    lower_jit(cx, binds, out, [batch, m, n, k], a, b, post, *acc)
}

type BoundSide = Vec<(Arc<BufferDecl>, [u32; 3])>;

fn side(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    side: &ContractSide,
    groups: [u32; 3],
) -> Option<BoundSide> {
    side.ops
        .iter()
        .map(|operand| {
            if fusor_tile::build::const_splat(cx, operand.src).is_some() {
                return None;
            }
            // A GEMM call reads from the buffer's start; an offset layout —
            // a slot of a multi-slot fold, a window — goes to the JIT.
            if super::resolved_layout(cx, &operand.layout).ok()?.0 != 0 {
                return None;
            }
            let strides = collapsed_strides(cx, &operand.layout, groups)?;
            Some((binds.of(operand.src).ok()?, strides))
        })
        .collect()
}

/// A contraction is more general than matrix multiplication: attention masks,
/// sampling rows, absorbed producers and fused epilogues all use the same
/// launch node. Only the small subset recognized above is a BLAS call. This
/// kernel is the native Cranelift path for the rest: one output per lane and a
/// private accumulator across `k`.
#[allow(clippy::too_many_arguments)]
fn lower_jit(
    cx: &LowerCtx<'_>,
    binds: Binds,
    out: std::sync::Arc<BufferDecl>,
    [batch, m, n, k]: [u32; 4],
    a: &ContractSide,
    b_side: &ContractSide,
    post: &ScalarExpr,
    acc: fusor_ir::dtype::Dtype,
) -> Result<KernelIr> {
    let b = Kernel::new();
    let block = DEFAULT_BLOCK;
    let total = u64::from(batch) * u64::from(m) * u64::from(n);
    let total_u32 = u32::try_from(total)
        .map_err(|_| Error::Legality("CPU JIT contraction output exceeds u32 indexing".into()))?;
    let grid = grid_for(total, block);
    let flat = global_lane(&b, block);
    let valid = b.lt(flat.clone(), b.u32(total_u32));
    let (rest, col) = b.divrem(flat.clone(), b.u32(n));
    let (batch_idx, row) = b.divrem(rest, b.u32(m));
    let k_local = b.local(ScalarElement::U32.element());
    let k_idx = b.load_local(k_local.clone());
    let uniforms = binds.buffers.first().cloned();

    let a_srcs = jit_side(&b, cx, &binds, a, [batch, m, k])?;
    let b_srcs = jit_side(&b, cx, &binds, b_side, [batch, k, n])?;
    let a_value = side_value(
        &b,
        cx,
        a,
        &a_srcs,
        [batch, m, k],
        [&batch_idx, &row, &k_idx],
        valid.clone(),
        uniforms.clone(),
    )?;
    let b_value = side_value(
        &b,
        cx,
        b_side,
        &b_srcs,
        [batch, k, n],
        [&batch_idx, &k_idx, &col],
        valid.clone(),
        uniforms.clone(),
    )?;
    let acc_ty = ElementType::Scalar(super::elem_of(acc)?);
    let local = b.local(acc_ty);
    let previous = b.load_local(local.clone());
    let update = b.add(
        previous.clone(),
        b.mul(b.cast(a_value, acc_ty), b.cast(b_value, acc_ty)),
    );
    let value = Translate {
        b: &b,
        args: &[previous],
        coords: &[],
        uniforms,
    }
    .run(post)?;
    let body = vec![
        Stmt::Loop {
            count: Some(b.u32(k)),
            index: Some(k_local),
            accumulators: vec![Accumulator {
                local,
                init: b.cast(b.f32(0.0), acc_ty),
                update,
            }],
            body: Vec::new(),
        },
        Stmt::Store {
            dst: view(&out),
            addr: Addr::Linear(flat),
            value,
            mask: valid,
        },
    ];
    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_contract_jit",
    })
}

/// One operand's source and how its `(batch, row, col)` coordinates reach
/// an element: three collapsed strides when each axis group is one dense
/// run, else the layout's own axes, decomposed per group — a permuted head
/// split, a broadcast, a merged axis, an offset slot are all just strides.
struct JitOperand {
    src: OperandSrc,
    offset: u32,
    addressing: Addressing,
}

enum Addressing {
    Collapsed([u32; 3]),
    Axes {
        extents: Vec<u32>,
        strides: Vec<u32>,
    },
}

type JitSide = Vec<JitOperand>;

fn jit_side(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    binds: &Binds,
    side: &ContractSide,
    groups: [u32; 3],
) -> Result<JitSide> {
    side.ops
        .iter()
        .map(|operand| {
            let (offset, extents, strides) = super::resolved_layout(cx, &operand.layout)?;
            let addressing = match collapse_resolved(&extents, &strides, groups) {
                Some(c) => Addressing::Collapsed(c),
                None => {
                    // The axes must partition into the groups, whatever
                    // their strides.
                    if group_axes(&extents, groups).is_none() {
                        return Err(Error::Legality(format!(
                            "CPU JIT contraction cannot partition layout {:?} into its groups",
                            operand.layout
                        )));
                    }
                    Addressing::Axes { extents, strides }
                }
            };
            Ok(JitOperand {
                src: super::operand_src(b, cx, binds, operand.src)?,
                offset,
                addressing,
            })
        })
        .collect()
}

/// The axis range each group covers, row-major with the last axis fastest,
/// and how many leading axes the three groups consumed.
fn group_ranges(extents: &[u32], groups: [u32; 3]) -> Option<([(usize, usize); 3], usize)> {
    let mut out = [(0, 0); 3];
    let mut axis = 0;
    for (group, wanted) in groups.into_iter().map(|v| v.max(1)).enumerate() {
        let start = axis;
        let mut product = 1u64;
        while product < u64::from(wanted) && axis < extents.len() {
            product = product.saturating_mul(u64::from(extents[axis]));
            axis += 1;
        }
        if product != u64::from(wanted) {
            return None;
        }
        out[group] = (start, axis);
    }
    Some((out, axis))
}

/// [`group_ranges`] over every axis, extent-0 axes counting as 1.
fn group_axes(extents: &[u32], groups: [u32; 3]) -> Option<[(usize, usize); 3]> {
    let extents: Vec<u32> = extents.iter().map(|e| (*e).max(1)).collect();
    group_ranges(&extents, groups).and_then(|(r, used)| (used == extents.len()).then_some(r))
}

/// `offset + Σ coord * stride`, the group coordinates decomposed over the
/// layout's axes.
fn axes_index(
    b: &Kernel,
    extents: &[u32],
    strides: &[u32],
    groups: [u32; 3],
    indices: [&TileExpr; 3],
    offset: u32,
) -> TileExpr {
    let ranges = group_axes(extents, groups).unwrap_or([(0, 0); 3]);
    let mut terms: Vec<TileExpr> = Vec::new();
    if offset != 0 {
        terms.push(b.u32(offset));
    }
    for (group, (start, end)) in ranges.into_iter().enumerate() {
        let mut rest = indices[group].clone();
        for i in (start..end).rev() {
            let extent = extents[i].max(1);
            let coord = match i == start {
                true => rest.clone(),
                false => b.rem(rest.clone(), b.u32(extent)),
            };
            match strides[i] {
                0 => {}
                1 => terms.push(coord),
                stride => terms.push(b.mul(coord, b.u32(stride))),
            }
            if i != start {
                rest = b.div(rest, b.u32(extent));
            }
        }
    }
    terms
        .into_iter()
        .reduce(|l, r| b.add(l, r))
        .unwrap_or_else(|| b.u32(0))
}

#[allow(clippy::too_many_arguments)]
fn side_value(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    side: &ContractSide,
    sources: &JitSide,
    groups: [u32; 3],
    indices: [&TileExpr; 3],
    mask: TileExpr,
    uniforms: Option<Arc<BufferDecl>>,
) -> Result<TileExpr> {
    let args = sources
        .iter()
        .map(|o| {
            let index = match &o.addressing {
                Addressing::Collapsed(strides) => {
                    let base = strided_index(b, indices, *strides);
                    match o.offset {
                        0 => base,
                        offset => b.add(b.u32(offset), base),
                    }
                }
                Addressing::Axes { extents, strides } => {
                    axes_index(b, extents, strides, groups, indices, o.offset)
                }
            };
            o.src.at(b, index, mask.clone())
        })
        .collect::<Vec<_>>();
    let coords = side_coords(b, cx, side, groups, indices).ok_or_else(|| {
        Error::Legality("CPU JIT contraction cannot state side coordinates".into())
    })?;
    Translate {
        b,
        args: &args,
        coords: &coords,
        uniforms,
    }
    .run(&side.pre)
}

fn strided_index(b: &Kernel, indices: [&TileExpr; 3], strides: [u32; 3]) -> TileExpr {
    indices
        .into_iter()
        .zip(strides)
        .filter(|(_, stride)| *stride != 0)
        .map(|(index, stride)| match stride {
            1 => index.clone(),
            _ => b.mul(index.clone(), b.u32(stride)),
        })
        .reduce(|left, right| b.add(left, right))
        .unwrap_or_else(|| b.u32(0))
}

/// The per-axis coordinates a side's `pre` reads, each group's flat index
/// split over the axes it covers.
fn side_coords(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    side: &ContractSide,
    groups: [u32; 3],
    indices: [&TileExpr; 3],
) -> Option<Vec<TileExpr>> {
    if !side.pre.reads_index_of() {
        return Some(Vec::new());
    }
    let extents: Vec<u32> = super::const_extents(cx, side.primary().layout.shape())
        .ok()?
        .into_iter()
        .map(|e| e.max(1))
        .collect();
    let (ranges, _) = group_ranges(&extents, groups)?;
    let mut coords = vec![b.u32(0); extents.len()];
    for ((start, end), index) in ranges.into_iter().zip(indices) {
        let mut rest = index.clone();
        for i in (start..end).rev() {
            let extent = b.u32(extents[i]);
            coords[i] = b.rem(rest.clone(), extent.clone());
            rest = b.div(rest, extent);
        }
    }
    Some(coords)
}

#[allow(clippy::too_many_arguments)]
fn gemm_name(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    out: &BufferDecl,
    a: &BoundSide,
    b: &BoundSide,
    a_pre: &ScalarExpr,
    b_pre: &ScalarExpr,
    post: &ScalarExpr,
) -> Result<&'static str> {
    let identity = |expr: &ScalarExpr| matches!(expr.kind(), ScalarKind::Arg(0));
    let f32_storage = ElementType::Scalar(ScalarElement::F32);
    let [(abuf, astrides)] = a.as_slice() else {
        return gelu_gemm_name(m, n, k, batch, out, a, b, a_pre, b_pre, post);
    };
    let [(bbuf, bstrides)] = b.as_slice() else {
        return Err(Error::Legality("CPU GEMM needs one B operand".into()));
    };
    if out.element != f32_storage || abuf.element != f32_storage || bbuf.element != f32_storage {
        return Err(Error::Legality(
            "CPU GEMM currently requires f32 storage".into(),
        ));
    }
    if !identity(a_pre) || !identity(b_pre) || !identity(post) {
        return Err(Error::Legality(
            "CPU GEMM requires its epilogue as a separate Cranelift map".into(),
        ));
    }
    compatible(*astrides, [m, k], [0, 1])?;
    compatible(*bstrides, [k, n], [1, 0])?;
    Ok(leak(format!(
        "cpu_contract_blas:{m},{n},{k},{batch},{},{},{},{},{},{},{},{},{}",
        out.binding,
        abuf.binding,
        astrides[0],
        astrides[1],
        astrides[2],
        bbuf.binding,
        bstrides[0],
        bstrides[1],
        bstrides[2]
    )))
}

#[allow(clippy::too_many_arguments)]
fn gelu_gemm_name(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    out: &BufferDecl,
    a: &BoundSide,
    b: &BoundSide,
    a_pre: &ScalarExpr,
    b_pre: &ScalarExpr,
    post: &ScalarExpr,
) -> Result<&'static str> {
    let [(abuf, astrides), (bias, bias_strides)] = a.as_slice() else {
        return Err(Error::Legality("CPU GEMM needs one A operand".into()));
    };
    let [(bbuf, bstrides)] = b.as_slice() else {
        return Err(Error::Legality("CPU GEMM needs one B operand".into()));
    };
    if *a_pre != tanh_gelu_of_bias()
        || !matches!(b_pre.kind(), ScalarKind::Arg(0))
        || !matches!(post.kind(), ScalarKind::Arg(0))
    {
        return Err(Error::Legality(
            "CPU GEMM requires its epilogue as a separate Cranelift map".into(),
        ));
    }
    let f32_storage = ElementType::Scalar(ScalarElement::F32);
    if [out, abuf.as_ref(), bias.as_ref(), bbuf.as_ref()]
        .iter()
        .any(|buffer| buffer.element != f32_storage)
        || *astrides != [0, k, 1]
        || *bias_strides != [0, 0, 1]
    {
        return Err(Error::Legality("unsupported fused CPU GEMM input".into()));
    }
    compatible(*bstrides, [k, n], [1, 0])?;
    Ok(leak(format!(
        "cpu_contract_gelu_blas:{m},{n},{k},{batch},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        out.binding,
        abuf.binding,
        astrides[0],
        astrides[1],
        astrides[2],
        bias.binding,
        bias_strides[0],
        bias_strides[1],
        bias_strides[2],
        bbuf.binding,
        bstrides[0],
        bstrides[1],
        bstrides[2]
    )))
}

/// The frontend's f32 tanh GELU (`composite::activations::gelu_expr`) over
/// `Arg(0) + Arg(1)`: the absorbed bias-add-GELU the platform kernel fuses.
fn tanh_gelu_of_bias() -> ScalarExpr {
    use fusor_ir::dtype::{Dtype, Splat};
    use fusor_ir::scalar::{BinOp, UnOp};
    let lit = |v: f32| ScalarExpr::lit(Splat::F32(v));
    let bin = ScalarExpr::bin;
    let clamp = |x, lo: f32, hi: f32| bin(BinOp::Min, bin(BinOp::Max, x, lit(lo)), lit(hi));
    let x = bin(
        BinOp::Add,
        ScalarExpr::arg(0, Dtype::F32),
        ScalarExpr::arg(1, Dtype::F32),
    );
    let x3 = bin(BinOp::Mul, x.clone(), bin(BinOp::Mul, x.clone(), x.clone()));
    let cubic = bin(BinOp::Add, x.clone(), bin(BinOp::Mul, lit(0.044_715), x3));
    let inner = clamp(bin(BinOp::Mul, lit(0.797_884_6), cubic), -15.0, 15.0);
    let p = ScalarExpr::un(UnOp::Exp, inner.clone());
    let n = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, inner));
    let tanh = bin(
        BinOp::Div,
        bin(BinOp::Sub, p.clone(), n.clone()),
        bin(BinOp::Add, p, n),
    );
    let one_plus = clamp(bin(BinOp::Add, lit(1.0), clamp(tanh, -1.0, 1.0)), 0.0, 2.0);
    bin(BinOp::Mul, bin(BinOp::Mul, lit(0.5), x), one_plus)
}

fn compatible(strides: [u32; 3], [rows, cols]: [u32; 2], broadcast: [u32; 2]) -> Result<()> {
    if (strides[2] == 1 && strides[1] >= cols)
        || (strides[1] == 1 && strides[2] >= rows)
        || strides[1..] == broadcast
    {
        Ok(())
    } else {
        Err(Error::Legality(format!(
            "CPU GEMM cannot address strides {strides:?}"
        )))
    }
}

fn leak(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn concrete(cx: &LowerCtx<'_>, dim: Dim, name: &str) -> Result<u32> {
    super::resolve_dim(cx, dim)
        .map_err(|_| Error::Legality(format!("CPU contraction needs a concrete {name}")))
}

fn collapsed_strides(cx: &LowerCtx<'_>, layout: &Layout, groups: [u32; 3]) -> Option<[u32; 3]> {
    let (_, extents, strides) = super::resolved_layout(cx, layout).ok()?;
    collapse_resolved(&extents, &strides, groups)
}

fn collapse_resolved(extents: &[u32], strides: &[u32], groups: [u32; 3]) -> Option<[u32; 3]> {
    let axes: Vec<(u32, u32)> = extents
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|(extent, _)| *extent != 1)
        .collect();
    let sizes: Vec<u32> = axes.iter().map(|(extent, _)| *extent).collect();
    let (ranges, used) = group_ranges(&sizes, groups)?;
    if used != axes.len() {
        return None;
    }
    let mut out = [0; 3];
    for (group, (start, end)) in ranges.into_iter().enumerate() {
        if start == end {
            continue;
        }
        if axes[start..end]
            .windows(2)
            .any(|pair| pair[0].1 as u64 != pair[1].1 as u64 * pair[1].0 as u64)
        {
            return None;
        }
        out[group] = axes[end - 1].1;
    }
    Some(out)
}

// Kept out of the public interface; tests cover layout collapsing and the
// platform GEMM adapter covers execution.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_dense_and_unit_axes() {
        assert_eq!(
            collapse_resolved(&[3, 8, 5], &[40, 5, 1], [3, 8, 5]),
            Some([40, 5, 1])
        );
        assert_eq!(
            collapse_resolved(&[16, 1], &[1, 1], [1, 16, 1]),
            Some([0, 1, 0])
        );
    }

    #[test]
    fn rejects_a_gapped_layout() {
        assert_eq!(collapse_resolved(&[4, 4], &[8, 1], [1, 16, 1]), None);
    }
}
