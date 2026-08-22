//! Direct platform-GEMM lowering for CPU contractions.

use std::sync::Arc;

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, BufferDecl, ElementType, KernelIr, LocalDecl, ScalarElement, Stmt,
    StorageView, TileExpr, TileExprKind,
};
use fusor_ir::ir::launch::{ContractSide, Launch, SchedPoint};
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{BinOp, CmpOp, ScalarExpr, ScalarKind};
use fusor_ir::shape::{Dim, Layout};
use fusor_ir::target::LowerCtx;

use super::{
    Binds, OperandSrc, Translate, bin, cmp, default_block, global_lane, grid_for, lit_f32, lit_u32,
    u32_ty,
};

pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    _theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
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
    let platform = side(cx, &binds, a, [batch, m, k])
        .and_then(|bound_a| side(cx, &binds, b, [batch, k, n]).map(|bound_b| (bound_a, bound_b)))
        .and_then(|(bound_a, bound_b)| {
            gemm_name(
                m, n, k, batch, &out, &bound_a, &bound_b, &a.pre, &b.pre, post,
            )
            .ok()
        });
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
    lower_jit(caps, cx, binds, out, [batch, m, n, k], a, b, post, *acc)
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
            if crate::lower::const_operand(cx, operand.src).is_some() {
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
    caps: &Caps,
    cx: &LowerCtx<'_>,
    binds: Binds,
    out: Arc<BufferDecl>,
    [batch, m, n, k]: [u32; 4],
    a: &ContractSide,
    b: &ContractSide,
    post: &ScalarExpr,
    acc: fusor_ir::dtype::Dtype,
) -> Result<KernelIr> {
    let block = default_block(caps);
    let total = u64::from(batch) * u64::from(m) * u64::from(n);
    let total_u32 = u32::try_from(total)
        .map_err(|_| Error::Legality("CPU JIT contraction output exceeds u32 indexing".into()))?;
    let grid = grid_for(total, block);
    let flat = global_lane(block);
    let valid = cmp(CmpOp::Lt, flat.clone(), lit_u32(total_u32));
    let col = bin(BinOp::Rem, flat.clone(), lit_u32(n), u32_ty());
    let rest = bin(BinOp::Div, flat.clone(), lit_u32(n), u32_ty());
    let row = bin(BinOp::Rem, rest.clone(), lit_u32(m), u32_ty());
    let batch_idx = bin(BinOp::Div, rest, lit_u32(m), u32_ty());
    let k_local = Arc::new(LocalDecl::new(u32_ty()));
    let k_idx = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&k_local)), u32_ty());
    let uniforms = binds.buffers.first().cloned();

    let a_srcs = jit_side(cx, &binds, a, [batch, m, k])?;
    let b_srcs = jit_side(cx, &binds, b, [batch, k, n])?;
    let a_value = side_value(
        cx,
        a,
        &a_srcs,
        [batch, m, k],
        [&batch_idx, &row, &k_idx],
        valid.clone(),
        uniforms.clone(),
    )?;
    let b_value = side_value(
        cx,
        b,
        &b_srcs,
        [batch, k, n],
        [&batch_idx, &k_idx, &col],
        valid.clone(),
        uniforms.clone(),
    )?;
    let acc_ty = ElementType::Scalar(super::elem_of(acc)?);
    let a_value = cast_to(a_value, acc_ty);
    let b_value = cast_to(b_value, acc_ty);
    let local = Arc::new(LocalDecl::new(acc_ty));
    let previous = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&local)), acc_ty);
    let product = bin(BinOp::Mul, a_value, b_value, acc_ty);
    let update = bin(BinOp::Add, previous, product, acc_ty);
    let zero = cast_to(lit_f32(0.0), acc_ty);
    let accumulated = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&local)), acc_ty);
    let value = Translate {
        args: &[accumulated],
        coords: &[],
        uniforms,
    }
    .run(post)?;
    let body = vec![
        Stmt::Loop {
            count: Some(lit_u32(k)),
            index: Some(k_local),
            accumulators: vec![Accumulator {
                local,
                init: zero,
                update,
            }],
            body: Vec::new(),
        },
        Stmt::Store {
            dst: StorageView {
                layout: out.layout.clone(),
                buffer: out,
                offset: 0,
            },
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

type JitSide = Vec<(OperandSrc, [u32; 3])>;

fn jit_side(
    cx: &LowerCtx<'_>,
    binds: &Binds,
    side: &ContractSide,
    groups: [u32; 3],
) -> Result<JitSide> {
    side.ops
        .iter()
        .map(|operand| {
            let strides = collapsed_strides(cx, &operand.layout, groups).ok_or_else(|| {
                Error::Legality(format!(
                    "CPU JIT contraction cannot collapse layout {:?}",
                    operand.layout
                ))
            })?;
            Ok((super::operand_src(cx, binds, operand.src)?, strides))
        })
        .collect()
}

fn side_value(
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
        .map(|(source, strides)| source.at(strided_index(indices, *strides), mask.clone()))
        .collect::<Vec<_>>();
    let coords = side_coords(cx, side, groups, indices).ok_or_else(|| {
        Error::Legality("CPU JIT contraction cannot state side coordinates".into())
    })?;
    Translate {
        args: &args,
        coords: &coords,
        uniforms,
    }
    .run(&side.pre)
}

fn strided_index(indices: [&TileExpr; 3], strides: [u32; 3]) -> TileExpr {
    indices
        .into_iter()
        .zip(strides)
        .filter(|(_, stride)| *stride != 0)
        .map(|(index, stride)| {
            if stride == 1 {
                index.clone()
            } else {
                bin(BinOp::Mul, index.clone(), lit_u32(stride), u32_ty())
            }
        })
        .reduce(|left, right| bin(BinOp::Add, left, right, u32_ty()))
        .unwrap_or_else(|| lit_u32(0))
}

fn side_coords(
    cx: &LowerCtx<'_>,
    side: &ContractSide,
    groups: [u32; 3],
    indices: [&TileExpr; 3],
) -> Option<Vec<TileExpr>> {
    if !side.pre.reads_index_of() {
        return Some(Vec::new());
    }
    let extents = super::const_extents(cx, side.primary().layout.shape()).ok()?;
    let mut coords = vec![lit_u32(0); extents.len()];
    let mut axis = 0;
    for (group, wanted) in groups.into_iter().enumerate() {
        let start = axis;
        let mut product = 1u64;
        while product < u64::from(wanted.max(1)) && axis < extents.len() {
            product = product.saturating_mul(u64::from(extents[axis].max(1)));
            axis += 1;
        }
        if product != u64::from(wanted.max(1)) {
            return None;
        }
        let mut rest = indices[group].clone();
        for i in (start..axis).rev() {
            let extent = lit_u32(extents[i].max(1));
            coords[i] = bin(BinOp::Rem, rest.clone(), extent.clone(), u32_ty());
            rest = bin(BinOp::Div, rest, extent, u32_ty());
        }
    }
    Some(coords)
}

fn cast_to(value: TileExpr, to: ElementType) -> TileExpr {
    if value.element() == to {
        value
    } else {
        TileExpr::new(
            TileExprKind::Cast {
                value: value.clone(),
                to,
            },
            to,
        )
    }
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
    if a_pre.structural_hash() != 17_166_440_295_432_690_555
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
        .into_iter()
        .filter(|(extent, _)| *extent != 1)
        .collect();
    let mut out = [0; 3];
    let mut axis = 0;
    for (group, wanted) in groups.into_iter().map(|value| value.max(1)).enumerate() {
        let start = axis;
        let mut product = 1u64;
        while product < wanted as u64 && axis < axes.len() {
            product *= axes[axis].0 as u64;
            axis += 1;
        }
        if product != wanted as u64 {
            return None;
        }
        if axis == start {
            continue;
        }
        if axes[start..axis]
            .windows(2)
            .any(|pair| pair[0].1 as u64 != pair[1].1 as u64 * pair[1].0 as u64)
        {
            return None;
        }
        out[group] = axes[axis - 1].1;
    }
    (axis == axes.len()).then_some(out)
}

// Kept out of the public interface; tests cover layout collapsing and the
// platform GEMM adapter covers execution.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_dense_and_unit_axes() {
        let dense = Layout::contiguous(&[Dim::Const(3), Dim::Const(8), Dim::Const(5)]);
        assert_eq!(
            collapse_resolved(&[3, 8, 5], &[40, 5, 1], [3, 8, 5]),
            Some([40, 5, 1])
        );
        let unit = Layout::contiguous(&[Dim::Const(16), Dim::Const(1)]);
        assert_eq!(
            collapse_resolved(&[16, 1], &[1, 1], [1, 16, 1]),
            Some([0, 1, 0])
        );
        let _ = (dense, unit);
    }

    #[test]
    fn rejects_a_gapped_layout() {
        let layout = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Const(4), Dim::Const(4)],
            &[Dim::Const(8), Dim::Const(1)],
        )
        .unwrap();
        assert_eq!(collapse_resolved(&[4, 4], &[8, 1], [1, 16, 1]), None);
        let _ = layout;
    }
}
