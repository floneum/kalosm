use std::{cell::RefCell, sync::Arc};

use fusor_gguf::blocks::BlockDecodeArgs;
use fusor_ir::Result;
use fusor_ir::dtype::{NumericContract, QFmt, QLayout};
use fusor_ir::ir::kernel::{
    BufferAccess, BufferDecl, Builtin, LocalDecl, MemoryLevel, ScalarElement, StorageView,
    TileBinaryOp, TileExpr, TileExprKind, TileLayout, TileLiteral, WorkgroupAxis, simplify_index,
};
use fusor_ir::ir::launch::SgemvParams;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DecodeKey {
    pub fmt: QFmt,
    pub layout: QLayout,
    pub k: u32,
    pub reduction: u32,
    pub elements: u32,
    pub m: u32,
    pub n: u32,
    pub offset: u32,
    pub params: SgemvParams,
    pub width: u32,
}

thread_local! {
    static DECODE_CACHE: RefCell<FxHashMap<DecodeKey, (u64, u64)>> = RefCell::new(FxHashMap::default());
}

fn u32_lit(v: u32) -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::U32(v)),
        ScalarElement::U32.element(),
    )
}

fn binary(op: TileBinaryOp, left: TileExpr, right: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left,
            right,
            numeric: NumericContract::RELAXED,
        },
        ScalarElement::U32.element(),
    )
}

/// Scalar instructions and word loads in one lane's SGEMV reduction. Full
/// passes and contiguous tails use the emitter's index simplification and CSE.
pub(crate) fn decode_window(key: DecodeKey) -> Result<(u64, u64)> {
    if let Some(work) = DECODE_CACHE.with(|cache| cache.borrow().get(&key).copied()) {
        return Ok(work);
    }
    let work = decode_window_uncached(key)?;
    DECODE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= 512 {
            cache.clear();
        }
        cache.insert(key, work);
    });
    Ok(work)
}

fn decode_window_uncached(key: DecodeKey) -> Result<(u64, u64)> {
    use fusor_ir::scalar::BinOp::{Add, Div, Min, Mul, Rem};
    let p = key.params;
    let variable = |v| TileExpr::new(TileExprKind::Builtin(v), ScalarElement::U32.element());
    let boolean = |v| {
        TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(v)),
            ScalarElement::Bool.element(),
        )
    };
    let less = |left, right| {
        TileExpr::new(
            TileExprKind::Compare {
                op: fusor_ir::scalar::CmpOp::Lt,
                left,
                right,
            },
            ScalarElement::Bool.element(),
        )
    };
    let wg = variable(Builtin::ProgramId(WorkgroupAxis::X));
    let groups = key.n.div_ceil(p.cols.max(1));
    let column_group = binary(Rem, wg.clone(), u32_lit(groups));
    let batch_base = binary(
        Mul,
        binary(Div, binary(Div, wg, u32_lit(groups)), u32_lit(key.m)),
        u32_lit(key.reduction),
    );
    let (lane, columns, column_base) = if p.cols > 1 {
        let columns = p.cols / p.subgroups;
        (
            variable(Builtin::SubgroupLane),
            columns,
            binary(
                Add,
                binary(Mul, column_group, u32_lit(p.cols)),
                binary(Mul, variable(Builtin::SubgroupId), u32_lit(columns)),
            ),
        )
    } else {
        (variable(Builtin::Lane), 1, column_group)
    };
    let storage = TileLayout::contiguous(MemoryLevel::Storage, &[1]);
    let src = StorageView {
        buffer: Arc::new(BufferDecl {
            binding: 0,
            element: ScalarElement::U32.element(),
            layout: storage.clone(),
            access: BufferAccess::Read,
        }),
        offset: 0,
        layout: storage,
    };
    let pass_work = |step, vector, contiguous, masked| -> Result<(u64, u64)> {
        let local = if contiguous || p.cols <= 1 || p.parts <= 1 {
            binary(Mul, lane.clone(), u32_lit(vector))
        } else {
            let run = p.run();
            let lanes_per_gap = u32_lit(p.gap / run);
            binary(
                Add,
                binary(
                    Mul,
                    binary(Div, lane.clone(), lanes_per_gap.clone()),
                    u32_lit(p.gap * p.parts),
                ),
                binary(Mul, binary(Rem, lane.clone(), lanes_per_gap), u32_lit(run)),
            )
        };
        let lane_base = binary(Add, step, local);
        let mut seen = FxHashSet::default();
        let mut work = (0, 0);
        for column in 0..columns {
            let column = binary(Add, column_base.clone(), u32_lit(column));
            let row = binary(Mul, column.clone(), u32_lit(key.k));
            for v in 0..vector {
                let off = if contiguous || p.cols <= 1 || p.parts <= 1 {
                    v
                } else {
                    v / p.run() * p.gap + v % p.run()
                };
                let k = binary(Add, lane_base.clone(), u32_lit(off));
                let mut mask = if masked {
                    less(k.clone(), u32_lit(key.reduction))
                } else {
                    boolean(true)
                };
                if !key.n.is_multiple_of(p.cols.max(1)) {
                    let col_ok = less(column.clone(), u32_lit(key.n));
                    mask = if masked {
                        TileExpr::new(
                            TileExprKind::Binary {
                                op: fusor_ir::scalar::BinOp::LogicalAnd,
                                left: mask,
                                right: col_ok,
                                numeric: NumericContract::RELAXED,
                            },
                            ScalarElement::Bool.element(),
                        )
                    } else {
                        col_ok
                    };
                }
                let mut flat = binary(
                    Add,
                    binary(Add, row.clone(), binary(Add, batch_base.clone(), k)),
                    u32_lit(key.offset),
                );
                if !mask.is_constant_true()
                    && let Some(last) = key.elements.checked_sub(1)
                {
                    flat = binary(Min, flat, u32_lit(last));
                }
                let args = BlockDecodeArgs {
                    src: &src,
                    layout: key.layout,
                    k_base: u32_lit(0),
                    col: flat,
                    mask,
                    fill: TileExpr::new(
                        TileExprKind::Literal(TileLiteral::F32(0)),
                        ScalarElement::F32.element(),
                    ),
                };
                let decoded = (fusor_gguf::block_spec(key.fmt, key.layout).decode.emit)(&args)?;
                count(&simplify_index(&decoded), &mut seen, &mut work);
            }
        }
        Ok(work)
    };
    let pass = key.width * p.vector.max(1);
    let mut work = (0, 0);
    if key.reduction >= pass {
        let index = Arc::new(LocalDecl::new(ScalarElement::U32.element()));
        let step = binary(
            Mul,
            TileExpr::new(TileExprKind::LoadLocal(index), ScalarElement::U32.element()),
            u32_lit(pass),
        );
        let full = pass_work(step, p.vector.max(1), false, false)?;
        let repeats = u64::from(key.reduction / pass);
        work = (full.0 * repeats, full.1 * repeats);
    }
    let rem = key.reduction % pass;
    let mut at = key.reduction - rem;
    for (vector, masked) in [
        (rem / key.width, false),
        (u32::from(!rem.is_multiple_of(key.width)), true),
    ] {
        if vector == 0 {
            continue;
        }
        let tail = pass_work(u32_lit(at), vector, true, masked)?;
        work.0 += tail.0;
        work.1 += tail.1;
        at += vector * key.width;
    }
    Ok(work)
}

fn count(expr: &TileExpr, seen: &mut FxHashSet<TileExpr>, work: &mut (u64, u64)) {
    if !seen.insert(expr.clone()) {
        return;
    }
    match expr.kind() {
        TileExprKind::Unary { .. }
        | TileExprKind::Binary { .. }
        | TileExprKind::Compare { .. }
        | TileExprKind::Select { .. }
        | TileExprKind::Cast { .. } => work.0 += 1,
        TileExprKind::Load { .. } => work.1 += 1,
        _ => {}
    }
    expr.kind().visit_children(&mut |e| count(e, seen, work));
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};
    use fusor_ir::dtype::Dtype;
    use fusor_ir::egraph::EGraph;
    use fusor_ir::extract::Extraction;
    use fusor_ir::ir::Op;
    use fusor_ir::ir::launch::{
        AccessPlan, ContractSide, Family, IndexSpace, Launch, Operand, SchedPoint, ScheduleDomain,
        SgemvDomain,
    };
    use fusor_ir::ir::logical::{BufferId, LeafKind, Logical};
    use fusor_ir::scalar::ScalarExpr;
    use fusor_ir::shape::{Dim, Layout};

    #[test]
    fn aligned_q6_scales_trade_payload_bytes_for_less_decode_work() -> Result<()> {
        let (fmt, n, k) = (QFmt::Q6K, 4, 14336);
        let params = SgemvParams {
            vector: 32,
            subgroups: 2,
            cols: 4,
            parts: 4,
            gap: 32,
        };
        let key = DecodeKey {
            fmt,
            layout: QLayout::Native,
            k,
            reduction: k,
            elements: k * n,
            m: 1,
            n,
            offset: 0,
            params,
            width: 32,
        };
        let native_work = decode_window(key)?;
        let aligned_work = decode_window(DecodeKey {
            layout: QLayout::F32Scales,
            ..key
        })?;
        assert!(native_work.0 > aligned_work.0);
        assert!(native_work.1 > aligned_work.1);

        let mut native = vec![0u8; (n * k / 256 * fmt.block_bytes(QLayout::Native)) as usize];
        for (block, bytes) in native.as_chunks_mut::<210>().0.iter_mut().enumerate() {
            for (i, byte) in bytes[..208].iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(29).wrapping_add(block as u8);
            }
            let scale = 0x3000u16 + ((block % 7) as u16) * 32;
            bytes[208..].copy_from_slice(&scale.to_le_bytes());
        }
        let mut aligned = Vec::new();
        fusor_gguf::repack(
            fmt,
            QLayout::Native,
            QLayout::F32Scales,
            &native,
            &mut aligned,
        )?;
        assert!(aligned.len() > native.len());
        for (a, b) in native
            .as_chunks::<210>()
            .0
            .iter()
            .zip(aligned.as_chunks::<212>().0)
        {
            let (mut av, mut bv) = ([0.0f32; 256], [0.0f32; 256]);
            fusor_gguf::blocks::cpu_dequantize_block(fmt, QLayout::Native, a, &mut av);
            fusor_gguf::blocks::cpu_dequantize_block(fmt, QLayout::F32Scales, b, &mut bv);
            assert_eq!(av.map(f32::to_bits), bv.map(f32::to_bits));
        }

        let caps = Caps {
            kind: DeviceKind::Gpu,
            name: "quantized layout pricing".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: false,
            bf16: false,
            coop: Default::default(),
            atomic_f32: false,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        };
        let arena = Arc::new(fusor_tile::Planner::new());
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(arena.clone()));
        let (n, k) = (Dim::Const(n.into()), Dim::Const(k.into()));
        let x = graph.add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(0),
            dtype: Dtype::F32,
            shape: [Dim::ONE, k].into_iter().collect(),
        })))?;
        let mut variants = Vec::new();
        for params in [
            params,
            SgemvParams {
                cols: 1,
                parts: 1,
                gap: 0,
                ..params
            },
        ] {
            for split in [1, 2, 4] {
                let batch = Dim::Const(split);
                let chunk = Dim::Const(k.as_const().unwrap() / split);
                let mut pair = Vec::new();
                for layout in [QLayout::Native, QLayout::F32Scales] {
                    let weight = graph.add(Op::Logical(Logical::Leaf(LeafKind::Quantized {
                        name: BufferId(1),
                        fmt,
                        layout,
                        shape: [n, k].into_iter().collect(),
                    })))?;
                    let side = |src, layout| {
                        ContractSide::one(
                            ScalarExpr::arg(0, Dtype::F32),
                            Operand {
                                src,
                                layout,
                                access: AccessPlan::Alias,
                            },
                        )
                    };
                    let root = graph.add(Op::Launch(Launch::Contract {
                        output: IndexSpace::new(if split == 1 {
                            vec![Dim::ONE, n]
                        } else {
                            vec![batch, Dim::ONE, n]
                        }),
                        m: Dim::ONE,
                        n,
                        k: chunk,
                        batch,
                        family: Family::Sgemv,
                        a: side(
                            x,
                            if split == 1 {
                                Layout::contiguous(&[Dim::ONE, chunk])
                            } else {
                                Layout::contiguous(&[batch, Dim::ONE, chunk])
                            },
                        ),
                        b: side(
                            weight,
                            if split == 1 {
                                Layout::from_parts(Dim::Const(0), &[chunk, n], &[Dim::ONE, k])?
                            } else {
                                Layout::from_parts(
                                    Dim::Const(0),
                                    &[batch, chunk, n],
                                    &[chunk, Dim::ONE, k],
                                )?
                            },
                        ),
                        acc: Dtype::F32,
                        post: ScalarExpr::arg(0, Dtype::F32),
                        sched: ScheduleDomain::Sgemv(
                            SgemvDomain {
                                params: [params].into_iter().collect(),
                            }
                            .into(),
                        ),
                    }))?;
                    let mut packed = graph.node(root).op.clone();
                    let Op::Launch(Launch::Contract { b, .. }) = &mut packed else {
                        unreachable!()
                    };
                    b.ops[0].access = AccessPlan::Pack {
                        into: Layout::contiguous(b.ops[0].layout.shape()),
                    };
                    let packed = graph.add(packed)?;
                    graph.union(root, packed)?;
                    pair.push((weight, [root, packed]));
                }
                graph.union(pair[0].1[0], pair[1].1[0])?;
                variants.push((params, pair));
            }
        }
        let cost = crate::Roofline::new(crate::facts::seed_facts_gpu(&caps));
        let search = crate::LocalSearch::new(arena.clone(), caps);
        let mut cache = crate::realize::NodeCache::default();
        for (params, pair) in variants {
            let mut seen = Vec::new();
            for (index, access) in [(0, 0), (1, 0), (0, 0), (0, 1), (1, 1)] {
                let (weight, roots) = pair[index];
                let root = roots[access];
                let mut ex = Extraction {
                    sigma: [x, weight, root]
                        .into_iter()
                        .map(|id| (graph.class_of(id), id))
                        .collect(),
                    m: Default::default(),
                    theta: [(root, SchedPoint::Sgemv(params))].into_iter().collect(),
                };
                let plan = search.replan(&graph, &[root], &mut ex, &cost, &mut cache)?;
                let result = crate::realize::realize_with(
                    &graph,
                    &[root],
                    &ex,
                    &cost,
                    arena.as_ref(),
                    &mut cache,
                )?;
                assert_eq!(result.components.len(), 1);
                let component = &result.components[0];
                let read = component
                    .external
                    .iter()
                    .position(|id| *id == weight)
                    .unwrap();
                assert_eq!(
                    component.reads[read].0,
                    [native.len(), aligned.len()][index] as u64
                );
                seen.push((component.work, plan.cost));
            }
            assert!(seen[0].0.index_ops > seen[1].0.index_ops);
            assert_eq!(seen[0], seen[2]);
            assert_eq!(seen[0], seen[3]);
            assert_eq!(seen[1], seen[4]);
        }
        Ok(())
    }
}
