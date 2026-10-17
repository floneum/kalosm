//! Coalesced Q6 payloads: each lane reads one low-bit word and its high-bit
//! plane, retaining both activations across the output columns it owns.
use super::*;
use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{QuantizedView, TileBinaryOp};

pub(super) fn lower(
    mut ctx: Ctx<'_>,
    c: &Contract<'_>,
    p: SgemvParams,
    q: &QuantizedView,
) -> Result<KernelIr> {
    let blocked_view = (ctx.q4_variant == 3)
        .then(|| ctx.block_storage_view(&q.data, 53))
        .flatten();
    let blocked = blocked_view.is_some();
    let weight_view = blocked_view.unwrap_or_else(|| q.data.clone());
    let b = &ctx.b;
    let f32_ty = ScalarElement::F32.element();
    let shape = c.shape;
    let rows = c.rows();
    let cps = p.cols / p.subgroups;
    let passes = (shape.k.as_const().unwrap() / 256) as u32;
    let max_splits = (ctx
        .caps
        .limits
        .max_compute_invocations_per_workgroup
        .min(ctx.caps.limits.max_compute_workgroup_size[0])
        / (p.subgroups * 32))
        .max(1);
    let mut splits = crate::flags::for_profile(ctx.caps.execution_profile)
        .q6_splits
        .min(max_splits);
    if crate::flags::for_profile(ctx.caps.execution_profile)
        .q6_split_max_n
        .is_some_and(|limit| shape.n.as_const().is_none_or(|n| n > limit as u64))
    {
        splits = 1;
    }
    while !passes.is_multiple_of(splits) {
        splits /= 2;
    }
    let physical_subgroups = p.subgroups * splits;
    let grid = crate::lower::tiled_grid_for(
        &IndexSpace::new([Dim::Const(u64::from(rows)), shape.n]),
        1,
        p.cols,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let wg = column_workgroup_index(&ctx, grid, rows, shape.n, p.cols)?;
    let (row, col_group) = row_col(&ctx, wg, ceil_extent(&ctx, shape.n, p.cols)?, rows);
    let (batch, b_row_base) = c.batch_of(&ctx, &row)?;
    let lane = b.builtin(Builtin::SubgroupLane);
    let physical_sg = b.builtin(Builtin::SubgroupId);
    let sg = b.div(physical_sg.clone(), b.u32(splits));
    let split = b.rem(physical_sg.clone(), b.u32(splits));
    let col_base = b.add(
        b.mul(col_group, b.u32(p.cols)),
        b.mul(sg.clone(), b.u32(cps)),
    );
    let index = b.local(ScalarElement::U32.element());
    let block_k = b.add(
        b.mul(b.load_local(index.clone()), b.u32(splits)),
        split.clone(),
    );
    let half = b.div(lane.clone(), b.u32(16));
    let within = b.rem(lane.clone(), b.u32(16));
    let input_base = b.add(
        b.mul(block_k.clone(), b.u32(256)),
        b.add(
            b.mul(half.clone(), b.u32(128)),
            b.mul(within.clone(), b.u32(4)),
        ),
    );
    let scale_group = b.add(
        b.mul(half.clone(), b.u32(8)),
        b.div(within.clone(), b.u32(4)),
    );
    let qh_addr = b.add(
        b.u32(32),
        b.add(
            b.mul(half.clone(), b.u32(8)),
            b.rem(within.clone(), b.u32(8)),
        ),
    );
    let high_shift = b.mul(b.div(within.clone(), b.u32(8)), b.u32(2));
    let bit = |op, a, v| b.binary(op, a, b.u32(v), NumericContract::RELAXED);
    let and = |a, v| bit(TileBinaryOp::BitAnd, a, v);
    let shr = |a, s| b.binary(TileBinaryOp::Shr, a, s, NumericContract::RELAXED);
    let mut activation = Vec::new();
    for offset in [0, 64] {
        let mut xs = Vec::new();
        for i in 0..4 {
            xs.push(c.a.value(
                &ctx,
                &row,
                &b.add(input_base.clone(), b.u32(offset + i)),
                &b.bool(true),
                &batch,
                c.acc,
            )?);
        }
        activation.push(xs);
    }
    let mut accumulators = Vec::new();
    let mut closes = Vec::new();
    let scratch =
        (splits > 1).then(|| b.tile("q6_coalesced_partials", f32_ty, &[physical_subgroups * cps]));
    let mut stores = Vec::new();
    let out = ctx.linear_view(ctx.output()?)?;
    let n = ctx.dim_expr(shape.n)?;
    let stride_blocks = q.data.layout.indexing.groups[1].sub_axes[0].stride / 256;
    for j in 0..cps {
        let col = b.add(col_base.clone(), b.u32(j));
        let ok = if shape
            .n
            .as_const()
            .is_some_and(|n| n.is_multiple_of(u64::from(p.cols)))
        {
            b.bool(true)
        } else {
            b.lt(col.clone(), n.clone())
        };
        let block = b.add(
            b.add(block_k.clone(), b.div(b_row_base.clone(), b.u32(256))),
            b.mul(col.clone(), b.u32(stride_blocks)),
        );
        let word_base = b.mul(block.clone(), b.u32(53));
        let word = |off| {
            b.load(
                Source::Storage(weight_view.clone()),
                if blocked {
                    Addr::Rc2 {
                        row: block.clone(),
                        col: off,
                    }
                } else {
                    Addr::Linear(b.add(word_base.clone(), off))
                },
                ok.clone(),
                b.u32(0),
            )
        };
        let d = b.bitcast(word(b.u32(52)), f32_ty);
        let local = b.local(f32_ty);
        let mut result = b.load_local(local.clone());
        for (part, xs) in activation.iter().enumerate() {
            let low = word(b.add(b.mul(half.clone(), b.u32(16)), within.clone()));
            let high = word(qh_addr.clone());
            let group = b.add(scale_group.clone(), b.u32(part as u32 * 4));
            let scale_word = word(b.add(b.u32(48), b.div(group.clone(), b.u32(4))));
            let scale_byte = and(
                shr(scale_word, b.mul(b.rem(group, b.u32(4)), b.u32(8))),
                255,
            );
            let signed = b.bitcast(
                bit(TileBinaryOp::Shl, scale_byte, 24),
                ScalarElement::I32.element(),
            );
            let scale = b.cast(shr(signed, b.u32(24)), f32_ty);
            let lo = and(shr(low.clone(), b.u32(part as u32 * 4)), 0x0f0f0f0f);
            let hi = bit(
                TileBinaryOp::Shl,
                and(
                    shr(
                        high.clone(),
                        b.add(high_shift.clone(), b.u32(part as u32 * 4)),
                    ),
                    0x03030303,
                ),
                4,
            );
            let packed = b.binary(TileBinaryOp::BitOr, lo, hi, NumericContract::RELAXED);
            if crate::flags::for_profile(ctx.caps.execution_profile).q6_float_unpack {
                let weights = (0..4u32)
                    .map(|byte| {
                        let q = and(shr(packed.clone(), b.u32(byte * 8)), 63);
                        let encoded = bit(TileBinaryOp::BitOr, q, 0x4b000000);
                        b.sub(b.bitcast(encoded, f32_ty), b.f32(8388640.0))
                    })
                    .collect();
                let dot = b.dot(
                    b.vector(ScalarElement::F32, xs.clone()),
                    b.vector(ScalarElement::F32, weights),
                );
                result = b.fma(dot, b.mul(scale, d.clone()), result);
                continue;
            }
            let mut scaled_x = Vec::new();
            let mut weights = Vec::new();
            for (byte, x) in xs.iter().enumerate() {
                let shift = byte as u32 * 8;
                weights.push(b.cast(and(packed.clone(), 63u32 << shift), f32_ty));
                scaled_x.push(b.mul(x.clone(), b.f32(2f32.powi(-(shift as i32)))));
            }
            let dot = b.dot(
                b.vector(ScalarElement::F32, scaled_x),
                b.vector(ScalarElement::F32, weights),
            );
            let sum_x = xs.iter().cloned().reduce(|a, v| b.add(a, v)).unwrap();
            let centered = b.sub(dot, b.mul(b.f32(32.0), sum_x));
            result = b.fma(centered, b.mul(scale, d.clone()), result);
        }
        accumulators.push(Accumulator {
            local: local.clone(),
            init: b.f32(0.0),
            update: result,
        });
        let total = b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, b.load_local(local));
        let total = if let Some(scratch) = &scratch {
            // Evaluate the subgroup collective with all lanes, then publish
            // one partial. Every subgroup reaches the later workgroup barrier.
            let reduced = b.local(f32_ty);
            closes.push(Stmt::StoreLocal {
                dst: reduced.clone(),
                value: total,
            });
            closes.push(Stmt::If {
                condition: b.eq(lane.clone(), b.u32(0)),
                accept: vec![Stmt::StoreTile {
                    dst: scratch.clone(),
                    index: b.add(b.u32(j * physical_subgroups), physical_sg.clone()),
                    value: b.load_local(reduced),
                }],
                reject: Vec::new(),
            });
            (0..splits)
                .map(|i| {
                    b.load_tile(
                        scratch.clone(),
                        b.add(
                            b.u32(j * physical_subgroups + i),
                            b.mul(sg.clone(), b.u32(splits)),
                        ),
                    )
                })
                .reduce(|a, v| b.add(a, v))
                .unwrap()
        } else {
            total
        };
        let addr = Addr::Linear(b.add(b.mul(row.clone(), n.clone()), col.clone()));
        stores.push(c.store_post(
            &ctx,
            &out,
            total,
            row.clone(),
            col,
            addr,
            b.and(
                ok,
                b.and(b.eq(lane.clone(), b.u32(0)), b.eq(split.clone(), b.u32(0))),
            ),
        )?);
    }
    let mut body = vec![Stmt::Loop {
        count: Some(b.u32(passes / splits)),
        index: Some(index),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(closes);
    if splits > 1 {
        body.push(Stmt::Barrier);
    }
    body.extend(stores);
    let name = match splits {
        2 => "sgemv_q6_coalesced_split2",
        4 => "sgemv_q6_coalesced_split4",
        8 => "sgemv_q6_coalesced_split8",
        _ => "sgemv_q6_coalesced",
    };
    Ok(ctx.finish(name, grid, physical_subgroups * 32, body))
}
