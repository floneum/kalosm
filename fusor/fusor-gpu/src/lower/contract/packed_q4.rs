//! Packed Q4_K dot-product probe. Four lanes cover each 32-value scale group;
//! eight lanes cover a block. Masks retain each nibble's position in its word,
//! with exact powers of two applied to the shared activations instead.
use super::*;
use fusor_ir::dtype::{NumericContract, QFmt, QLayout};
use fusor_ir::ir::kernel::{QuantizedView, TileBinaryOp, TileExprKind, TileUnaryOp};

pub(super) fn packed_q4_source(
    c: &Contract<'_>,
    width: u32,
    p: SgemvParams,
) -> Option<QuantizedView> {
    if c.shape.m != 1
        || c.shape.batch != 1
        || width != 32
        || p.vector != 32
        || c.acc != ScalarElement::F32
        || !matches!(c.b.side.pre.kind(), ScalarKind::Arg(0))
        || !c.shape.k.as_const()?.is_multiple_of(1024)
    {
        return None;
    }
    let [StagedSource::Mem(Source::Quantized(q))] = &c.b.sources[..] else {
        return None;
    };
    if q.fmt != QFmt::Q4K || q.layout != QLayout::Native {
        return None;
    }
    let [ka, na] = &q.data.layout.indexing.groups[..] else {
        return None;
    };
    let [ks] = &ka.sub_axes[..] else {
        return None;
    };
    let [ns] = &na.sub_axes[..] else {
        return None;
    };
    // B's K dimension must be contiguous and each N column block-aligned.
    if ks.stride != 1 || !ns.stride.is_multiple_of(256) {
        return None;
    }
    Some(q.clone())
}

pub(super) fn lower_packed_q4(
    mut ctx: Ctx<'_>,
    c: &Contract<'_>,
    p: SgemvParams,
    q: &QuantizedView,
) -> Result<KernelIr> {
    let blocked_view = (ctx.q4_variant == 3)
        .then(|| ctx.block_storage_view(&q.data, 36))
        .flatten();
    let blocked = blocked_view.is_some();
    let weight_view = blocked_view.unwrap_or_else(|| q.data.clone());
    let b = &ctx.b;
    let f32_ty = ScalarElement::F32.element();
    let shape = c.shape;
    let arithmetic = if shape.n.as_const().is_some_and(|n| n >= 16384) {
        ctx.q4_variant
    } else {
        0
    };
    let rows = c.rows();
    let cps = p.cols / p.subgroups;
    let lanes_per_block = 8;
    let values_per_group = 64 / lanes_per_block;
    let blocks_per_subgroup = 32 / lanes_per_block;
    let passes = (shape.k.as_const().unwrap() / u64::from(256 * blocks_per_subgroup)) as u32;
    let max_splits = (ctx
        .caps
        .limits
        .max_compute_invocations_per_workgroup
        .min(ctx.caps.limits.max_compute_workgroup_size[0])
        / (p.subgroups * 32))
        .max(1);
    let mut splits = crate::flags::for_profile(ctx.caps.execution_profile)
        .packed_q4_splits
        .min(max_splits);
    if crate::flags::for_profile(ctx.caps.execution_profile)
        .packed_q4_split_max_n
        .is_some_and(|limit| shape.n.as_const().is_none_or(|n| n > limit as u64))
        || crate::flags::for_profile(ctx.caps.execution_profile)
            .packed_q4_split_max_k
            .is_some_and(|limit| shape.k.as_const().is_none_or(|k| k > limit as u64))
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
    let block_lane = b.div(lane.clone(), b.u32(lanes_per_block));
    let group_pair = b.rem(b.div(lane.clone(), b.u32(lanes_per_block / 2)), b.u32(2));
    let byte_run = b.mul(
        b.rem(lane.clone(), b.u32(lanes_per_block / 2)),
        b.u32(values_per_group),
    );
    let index = b.local(ScalarElement::U32.element());
    let block_k = b.add(
        b.mul(
            b.load_local(index.clone()),
            b.u32(blocks_per_subgroup * splits),
        ),
        b.add(b.mul(split.clone(), b.u32(blocks_per_subgroup)), block_lane),
    );
    let input_base = b.add(
        b.mul(block_k.clone(), b.u32(256)),
        b.add(b.mul(group_pair.clone(), b.u32(64)), byte_run.clone()),
    );
    let bit = |op, a, v| b.binary(op, a, b.u32(v), NumericContract::RELAXED);
    let and = |a, v| bit(TileBinaryOp::BitAnd, a, v);
    let shr = |a, v| bit(TileBinaryOp::Shr, a, v);
    let or = |a, v| b.binary(TileBinaryOp::BitOr, a, v, NumericContract::RELAXED);
    let comp = |v, component| {
        TileExpr::new(
            TileExprKind::VecComponent {
                vector: v,
                component,
            },
            f32_ty,
        )
    };
    let mut activation = Vec::new();
    let mut sums = Vec::new();
    // Each lane owns a contiguous run in each of four scale groups.
    for offset in [0, 32, 128, 160] {
        let mut vals = Vec::new();
        for i in 0..values_per_group {
            vals.push(c.a.value(
                &ctx,
                &row,
                &b.add(input_base.clone(), b.u32(offset + i)),
                &b.bool(true),
                &batch,
                c.acc,
            )?);
        }
        sums.push(vals.iter().cloned().reduce(|a, v| b.add(a, v)).unwrap());
        activation.push(vals);
    }
    let mut accumulators = Vec::new();
    let mut closes = Vec::new();
    let scratch =
        (splits > 1).then(|| b.tile("packed_q4_partials", f32_ty, &[physical_subgroups * cps]));
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
        let word_base = b.mul(block.clone(), b.u32(36));
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
        let scales = b.unary(
            TileUnaryOp::Unpack2x16Float,
            word(b.u32(0)),
            NumericContract::RELAXED,
        );
        let packed_scale = word(b.u32(1));
        let packed_min = word(b.u32(2));
        let packed_extra = word(b.u32(3));
        let mut result = b.f32(0.0);
        for group in 0..4usize {
            let high_nibble = group % 2 == 1;
            let high_group = group >= 2;
            let scale_shift = b.add(
                b.mul(group_pair.clone(), b.u32(16)),
                b.u32((group as u32 % 2) * 8),
            );
            let extract = |v| {
                and(
                    b.binary(
                        TileBinaryOp::Shr,
                        v,
                        scale_shift.clone(),
                        NumericContract::RELAXED,
                    ),
                    255,
                )
            };
            let s = extract(packed_scale.clone());
            let m = extract(packed_min.clone());
            let extra = extract(packed_extra.clone());
            let gs = if high_group {
                or(and(extra.clone(), 15), shr(and(s, 192), 2))
            } else {
                and(s, 63)
            };
            let gm = if high_group {
                or(shr(extra, 4), shr(and(m, 192), 2))
            } else {
                and(m, 63)
            };
            let mut dots = Vec::new();
            for chunk in 0..values_per_group / 4 {
                let off = b.add(
                    b.u32(4 + (group as u32 / 2) * 16 + chunk),
                    b.add(
                        b.mul(group_pair.clone(), b.u32(8)),
                        b.div(byte_run.clone(), b.u32(4)),
                    ),
                );
                let packed = word(off);
                if crate::flags::for_profile(ctx.caps.execution_profile).q4_half_unpack
                    || arithmetic == 2
                {
                    // Encode two integer nibbles as normal half mantissas.
                    // unpack(1 + q/1024) followed by FMA gives exactly q in
                    // F32. Activations and accumulation retain F32 precision.
                    let bits = shr(packed, if high_nibble { 4 } else { 0 });
                    let pairs: Vec<_> = [0, 8]
                        .into_iter()
                        .map(|shift| {
                            if arithmetic == 2 {
                                // 16+q is exactly representable as a normal
                                // half. The dot's extra 16*sum(x) is corrected
                                // in F32 below, together with the block offset.
                                let positioned = if shift == 0 {
                                    bit(TileBinaryOp::Shl, bits.clone(), 6)
                                } else {
                                    shr(bits.clone(), 2)
                                };
                                let encoded = bit(
                                    TileBinaryOp::BitOr,
                                    and(positioned, 0x03c003c0),
                                    0x4c004c00,
                                );
                                return b.unary(
                                    TileUnaryOp::Unpack2x16Float,
                                    encoded,
                                    NumericContract::RELAXED,
                                );
                            }
                            let encoded = bit(
                                TileBinaryOp::BitOr,
                                and(shr(bits.clone(), shift), 0x000f000f),
                                0x3c003c00,
                            );
                            let values = b.unary(
                                TileUnaryOp::Unpack2x16Float,
                                encoded,
                                NumericContract::RELAXED,
                            );
                            b.fma(
                                values,
                                b.vector(ScalarElement::F32, vec![b.f32(1024.0); 2]),
                                b.vector(ScalarElement::F32, vec![b.f32(-1024.0); 2]),
                            )
                        })
                        .collect();
                    let weights = (0..4)
                        .map(|byte| comp(pairs[byte % 2].clone(), (byte / 2) as u32))
                        .collect();
                    let xs = activation[group][chunk as usize * 4..chunk as usize * 4 + 4].to_vec();
                    dots.push(b.dot(
                        b.vector(ScalarElement::F32, xs),
                        b.vector(ScalarElement::F32, weights),
                    ));
                    continue;
                }
                let mut weights = Vec::new();
                let mut xs = Vec::new();
                for byte in 0..4u32 {
                    let shift = byte * 8 + if high_nibble { 4 } else { 0 };
                    weights.push(b.cast(and(packed.clone(), 15u32 << shift), f32_ty));
                    xs.push(b.mul(
                        activation[group][chunk as usize * 4 + byte as usize].clone(),
                        b.f32(2f32.powi(-(shift as i32))),
                    ));
                }
                dots.push(b.dot(
                    b.vector(ScalarElement::F32, xs),
                    b.vector(ScalarElement::F32, weights),
                ));
            }
            let dot = dots.into_iter().reduce(|a, v| b.add(a, v)).unwrap();
            let scale = b.mul(b.cast(gs, f32_ty), comp(scales.clone(), 0));
            let minimum = b.mul(b.cast(gm, f32_ty), comp(scales.clone(), 1));
            result = if arithmetic == 2 {
                b.fma(
                    dot,
                    scale.clone(),
                    b.fma(
                        b.sub(b.f32(0.), sums[group].clone()),
                        b.fma(b.f32(16.), scale, minimum),
                        result,
                    ),
                )
            } else if arithmetic == 1 {
                b.fma(
                    dot,
                    scale,
                    b.fma(b.sub(b.f32(0.), sums[group].clone()), minimum, result),
                )
            } else {
                b.add(
                    result,
                    b.sub(b.mul(dot, scale), b.mul(sums[group].clone(), minimum)),
                )
            };
        }
        let local = b.local(f32_ty);
        accumulators.push(Accumulator {
            local: local.clone(),
            init: b.f32(0.0),
            update: b.add(b.load_local(local.clone()), result),
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
    Ok(ctx.finish(
        if splits > 1 {
            "sgemv_packed_q4_split"
        } else {
            "sgemv_packed_q4"
        },
        grid,
        physical_subgroups * 32,
        body,
    ))
}
