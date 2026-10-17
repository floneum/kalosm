//! One workgroup per query head: subgroup QK dots, stable softmax in shared
//! storage, then column-coalesced PV. All arithmetic and cache storage are F32.
use super::{Ctx, distribute_workgroups};
use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, KernelIr, ReduceKind, ScalarElement, Stmt, Tile, TileBinaryOp,
    TileExpr, TileReduceOp, TileUnaryOp,
};
use fusor_ir::ir::launch::Launch;
use fusor_ir::{Error, Result};

mod online;

pub(super) fn lower_merge(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    let Launch::DecodeAttentionMerge {
        output,
        splits,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected attention merge".into()));
    };
    let b = &ctx.b;
    let f = ScalarElement::F32.element();
    let u = ScalarElement::U32.element();
    let heads = output.iterations().unwrap() as u32 / 128;
    let group = ctx.linear_workgroup();
    let lane = b.builtin(Builtin::Lane);
    let grid = distribute_workgroups(heads, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let total = u64::from(heads) * u64::from(*splits) * 130;
    let it = b.local(u);
    let base = b.mul(
        b.add(
            b.mul(group.clone(), b.u32(*splits)),
            b.load_local(it.clone()),
        ),
        b.u32(130),
    );
    let load = |offset| ctx.load_mapped(&ops[0], b.add(base.clone(), offset), total);
    let maximum = b.local(f);
    let mut body = vec![Stmt::Loop {
        count: Some(b.u32(*splits)),
        index: Some(it.clone()),
        accumulators: vec![Accumulator {
            local: maximum.clone(),
            init: b.f32(-f32::MAX),
            update: b.binary(
                TileBinaryOp::Max,
                b.load_local(maximum.clone()),
                load(b.u32(128))?,
                NumericContract::RELAXED,
            ),
        }],
        body: Vec::new(),
    }];
    let weight = b.local(f);
    let numerator = b.local(f);
    let denominator = b.local(f);
    body.push(Stmt::Loop {
        count: Some(b.u32(*splits)),
        index: Some(it),
        accumulators: vec![
            Accumulator {
                local: numerator.clone(),
                init: b.f32(0.0),
                update: b.fma(
                    b.load_local(weight.clone()),
                    load(lane.clone())?,
                    b.load_local(numerator.clone()),
                ),
            },
            Accumulator {
                local: denominator.clone(),
                init: b.f32(0.0),
                update: b.fma(
                    b.load_local(weight.clone()),
                    load(b.u32(129))?,
                    b.load_local(denominator.clone()),
                ),
            },
        ],
        body: vec![Stmt::StoreLocal {
            dst: weight,
            value: b.unary(
                TileUnaryOp::Exp,
                b.sub(load(b.u32(128))?, b.load_local(maximum)),
                NumericContract::RELAXED,
            ),
        }],
    });
    body.push(Stmt::Store {
        dst: ctx.linear_view(ctx.output()?)?,
        addr: Addr::Linear(b.add(b.mul(group.clone(), b.u32(128)), lane)),
        value: b.div(b.load_local(numerator), b.load_local(denominator)),
        mask: b.bool(true),
    });
    let body = vec![Stmt::If {
        condition: b.lt(group, b.u32(heads)),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish("decode_attention_merge", grid, 128, body))
}

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    if crate::flags::for_profile(ctx.caps.execution_profile).attention_online
        && matches!(
            op,
            Launch::DecodeAttention {
                heads_per_group: 1,
                ..
            }
        )
    {
        return online::lower(ctx, op);
    }
    let Launch::DecodeAttention {
        output,
        kv_heads,
        keys,
        block,
        heads_per_group,
        chunk,
        key_major,
        scale,
        prefix,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected decode attention".into()));
    };
    let b = &ctx.b;
    let block = *block;
    let heads_per_group = *heads_per_group;
    let subgroups = block / 32;
    let f = ScalarElement::F32.element();
    let u = ScalarElement::U32.element();
    let heads = output.dims[1].as_const().unwrap() as u32;
    let batches = output.dims[0].as_const().unwrap() as u32;
    let capacity = if *chunk == 0 { *keys } else { *chunk };
    let splits = keys.div_ceil(capacity);
    let count = batches * heads * splits / heads_per_group;
    let physical_block = block * heads_per_group;
    let head_slot = if heads_per_group == 1 {
        b.u32(0)
    } else {
        b.div(b.builtin(Builtin::Lane), b.u32(block))
    };
    let grid = distribute_workgroups(count, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let group = ctx.linear_workgroup();
    let (head_row, split) = if *key_major {
        let (split, head) = b.divrem(group.clone(), b.u32(batches * heads / heads_per_group));
        (head, split)
    } else {
        b.divrem(group.clone(), b.u32(splits))
    };
    let head_row = if heads_per_group == 1 {
        head_row
    } else {
        b.add(b.mul(head_row, b.u32(heads_per_group)), head_slot.clone())
    };
    let output_group = b.add(b.mul(head_row.clone(), b.u32(splits)), split.clone());
    let start = b.mul(split, b.u32(capacity));
    let (batch, head) = b.divrem(head_row.clone(), b.u32(heads));
    let kv_row = b.add(
        b.mul(batch, b.u32(*kv_heads)),
        b.div(head, b.u32(heads / kv_heads)),
    );
    let kv_base = b.mul(kv_row, b.u32(keys * 128));
    let lane = if heads_per_group == 1 {
        b.builtin(Builtin::Lane)
    } else {
        b.rem(b.builtin(Builtin::Lane), b.u32(block))
    };
    let sl = b.builtin(Builtin::SubgroupLane);
    let sg = if heads_per_group == 1 {
        b.builtin(Builtin::SubgroupId)
    } else {
        b.rem(b.builtin(Builtin::SubgroupId), b.u32(subgroups))
    };
    let scale = ctx.eval_scalar(scale, &[], &[])?;
    let prefix = ctx.eval_scalar(prefix, &[], &[])?;
    let prefix = b.min(b.cast(prefix, u), b.u32(*keys));
    let length = b.min(
        b.sub(
            b.binary(
                TileBinaryOp::Max,
                prefix,
                start.clone(),
                NumericContract::RELAXED,
            ),
            start.clone(),
        ),
        b.u32(capacity),
    );
    let passes = |width: u32| b.div(b.add(length.clone(), b.u32(width - 1)), b.u32(width));
    let scores = b.tile("attention_scores", f, &[capacity * heads_per_group]);
    let scratch = b.tile("attention_reduce", f, &[physical_block]);
    let partials = b.tile("attention_partial", f, &[physical_block * 4]);
    let score_at = |i| {
        if heads_per_group == 1 {
            i
        } else {
            b.add(b.mul(head_slot.clone(), b.u32(capacity)), i)
        }
    };
    let partial_at = |i| {
        if heads_per_group == 1 {
            i
        } else {
            b.add(b.mul(head_slot.clone(), b.u32(block * 4)), i)
        }
    };
    let mut body = Vec::new();
    let mut q = Vec::new();
    for c in 0..4 {
        let at = b.add(
            b.mul(head_row.clone(), b.u32(128)),
            b.add(sl.clone(), b.u32(c * 32)),
        );
        let local = b.local(f);
        body.push(Stmt::StoreLocal {
            dst: local.clone(),
            value: ctx.load_mapped(&ops[0], at, u64::from(batches) * u64::from(heads) * 128)?,
        });
        q.push(b.load_local(local));
    }
    let it = b.local(u);
    let key = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups)),
        sg.clone(),
    );
    let valid = b.lt(key.clone(), length.clone());
    let mut dot = b.f32(0.0);
    let kv_total = u64::from(batches) * u64::from(*kv_heads) * u64::from(*keys) * 128;
    for (c, q) in q.into_iter().enumerate() {
        let at = b.add(
            kv_base.clone(),
            b.add(
                b.mul(b.add(start.clone(), key.clone()), b.u32(128)),
                b.add(sl.clone(), b.u32(c as u32 * 32)),
            ),
        );
        dot = b.fma(q, ctx.load_mapped(&ops[1], at, kv_total)?, dot);
    }
    let dot_total = b.local(f);
    let lane_zero = b.eq(sl.clone(), b.u32(0));
    body.push(Stmt::Loop {
        count: Some(passes(subgroups)),
        index: Some(it),
        accumulators: Vec::new(),
        body: vec![Stmt::If {
            condition: b.lt(key.clone(), b.u32(capacity)),
            accept: vec![Stmt::If {
                condition: valid,
                accept: vec![
                    Stmt::StoreLocal {
                        dst: dot_total.clone(),
                        value: b.mul(
                            b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, dot),
                            scale,
                        ),
                    },
                    Stmt::If {
                        condition: lane_zero.clone(),
                        accept: vec![Stmt::StoreTile {
                            dst: scores.clone(),
                            index: score_at(key.clone()),
                            value: b.load_local(dot_total),
                        }],
                        reject: Vec::new(),
                    },
                ],
                reject: vec![Stmt::If {
                    condition: lane_zero,
                    accept: vec![Stmt::StoreTile {
                        dst: scores.clone(),
                        index: score_at(key),
                        value: b.f32(-f32::MAX),
                    }],
                    reject: Vec::new(),
                }],
            }],
            reject: Vec::new(),
        }],
    });
    body.push(Stmt::Barrier);

    // Stable maximum, broadcast to every lane. Clamp masked tile reads so
    // non-multiple-of-block capacities never address past shared storage.
    let it = b.local(u);
    let key = b.add(b.mul(b.load_local(it.clone()), b.u32(block)), lane.clone());
    let live = b.lt(key.clone(), length.clone());
    let at = b.min(key, b.sub(length.clone(), b.u32(1)));
    let local_max = b.local(f);
    body.push(Stmt::Loop {
        count: Some(passes(block)),
        index: Some(it),
        accumulators: vec![Accumulator {
            local: local_max.clone(),
            init: b.f32(-f32::MAX),
            update: b.binary(
                TileBinaryOp::Max,
                b.load_local(local_max.clone()),
                b.select(
                    live,
                    b.load_tile(scores.clone(), score_at(at)),
                    b.f32(-f32::MAX),
                ),
                NumericContract::RELAXED,
            ),
        }],
        body: Vec::new(),
    });
    let max = b.local(f);
    let reduced = head_reduce(
        &ctx,
        &mut body,
        block,
        heads_per_group,
        scratch.clone(),
        TileReduceOp::Max,
        b.load_local(local_max),
    );
    body.push(Stmt::StoreLocal {
        dst: max.clone(),
        value: reduced,
    });
    let it = b.local(u);
    let key = b.add(b.mul(b.load_local(it.clone()), b.u32(block)), lane.clone());
    let live = b.lt(key.clone(), length.clone());
    let score = b.load_tile(
        scores.clone(),
        score_at(b.min(key.clone(), b.sub(length.clone(), b.u32(1)))),
    );
    let exp = b.local(f);
    let sum = b.local(f);
    body.push(Stmt::Loop {
        count: Some(passes(block)),
        index: Some(it),
        accumulators: vec![Accumulator {
            local: sum.clone(),
            init: b.f32(0.0),
            update: b.add(b.load_local(sum.clone()), b.load_local(exp.clone())),
        }],
        body: vec![
            Stmt::StoreLocal {
                dst: exp.clone(),
                value: b.select(
                    live.clone(),
                    b.unary(
                        TileUnaryOp::Exp,
                        b.sub(score, b.load_local(max.clone())),
                        NumericContract::RELAXED,
                    ),
                    b.f32(0.0),
                ),
            },
            Stmt::If {
                condition: live,
                accept: vec![Stmt::StoreTile {
                    dst: scores.clone(),
                    index: score_at(key),
                    value: b.load_local(exp),
                }],
                reject: Vec::new(),
            },
        ],
    });
    let denominator = b.local(f);
    let reduced = head_reduce(
        &ctx,
        &mut body,
        block,
        heads_per_group,
        scratch,
        TileReduceOp::Sum,
        b.load_local(sum),
    );
    body.push(Stmt::StoreLocal {
        dst: denominator.clone(),
        value: reduced,
    });
    body.push(Stmt::Barrier);

    // Every subgroup owns a K partition; adjacent lanes load adjacent V
    // channels. Four independent accumulators cover all 128 channels.
    let it = b.local(u);
    let key = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups)),
        sg.clone(),
    );
    let valid = b.lt(key.clone(), length.clone());
    let mut updates = Vec::new();
    let mut accumulators = Vec::new();
    let probability = b.local(f);
    updates.push(Stmt::StoreLocal {
        dst: probability.clone(),
        value: b.load_tile(scores, score_at(key.clone())),
    });
    for c in 0..4 {
        let local = b.local(f);
        body.push(Stmt::StoreLocal {
            dst: local.clone(),
            value: b.f32(0.0),
        });
        let at = b.add(
            kv_base.clone(),
            b.add(
                b.mul(b.add(start.clone(), key.clone()), b.u32(128)),
                b.add(sl.clone(), b.u32(c * 32)),
            ),
        );
        let value = ctx.load_mapped(&ops[2], at, kv_total)?;
        updates.push(Stmt::StoreLocal {
            dst: local.clone(),
            value: b.fma(
                b.load_local(probability.clone()),
                value,
                b.load_local(local.clone()),
            ),
        });
        accumulators.push(local);
    }
    body.push(Stmt::Loop {
        count: Some(passes(subgroups)),
        index: Some(it),
        accumulators: Vec::new(),
        body: vec![Stmt::If {
            condition: valid,
            accept: updates,
            reject: Vec::new(),
        }],
    });
    body.push(Stmt::Barrier);
    for (c, acc) in accumulators.into_iter().enumerate() {
        body.push(Stmt::StoreTile {
            dst: partials.clone(),
            index: partial_at(b.add(lane.clone(), b.u32(c as u32 * block))),
            value: b.load_local(acc),
        });
    }
    body.push(Stmt::Barrier);
    let out = ctx.linear_view(ctx.output()?)?;
    let mut stores = Vec::new();
    for c in 0..4 {
        let sum = (0..subgroups)
            .map(|i| {
                b.load_tile(
                    partials.clone(),
                    partial_at(b.add(sl.clone(), b.u32(c * block + i * 32))),
                )
            })
            .reduce(|a, v| b.add(a, v))
            .unwrap();
        stores.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(b.add(
                b.mul(
                    output_group.clone(),
                    b.u32(if *chunk == 0 { 128 } else { 130 }),
                ),
                b.add(sl.clone(), b.u32(c * 32)),
            )),
            value: if *chunk == 0 {
                b.div(sum, b.load_local(denominator.clone()))
            } else {
                sum
            },
            mask: b.bool(true),
        });
    }
    body.push(Stmt::If {
        condition: b.eq(sg, b.u32(0)),
        accept: stores,
        reject: Vec::new(),
    });
    if *chunk != 0 {
        body.push(Stmt::If {
            condition: b.eq(lane, b.u32(0)),
            accept: vec![
                Stmt::Store {
                    dst: out.clone(),
                    addr: Addr::Linear(b.add(b.mul(output_group.clone(), b.u32(130)), b.u32(128))),
                    value: b.load_local(max),
                    mask: b.bool(true),
                },
                Stmt::Store {
                    dst: out,
                    addr: Addr::Linear(b.add(b.mul(output_group.clone(), b.u32(130)), b.u32(129))),
                    value: b.load_local(denominator),
                    mask: b.bool(true),
                },
            ],
            reject: Vec::new(),
        });
    }
    let body = vec![Stmt::If {
        condition: b.lt(group, b.u32(count)),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish(
        if heads_per_group > 1 {
            if *chunk == 0 {
                "decode_attention_gqa"
            } else {
                "decode_attention_gqa_partial"
            }
        } else if *chunk == 0 {
            "decode_attention"
        } else {
            "decode_attention_partial"
        },
        grid,
        physical_block,
        body,
    ))
}

/// Partition a physical workgroup into independent query heads. Every lane
/// reaches both barriers, and each head reads only its own subgroup partials.
fn head_reduce(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    block: u32,
    heads: u32,
    scratch: Tile,
    op: TileReduceOp,
    value: TileExpr,
) -> TileExpr {
    let b = &ctx.b;
    if heads == 1 {
        return b.reduce(
            op,
            ReduceKind::Workgroup {
                scratch,
                group_size: block,
            },
            value,
        );
    }
    let subgroups = block / 32;
    let partial = b.local(ScalarElement::F32.element());
    body.push(Stmt::StoreLocal {
        dst: partial.clone(),
        value: b.reduce(op, ReduceKind::Subgroup, value),
    });
    body.push(Stmt::Barrier);
    body.push(Stmt::If {
        condition: b.eq(b.builtin(Builtin::SubgroupLane), b.u32(0)),
        accept: vec![Stmt::StoreTile {
            dst: scratch.clone(),
            index: b.builtin(Builtin::SubgroupId),
            value: b.load_local(partial),
        }],
        reject: Vec::new(),
    });
    body.push(Stmt::Barrier);
    let base = b.mul(
        b.div(b.builtin(Builtin::SubgroupId), b.u32(subgroups)),
        b.u32(subgroups),
    );
    (0..subgroups)
        .map(|i| b.load_tile(scratch.clone(), b.add(base.clone(), b.u32(i))))
        .reduce(|a, v| match op {
            TileReduceOp::Sum => b.add(a, v),
            TileReduceOp::Max => b.binary(TileBinaryOp::Max, a, v, NumericContract::RELAXED),
            _ => unreachable!("attention only reduces sums and maxima"),
        })
        .unwrap()
}
