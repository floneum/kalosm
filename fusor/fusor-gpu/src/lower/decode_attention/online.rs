//! Independent subgroup online softmax/PV, followed by one workgroup merge.
//! Each subgroup publishes all channel and statistic values before the sole barrier.
//! Empty key partitions publish zero numerator/denominator and a finite floor.
use super::*;
use fusor_ir::ir::kernel::TileExpr;

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    let Launch::DecodeAttention {
        output,
        kv_heads,
        keys,
        block,
        chunk,
        scale,
        prefix,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected online decode attention".into()));
    };
    let b = &ctx.b;
    let subgroup = ctx.caps.subgroup_width();
    let width = ops[0].layout.shape()[3].as_const().unwrap() as u32;
    let (block, subgroups) = (*block, *block / subgroup);
    let f = ScalarElement::F32.element();
    let u = ScalarElement::U32.element();
    let heads = output.dims[1].as_const().unwrap() as u32;
    let batches = output.dims[0].as_const().unwrap() as u32;
    let capacity = if *chunk == 0 {
        u32::try_from(
            keys.as_const()
                .ok_or_else(|| Error::Plan("symbolic attention needs a fixed chunk".into()))?,
        )
        .map_err(|_| Error::Plan("attention capacity exceeds u32".into()))?
    } else {
        *chunk
    };
    let split_dim = keys.div_ceil(u64::from(capacity));
    let splits = ctx.dim_expr(split_dim)?;
    let groups_per_split = batches * heads;
    let count = b.mul(b.u32(groups_per_split), splits.clone());
    let grid = super::super::grid_for(
        &fusor_ir::ir::launch::IndexSpace::new([
            fusor_ir::shape::Dim::Const(u64::from(groups_per_split)),
            split_dim,
        ]),
        1,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let group = ctx.linear_workgroup();
    let (head, split) = b.divrem(group.clone(), splits.clone());
    let output_group = b.add(b.mul(head.clone(), splits.clone()), split.clone());
    let start = b.mul(split, b.u32(capacity));
    let (batch, head_in_batch) = b.divrem(head.clone(), b.u32(heads));
    let kv_row = b.add(
        b.mul(batch, b.u32(*kv_heads)),
        b.div(head_in_batch, b.u32(heads / kv_heads)),
    );
    let kv_base = b.mul(kv_row, b.mul(ctx.dim_expr(*keys)?, b.u32(width)));
    let sg = b.builtin(Builtin::SubgroupId);
    let lane = b.builtin(Builtin::SubgroupLane);
    let scale = ctx.eval_scalar(scale, &[], &[])?;
    let prefix = b.min(
        b.cast(ctx.eval_scalar(prefix, &[], &[])?, u),
        ctx.dim_expr(*keys)?,
    );
    let maximum = |a, c| b.binary(TileBinaryOp::Max, a, c, NumericContract::RELAXED);
    let exp = |v| b.unary(TileUnaryOp::Exp, v, NumericContract::RELAXED);
    let length = b.min(
        b.sub(maximum(prefix, start.clone()), start.clone()),
        b.u32(capacity),
    );
    let partials = b.tile("online_attention_partials", f, &[subgroups * (width + 2)]);
    let mut body = Vec::new();
    let mut query = Vec::new();
    let mut numerators = Vec::new();
    for channel in 0..width / subgroup {
        let q = b.local(f);
        let index = b.add(
            b.mul(head.clone(), b.u32(width)),
            b.add(lane.clone(), b.u32(channel * subgroup)),
        );
        body.push(Stmt::StoreLocal {
            dst: q.clone(),
            value: ctx.load_mapped(
                &ops[0],
                index,
                u64::from(batches) * u64::from(heads) * u64::from(width),
            )?,
        });
        query.push(b.load_local(q));
        let numerator = b.local(f);
        body.push(Stmt::StoreLocal {
            dst: numerator.clone(),
            value: b.f32(0.),
        });
        numerators.push(numerator);
    }
    let running_max = b.local(f);
    let running_sum = b.local(f);
    body.push(Stmt::StoreLocal {
        dst: running_max.clone(),
        value: b.f32(-f32::MAX),
    });
    body.push(Stmt::StoreLocal {
        dst: running_sum.clone(),
        value: b.f32(0.),
    });
    // Batch several keys in registers. One rescale applies to the whole batch,
    // shortening the online dependency chain without workgroup synchronization.
    const KEYS_PER_STEP: u32 = 4;
    let it = b.local(u);
    let first = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups * KEYS_PER_STEP)),
        sg.clone(),
    );
    let kv_total =
        u64::from(batches) * u64::from(*kv_heads) * keys.as_const().unwrap_or(0) * u64::from(width);
    let index = |key: TileExpr, channel| {
        b.add(
            kv_base.clone(),
            b.add(
                b.mul(b.add(start.clone(), key), b.u32(width)),
                b.add(lane.clone(), b.u32(channel * subgroup)),
            ),
        )
    };
    let next_max = b.local(f);
    let alpha = b.local(f);
    let mut update = vec![Stmt::StoreLocal {
        dst: next_max.clone(),
        value: b.load_local(running_max.clone()),
    }];
    let mut scores = Vec::new();
    for item in 0..KEYS_PER_STEP {
        let key = b.add(first.clone(), b.u32(item * subgroups));
        let live = b.lt(key.clone(), length.clone());
        let score = b.local(f);
        let mut dot = b.f32(0.);
        for (channel, q) in query.iter().enumerate() {
            dot = b.fma(
                q.clone(),
                ctx.load_mapped(&ops[1], index(key.clone(), channel as u32), kv_total)?,
                dot,
            );
        }
        update.push(Stmt::If {
            // Subgroup-uniform: inactive keys never load K or participate in
            // the score arithmetic, including masked NaNs and infinities.
            condition: live.clone(),
            accept: vec![
                Stmt::StoreLocal {
                    dst: score.clone(),
                    value: b.mul(
                        b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, dot),
                        scale.clone(),
                    ),
                },
                Stmt::StoreLocal {
                    dst: next_max.clone(),
                    value: maximum(b.load_local(next_max.clone()), b.load_local(score.clone())),
                },
            ],
            reject: Vec::new(),
        });
        scores.push((key, live, score));
    }
    update.push(Stmt::StoreLocal {
        dst: alpha.clone(),
        value: exp(b.sub(
            b.load_local(running_max.clone()),
            b.load_local(next_max.clone()),
        )),
    });
    for numerator in &numerators {
        update.push(Stmt::StoreLocal {
            dst: numerator.clone(),
            value: b.mul(b.load_local(alpha.clone()), b.load_local(numerator.clone())),
        });
    }
    update.push(Stmt::StoreLocal {
        dst: running_sum.clone(),
        value: b.mul(b.load_local(alpha), b.load_local(running_sum.clone())),
    });
    for (key, live, score) in scores {
        let beta = b.local(f);
        let mut accumulate = vec![Stmt::StoreLocal {
            dst: beta.clone(),
            value: exp(b.sub(b.load_local(score), b.load_local(next_max.clone()))),
        }];
        for (channel, numerator) in numerators.iter().enumerate() {
            accumulate.push(Stmt::StoreLocal {
                dst: numerator.clone(),
                value: b.fma(
                    b.load_local(beta.clone()),
                    ctx.load_mapped(&ops[2], index(key.clone(), channel as u32), kv_total)?,
                    b.load_local(numerator.clone()),
                ),
            });
        }
        accumulate.push(Stmt::StoreLocal {
            dst: running_sum.clone(),
            value: b.add(b.load_local(running_sum.clone()), b.load_local(beta)),
        });
        update.push(Stmt::If {
            condition: live,
            accept: accumulate,
            reject: Vec::new(),
        });
    }
    update.push(Stmt::StoreLocal {
        dst: running_max.clone(),
        value: b.load_local(next_max),
    });
    let iterations = b.div(
        b.sub(
            b.add(length, b.u32(subgroups * KEYS_PER_STEP - 1)),
            sg.clone(),
        ),
        b.u32(subgroups * KEYS_PER_STEP),
    );
    body.push(Stmt::Loop {
        count: Some(iterations),
        index: Some(it),
        accumulators: Vec::new(),
        body: update,
    });
    let partial_base = b.mul(sg.clone(), b.u32(width + 2));
    for (channel, numerator) in numerators.into_iter().enumerate() {
        body.push(Stmt::StoreTile {
            dst: partials.clone(),
            index: b.add(
                partial_base.clone(),
                b.add(lane.clone(), b.u32(channel as u32 * subgroup)),
            ),
            value: b.load_local(numerator),
        });
    }
    body.push(Stmt::If {
        condition: b.eq(lane.clone(), b.u32(0)),
        accept: vec![
            Stmt::StoreTile {
                dst: partials.clone(),
                index: b.add(partial_base.clone(), b.u32(width)),
                value: b.load_local(running_max),
            },
            Stmt::StoreTile {
                dst: partials.clone(),
                index: b.add(partial_base, b.u32(width + 1)),
                value: b.load_local(running_sum),
            },
        ],
        reject: Vec::new(),
    });
    body.push(Stmt::Barrier);
    let load = |part, at| b.load_tile(partials.clone(), b.add(b.u32(part * (width + 2)), at));
    let max_value = (0..subgroups)
        .map(|i| load(i, b.u32(width)))
        .reduce(maximum)
        .unwrap();
    let max = b.local(f);
    let denominator = b.local(f);
    let mut merge = vec![
        Stmt::StoreLocal {
            dst: max.clone(),
            value: max_value,
        },
        Stmt::StoreLocal {
            dst: denominator.clone(),
            value: b.f32(0.),
        },
    ];
    let mut factors = Vec::new();
    for i in 0..subgroups {
        let weight = b.local(f);
        merge.push(Stmt::StoreLocal {
            dst: weight.clone(),
            value: exp(b.sub(load(i, b.u32(width)), b.load_local(max.clone()))),
        });
        merge.push(Stmt::StoreLocal {
            dst: denominator.clone(),
            value: b.fma(
                b.load_local(weight.clone()),
                load(i, b.u32(width + 1)),
                b.load_local(denominator.clone()),
            ),
        });
        factors.push(b.load_local(weight));
    }
    let out = ctx.linear_view(ctx.output()?)?;
    let output_base = b.mul(
        output_group,
        b.u32(if *chunk == 0 { width } else { width + 2 }),
    );
    for channel in 0..width / subgroup {
        let offset = b.add(lane.clone(), b.u32(channel * subgroup));
        let mut value = b.f32(0.);
        for (i, weight) in factors.iter().enumerate() {
            value = b.fma(weight.clone(), load(i as u32, offset.clone()), value);
        }
        if *chunk == 0 {
            value = b.div(value, b.load_local(denominator.clone()));
        }
        merge.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(b.add(output_base.clone(), offset)),
            value,
            mask: b.bool(true),
        });
    }
    if *chunk != 0 {
        merge.push(Stmt::If {
            condition: b.eq(lane, b.u32(0)),
            accept: vec![
                Stmt::Store {
                    dst: out.clone(),
                    addr: Addr::Linear(b.add(output_base.clone(), b.u32(width))),
                    value: b.load_local(max),
                    mask: b.bool(true),
                },
                Stmt::Store {
                    dst: out,
                    addr: Addr::Linear(b.add(output_base, b.u32(width + 1))),
                    value: b.load_local(denominator),
                    mask: b.bool(true),
                },
            ],
            reject: Vec::new(),
        });
    }
    body.push(Stmt::If {
        condition: b.eq(sg, b.u32(0)),
        accept: merge,
        reject: Vec::new(),
    });
    let body = vec![Stmt::If {
        condition: b.lt(group, count),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish(
        if *chunk == 0 {
            "decode_attention_online"
        } else {
            "decode_attention_online_partial"
        },
        grid,
        block,
        body,
    ))
}
