//! Independent subgroup online softmax/PV, followed by one workgroup merge.
//! Each subgroup publishes all 130 partial values before the sole barrier.
//! Empty key partitions publish zero numerator/denominator and a finite floor.
use super::*;

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    let Launch::DecodeAttention {
        output,
        kv_heads,
        keys,
        block,
        chunk,
        key_major,
        scale,
        prefix,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected online decode attention".into()));
    };
    let b = &ctx.b;
    let (block, subgroups) = (*block, *block / 32);
    let f = ScalarElement::F32.element();
    let u = ScalarElement::U32.element();
    let heads = output.dims[1].as_const().unwrap() as u32;
    let batches = output.dims[0].as_const().unwrap() as u32;
    let capacity = if *chunk == 0 { *keys } else { *chunk };
    let splits = keys.div_ceil(capacity);
    let count = batches * heads * splits;
    let grid = distribute_workgroups(count, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let group = ctx.linear_workgroup();
    let (head, split) = if *key_major {
        let (split, head) = b.divrem(group.clone(), b.u32(batches * heads));
        (head, split)
    } else {
        b.divrem(group.clone(), b.u32(splits))
    };
    let output_group = b.add(b.mul(head.clone(), b.u32(splits)), split.clone());
    let start = b.mul(split, b.u32(capacity));
    let (batch, head_in_batch) = b.divrem(head.clone(), b.u32(heads));
    let kv_row = b.add(
        b.mul(batch, b.u32(*kv_heads)),
        b.div(head_in_batch, b.u32(heads / kv_heads)),
    );
    let kv_base = b.mul(kv_row, b.u32(keys * 128));
    let sg = b.builtin(Builtin::SubgroupId);
    let lane = b.builtin(Builtin::SubgroupLane);
    let scale = ctx.eval_scalar(scale, &[], &[])?;
    let prefix = b.min(b.cast(ctx.eval_scalar(prefix, &[], &[])?, u), b.u32(*keys));
    let maximum = |a, c| b.binary(TileBinaryOp::Max, a, c, NumericContract::RELAXED);
    let exp = |v| b.unary(TileUnaryOp::Exp, v, NumericContract::RELAXED);
    let length = b.min(
        b.sub(maximum(prefix, start.clone()), start.clone()),
        b.u32(capacity),
    );
    let partials = b.tile("online_attention_partials", f, &[subgroups * 130]);
    let mut body = Vec::new();
    let mut query = Vec::new();
    let mut numerators = Vec::new();
    for channel in 0..4 {
        let q = b.local(f);
        let index = b.add(
            b.mul(head.clone(), b.u32(128)),
            b.add(lane.clone(), b.u32(channel * 32)),
        );
        body.push(Stmt::StoreLocal {
            dst: q.clone(),
            value: ctx.load_mapped(&ops[0], index, u64::from(batches) * u64::from(heads) * 128)?,
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
    let it = b.local(u);
    let key = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups)),
        sg.clone(),
    );
    let kv_total = u64::from(batches) * u64::from(*kv_heads) * u64::from(*keys) * 128;
    let index = |channel| {
        b.add(
            kv_base.clone(),
            b.add(
                b.mul(b.add(start.clone(), key.clone()), b.u32(128)),
                b.add(lane.clone(), b.u32(channel * 32)),
            ),
        )
    };
    let mut dot = b.f32(0.);
    for (channel, q) in query.into_iter().enumerate() {
        dot = b.fma(
            q,
            ctx.load_mapped(&ops[1], index(channel as u32), kv_total)?,
            dot,
        );
    }
    let score = b.local(f);
    let next_max = b.local(f);
    let alpha = b.local(f);
    let beta = b.local(f);
    let mut update = vec![
        Stmt::StoreLocal {
            dst: score.clone(),
            value: b.mul(
                b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, dot),
                scale,
            ),
        },
        Stmt::StoreLocal {
            dst: next_max.clone(),
            value: maximum(
                b.load_local(running_max.clone()),
                b.load_local(score.clone()),
            ),
        },
        Stmt::StoreLocal {
            dst: alpha.clone(),
            value: exp(b.sub(
                b.load_local(running_max.clone()),
                b.load_local(next_max.clone()),
            )),
        },
        Stmt::StoreLocal {
            dst: beta.clone(),
            value: exp(b.sub(b.load_local(score.clone()), b.load_local(next_max.clone()))),
        },
    ];
    for (channel, numerator) in numerators.iter().enumerate() {
        update.push(Stmt::StoreLocal {
            dst: numerator.clone(),
            value: b.fma(
                b.load_local(beta.clone()),
                ctx.load_mapped(&ops[2], index(channel as u32), kv_total)?,
                b.mul(b.load_local(alpha.clone()), b.load_local(numerator.clone())),
            ),
        });
    }
    update.extend([
        Stmt::StoreLocal {
            dst: running_sum.clone(),
            value: b.fma(
                b.load_local(alpha),
                b.load_local(running_sum.clone()),
                b.load_local(beta),
            ),
        },
        Stmt::StoreLocal {
            dst: running_max.clone(),
            value: b.load_local(next_max),
        },
    ]);
    let (iterations, loop_body) =
        if crate::flags::for_profile(ctx.caps.execution_profile).attention_exact_partition {
            // Subgroup s owns s, s + S, ... < length. All of its lanes execute
            // floor((length + S - 1 - s) / S) iterations, so no per-key guard is
            // needed. The numerator stays nonnegative even for an empty prefix.
            // Only subgroup collectives occur inside; the workgroup rendezvous
            // remains outside the loop, where every subgroup reaches it.
            (
                b.div(
                    b.sub(b.add(length.clone(), b.u32(subgroups - 1)), sg.clone()),
                    b.u32(subgroups),
                ),
                update,
            )
        } else {
            (
                b.div(
                    b.add(length.clone(), b.u32(subgroups - 1)),
                    b.u32(subgroups),
                ),
                vec![Stmt::If {
                    condition: b.lt(key, length),
                    accept: update,
                    reject: Vec::new(),
                }],
            )
        };
    body.push(Stmt::Loop {
        count: Some(iterations),
        index: Some(it),
        accumulators: Vec::new(),
        body: loop_body,
    });
    let partial_base = b.mul(sg.clone(), b.u32(130));
    for (channel, numerator) in numerators.into_iter().enumerate() {
        body.push(Stmt::StoreTile {
            dst: partials.clone(),
            index: b.add(
                partial_base.clone(),
                b.add(lane.clone(), b.u32(channel as u32 * 32)),
            ),
            value: b.load_local(numerator),
        });
    }
    body.push(Stmt::If {
        condition: b.eq(lane.clone(), b.u32(0)),
        accept: vec![
            Stmt::StoreTile {
                dst: partials.clone(),
                index: b.add(partial_base.clone(), b.u32(128)),
                value: b.load_local(running_max),
            },
            Stmt::StoreTile {
                dst: partials.clone(),
                index: b.add(partial_base, b.u32(129)),
                value: b.load_local(running_sum),
            },
        ],
        reject: Vec::new(),
    });
    body.push(Stmt::Barrier);
    let load = |part, at| b.load_tile(partials.clone(), b.add(b.u32(part * 130), at));
    let max_value = (0..subgroups)
        .map(|i| load(i, b.u32(128)))
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
            value: exp(b.sub(load(i, b.u32(128)), b.load_local(max.clone()))),
        });
        merge.push(Stmt::StoreLocal {
            dst: denominator.clone(),
            value: b.fma(
                b.load_local(weight.clone()),
                load(i, b.u32(129)),
                b.load_local(denominator.clone()),
            ),
        });
        factors.push(b.load_local(weight));
    }
    let out = ctx.linear_view(ctx.output()?)?;
    let output_base = b.mul(output_group, b.u32(if *chunk == 0 { 128 } else { 130 }));
    for channel in 0..4 {
        let offset = b.add(lane.clone(), b.u32(channel * 32));
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
                    addr: Addr::Linear(b.add(output_base.clone(), b.u32(128))),
                    value: b.load_local(max),
                    mask: b.bool(true),
                },
                Stmt::Store {
                    dst: out,
                    addr: Addr::Linear(b.add(output_base, b.u32(129))),
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
        condition: b.lt(group, b.u32(count)),
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
