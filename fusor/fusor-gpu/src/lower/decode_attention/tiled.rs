//! Tiled stable softmax: compute scores once, normalize cooperatively, then PV.
//! Invalid keys never load K/V; empty partitions publish zero mass.
use super::*;

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
    let scores = b.tile("attention_scores", f, &[capacity]);
    let scratch = b.tile("attention_reduce", f, &[block]);
    let partials = b.tile("attention_partials", f, &[subgroups * width]);
    let mut body = Vec::new();
    let mut query = Vec::new();
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
    }
    let it = b.local(u);
    let key = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups)),
        sg.clone(),
    );
    // Subgroup-uniform trip counts allow the final partial wave to avoid
    // reading masked NaNs or a key beyond physical capacity.
    let iterations = b.div(
        b.sub(b.add(length.clone(), b.u32(subgroups - 1)), sg.clone()),
        b.u32(subgroups),
    );
    let kv_total =
        u64::from(batches) * u64::from(*kv_heads) * keys.as_const().unwrap_or(0) * u64::from(width);
    let index = |channel| {
        b.add(
            kv_base.clone(),
            b.add(
                b.mul(b.add(start.clone(), key.clone()), b.u32(width)),
                b.add(lane.clone(), b.u32(channel * subgroup)),
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
    body.push(Stmt::Loop {
        count: Some(iterations.clone()),
        index: Some(it.clone()),
        accumulators: Vec::new(),
        body: vec![
            Stmt::StoreLocal {
                dst: score.clone(),
                value: b.mul(
                    b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, dot),
                    scale,
                ),
            },
            Stmt::If {
                condition: b.eq(lane.clone(), b.u32(0)),
                accept: vec![Stmt::StoreTile {
                    dst: scores.clone(),
                    index: key.clone(),
                    value: b.load_local(score),
                }],
                reject: Vec::new(),
            },
        ],
    });
    body.push(Stmt::Barrier);

    // All lanes participate in both reductions. Each score gets exactly one
    // exponential; the result is shared by every output channel.
    let physical_lane = b.builtin(Builtin::Lane);
    let tile_it = b.local(u);
    let tile_key = b.add(
        b.mul(b.load_local(tile_it.clone()), b.u32(block)),
        physical_lane.clone(),
    );
    let active = b.lt(tile_key.clone(), length.clone());
    // The count is zero for empty partitions. Clamping to capacity also makes
    // the inactive lanes' speculative shared-memory reads valid.
    let at = b.min(tile_key.clone(), b.u32(capacity - 1));
    let passes = b.div(b.add(length.clone(), b.u32(block - 1)), b.u32(block));
    let local_max = b.local(f);
    body.push(Stmt::Loop {
        count: Some(passes.clone()),
        index: Some(tile_it.clone()),
        accumulators: vec![Accumulator {
            local: local_max.clone(),
            init: b.f32(-f32::MAX),
            update: maximum(
                b.load_local(local_max.clone()),
                b.select(
                    active.clone(),
                    b.load_tile(scores.clone(), at.clone()),
                    b.f32(-f32::MAX),
                ),
            ),
        }],
        body: Vec::new(),
    });
    let max = b.local(f);
    let reduce_kind = ReduceKind::Workgroup {
        scratch: scratch.clone(),
        group_size: block,
    };
    body.push(Stmt::StoreLocal {
        dst: max.clone(),
        value: b.reduce(
            TileReduceOp::Max,
            reduce_kind.clone(),
            b.load_local(local_max),
        ),
    });
    let weight = b.local(f);
    let local_sum = b.local(f);
    body.push(Stmt::Loop {
        count: Some(passes),
        index: Some(tile_it),
        accumulators: vec![Accumulator {
            local: local_sum.clone(),
            init: b.f32(0.),
            update: b.add(
                b.load_local(local_sum.clone()),
                b.load_local(weight.clone()),
            ),
        }],
        body: vec![
            Stmt::StoreLocal {
                dst: weight.clone(),
                value: b.select(
                    active.clone(),
                    exp(b.sub(b.load_tile(scores.clone(), at), b.load_local(max.clone()))),
                    b.f32(0.),
                ),
            },
            Stmt::If {
                condition: active,
                accept: vec![Stmt::StoreTile {
                    dst: scores.clone(),
                    index: tile_key,
                    value: b.load_local(weight),
                }],
                reject: Vec::new(),
            },
        ],
    });
    let denominator = b.local(f);
    body.push(Stmt::StoreLocal {
        dst: denominator.clone(),
        value: b.reduce(TileReduceOp::Sum, reduce_kind, b.load_local(local_sum)),
    });
    body.push(Stmt::Barrier);

    let probability = b.local(f);
    let mut numerators = Vec::new();
    let mut accumulators = Vec::new();
    for channel in 0..width / subgroup {
        let numerator = b.local(f);
        accumulators.push(Accumulator {
            local: numerator.clone(),
            init: b.f32(0.),
            update: b.fma(
                b.load_local(probability.clone()),
                ctx.load_mapped(&ops[2], index(channel), kv_total)?,
                b.load_local(numerator.clone()),
            ),
        });
        numerators.push(numerator);
    }
    body.push(Stmt::Loop {
        count: Some(iterations),
        index: Some(it),
        accumulators,
        body: vec![Stmt::StoreLocal {
            dst: probability,
            value: b.load_tile(scores, key),
        }],
    });
    for (channel, numerator) in numerators.into_iter().enumerate() {
        body.push(Stmt::StoreTile {
            dst: partials.clone(),
            index: b.add(
                b.mul(sg.clone(), b.u32(width)),
                b.add(lane.clone(), b.u32(channel as u32 * subgroup)),
            ),
            value: b.load_local(numerator),
        });
    }
    body.push(Stmt::Barrier);
    let out = ctx.linear_view(ctx.output()?)?;
    let output_base = b.mul(
        output_group,
        b.u32(if *chunk == 0 { width } else { width + 2 }),
    );
    let mut stores = Vec::new();
    for channel in 0..width / subgroup {
        let offset = b.add(lane.clone(), b.u32(channel * subgroup));
        let sum = (0..subgroups)
            .map(|part| b.load_tile(partials.clone(), b.add(b.u32(part * width), offset.clone())))
            .reduce(|a, c| b.add(a, c))
            .unwrap();
        stores.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(b.add(output_base.clone(), offset)),
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
            condition: b.eq(physical_lane, b.u32(0)),
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
    let body = vec![Stmt::If {
        condition: b.lt(group, count),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish(
        if *chunk == 0 {
            "decode_attention_tiled"
        } else {
            "decode_attention_tiled_partial"
        },
        grid,
        block,
        body,
    ))
}
