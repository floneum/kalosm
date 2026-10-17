//! One complete Q6_K scale group per lane: decode one signed scale for
//! sixteen weights, instead of distributing four scale groups over a lane.
use super::*;
use fusor_ir::dtype::{NumericContract, QFmt, QLayout};
use fusor_ir::ir::kernel::{QuantizedView, TileBinaryOp};

pub(super) fn source(c: &Contract<'_>, width: u32, p: SgemvParams) -> Option<QuantizedView> {
    if c.shape.m != 1
        || c.shape.batch != 1
        || width != 32
        || p.vector != 32
        || c.acc != ScalarElement::F32
        || !matches!(c.b.side.pre.kind(), ScalarKind::Arg(0))
        || !c.shape.k.as_const()?.is_multiple_of(512)
    {
        return None;
    }
    let [StagedSource::Mem(Source::Quantized(q))] = &c.b.sources[..] else {
        return None;
    };
    if q.fmt != QFmt::Q6K || q.layout != QLayout::F32Scales {
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
    if ks.stride != 1 || !ns.stride.is_multiple_of(256) {
        return None;
    }
    Some(q.clone())
}

pub(super) fn lower(
    ctx: Ctx<'_>,
    c: &Contract<'_>,
    p: SgemvParams,
    q: &QuantizedView,
) -> Result<KernelIr> {
    let b = &ctx.b;
    let f32_ty = ScalarElement::F32.element();
    let shape = c.shape;
    let rows = c.rows();
    let cps = p.cols / p.subgroups;
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
    let sg = b.builtin(Builtin::SubgroupId);
    let col_base = b.add(b.mul(col_group, b.u32(p.cols)), b.mul(sg, b.u32(cps)));
    let group = b.rem(lane.clone(), b.u32(16));
    let index = b.local(ScalarElement::U32.element());
    let block_k = b.add(
        b.mul(b.load_local(index.clone()), b.u32(2)),
        b.div(lane.clone(), b.u32(16)),
    );
    let bit = |op, a, v| b.binary(op, a, b.u32(v), NumericContract::RELAXED);
    let and = |a, v| bit(TileBinaryOp::BitAnd, a, v);
    let shr = |a, s| b.binary(TileBinaryOp::Shr, a, s, NumericContract::RELAXED);
    let input_base = b.add(
        b.mul(block_k.clone(), b.u32(256)),
        b.mul(group.clone(), b.u32(16)),
    );
    let mut activation = Vec::new();
    for i in 0..16 {
        activation.push(c.a.value(
            &ctx,
            &row,
            &b.add(input_base.clone(), b.u32(i)),
            &b.bool(true),
            &batch,
            c.acc,
        )?);
    }
    let sum_x = activation
        .iter()
        .cloned()
        .reduce(|a, v| b.add(a, v))
        .unwrap();
    let ql_base = b.add(
        b.mul(b.div(group.clone(), b.u32(8)), b.u32(16)),
        b.mul(b.rem(group.clone(), b.u32(4)), b.u32(4)),
    );
    let qh_base = b.add(
        b.u32(32),
        b.add(
            b.mul(b.div(group.clone(), b.u32(8)), b.u32(8)),
            b.mul(b.rem(group.clone(), b.u32(2)), b.u32(4)),
        ),
    );
    let low_shift = b.mul(b.rem(b.div(group.clone(), b.u32(4)), b.u32(2)), b.u32(4));
    let high_shift = b.mul(b.rem(b.div(group.clone(), b.u32(2)), b.u32(4)), b.u32(2));
    let mut accumulators = Vec::new();
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
        let word_base = b.mul(block, b.u32(53));
        let word = |off| {
            b.load(
                Source::Storage(q.data.clone()),
                Addr::Linear(b.add(word_base.clone(), off)),
                ok.clone(),
                b.u32(0),
            )
        };
        let d = b.bitcast(word(b.u32(52)), f32_ty);
        let scale_word = word(b.add(b.u32(48), b.div(group.clone(), b.u32(4))));
        let scale_byte = and(
            shr(scale_word, b.mul(b.rem(group.clone(), b.u32(4)), b.u32(8))),
            255,
        );
        let signed = b.bitcast(
            bit(TileBinaryOp::Shl, scale_byte, 24),
            ScalarElement::I32.element(),
        );
        let scale = b.cast(shr(signed, b.u32(24)), f32_ty);
        let mut dots = Vec::new();
        for chunk in 0..4 {
            let low = and(
                shr(
                    word(b.add(ql_base.clone(), b.u32(chunk))),
                    low_shift.clone(),
                ),
                0x0f0f0f0f,
            );
            let high = bit(
                TileBinaryOp::Shl,
                and(
                    shr(
                        word(b.add(qh_base.clone(), b.u32(chunk))),
                        high_shift.clone(),
                    ),
                    0x03030303,
                ),
                4,
            );
            let packed = b.binary(TileBinaryOp::BitOr, low, high, NumericContract::RELAXED);
            let mut xs = Vec::new();
            let mut weights = Vec::new();
            for byte in 0..4 {
                let shift = byte * 8;
                weights.push(b.cast(and(packed.clone(), 63u32 << shift), f32_ty));
                xs.push(b.mul(
                    activation[chunk as usize * 4 + byte as usize].clone(),
                    b.f32(2f32.powi(-shift)),
                ));
            }
            dots.push(b.dot(
                b.vector(ScalarElement::F32, xs),
                b.vector(ScalarElement::F32, weights),
            ));
        }
        let dot = dots.into_iter().reduce(|a, v| b.add(a, v)).unwrap();
        let centered = b.sub(dot, b.mul(b.f32(32.0), sum_x.clone()));
        let local = b.local(f32_ty);
        accumulators.push(Accumulator {
            local: local.clone(),
            init: b.f32(0.0),
            update: b.fma(centered, b.mul(scale, d), b.load_local(local.clone())),
        });
        let total = b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, b.load_local(local));
        let addr = Addr::Linear(b.add(b.mul(row.clone(), n.clone()), col.clone()));
        stores.push(c.store_post(
            &ctx,
            &out,
            total,
            row.clone(),
            col,
            addr,
            b.and(ok, b.eq(lane.clone(), b.u32(0))),
        )?);
    }
    let mut body = vec![Stmt::Loop {
        count: Some(b.u32((shape.k.as_const().unwrap() / 512) as u32)),
        index: Some(index),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);
    Ok(ctx.finish("sgemv_q6_group16", grid, p.subgroups * 32, body))
}
