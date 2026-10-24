//! Dense small-M contractions whose B columns are contiguous. Lanes cover
//! neighboring output columns; subgroups split K and close through scratch.
//! The ordinary quantized GEMV layout instead places lanes along K.
use super::*;

pub(super) fn supported(c: &Contract<'_>, caps: &fusor_ir::device::Caps) -> bool {
    // The body assigns columns using actual subgroup lane/id builtins. An
    // assumed width is not enough: absent subgroups also have an assumed
    // width of 32, and a ranged adapter may execute a wider subgroup.
    if caps
        .subgroups
        .is_none_or(|s| !s.is_fixed() || s.assumed() != 32)
        || c.shape.m > 4
        || c.acc != ScalarElement::F32
        || c.shape.n.as_const().is_none_or(|n| n <= 1)
    {
        return false;
    }
    match &c.b.sources[..] {
        [StagedSource::Mem(Source::Storage(v))] => {
            let [ka, na] = &v.layout.indexing.groups[..] else {
                return false;
            };
            let [ns] = &na.sub_axes[..] else {
                return false;
            };
            ns.stride == 1 && ka.sub_axes.iter().any(|axis| axis.stride > 1)
        }
        [
            StagedSource::Indexed {
                operand,
                axes: Some((_, column)),
                ..
            },
        ] => {
            // Fixed KV capacity can leave symbolic leading strides even
            // when N is concrete. Preserve the generic row-address proof;
            // only the final, independent column axis must be contiguous.
            *column + 1 == operand.layout.rank()
                && operand.layout.shape()[*column].known_eq(c.shape.n)
                && operand.layout.strides()[*column].as_const() == Some(1)
        }
        _ => false,
    }
}

pub(super) fn lower(ctx: Ctx<'_>, c: &Contract<'_>) -> Result<KernelIr> {
    let b = &ctx.b;
    let shape = c.shape;
    let rows = c.rows();
    let subgroups = 8.min(
        ctx.caps
            .limits
            .max_compute_invocations_per_workgroup
            .min(ctx.caps.limits.max_compute_workgroup_size[0])
            / 32,
    );
    let block = subgroups * 32;
    let grid = crate::lower::tiled_grid_for(
        &IndexSpace::new([Dim::Const(u64::from(rows)), shape.n]),
        1,
        32,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let wg = column_workgroup_index(&ctx, grid, rows, shape.n, 32)?;
    let (row, col_group) = row_col(&ctx, wg, ceil_extent(&ctx, shape.n, 32)?, rows);
    let (batch, b_row_base) = c.batch_of(&ctx, &row)?;
    let lane = b.builtin(Builtin::SubgroupLane);
    let sg = b.builtin(Builtin::SubgroupId);
    let col = b.add(b.mul(col_group, b.u32(32)), lane.clone());
    let n = ctx.dim_expr(shape.n)?;
    let col_ok = b.lt(col.clone(), n.clone());
    let it = b.local(ScalarElement::U32.element());
    let k = b.add(
        b.mul(b.load_local(it.clone()), b.u32(subgroups)),
        sg.clone(),
    );
    let k_ok = b.lt(k.clone(), ctx.dim_expr(shape.k)?);
    let av = c.a.value(&ctx, &row, &k, &k_ok, &batch, c.acc)?;
    let bv = c.b.value(
        &ctx,
        &b.add(b_row_base, k),
        &col,
        &b.and(k_ok.clone(), col_ok.clone()),
        &batch,
        c.acc,
    )?;
    let av = zero_unless(&ctx, true, &k_ok, av, c.acc);
    let bv = zero_unless(&ctx, true, &k_ok, bv, c.acc);
    let acc = b.local(ScalarElement::F32.element());
    let scratch = b.tile(
        "contiguous_n_partial",
        ScalarElement::F32.element(),
        &[block],
    );
    let mut body = vec![
        Stmt::Loop {
            count: Some(ceil_extent(&ctx, shape.k, subgroups)?),
            index: Some(it),
            accumulators: vec![Accumulator {
                local: acc.clone(),
                init: b.f32(0.0),
                update: b.fma(av, bv, b.load_local(acc.clone())),
            }],
            body: Vec::new(),
        },
        Stmt::StoreTile {
            dst: scratch.clone(),
            index: b.add(b.mul(sg.clone(), b.u32(32)), lane.clone()),
            value: b.load_local(acc),
        },
        Stmt::Barrier,
    ];
    let total = (0..subgroups)
        .map(|i| b.load_tile(scratch.clone(), b.add(lane.clone(), b.u32(i * 32))))
        .reduce(|a, v| b.add(a, v))
        .unwrap();
    let out = ctx.linear_view(ctx.output()?)?;
    let addr = Addr::Linear(b.add(b.mul(row.clone(), n), col.clone()));
    body.push(c.store_post(
        &ctx,
        &out,
        total,
        row,
        col,
        addr,
        b.and(col_ok, b.eq(sg, b.u32(0))),
    )?);
    Ok(ctx.finish("sgemv_contiguous_n", grid, block, body))
}
