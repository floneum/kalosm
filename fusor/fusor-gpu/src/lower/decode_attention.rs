//! One workgroup per query head: subgroup QK dots, stable softmax in shared
//! storage, then column-coalesced PV. All arithmetic and cache storage are F32.
use super::{Ctx, distribute_workgroups};
use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, KernelIr, ReduceKind, ScalarElement, Stmt, TileBinaryOp,
    TileReduceOp, TileUnaryOp,
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
    let width = output.dims[3].as_const().unwrap() as u32;
    let b = &ctx.b;
    let f = ScalarElement::F32.element();
    let u = ScalarElement::U32.element();
    let heads = output.iterations().unwrap() as u32 / width;
    let group = ctx.linear_workgroup();
    let lane = b.builtin(Builtin::Lane);
    let grid = distribute_workgroups(heads, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let total = u64::from(heads) * splits.as_const().unwrap_or(0) * u64::from(width + 2);
    let split_count = ctx.dim_expr(*splits)?;
    let it = b.local(u);
    let base = b.mul(
        b.add(
            b.mul(group.clone(), split_count.clone()),
            b.load_local(it.clone()),
        ),
        b.u32(width + 2),
    );
    let load = |offset| ctx.load_mapped(&ops[0], b.add(base.clone(), offset), total);
    let maximum = b.local(f);
    let mut body = vec![Stmt::Loop {
        count: Some(split_count.clone()),
        index: Some(it.clone()),
        accumulators: vec![Accumulator {
            local: maximum.clone(),
            init: b.f32(-f32::MAX),
            update: b.binary(
                TileBinaryOp::Max,
                b.load_local(maximum.clone()),
                load(b.u32(width))?,
                NumericContract::RELAXED,
            ),
        }],
        body: Vec::new(),
    }];
    let weight = b.local(f);
    let numerator = b.local(f);
    let denominator = b.local(f);
    body.push(Stmt::Loop {
        count: Some(split_count.clone()),
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
                    load(b.u32(width + 1))?,
                    b.load_local(denominator.clone()),
                ),
            },
        ],
        body: vec![Stmt::StoreLocal {
            dst: weight,
            value: b.unary(
                TileUnaryOp::Exp,
                b.sub(load(b.u32(width))?, b.load_local(maximum)),
                NumericContract::RELAXED,
            ),
        }],
    });
    body.push(Stmt::Store {
        dst: ctx.linear_view(ctx.output()?)?,
        addr: Addr::Linear(b.add(b.mul(group.clone(), b.u32(width)), lane)),
        value: b.div(b.load_local(numerator), b.load_local(denominator)),
        mask: b.bool(true),
    });
    let body = vec![Stmt::If {
        condition: b.lt(group, b.u32(heads)),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish("decode_attention_merge", grid, width, body))
}

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    online::lower(ctx, op)
}
