//! One workgroup owns a normalization row, retaining its inputs across the
//! square-sum reduction so no intermediate normalization buffers are needed.
use super::{Ctx, distribute_workgroups};
use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{
    Addr, Builtin, KernelIr, ReduceKind, ScalarElement, Stmt, TileReduceOp, TileUnaryOp,
};
use fusor_ir::ir::launch::Launch;
use fusor_ir::{Error, Result};

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    let Launch::RmsNorm {
        output,
        width,
        block,
        epsilon,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected RMS normalization".into()));
    };
    let b = &ctx.b;
    let block = *block;
    let f = ScalarElement::F32.element();
    let elements = output.iterations().unwrap();
    let rows = elements as u32 / *width;
    let group = ctx.linear_workgroup();
    let lane = b.builtin(Builtin::Lane);
    let base = b.mul(group.clone(), b.u32(*width));
    let mut body = Vec::new();
    let mut held = Vec::new();
    let mut squares = b.f32(0.0);
    for slot in 0..width.div_ceil(block) {
        let col = b.add(lane.clone(), b.u32(slot * block));
        let ok = b.lt(col.clone(), b.u32(*width));
        let x = b.local(f);
        let flat = b.add(base.clone(), col.clone());
        let mut value = ctx.load_mapped(&ops[0], flat.clone(), elements)?;
        if let Some(residual) = ops.get(2) {
            value = b.add(value, ctx.load_mapped(residual, flat, elements)?);
        }
        body.push(Stmt::StoreLocal {
            dst: x.clone(),
            value: b.f32(0.0),
        });
        body.push(Stmt::If {
            condition: ok.clone(),
            accept: vec![Stmt::StoreLocal {
                dst: x.clone(),
                value,
            }],
            reject: Vec::new(),
        });
        squares = b.fma(b.load_local(x.clone()), b.load_local(x.clone()), squares);
        held.push((col, ok, x));
    }
    let subgroup_width = ctx.caps.subgroup_width();
    let sum = if ctx.caps.subgroups.is_some_and(|s| s.is_fixed())
        && block.is_multiple_of(subgroup_width)
        && block / subgroup_width <= subgroup_width
    {
        let subgroup_lane = b.builtin(Builtin::SubgroupLane);
        let subgroup = b.builtin(Builtin::SubgroupId);
        let partial = b.local(f);
        body.push(Stmt::StoreLocal {
            dst: partial.clone(),
            value: b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, squares),
        });
        if block == subgroup_width {
            b.load_local(partial)
        } else {
            // This scratch is fresh and used once, so only the publication
            // barrier is needed. Each subgroup reads the partials across its
            // lanes, then broadcasts their collective sum to all its lanes.
            let scratch = b.tile("rms_subgroup_partials", f, &[block / subgroup_width]);
            body.push(Stmt::If {
                condition: b.eq(subgroup_lane.clone(), b.u32(0)),
                accept: vec![Stmt::StoreTile {
                    dst: scratch.clone(),
                    index: subgroup,
                    value: b.load_local(partial),
                }],
                reject: Vec::new(),
            });
            body.push(Stmt::Barrier);
            let lane_partial = b.local(f);
            body.push(Stmt::StoreLocal {
                dst: lane_partial.clone(),
                value: b.f32(0.),
            });
            body.push(Stmt::If {
                condition: b.lt(subgroup_lane.clone(), b.u32(block / subgroup_width)),
                accept: vec![Stmt::StoreLocal {
                    dst: lane_partial.clone(),
                    value: b.load_tile(scratch, subgroup_lane),
                }],
                reject: Vec::new(),
            });
            b.reduce(
                TileReduceOp::Sum,
                ReduceKind::Subgroup,
                b.load_local(lane_partial),
            )
        }
    } else {
        b.reduce(
            TileReduceOp::Sum,
            ReduceKind::Workgroup {
                scratch: b.tile("rms_scratch", f, &[block]),
                group_size: block,
            },
            squares,
        )
    };
    let inv = b.local(f);
    body.push(Stmt::StoreLocal {
        dst: inv.clone(),
        value: b.unary(
            TileUnaryOp::InverseSqrt,
            b.add(
                b.mul(sum, b.f32(1.0 / *width as f32)),
                ctx.eval_scalar(epsilon, &[], &[])?,
            ),
            NumericContract::RELAXED,
        ),
    });
    let out = ctx.linear_view(ctx.output()?)?;
    for (col, ok, x) in held {
        body.push(Stmt::If {
            condition: ok,
            accept: vec![Stmt::Store {
                dst: out.clone(),
                addr: Addr::Linear(b.add(base.clone(), col.clone())),
                value: b.mul(
                    b.mul(b.load_local(x), b.load_local(inv.clone())),
                    ctx.load_mapped(&ops[1], col, u64::from(*width))?,
                ),
                mask: b.bool(true),
            }],
            reject: Vec::new(),
        });
    }
    let grid = distribute_workgroups(rows, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let body = vec![Stmt::If {
        condition: b.lt(group, b.u32(rows)),
        accept: body,
        reject: Vec::new(),
    }];
    Ok(ctx.finish(
        if ops.len() == 3 {
            "rms_norm_residual"
        } else {
            "rms_norm"
        },
        grid,
        block,
        body,
    ))
}
