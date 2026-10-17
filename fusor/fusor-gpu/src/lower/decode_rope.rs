//! Direct rotary embedding for one position, including strided Q/K views.
use super::{Ctx, distribute_workgroups};
use fusor_ir::ir::kernel::{Addr, Builtin, KernelIr, ScalarElement, Stmt};
use fusor_ir::ir::launch::Launch;
use fusor_ir::{Error, Result};

pub(super) fn lower(ctx: Ctx<'_>, op: &Launch) -> Result<KernelIr> {
    let Launch::DecodeRope {
        output,
        head_dim,
        interleaved,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("expected decode rotary embedding".into()));
    };
    let b = &ctx.b;
    let f = ScalarElement::F32.element();
    let elements = output.iterations().unwrap();
    let index = b.add(
        b.mul(ctx.linear_workgroup(), b.u32(256)),
        b.builtin(Builtin::Lane),
    );
    let column = b.rem(index.clone(), b.u32(*head_dim));
    let half = *head_dim / 2;
    let negative = if *interleaved {
        b.eq(b.rem(column.clone(), b.u32(2)), b.u32(0))
    } else {
        b.lt(column.clone(), b.u32(half))
    };
    let rotated_column = if *interleaved {
        b.select(
            negative.clone(),
            b.add(column.clone(), b.u32(1)),
            b.sub(column.clone(), b.u32(1)),
        )
    } else {
        b.rem(b.add(column.clone(), b.u32(half)), b.u32(*head_dim))
    };
    let rotated_index = b.add(b.sub(index.clone(), column.clone()), rotated_column);
    let frequency = if *interleaved {
        b.div(column, b.u32(2))
    } else {
        b.rem(column, b.u32(half))
    };
    let position = ctx.load_mapped(&ops[3], b.u32(0), 1)?;
    let rows = ops[1].layout.shape()[0].as_const().unwrap() as u32;
    let table_elements = u64::from(rows) * u64::from(half);
    let table_index = b.add(b.mul(position.clone(), b.u32(half)), frequency);
    let cos = b.local(f);
    let sin = b.local(f);
    let x = ctx.load_mapped(&ops[0], index.clone(), elements)?;
    let rotated = ctx.load_mapped(&ops[0], rotated_index, elements)?;
    let signed = b.mul(rotated, b.select(negative, b.f32(-1.0), b.f32(1.0)));
    let out = ctx.linear_view(ctx.output()?)?;
    let body = vec![Stmt::If {
        condition: b.lt(index.clone(), b.u32(elements as u32)),
        accept: vec![
            Stmt::StoreLocal {
                dst: cos.clone(),
                value: b.f32(0.0),
            },
            Stmt::StoreLocal {
                dst: sin.clone(),
                value: b.f32(0.0),
            },
            // Gather's out-of-range table row is zero, with no invalid load.
            Stmt::If {
                condition: b.lt(position, b.u32(rows)),
                accept: vec![
                    Stmt::StoreLocal {
                        dst: cos.clone(),
                        value: ctx.load_mapped(&ops[1], table_index.clone(), table_elements)?,
                    },
                    Stmt::StoreLocal {
                        dst: sin.clone(),
                        value: ctx.load_mapped(&ops[2], table_index, table_elements)?,
                    },
                ],
                reject: Vec::new(),
            },
            Stmt::Store {
                dst: out,
                addr: Addr::Linear(index),
                value: b.add(
                    b.mul(x, b.load_local(cos)),
                    b.mul(signed, b.load_local(sin)),
                ),
                mask: b.bool(true),
            },
        ],
        reject: Vec::new(),
    }];
    let grid = distribute_workgroups(
        elements.div_ceil(256) as u32,
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    Ok(ctx.finish("decode_rope", grid, 256, body))
}
