//! `Group`: independent launches as one dispatch. Member `i` owns workgroups
//! `[off_i, off_i + n_i)` and runs its own lowering's body with the workgroup
//! index restated as `linear - off_i`, at the widest member's block.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::extract::Dispatch;
use fusor_ir::ir::kernel::{KernelIr, Stmt, TileExpr};
use fusor_ir::ir::launch::{Launch, SchedPoint};
use fusor_ir::target::LowerCtx;

use crate::lower::{Ctx, GridSpec, Kernel, distribute_workgroups, lower_member};

fn ceil_div(b: &Kernel, value: TileExpr, divisor: TileExpr) -> TileExpr {
    let (quotient, remainder) = b.divrem(value, divisor);
    b.add(
        quotient,
        b.select(b.lt(b.u32(0), remainder), b.u32(1), b.u32(0)),
    )
}

fn nonzero(b: &Kernel, value: TileExpr) -> TileExpr {
    b.select(b.lt(b.u32(0), value), b.u32(1), b.u32(0))
}

fn at_least_one(b: &Kernel, value: TileExpr) -> TileExpr {
    b.select(b.lt(value.clone(), b.u32(1)), b.u32(1), value)
}

fn distributed_span(b: &Kernel, groups: TileExpr, max_per_dim: u32) -> TileExpr {
    let max = b.u32(max_per_dim.max(1));
    let y = at_least_one(
        b,
        b.min(ceil_div(b, groups.clone(), max.clone()), max.clone()),
    );
    let x = b.min(ceil_div(b, groups.clone(), y.clone()), max);
    let xy = b.mul(x.clone(), y.clone());
    let divisor = at_least_one(b, xy);
    let z = at_least_one(b, ceil_div(b, groups, divisor));
    b.mul(b.mul(x, y), z)
}

fn member_span(ctx: &Ctx<'_>, spec: &GridSpec) -> Result<TileExpr> {
    let b = &ctx.b;
    let block = spec.block.max(1);
    let divisor = b.u32(block);
    // Keep product/block as quotient and remainder; never form the full element product.
    let mut quotient = b.u32(1 / block);
    let mut remainder = b.u32(1 % block);
    let rank = spec.space.rank();
    for (axis, dim) in spec.space.dims.iter().enumerate() {
        let mut extent = ctx.dim_expr(*dim)?;
        if axis + 1 == rank {
            extent = ceil_div(b, extent, b.u32(spec.inner_tile.max(1)));
        }
        let (extent_quotient, extent_remainder) = b.divrem(extent.clone(), divisor.clone());
        let remainder_quotient = b.mul(remainder.clone(), extent_quotient);
        let remainder_remainder = b.mul(remainder, extent_remainder);
        let carried = b.add(
            remainder_quotient,
            b.div(remainder_remainder.clone(), divisor.clone()),
        );
        quotient = b.add(b.mul(quotient, extent), carried);
        remainder = b.rem(remainder_remainder, divisor.clone());
    }
    let groups = b.add(quotient, nonzero(b, remainder));
    Ok(distributed_span(
        b,
        groups,
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    ))
}

pub(crate) fn lower_kgroup(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Group { members, .. } = op else {
        return Err(Error::Plan("lower_kgroup on a non-Group node".into()));
    };
    if theta != SchedPoint::Point {
        return Err(Error::Plan(format!(
            "a Group lowers at Point, not {theta:?}"
        )));
    }
    let Some(last) = members.last().copied() else {
        return Err(Error::Plan("a Group has no members".into()));
    };
    let linear = ctx.linear_workgroup();
    let cx = ctx.cx;
    let lower_at = |m: fusor_ir::egraph::Id,
                    local: fusor_ir::ir::kernel::TileExpr,
                    floor: u32|
     -> Result<(KernelIr, Option<GridSpec>)> {
        // The last member's store lands in the launch root's buffer.
        let root = if m == last { cx.launch.root } else { m };
        let dispatch = Dispatch {
            root,
            ..cx.launch.clone()
        };
        let mcx = LowerCtx {
            plan: cx.plan,
            launch: &dispatch,
            graph: cx.graph,
            symbols: cx.symbols,
            dim_bindings: cx.dim_bindings,
        };
        let theta_m = cx
            .plan
            .extraction
            .theta
            .get(&m)
            .copied()
            .unwrap_or(SchedPoint::Point);
        let binding = ctx.binding.fork_group_member();
        let kernel = lower_member(
            ctx.caps,
            cx.graph.node(m),
            theta_m,
            &mcx,
            binding.clone(),
            ctx.pack.clone(),
            local,
            floor,
            &ctx.buffers,
        )?;
        let grid = binding.grid_derivation(kernel.grid, &ctx.caps.limits);
        ctx.binding.merge_group_grid_symbols(&binding);
        Ok((kernel, grid))
    };
    // Find the common block first: widening changes a member's grid, so
    // offsets must use the widened grid.
    let mut block = 1;
    for m in members.iter().copied() {
        let (kernel, _) = lower_at(m, linear.clone(), 0)?;
        block = block.max(kernel.block);
    }
    let mut offset = 0u32;
    let mut dynamic_offset = ctx.b.u32(0);
    let mut body: Vec<Stmt> = Vec::with_capacity(members.len());
    for m in members.iter().copied() {
        let local = ctx.b.sub(linear.clone(), dynamic_offset.clone());
        let (k, grid) = lower_at(m, local.clone(), block)?;
        if k.block != block {
            return Err(Error::Plan(format!(
                "group member {m} lowers at {} lanes, the group at {block}",
                k.block
            )));
        }
        let n = k.grid[0]
            .checked_mul(k.grid[1])
            .and_then(|v| v.checked_mul(k.grid[2]))
            .ok_or_else(|| Error::Plan("a group member's grid overflows a u32".into()))?;
        offset = offset
            .checked_add(n)
            .ok_or_else(|| Error::Plan("a group's grid overflows a u32".into()))?;
        let n = match grid {
            Some(spec) => member_span(&ctx, &spec)?,
            None => ctx.b.u32(n),
        };
        let inside = ctx.b.lt(local, n.clone());
        // A uniform barrier between members lets the arena alias their tiles.
        if !body.is_empty() {
            body.push(Stmt::Barrier);
        }
        body.push(Stmt::If {
            condition: inside,
            accept: k.body,
            reject: Vec::new(),
        });
        dynamic_offset = ctx.b.add(dynamic_offset, n);
    }
    let grid = distribute_workgroups(
        offset.max(1),
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    Ok(ctx.finish("kgroup", grid, block, body))
}
