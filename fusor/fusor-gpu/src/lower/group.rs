//! `Group`: independent launches as one dispatch. Member `i` owns workgroups
//! `[off_i, off_i + n_i)` and runs its own lowering's body with the workgroup
//! index restated as `linear - off_i`, at the widest member's block.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::extract::Dispatch;
use fusor_ir::ir::kernel::{KernelIr, Stmt};
use fusor_ir::ir::launch::{Launch, SchedPoint};
use fusor_ir::target::LowerCtx;

use crate::lower::{Ctx, distribute_workgroups, lower_member};

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
     -> Result<KernelIr> {
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
        lower_member(
            ctx.caps,
            cx.graph.node(m),
            theta_m,
            &mcx,
            ctx.binding.clone(),
            ctx.pack.clone(),
            local,
            floor,
            &ctx.buffers,
        )
    };
    // Find the common block first: widening changes a member's grid, so
    // offsets must use the widened grid.
    let mut block = 1;
    for m in members.iter().copied() {
        block = block.max(lower_at(m, linear.clone(), 0)?.block);
    }
    let mut offset = 0u32;
    let mut body: Vec<Stmt> = Vec::with_capacity(members.len());
    for m in members.iter().copied() {
        let local = ctx.b.sub(linear.clone(), ctx.b.u32(offset));
        let k = lower_at(m, local.clone(), block)?;
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
        let inside = ctx.b.lt(local, ctx.b.u32(n));
        // A uniform barrier between members lets the arena alias their tiles.
        if !body.is_empty() {
            body.push(Stmt::Barrier);
        }
        body.push(Stmt::If {
            condition: inside,
            accept: k.body,
            reject: Vec::new(),
        });
    }
    let grid = distribute_workgroups(
        offset.max(1),
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    Ok(ctx.finish("kgroup", grid, block, body))
}
