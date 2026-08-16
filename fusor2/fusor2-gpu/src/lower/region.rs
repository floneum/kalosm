//! `Region`: the multi-output fusion primitive, several members run in one
//! dispatch over one linearized index.
//!
//! ## The schedule
//!
//! The body carries a `sched` field, so extraction resolves it like any other
//! node's, and the body consumes the selected [`MapTiling`]. The workgroup width
//! is a whole number of subgroups, never wider than the work there is, computed
//! via [`block_for`].

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
use fusor2_ir::ir::launch::{Launch, MapTiling, SchedPoint};
use fusor2_ir::ir::kernel::{
    Addr, ElementType, KernelIr, ScalarElement, Stmt, TileBinaryOp, TileCompareOp,
};

use crate::lower::{Ctx, distribute_workgroups};


/// The workgroup width for a linear body needing `lanes` lanes: a whole
/// number of subgroups, never wider than the work and never wider than the
/// device allows. The ceiling is [`fusor2_tile::domains::emitted_block`],
/// shared with the fold domain.
pub(crate) fn block_for(caps: &Caps, lanes: u64) -> u32 {
    let cap = fusor2_tile::domains::emitted_block(1, caps)
        .min(caps.limits.max_compute_workgroup_size[0])
        .max(1);
    let sgw = caps.subgroup_width().max(1).min(cap);
    let want = u32::try_from(lanes.max(1).min(u64::from(cap))).unwrap_or(cap);
    want.div_ceil(sgw).max(1).saturating_mul(sgw).min(cap)
}



/// Read the selected point.
///
/// `Point` is the untiled member of this node's own domain — the fallback
/// `Session::run` supplies when `Extraction::theta` has no entry. A `Map`
/// point naming an axis is refused: this body has no axis to tile.
fn tiling_of(theta: SchedPoint) -> Result<MapTiling> {
    match theta {
        SchedPoint::Point => Ok(MapTiling {
            dim: None,
            tm: 1,
            vector: 1,
        }),
        SchedPoint::Map(t) if t.dim.is_none() => Ok(MapTiling {
            tm: t.tm.max(1),
            ..t
        }),
        SchedPoint::Map(t) => Err(Error::Plan(format!(
            "a region walks one linearized index and has no axis {:?} to tile",
            t.dim
        ))),
        other => Err(Error::Plan(format!(
            "a region needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

/// One pass over the shared index space, one store per `live_out`.
pub(crate) fn lower_kregion(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Region {
        members, live_outs, ..
    } = op
    else {
        return Err(Error::Plan("lower_kregion on a non-Region node".into()));
    };
    if members.is_empty() {
        return Err(Error::Plan("a region has no members".into()));
    }
    let limits = ctx.caps.limits;

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let count = out_view.layout.element_count();
    let elements = u32::try_from(count)
        .map_err(|_| Error::Plan("region output exceeds a u32 element count".into()))?
        .max(1);

    // The members share one index space, so the live-outs' element count is
    // the domain the tiling is derived from.
    let tm = tiling_of(theta)?.tm;
    let block = block_for(ctx.caps, u64::from(elements).div_ceil(u64::from(tm)));
    let per_group = block.saturating_mul(tm);

    let grid = distribute_workgroups(
        elements.div_ceil(per_group),
        limits.max_compute_workgroups_per_dimension,
    );
    let base = ctx.global_index(block, grid);
    // At `tm == 1` a group covers exactly `block`.
    let tile_base = if tm == 1 {
        base
    } else {
        let block_e = ctx.b.u32(block);
        let group = ctx.b.binary(
            TileBinaryOp::Div,
            base.clone(),
            block_e.clone(),
            NumericContract::RELAXED,
        );
        let lane = ctx
            .b
            .binary(TileBinaryOp::Rem, base, block_e, NumericContract::RELAXED);
        let group_e = ctx.b.u32(per_group);
        let scaled = ctx.b.mul(group, group_e);
        ctx.b.add(scaled, lane)
    };
    let bound = ctx.b.u32(elements);

    let mut body = Vec::new();
    let zero_elem = match out_elem {
        ElementType::Scalar(s) => s,
        _ => ScalarElement::F32,
    };
    for t in 0..tm {
        let index = if t == 0 {
            tile_base.clone()
        } else {
            let step = ctx.b.u32(t.saturating_mul(block));
            ctx.b.add(tile_base.clone(), step)
        };
        let live = ctx
            .b
            .compare(TileCompareOp::Lt, index.clone(), bound.clone());

        // The shared value each member computes, read once into a register
        // and written to every live-out.
        let zero = ctx.b.zero(zero_elem);
        let shared = ctx.b.load(
            fusor2_ir::ir::kernel::Source::Storage(out_view.clone()),
            Addr::Linear(index.clone()),
            live.clone(),
            zero,
        );
        let local = ctx.b.local(shared.element());
        body.push(Stmt::StoreLocal {
            dst: local.clone(),
            value: shared,
        });
        let value = ctx.b.load_local(local);

        for slot in live_outs {
            let member = members
                .get(*slot as usize)
                .copied()
                .ok_or_else(|| Error::Plan(format!("region live-out {slot} names no member")))?;
            let view = ctx.linear_view(member).unwrap_or_else(|_| out_view.clone());
            let v = ctx.b.cast(value.clone(), view.buffer.element);
            body.push(Stmt::Store {
                dst: view,
                addr: Addr::Linear(index.clone()),
                value: v,
                mask: live.clone(),
            });
        }
        if live_outs.is_empty() {
            let v = ctx.b.cast(value, out_elem);
            body.push(Stmt::Store {
                dst: out_view.clone(),
                addr: Addr::Linear(index),
                value: v,
                mask: live,
            });
        }
    }

    Ok(ctx.finish("kregion", grid, block, body))
}
