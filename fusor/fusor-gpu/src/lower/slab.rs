//! `Slab`: a chain of launches as the stages of one kernel, one workgroup
//! per slab of the leading rows they all keep independent.
//!
//! Workgroup `s` runs every stage over slab `s` of that stage's index space,
//! with the workgroup's lanes striding the slab's elements — or, for a fold,
//! a group of lanes per output row striding the reduced axis and closing
//! over a subgroup collective or workgroup tree — and a barrier between stages, since a
//! stage reads only what this workgroup wrote. Every stage but the last
//! stores into its member's own buffer; the last stores into the slab's.
//! The block is [`slab_block`] of the widest stage's share of one slab.
//!
//! The stages are the plain per-element and per-row loops of a map and a
//! fold: no tiling, no cooperative loads. What a slab buys is the dispatch
//! count, which at the shapes where the extractor chooses it is most of what
//! the chain costs.

use fusor_ir::Result;
use fusor_ir::egraph::ClassId;
use fusor_ir::error::Error;
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, ReduceKind, ScalarElement, Stmt,
    StorageView, Tile, TileExpr,
};
use fusor_ir::ir::launch::{
    IndexSpace, Launch, Operand, SchedPoint, slab_block, slab_lanes_per_row, slab_subgroup_width,
};
use rustc_hash::FxHashMap;

use crate::lower::map_fold::merge_body;
use crate::lower::{Ctx, distribute_workgroups, scalar_element};

/// One workgroup's coordinates: which slab it owns and which lane this is,
/// and the scratch its fold stages close over, one tile per carrier slot,
/// shared by every stage.
struct Lanes {
    slab: TileExpr,
    lane: TileExpr,
    block: u32,
    scratch: Vec<Tile>,
    /// Members kept in workgroup memory, by class: the tile and this slab's
    /// share of the member's elements, so a member-space index `i` lands at
    /// `i - slab * share`.
    private: FxHashMap<ClassId, (Tile, u32)>,
}

impl Lanes {
    /// Member-space index `at` inside this slab's `share` of a member.
    fn local(&self, ctx: &Ctx<'_>, at: TileExpr, share: u32) -> TileExpr {
        ctx.b
            .sub(at, ctx.b.mul(self.slab.clone(), ctx.b.u32(share)))
    }

    /// `slab * per + it * step + lane`, and whether it is inside the slab.
    fn index(
        &self,
        ctx: &Ctx<'_>,
        it: TileExpr,
        step: u32,
        lane: TileExpr,
        per: u32,
    ) -> (TileExpr, TileExpr) {
        let b = &ctx.b;
        let within = b.add(b.mul(it, b.u32(step)), lane);
        let per = b.u32(per);
        let live = b.lt(within.clone(), per.clone());
        (b.add(b.mul(self.slab.clone(), per), within), live)
    }
}

/// Where a stage stores: the member's buffer, or its workgroup tile.
enum Dst {
    Buffer(StorageView),
    Tile(Tile, u32),
}

impl Dst {
    fn element(&self) -> ElementType {
        match self {
            Dst::Buffer(v) => v.buffer.element,
            Dst::Tile(t, _) => t.element,
        }
    }

    /// `value` at member-space index `at`, under `live`.
    fn store(
        &self,
        ctx: &Ctx<'_>,
        lanes: &Lanes,
        at: TileExpr,
        value: TileExpr,
        live: TileExpr,
    ) -> Stmt {
        match self {
            Dst::Buffer(view) => Stmt::Store {
                dst: view.clone(),
                addr: Addr::Linear(at),
                value,
                mask: live,
            },
            Dst::Tile(tile, share) => Stmt::If {
                condition: live,
                accept: vec![Stmt::StoreTile {
                    dst: tile.clone(),
                    index: lanes.local(ctx, at, *share),
                    value,
                }],
                reject: Vec::new(),
            },
        }
    }
}

/// One operand of a stage at flat index `index` of the stage's space: from
/// the member's tile when the slab keeps it, else through the buffer.
fn load(
    ctx: &Ctx<'_>,
    lanes: &Lanes,
    operand: &Operand,
    index: TileExpr,
    total: u64,
) -> Result<TileExpr> {
    let class = ctx.cx.graph.class_of(operand.src);
    if let Some((tile, share)) = lanes.private.get(&class) {
        let at = ctx.operand_address(operand, index, total)?;
        return Ok(ctx.b.load_tile(tile.clone(), lanes.local(ctx, at, *share)));
    }
    ctx.load_mapped(operand, index, total)
}

pub(crate) fn lower_kslab(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Slab { slabs, members, .. } = op else {
        return Err(Error::Plan("lower_kslab on a non-Slab node".into()));
    };
    if theta != SchedPoint::Point {
        return Err(Error::Plan(format!(
            "a Slab lowers at Point, not {theta:?}"
        )));
    }
    let Some(last) = members.last().copied() else {
        return Err(Error::Plan("a Slab has no members".into()));
    };
    let stage = |m| match &ctx.cx.graph.node(m).op {
        Op::Launch(stage) => Ok(stage),
        _ => Err(Error::Plan(format!("slab member {m} is not a Launch node"))),
    };
    let mut widest = 0u64;
    for m in members.iter().copied() {
        let total = match stage(m)? {
            Launch::Map { space, .. } | Launch::Fold { space, .. } => space.iterations(),
            _ => None,
        }
        .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))?;
        widest = widest.max(total / u64::from((*slabs).max(1)));
    }
    let block = ctx.block(slab_block(widest, ctx.caps));
    let grid = distribute_workgroups(*slabs, ctx.caps.limits.max_compute_workgroups_per_dimension);
    let (slab, lane) = ctx.b.divrem(ctx.global_index(block), ctx.b.u32(block));
    let mut lanes = Lanes {
        slab,
        lane,
        block,
        scratch: Vec::new(),
        private: FxHashMap::default(),
    };

    let out = ctx.output()?;
    let mut body: Vec<Stmt> = Vec::new();
    for m in members.iter().copied() {
        let stage = stage(m)?;
        // A middle member the plan gave no buffer lives in a tile of this
        // slab's share of it.
        let dst = if m == last {
            Dst::Buffer(ctx.linear_view(out)?)
        } else if ctx.has_buffer(m) {
            Dst::Buffer(ctx.linear_view(m)?)
        } else {
            let facts = ctx.cx.graph.facts(m);
            let elements = facts
                .shape
                .iter()
                .try_fold(1u64, |a, d| d.as_const().map(|d| a * d))
                .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))?;
            let share = per_slab(elements, *slabs)?;
            let tile = ctx.b.tile(
                "slab_member",
                ElementType::Scalar(scalar_element(facts.dtype)),
                &[share],
            );
            lanes
                .private
                .insert(ctx.cx.graph.class_of(m), (tile.clone(), share));
            Dst::Tile(tile, share)
        };
        match stage {
            Launch::Map { .. } => map_stage(&ctx, &mut body, stage, &dst, *slabs, &lanes)?,
            Launch::Fold { .. } => fold_stage(&ctx, &mut body, stage, &dst, *slabs, &mut lanes)?,
            other => {
                return Err(Error::Plan(format!(
                    "a slab stage is a Map or a Fold, not {:?}",
                    other.tag()
                )));
            }
        }
        if m != last {
            body.push(match dst {
                Dst::Tile(..) => Stmt::Barrier,
                Dst::Buffer(_) => Stmt::StorageBarrier,
            });
        }
    }
    Ok(ctx.finish("kslab", grid, block, body))
}

/// `count / slabs`, refusing a partition the rule should never have produced.
fn per_slab(count: u64, slabs: u32) -> Result<u32> {
    let slabs = u64::from(slabs.max(1));
    if !count.is_multiple_of(slabs) {
        return Err(Error::Plan(format!(
            "{count} elements do not partition into {slabs} slabs"
        )));
    }
    u32::try_from(count / slabs).map_err(|_| Error::Plan("a slab exceeds u32 addressing".into()))
}

/// A map stage: the lanes stride the slab's elements.
fn map_stage(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    stage: &Launch,
    dst: &Dst,
    slabs: u32,
    lanes: &Lanes,
) -> Result<()> {
    let Launch::Map {
        space,
        body: expr,
        ops,
        ..
    } = stage
    else {
        return Err(Error::Plan("map_stage on a non-Map".into()));
    };
    let b = &ctx.b;
    let total = space
        .iterations()
        .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))?;
    let per = per_slab(total, slabs)?;
    let iters = per.div_ceil(lanes.block).max(1);

    let it = b.local(ScalarElement::U32.element());
    let (index, live) = lanes.index(
        ctx,
        b.load_local(it.clone()),
        lanes.block,
        lanes.lane.clone(),
        per,
    );
    let coords = ctx.coords_from_linear(index.clone(), space)?;
    let args = ops
        .iter()
        .map(|operand| load(ctx, lanes, operand, index.clone(), total))
        .collect::<Result<Vec<_>>>()?;
    let value = b.cast(ctx.eval_scalar(expr, &args, &coords)?, dst.element());
    body.push(Stmt::Loop {
        count: Some(b.u32(iters)),
        index: Some(it),
        accumulators: Vec::new(),
        body: vec![dst.store(ctx, lanes, index, value, live)],
    });
    Ok(())
}

/// A fold stage: `lpr` lanes per output row stride the slab's rows, each
/// lane walking every `lpr`th element of the reduced axis with the
/// carrier's lift and merge, then closing with a collective or a tree.
fn fold_stage(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    stage: &Launch,
    dst: &Dst,
    slabs: u32,
    lanes: &mut Lanes,
) -> Result<()> {
    if let Launch::Fold {
        space,
        axis,
        carrier,
        ..
    } = stage
    {
        let dims = constant_dims(space)?;
        let k = *dims.get(*axis as usize).ok_or_else(|| {
            Error::Plan(format!("fold axis {axis} of a rank-{} space", dims.len()))
        })?;
        let rows = dims.iter().product::<u64>() / k.max(1);
        let per = per_slab(rows, slabs)?;
        if let Some(width) = slab_subgroup_width(lanes.block, u64::from(per), k, carrier, ctx.caps)
        {
            // Local invocation indices have no specified subgroup mapping.
            // Assign rows using actual subgroup IDs and lanes. Only use this
            // path when every slot is occupied; otherwise the original tree
            // covers the workgroup's contiguous local invocation indices.
            let full = ctx.b.eq(
                ctx.b.builtin(Builtin::NumSubgroups),
                ctx.b.u32(lanes.block / width),
            );
            let mut accept = Vec::new();
            let mut reject = Vec::new();
            fold_stage_impl(ctx, &mut accept, stage, dst, slabs, lanes, Some(width))?;
            fold_stage_impl(ctx, &mut reject, stage, dst, slabs, lanes, None)?;
            body.push(Stmt::If {
                condition: full,
                accept,
                reject,
            });
            return Ok(());
        }
    }
    fold_stage_impl(ctx, body, stage, dst, slabs, lanes, None)
}

fn fold_stage_impl(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    stage: &Launch,
    dst: &Dst,
    slabs: u32,
    lanes: &mut Lanes,
    subgroup: Option<u32>,
) -> Result<()> {
    let Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        ..
    } = stage
    else {
        return Err(Error::Plan("fold_stage on a non-Fold".into()));
    };
    let width = carrier.width();
    if !vec_axes.is_empty() || width == 0 || post.is_empty() {
        return Err(Error::Plan(
            "a slab fold stage has scalar slots, at least one post and no promoted axis".into(),
        ));
    }
    let b = &ctx.b;
    let axis = *axis as usize;
    let dims = constant_dims(space)?;
    if axis >= dims.len() {
        return Err(Error::Plan(format!(
            "fold axis {axis} of a rank-{} space",
            dims.len()
        )));
    }
    let total: u64 = dims.iter().product();
    let k = dims[axis];
    let inner: u64 = dims[axis + 1..].iter().product();
    let rows = total / k.max(1);
    let per = per_slab(rows, slabs)?;
    let lpr = subgroup.unwrap_or_else(|| slab_lanes_per_row(lanes.block, u64::from(per), k));
    let groups = lanes.block / lpr;
    let iters = per.div_ceil(groups).max(1);
    let clamp = |v: u64| b.u32(u32::try_from(v).unwrap_or(u32::MAX));

    // Lane `l` of iteration `it` serves row `slab * per + it * groups + l /
    // lpr` at sub-lane `l % lpr`.
    let it = b.local(ScalarElement::U32.element());
    let lpr_e = b.u32(lpr);
    let (group, sub) = match subgroup {
        Some(_) => (
            b.builtin(Builtin::SubgroupId),
            b.builtin(Builtin::SubgroupLane),
        ),
        None => b.divrem(lanes.lane.clone(), lpr_e.clone()),
    };
    let (row, live) = lanes.index(ctx, b.load_local(it.clone()), groups, group, per);

    // The row's element at `kk`: `(row / inner) * (k * inner) + kk * inner +
    // row % inner`.
    let inner_e = clamp(inner);
    let base = b.row_base(row.clone(), inner_e.clone(), clamp(k * inner));
    let j = b.local(ScalarElement::U32.element());
    let kk = b.add(b.mul(b.load_local(j.clone()), lpr_e), sub.clone());
    let in_axis = b.lt(kk.clone(), clamp(k));
    let index = b.at(base, kk, inner_e);

    let coords = ctx.coords_from_linear(index.clone(), space)?;
    let args = ops
        .iter()
        .map(|operand| load(ctx, lanes, operand, index.clone(), total))
        .collect::<Result<Vec<_>>>()?;
    // One accumulator per slot. `merge` reads the accumulators as its first
    // `width` arguments and the lifted element as the next `width`; a
    // sub-lane past the axis lifts the identity.
    let acc_elem = scalar_element(*acc);
    let acc_ty = ElementType::Scalar(acc_elem);
    let mut lifted = Vec::with_capacity(width);
    for (slot, lift) in carrier.lift.iter().take(width).enumerate() {
        let v = b.cast(ctx.eval_scalar(lift, &args, &coords)?, acc_ty);
        lifted.push(b.select(
            in_axis.clone(),
            v,
            b.identity(carrier.identity[slot], acc_elem),
        ));
    }
    let locals: Vec<_> = (0..width).map(|_| b.local(acc_ty)).collect();
    let partials: Vec<TileExpr> = locals.iter().map(|l| b.load_local(l.clone())).collect();
    let mut merge_args = partials.clone();
    merge_args.extend(lifted);
    let mut accumulators = Vec::with_capacity(width);
    for (slot, local) in locals.iter().enumerate() {
        let update = ctx.eval_scalar(&carrier.merge[slot], &merge_args, &[])?;
        accumulators.push(Accumulator {
            local: local.clone(),
            init: b.identity(carrier.identity[slot], acc_elem),
            update: b.cast(update, acc_ty),
        });
    }
    let mut stmts = vec![Stmt::Loop {
        count: Some(clamp(k.div_ceil(u64::from(lpr)))),
        index: Some(j),
        accumulators,
        body: Vec::new(),
    }];

    // The cross-lane close over each row's `lpr` lanes: one scratch tile
    // per slot, shared across the kernel's fold stages. A one-lane group
    // already holds its row.
    let reduced: Vec<TileExpr> = if lpr <= 1 {
        partials
    } else if subgroup.is_some() {
        let op = fusor_ir::ir::kernel::fast_reduce_op(carrier)
            .expect("subgroup admission checked the scalar carrier");
        // Evaluate the collective before the leader-only store. A tile
        // destination wraps its value in an If, where a lazy reduction would
        // run on only the leader and lose the other lanes' contributions.
        let total = b.local(acc_ty);
        stmts.push(Stmt::StoreLocal {
            dst: total.clone(),
            value: b.reduce(op, ReduceKind::Subgroup, partials[0].clone()),
        });
        vec![b.load_local(total)]
    } else {
        while lanes.scratch.len() < width {
            lanes
                .scratch
                .push(b.tile("slab_scratch", acc_ty, &[lanes.block]));
        }
        let scratch = lanes.scratch[..width].iter().cloned().collect();
        let (reduce, outs) = b.merge_tree(scratch, lpr, partials, None, acc_ty, |args| {
            merge_body(ctx, &carrier.merge[..width], args, Some(acc_ty))
        })?;
        stmts.push(reduce);
        outs
    };

    // A post lands at `row * posts + slot`: the axis the fold's facts append
    // to its shape when there is more than one. The group's first lane
    // stores.
    let mask = b.and(live, b.eq(sub, b.u32(0)));
    let base = b.mul(row.clone(), b.u32(post.len() as u32));
    for (slot, post) in post.iter().enumerate() {
        let value = ctx.eval_scalar(post, &reduced, std::slice::from_ref(&row))?;
        let at = b.add(base.clone(), b.u32(slot as u32));
        stmts.push(dst.store(ctx, lanes, at, b.cast(value, dst.element()), mask.clone()));
    }
    body.push(Stmt::Loop {
        count: Some(b.u32(iters)),
        index: Some(it),
        accumulators: Vec::new(),
        body: stmts,
    });
    Ok(())
}

fn constant_dims(space: &IndexSpace) -> Result<Vec<u64>> {
    space
        .dims
        .iter()
        .map(|d| {
            d.as_const()
                .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))
        })
        .collect()
}
