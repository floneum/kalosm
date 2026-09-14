//! `Slab`: a chain of launches as the stages of one kernel, one workgroup
//! per slab of the leading rows they all keep independent.
//!
//! Workgroup `s` runs every stage over slab `s` of that stage's index space,
//! with the workgroup's lanes striding the slab's elements — or, for a fold,
//! a group of lanes per output row striding the reduced axis and closing
//! over a workgroup tree — and a storage barrier between stages, since a
//! stage reads only what this workgroup wrote. Every stage but the last
//! stores into its member's own buffer; the last stores into the slab's.
//! The block is [`slab_block`] of the widest stage's share of one slab.
//!
//! The stages are the plain per-element and per-row loops of a map and a
//! fold: no tiling, no cooperative loads. What a slab buys is the dispatch
//! count, which at the shapes where the extractor chooses it is most of what
//! the chain costs.

use fusor_ir::Result;
use fusor_ir::dtype::NumericContract;
use fusor_ir::error::Error;
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ScalarElement, Stmt, StorageView, TileBinaryOp,
    TileCompareOp, TileExpr,
};
use fusor_ir::ir::launch::{IndexSpace, Launch, SchedPoint, slab_block, slab_lanes_per_row};
use fusor_ir::ir::kernel::{MergeBody, ReduceKind, Tile};
use fusor_ir::egraph::ClassId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::lower::map_fold::identity_expr;
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
}

/// `value` at member-space index `at`, under `live`.
fn store(ctx: &mut Ctx<'_>, lanes: &Lanes, dst: &Dst, at: TileExpr, value: TileExpr, live: TileExpr) -> Stmt {
    match dst {
        Dst::Buffer(view) => Stmt::Store {
            dst: view.clone(),
            addr: Addr::Linear(at),
            value,
            mask: live,
        },
        Dst::Tile(tile, share) => {
            let share_e = ctx.b.u32(*share);
            let base = ctx.b.mul(lanes.slab.clone(), share_e);
            let local = ctx.b.sub(at, base);
            Stmt::If {
                condition: live,
                accept: vec![Stmt::StoreTile {
                    dst: tile.clone(),
                    index: local,
                    value,
                }],
                reject: Vec::new(),
            }
        }
    }
}

/// One operand of a stage at flat index `index` of the stage's space: from
/// the member's tile when the slab keeps it, else through the buffer.
fn load(
    ctx: &mut Ctx<'_>,
    lanes: &Lanes,
    operand: &fusor_ir::ir::launch::Operand,
    index: TileExpr,
    total: u64,
) -> Result<TileExpr> {
    let class = ctx.cx.graph.class_of(operand.src);
    if let Some((tile, share)) = lanes.private.get(&class) {
        let at = ctx.operand_address(operand, index, total)?;
        let share_e = ctx.b.u32(*share);
        let base = ctx.b.mul(lanes.slab.clone(), share_e);
        let local = ctx.b.sub(at, base);
        return Ok(ctx.b.load_tile(tile.clone(), local));
    }
    ctx.load_mapped(operand, index, total)
}

pub(crate) fn lower_kslab(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Slab { slabs, members, .. } = op else {
        return Err(Error::Plan("lower_kslab on a non-Slab node".into()));
    };
    if theta != SchedPoint::Point {
        return Err(Error::Plan(format!("a Slab lowers at Point, not {theta:?}")));
    }
    let Some(last) = members.last().copied() else {
        return Err(Error::Plan("a Slab has no members".into()));
    };
    let mut widest = 0u64;
    for m in members.iter().copied() {
        let Op::Launch(stage) = &ctx.cx.graph.node(m).op else {
            return Err(Error::Plan(format!("slab member {m} is not a Launch node")));
        };
        let total = match stage {
            Launch::Map { space, .. } | Launch::Fold { space, .. } => space.iterations(),
            _ => None,
        }
        .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))?;
        widest = widest.max(total / u64::from((*slabs).max(1)));
    }
    let block = slab_block(widest, ctx.caps).max(ctx.block_floor);
    let grid = distribute_workgroups(*slabs, ctx.caps.limits.max_compute_workgroups_per_dimension);

    let global = ctx.global_index(block, grid);
    let block_e = ctx.b.u32(block);
    let slab = ctx
        .b
        .binary(TileBinaryOp::Div, global.clone(), block_e.clone(), NumericContract::RELAXED);
    let lane = ctx
        .b
        .binary(TileBinaryOp::Rem, global, block_e, NumericContract::RELAXED);
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
        let stage = ctx.cx.graph.node(m).op.clone();
        let Op::Launch(stage) = stage else {
            return Err(Error::Plan(format!("slab member {m} is not a Launch node")));
        };
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
            let tile = ctx
                .b
                .tile("slab_member", ElementType::Scalar(scalar_element(facts.dtype)), &[share]);
            lanes
                .private
                .insert(ctx.cx.graph.class_of(m), (tile.clone(), share));
            Dst::Tile(tile, share)
        };
        let to_tile = matches!(dst, Dst::Tile(..));
        match &stage {
            Launch::Map { .. } => map_stage(&mut ctx, &mut body, &stage, &dst, *slabs, &lanes)?,
            Launch::Fold { .. } => {
                fold_stage(&mut ctx, &mut body, &stage, &dst, *slabs, &mut lanes)?
            }
            other => {
                return Err(Error::Plan(format!(
                    "a slab stage is a Map or a Fold, not {:?}",
                    other.tag()
                )));
            }
        }
        if m != last {
            body.push(if to_tile { Stmt::Barrier } else { Stmt::StorageBarrier });
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

/// `slab * per + it * block + lane`, and whether that lane is live.
fn strided(
    ctx: &mut Ctx<'_>,
    lanes: &Lanes,
    it: &TileExpr,
    per: u32,
) -> (TileExpr, TileExpr) {
    let block_e = ctx.b.u32(lanes.block);
    let step = ctx.b.mul(it.clone(), block_e);
    let within = ctx.b.add(step, lanes.lane.clone());
    let per_e = ctx.b.u32(per);
    let live = ctx
        .b
        .compare(TileCompareOp::Lt, within.clone(), per_e.clone());
    let base = ctx.b.mul(lanes.slab.clone(), per_e);
    let index = ctx.b.add(base, within);
    (index, live)
}

/// A map stage: the lanes stride the slab's elements.
fn map_stage(
    ctx: &mut Ctx<'_>,
    body: &mut Vec<Stmt>,
    stage: &Launch,
    dst: &Dst,
    slabs: u32,
    lanes: &Lanes,
) -> Result<()> {
    let Launch::Map {
        space, body: expr, ops, ..
    } = stage
    else {
        return Err(Error::Plan("map_stage on a non-Map".into()));
    };
    let total = space
        .iterations()
        .ok_or_else(|| Error::Plan("a slab stage needs constant extents".into()))?;
    let per = per_slab(total, slabs)?;
    let iters = per.div_ceil(lanes.block).max(1);

    let it = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let it_e = ctx.b.load_local(it.clone());
    let (index, live) = strided(ctx, lanes, &it_e, per);
    let coords = ctx.coords_from_linear(index.clone(), space)?;
    let mut args = Vec::with_capacity(ops.len());
    for operand in ops {
        args.push(load(ctx, lanes, operand, index.clone(), total)?);
    }
    let value = ctx.eval_scalar(expr, &args, &coords)?;
    let value = ctx.b.cast(value, dst.element());
    let count = ctx.b.u32(iters);
    let st = store(ctx, lanes, dst, index, value, live);
    body.push(Stmt::Loop {
        count: Some(count),
        index: Some(it),
        accumulators: Vec::new(),
        body: vec![st],
    });
    Ok(())
}

/// A fold stage: `lpr` lanes per output row stride the slab's rows, each
/// lane walking every `lpr`th element of the reduced axis with the
/// carrier's lift and merge, then the group closes over a workgroup tree.
fn fold_stage(
    ctx: &mut Ctx<'_>,
    body: &mut Vec<Stmt>,
    stage: &Launch,
    dst: &Dst,
    slabs: u32,
    lanes: &mut Lanes,
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
    let axis = *axis as usize;
    let dims = constant_dims(space)?;
    if axis >= dims.len() {
        return Err(Error::Plan(format!("fold axis {axis} of a rank-{} space", dims.len())));
    }
    let total: u64 = dims.iter().product();
    let k = dims[axis];
    let inner: u64 = dims[axis + 1..].iter().product();
    let rows = total / k.max(1);
    let per = per_slab(rows, slabs)?;
    let lpr = slab_lanes_per_row(lanes.block, u64::from(per), k);
    let groups = lanes.block / lpr;
    let iters = per.div_ceil(groups).max(1);

    // Lane `l` of iteration `it` serves row `slab * per + it * groups + l /
    // lpr` at sub-lane `l % lpr`.
    let it = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let it_e = ctx.b.load_local(it.clone());
    let lpr_e = ctx.b.u32(lpr);
    let group = ctx.b.binary(
        TileBinaryOp::Div,
        lanes.lane.clone(),
        lpr_e.clone(),
        NumericContract::RELAXED,
    );
    let sub = ctx.b.binary(
        TileBinaryOp::Rem,
        lanes.lane.clone(),
        lpr_e.clone(),
        NumericContract::RELAXED,
    );
    let groups_e = ctx.b.u32(groups);
    let step = ctx.b.mul(it_e, groups_e);
    let within = ctx.b.add(step, group);
    let per_e = ctx.b.u32(per);
    let live = ctx
        .b
        .compare(TileCompareOp::Lt, within.clone(), per_e.clone());
    let base = ctx.b.mul(lanes.slab.clone(), per_e);
    let row = ctx.b.add(base, within);

    // The row's first element in the space, and its element at `kk`:
    // `(row / inner) * (k * inner) + kk * inner + row % inner`.
    let inner_e = ctx.b.u32(u32::try_from(inner).unwrap_or(u32::MAX));
    let outer = ctx.b.binary(
        TileBinaryOp::Div,
        row.clone(),
        inner_e.clone(),
        NumericContract::RELAXED,
    );
    let rem = ctx.b.binary(
        TileBinaryOp::Rem,
        row.clone(),
        inner_e.clone(),
        NumericContract::RELAXED,
    );
    let span = ctx.b.u32(u32::try_from(k * inner).unwrap_or(u32::MAX));
    let scaled = ctx.b.mul(outer, span);
    let base = ctx.b.add(scaled, rem);
    let j = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let j_e = ctx.b.load_local(j.clone());
    let strided_j = ctx.b.mul(j_e, lpr_e);
    let kk = ctx.b.add(strided_j, sub.clone());
    let k_e = ctx.b.u32(u32::try_from(k).unwrap_or(u32::MAX));
    let in_axis = ctx.b.compare(TileCompareOp::Lt, kk.clone(), k_e);
    let offset = ctx.b.mul(kk, inner_e);
    let index = ctx.b.add(base, offset);

    let coords = ctx.coords_from_linear(index.clone(), space)?;
    let mut args = Vec::with_capacity(ops.len());
    for operand in ops {
        args.push(load(ctx, lanes, operand, index.clone(), total)?);
    }
    // One accumulator per slot. `merge` reads the accumulators as its first
    // `width` arguments and the lifted element as the next `width`; a
    // sub-lane past the axis lifts the identity.
    let acc_elem = scalar_element(*acc);
    let acc_ty = ElementType::Scalar(acc_elem);
    let mut lifted = Vec::with_capacity(width);
    for (slot, lift) in carrier.lift.iter().take(width).enumerate() {
        let v = ctx.eval_scalar(lift, &args, &coords)?;
        let v = ctx.b.cast(v, acc_ty);
        let ident = identity_expr(ctx, carrier.identity[slot], acc_elem);
        lifted.push(ctx.b.select(in_axis.clone(), v, ident));
    }
    let locals: Vec<_> = (0..width).map(|_| ctx.b.local(acc_ty)).collect();
    let mut merge_args: Vec<TileExpr> = locals.iter().map(|l| ctx.b.load_local(l.clone())).collect();
    let partials = merge_args.clone();
    merge_args.extend(lifted);
    let mut accumulators = Vec::with_capacity(width);
    for (slot, local) in locals.iter().enumerate() {
        let update = ctx.eval_scalar(&carrier.merge[slot], &merge_args, &[])?;
        let update = ctx.b.cast(update, acc_ty);
        let init = identity_expr(ctx, carrier.identity[slot], acc_elem);
        accumulators.push(Accumulator {
            local: local.clone(),
            init,
            update,
        });
    }
    let count = ctx.b.u32(u32::try_from(k.div_ceil(u64::from(lpr))).unwrap_or(u32::MAX));
    let mut stmts = vec![Stmt::Loop {
        count: Some(count),
        index: Some(j),
        accumulators,
        body: Vec::new(),
    }];

    // The cross-lane close over each row's `lpr` lanes: one scratch tile
    // per slot, shared across the kernel's fold stages. A one-lane group
    // already holds its row.
    let reduced: Vec<TileExpr> = if lpr <= 1 {
        partials
    } else {
        while lanes.scratch.len() < width {
            lanes.scratch.push(ctx.b.tile("slab_scratch", acc_ty, &[lanes.block]));
        }
        let scratch: SmallVec<[Tile; 4]> = lanes.scratch[..width].iter().cloned().collect();
        let lhs: SmallVec<[_; 4]> = (0..width).map(|_| ctx.b.local(acc_ty)).collect();
        let rhs: SmallVec<[_; 4]> = (0..width).map(|_| ctx.b.local(acc_ty)).collect();
        let outs: SmallVec<[_; 4]> = (0..width).map(|_| ctx.b.local(acc_ty)).collect();
        let mut tree_args: Vec<TileExpr> = Vec::with_capacity(2 * width);
        for l in lhs.iter().chain(rhs.iter()) {
            tree_args.push(ctx.b.load_local(l.clone()));
        }
        let mut merged: SmallVec<[TileExpr; 4]> = SmallVec::new();
        for merge in carrier.merge.iter().take(width) {
            let v = ctx.eval_scalar(merge, &tree_args, &[])?;
            merged.push(ctx.b.cast(v, acc_ty));
        }
        stmts.push(Stmt::Reduce {
            kind: Box::new(ReduceKind::Workgroup {
                scratch: scratch[0].clone(),
                group_size: lpr,
            }),
            values: partials.into_iter().collect(),
            merge: Box::new(MergeBody {
                lhs,
                rhs,
                body: merged,
            }),
            fast: None,
            outs: outs.clone(),
            scratch,
        });
        outs.iter().map(|l| ctx.b.load_local(l.clone())).collect()
    };

    // A post lands at `row * posts + slot`: the axis the fold's facts append
    // to its shape when there is more than one. The group's first lane
    // stores.
    let zero = ctx.b.u32(0);
    let first = ctx.b.compare(TileCompareOp::Eq, sub, zero);
    let mask = ctx.b.and(live, first);
    let width_e = ctx.b.u32(post.len() as u32);
    let base = ctx.b.mul(row.clone(), width_e);
    for (slot, post) in post.iter().enumerate() {
        let value = ctx.eval_scalar(post, &reduced, std::slice::from_ref(&row))?;
        let value = ctx.b.cast(value, dst.element());
        let off = ctx.b.u32(slot as u32);
        let at = ctx.b.add(base.clone(), off);
        stmts.push(store(ctx, lanes, dst, at, value, mask.clone()));
    }
    let count = ctx.b.u32(iters);
    body.push(Stmt::Loop {
        count: Some(count),
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
