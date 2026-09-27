//! Dense contraction: the cooperative-matrix, SGEMM and SGEMV bodies.
//!
//! The arm that runs is the one extraction selected. `pre_a`/`pre_b`/`post`
//! fuse into the k-loop prologue and epilogue.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, CoopMatrixRole, CoopSrc, ElementType, KernelIr, Local, MemoryLevel,
    ReduceKind, ScalarElement, Source, Stmt, StorageView, Tile, TileCompareOp, TileExpr,
    TileLayout, TileReduceOp, cooperative_store_layout_supported,
};
use fusor_ir::ir::launch::{
    ContractSide, CoopGeom, Family, IndexSpace, Launch, SchedPoint, SgemmParams, SgemvParams,
};
use fusor_ir::scalar::{ScalarExpr, ScalarKind};
use fusor_ir::shape::Dim;

use crate::lower::{Ctx, StagedSource, distribute_workgroups, scalar_element};

/// Dispatch on the family this lowering was selected at.
pub(crate) fn lower_contract(
    ctx: Ctx<'_>,
    op: &Launch,
    family: Family,
    theta: SchedPoint,
) -> Result<KernelIr> {
    let contract = Contract::of(&ctx, op);
    // Which family and which point extraction actually resolved; the answer
    // is not in the graph, it is in `theta`.
    if crate::flags().dump_contract {
        let s = contract.as_ref().map(|c| c.shape);
        eprintln!(
            "CONTRACT family={family:?} theta={theta:?} shape={:?}",
            s.map(|s| (s.m, s.n, s.k, s.batch))
        );
    }
    let contract = contract?;
    match (family, theta) {
        (Family::Coop, SchedPoint::Coop { geom, staging }) => {
            lower_coop(ctx, &contract, geom, staging)
        }
        (Family::Sgemm, SchedPoint::Sgemm(p)) => lower_sgemm(ctx, &contract, p),
        (Family::Sgemv, SchedPoint::Sgemv(p)) if p.cols > 1 => {
            lower_sgemv_subgroup_cols(ctx, &contract, p)
        }
        (Family::Sgemv, SchedPoint::Sgemv(p)) => lower_sgemv(ctx, &contract, p),
        (f, t) => Err(Error::Plan(format!(
            "family {f:?} cannot run at schedule point {t:?}"
        ))),
    }
}

#[derive(Copy, Clone)]
struct Shape {
    m: u32,
    n: Dim,
    k: Dim,
    batch: u32,
}

/// One side of a contraction: its node side, a staging source per buffer it
/// reads, and the coordinates its `pre` reads, when it reads any.
struct Side<'o> {
    side: &'o ContractSide,
    sources: Vec<StagedSource>,
    coords: Option<SideCoords>,
}

/// What every family reads off a `Contract` node: A as `[batch * m, k]` and
/// B as `[batch * k, n]` in their own strides; a transposed rhs is a stride swap.
struct Contract<'o> {
    shape: Shape,
    post: &'o ScalarExpr,
    acc: ScalarElement,
    a: Side<'o>,
    b: Side<'o>,
}

impl<'o> Contract<'o> {
    fn of(ctx: &Ctx<'_>, op: &'o Launch) -> Result<Self> {
        let Launch::Contract {
            m,
            n,
            k,
            batch,
            post,
            acc,
            a,
            b,
            ..
        } = op
        else {
            return Err(Error::Plan(
                "contract lowering on a non-Contract node".into(),
            ));
        };
        let shape = Shape {
            m: bound_u32(ctx, *m)?,
            n: positive(*n),
            k: positive(*k),
            batch: bound_u32(ctx, *batch)?.max(1),
        };
        let a_rows = Dim::Const(u64::from(shape.batch.saturating_mul(shape.m).max(1)));
        let b_rows = Dim::Const(u64::from(shape.batch)) * shape.k;
        let m_rows = Dim::Const(u64::from(shape.m));
        Ok(Self {
            shape,
            post,
            acc: scalar_element(*acc),
            a: Side {
                side: a,
                sources: ctx.contract_side_sources(a, shape.batch, m_rows, shape.k)?,
                coords: SideCoords::for_side(ctx, a, a_rows, shape.k)?,
            },
            b: Side {
                side: b,
                sources: ctx.contract_side_sources(b, shape.batch, shape.k, shape.n)?,
                coords: SideCoords::for_side(ctx, b, b_rows, shape.n)?,
            },
        })
    }

    /// Output rows: `batch * m`.
    fn rows(&self) -> u32 {
        self.shape.m.saturating_mul(self.shape.batch).max(1)
    }

    /// `(batch index, B's first row)` of output row `row`: the batch index
    /// rides in `row`, and B's rows are `batch * k + kk`.
    fn batch_of(&self, ctx: &Ctx<'_>, row: &TileExpr) -> Result<(TileExpr, TileExpr)> {
        let b = &ctx.b;
        let batch = b.div(row.clone(), b.u32(self.shape.m.max(1)));
        let b_row_base = b.mul(batch.clone(), ctx.dim_expr(self.shape.k)?);
        Ok((batch, b_row_base))
    }

    /// `post` over one output element, stored at `addr` under `mask`.
    #[allow(clippy::too_many_arguments)]
    fn store_post(
        &self,
        ctx: &Ctx<'_>,
        out: &StorageView,
        total: TileExpr,
        row: TileExpr,
        col: TileExpr,
        addr: Addr,
        mask: TileExpr,
    ) -> Result<Stmt> {
        let value = ctx.eval_scalar(self.post, &[total], &[row, col])?;
        Ok(Stmt::Store {
            dst: out.clone(),
            addr,
            value: ctx.b.cast(value, out.buffer.element),
            mask,
        })
    }
}

impl Side<'_> {
    /// This side's `pre` at `(row, col)`, every source loaded under `mask`,
    /// cast to `elem`.
    fn value(
        &self,
        ctx: &Ctx<'_>,
        row: &TileExpr,
        col: &TileExpr,
        mask: &TileExpr,
        batch: &TileExpr,
        elem: ScalarElement,
    ) -> Result<TileExpr> {
        let raws = load_staged(ctx, &self.sources, row, col, mask, batch)?;
        let coords = match &self.coords {
            Some(c) => c.at(ctx, row, col)?,
            None => Vec::new(),
        };
        let value = ctx.eval_scalar(&self.side.pre, &raws, &coords)?;
        Ok(ctx.b.cast(value, elem.element()))
    }
}

/// A masked-out k lane contributes a zero, not `pre(0)`: `pre` may turn the
/// zero fill into `inf`, and `fma(inf, 0, acc)` poisons the k-sum.
fn zero_unless(
    ctx: &Ctx<'_>,
    masked: bool,
    mask: &TileExpr,
    value: TileExpr,
    elem: ScalarElement,
) -> TileExpr {
    match masked {
        true => ctx.b.select(mask.clone(), value, ctx.b.zero(elem)),
        false => value,
    }
}

fn positive(dim: Dim) -> Dim {
    match dim {
        Dim::Const(value) => Dim::Const(value.max(1)),
        _ => dim,
    }
}

fn bound_u32(ctx: &Ctx<'_>, dim: Dim) -> Result<u32> {
    let value = ctx.binding.require(dim)?;
    u32::try_from(value)
        .map_err(|_| Error::Plan(format!("contraction extent {value} exceeds a u32")))
}

fn ceil_extent(ctx: &Ctx<'_>, dim: Dim, divisor: u32) -> Result<TileExpr> {
    let b = &ctx.b;
    if let Some(value) = dim.as_const() {
        let value = u32::try_from(value.div_ceil(u64::from(divisor.max(1))))
            .map_err(|_| Error::Plan("contraction tile count exceeds a u32".into()))?;
        return Ok(b.u32(value));
    }
    let value = ctx.dim_expr(dim)?;
    if divisor <= 1 {
        return Ok(value);
    }
    let (whole, rem) = b.divrem(value, b.u32(divisor.max(1)));
    let zero = b.u32(0);
    let tail = b.select(
        b.compare(TileCompareOp::Gt, rem, zero.clone()),
        b.u32(1),
        zero,
    );
    Ok(b.add(whole, tail))
}

/// Grid swizzle group along M: the number of M blocks one traversal of the N
/// blocks keeps resident, computed from the plan-carried geometry.
pub(crate) fn swizzle_group_m(geom: CoopGeom, n: u32) -> u32 {
    let n_blocks = n.div_ceil(geom.bn.max(1)).max(1);
    n_blocks.clamp(1, 8)
}

/// Everything a [`CoopGeom`] implies but does not store.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CoopShape {
    /// Output columns one N pass covers.
    bn_pass: u32,
    /// Rows and columns of the sub-block one subgroup owns inside a pass.
    sg_rows: u32,
    sg_cols: u32,
    /// `COOP_DIM`-sided accumulator fragments that subgroup carries.
    frags_m: u32,
    frags_n: u32,
    /// `COOP_DIM` slices of one staged K tile.
    kk_steps: u32,
    /// Lanes in the workgroup.
    lanes: u32,
}

impl CoopShape {
    /// `CoopGeom::legal` plus whole-fragment K tiles and N passes; a fragment
    /// grid that does not tile its block is an error, not a truncation.
    fn of(geom: CoopGeom, width: u32, max_lanes: u32) -> Result<Self> {
        let dim = CoopGeom::COOP_DIM;
        if !geom.legal(width, max_lanes)
            || !geom.bn.is_multiple_of(geom.n_passes)
            || !geom.bk.is_multiple_of(dim)
        {
            return Err(Error::Plan(format!(
                "coop geometry {geom:?} is illegal at subgroup width {width}"
            )));
        }
        let bn_pass = geom.bn / geom.n_passes;
        let sg_rows = geom.bm / geom.rg;
        let sg_cols = bn_pass / geom.cg;
        Ok(Self {
            bn_pass,
            sg_rows,
            sg_cols,
            frags_m: sg_rows / dim,
            frags_n: sg_cols / dim,
            kk_steps: geom.bk / dim,
            lanes: geom.lanes(width),
        })
    }
}

/// `CoopLoad` / `CoopMma` / `CoopStore`.
///
/// One workgroup per `(split, batch, m_block, n_block)`; the k loop stages
/// `staging` K tiles per iteration between two barriers. `pre` is zeroed past
/// the logical extents; `post` fuses into the epilogue over this workgroup's
/// own disjoint output block.
fn lower_coop(ctx: Ctx<'_>, c: &Contract<'_>, geom: CoopGeom, staging: u8) -> Result<KernelIr> {
    let shape = c.shape;
    let n = bound_u32(&ctx, shape.n)?.max(1);
    let width = ctx.caps.subgroup_width();
    let max_lanes = ctx.caps.limits.max_compute_invocations_per_workgroup;
    let cs = CoopShape::of(geom, width, max_lanes)?;
    let depth = u32::from(staging.max(1));
    let dim = CoopGeom::COOP_DIM;
    let acc_elem = c.acc;
    let operand_elem = scalar_element(ctx.plan_dtype(c.a.side.primary().src)?);
    let b = &ctx.b;

    // Allocation maps logical output axes onto whole padded matrix tiles.
    // Cooperative stores use their flattened physical matrix coordinates.
    let out = ctx.output()?;
    let tiles_m = shape.m.max(1).div_ceil(geom.bm.max(1)).max(1);
    let tiles_n = n.div_ceil(geom.bn.max(1)).max(1);
    let m_padded = tiles_m.saturating_mul(geom.bm);
    let n_padded = tiles_n.saturating_mul(geom.bn);
    let out_view = StorageView {
        buffer: ctx.buffer(out)?,
        offset: ctx.offset_of(out),
        layout: TileLayout::contiguous(
            MemoryLevel::Storage,
            &[shape.batch.saturating_mul(m_padded).max(1), n_padded],
        ),
    };
    let out_elem = out_view.buffer.element;

    // Operand staging tiles: `staging` buffers stacked in one declaration, the
    // footprint `verify_launch::coop_tiles` admitted.
    let a_tile = b.tile(
        "coop_a",
        operand_elem.element(),
        &[depth.saturating_mul(geom.bm), geom.bk],
    );
    let b_tile = b.tile(
        "coop_b",
        operand_elem.element(),
        &[depth.saturating_mul(geom.bk), cs.bn_pass],
    );

    // A group member runs at the group's block: extra subgroups help stage
    // the tiles and mirror an owning subgroup's fragments, storing nothing.
    let block = ctx.block(cs.lanes);
    let groups = shape
        .batch
        .saturating_mul(tiles_m)
        .saturating_mul(tiles_n)
        .max(1);
    let grid = distribute_workgroups(groups, ctx.caps.limits.max_compute_workgroups_per_dimension);

    let lane = b.builtin(Builtin::Lane);
    let tile_id = workgroup_index(&ctx, grid, groups);
    let per_batch = tiles_m.saturating_mul(tiles_n).max(1);
    let (batch_index, local_tile) = split_const(&ctx, tile_id, shape.batch, per_batch);
    let group_m = swizzle_group_m(geom, n);
    let (m_tile, n_tile) = swizzle_tile(&ctx, local_tile, tiles_m, tiles_n, group_m);
    let row_block = b.mul(m_tile, b.u32(geom.bm.max(1)));
    let col_block = b.mul(n_tile, b.u32(geom.bn.max(1)));

    // Operand row origins of this batch element; `row` counts `(batch, m)`.
    let m_e = b.u32(shape.m.max(1));
    let k_e = ctx.dim_expr(shape.k)?;
    let a_batch_base = b.mul(batch_index.clone(), m_e.clone());
    let b_batch_base = b.mul(batch_index.clone(), k_e.clone());
    let a_row_base = b.add(a_batch_base.clone(), row_block.clone());
    let a_row_limit = b.add(a_batch_base, m_e);
    let b_row_limit = b.add(b_batch_base.clone(), k_e);
    // Output row origin: the batch index walks the *padded* row space.
    let out_row_base = b.add(
        b.mul(batch_index.clone(), b.u32(m_padded.max(1))),
        row_block,
    );

    // Subgroup fragment origin inside the block.
    let sg_raw = b.builtin(Builtin::SubgroupId);
    let owners = b.u32((cs.lanes / width.max(1)).max(1));
    let sg = match block > cs.lanes {
        true => b.rem(sg_raw.clone(), owners.clone()),
        false => sg_raw.clone(),
    };
    let owns = (block > cs.lanes).then(|| b.lt(sg_raw, owners));
    let (sg_row, sg_col) = b.divrem(sg, b.u32(geom.cg));
    let sg_row_base = b.mul(sg_row, b.u32(cs.sg_rows));
    let sg_col_base = b.mul(sg_col, b.u32(cs.sg_cols));
    let frag_origin = |r: u32, col: u32| {
        (
            b.add(sg_row_base.clone(), b.u32(r.saturating_mul(dim))),
            b.add(sg_col_base.clone(), b.u32(col.saturating_mul(dim))),
        )
    };

    let post_is_identity = matches!(c.post.kind(), ScalarKind::Arg(0));
    // A mixed-type arena has one untyped physical binding. Stage through
    // the declared accumulator tile so its scratch stays in the arena plan.
    let mixed_binding = ctx
        .buffers
        .iter()
        .any(|buffer| buffer.binding == out_view.buffer.binding && buffer.element != out_elem);
    let needs_stage =
        mixed_binding || (!ctx.caps.mixed_precision_coop_store && acc_elem.element() != out_elem);
    let stage_tile: Option<Tile> = (needs_stage
        || !cooperative_store_layout_supported(&out_view.layout))
    .then(|| b.tile("coop_acc", acc_elem.element(), &[geom.bm, cs.bn_pass]));

    let k_limit = ctx.dim_expr(shape.k)?;
    let n_limit = b.u32(n);
    let stage = StageNest {
        lane: &lane,
        batch: &batch_index,
        block,
        elem: operand_elem,
    };
    let mut body: Vec<Stmt> = Vec::new();
    for pass in 0..geom.n_passes {
        let pass_col_base = b.add(col_block.clone(), b.u32(pass.saturating_mul(cs.bn_pass)));

        // The k loop: two barriers around the staging copy separate the previous
        // iteration's fragment reads from this iteration's tile writes.
        let mut loop_body: Vec<Stmt> = vec![Stmt::Barrier];
        let k_index = b.local(ScalarElement::U32.element());
        let iter_base = b.mul(
            b.load_local(k_index.clone()),
            b.u32(depth.saturating_mul(geom.bk)),
        );
        for d in 0..depth {
            let k_base = b.add(iter_base.clone(), b.u32(d.saturating_mul(geom.bk)));
            stage.copy(
                &ctx,
                &mut loop_body,
                &c.a,
                &a_tile,
                d.saturating_mul(geom.bm).saturating_mul(geom.bk),
                [a_row_base.clone(), k_base.clone()],
                [a_row_limit.clone(), k_limit.clone()],
                [geom.bm, geom.bk],
            )?;
            stage.copy(
                &ctx,
                &mut loop_body,
                &c.b,
                &b_tile,
                d.saturating_mul(geom.bk).saturating_mul(cs.bn_pass),
                [b.add(b_batch_base.clone(), k_base), pass_col_base.clone()],
                [b_row_limit.clone(), n_limit.clone()],
                [geom.bk, cs.bn_pass],
            )?;
        }
        loop_body.push(Stmt::Barrier);

        // One accumulator per fragment of this subgroup's sub-block, each
        // folding every `COOP_DIM` K slice of every buffer in the iteration.
        let frag = |role, tile: &Tile, row, col| {
            let src = CoopSrc {
                tile: tile.clone(),
                row,
                col,
                transposed: false,
            };
            b.coop_load(role, operand_elem, dim, dim, src)
        };
        let mut accumulators = Vec::with_capacity((cs.frags_m * cs.frags_n) as usize);
        for r in 0..cs.frags_m {
            for col in 0..cs.frags_n {
                let (a_row, b_col) = frag_origin(r, col);
                let local = b.local(ElementType::CoopMatrix {
                    scalar: acc_elem,
                    role: CoopMatrixRole::C,
                    rows: dim,
                    cols: dim,
                });
                let mut update = b.load_local(local.clone());
                for d in 0..depth {
                    let a_row_d = b.add(b.u32(d.saturating_mul(geom.bm)), a_row.clone());
                    let b_buf = d.saturating_mul(geom.bk);
                    for step in 0..cs.kk_steps {
                        let offset = step.saturating_mul(dim);
                        let a_frag =
                            frag(CoopMatrixRole::A, &a_tile, a_row_d.clone(), b.u32(offset));
                        let b_row = b.u32(b_buf.saturating_add(offset));
                        let b_frag = frag(CoopMatrixRole::B, &b_tile, b_row, b_col.clone());
                        update = b.coop_mma(a_frag, b_frag, update);
                    }
                }
                // A fragment accumulator starts from a zero fragment: a
                // scalar zero has the wrong `ElementType`.
                accumulators.push(Accumulator {
                    local,
                    init: b.coop_zero(CoopMatrixRole::C, acc_elem, dim, dim),
                    update,
                });
            }
        }
        let locals: Vec<Local> = accumulators.iter().map(|a| a.local.clone()).collect();
        let count = ceil_extent(&ctx, shape.k, geom.bk.max(1).saturating_mul(depth))?;
        body.push(Stmt::Loop {
            count: Some(count),
            index: Some(k_index),
            accumulators,
            body: loop_body,
        });

        let mut stores: Vec<Stmt> = Vec::new();
        for (i, local) in (0u32..).zip(locals) {
            let (frag_row, frag_col) = frag_origin(i / cs.frags_n, i % cs.frags_n);
            let acc = b.load_local(local);
            stores.push(match &stage_tile {
                Some(tile) => Stmt::CoopStoreTile {
                    acc,
                    tile: tile.clone(),
                    row: frag_row,
                    col: frag_col,
                },
                None => Stmt::CoopStore {
                    acc,
                    dst: out_view.clone(),
                    addr: Addr::Rc2 {
                        row: b.add(out_row_base.clone(), frag_row),
                        col: b.add(pass_col_base.clone(), frag_col),
                    },
                },
            });
        }
        match &owns {
            Some(owns) => body.push(Stmt::If {
                condition: owns.clone(),
                accept: stores,
                reject: Vec::new(),
            }),
            None => body.extend(stores),
        }

        // `post` over this workgroup's own block, reading each element
        // through `value_at(flat, row, col, active)`.
        let epilogue =
            |body: &mut Vec<Stmt>,
             value_at: &dyn Fn(TileExpr, &TileExpr, &TileExpr, &TileExpr) -> TileExpr|
             -> Result<()> {
                let (rows, cols) = (geom.bm, cs.bn_pass);
                per_lane_block(
                    &ctx,
                    body,
                    &lane,
                    rows,
                    cols,
                    block,
                    |out, flat, lr, lc, active| {
                        let row = b.add(out_row_base.clone(), lr.clone());
                        let col = b.add(pass_col_base.clone(), lc);
                        let value = value_at(flat, &row, &col, &active);
                        let logical_row = b.add(a_row_base.clone(), lr);
                        let addr = Addr::Rc2 {
                            row,
                            col: col.clone(),
                        };
                        out.push(c.store_post(
                            &ctx,
                            &out_view,
                            value,
                            logical_row,
                            col,
                            addr,
                            active,
                        )?);
                        Ok(())
                    },
                )
            };
        match &stage_tile {
            // The staged path already reads every element per lane, so a fused
            // `post` costs nothing extra there.
            Some(tile) => {
                body.push(Stmt::Barrier);
                epilogue(&mut body, &|flat, _, _, _| b.load_tile(tile.clone(), flat))?;
                body.push(Stmt::Barrier);
            }
            // Fragments are opaque to scalar code, so a fused `post` stores, barriers,
            // then maps `post` in place over this workgroup's disjoint block.
            None if !post_is_identity => {
                body.push(Stmt::StorageBarrier);
                epilogue(&mut body, &|_, row, col, active| {
                    let addr = Addr::Rc2 {
                        row: row.clone(),
                        col: col.clone(),
                    };
                    let src = Source::Storage(out_view.clone());
                    b.load(src, addr, active.clone(), b.zero(acc_elem))
                })?;
            }
            None => {}
        }
    }

    Ok(ctx.finish("coop_matmul", grid, block, body))
}

/// The flat workgroup index, linearized against this grid. A cooperative op
/// needs uniform control flow, so an overhang workgroup is clamped onto the
/// last block (only when the grid over-covers) and stores the same values.
fn workgroup_index(ctx: &Ctx<'_>, grid: [u32; 3], groups: u32) -> TileExpr {
    let id = ctx.linear_workgroup();
    let covered = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
    match covered > u64::from(groups) {
        true => ctx.b.min(id, ctx.b.u32(groups.saturating_sub(1))),
        false => id,
    }
}

fn column_workgroup_index(
    ctx: &Ctx<'_>,
    grid: [u32; 3],
    rows: u32,
    columns: Dim,
    tile: u32,
) -> Result<TileExpr> {
    let b = &ctx.b;
    if let Some(n) = columns.as_const() {
        let groups = u64::from(rows).saturating_mul(n.div_ceil(u64::from(tile.max(1))));
        let groups = u32::try_from(groups)
            .map_err(|_| Error::Plan("contraction workgroup count exceeds a u32".into()))?;
        return Ok(workgroup_index(ctx, grid, groups));
    }
    let groups = b.mul(b.u32(rows), ceil_extent(ctx, columns, tile)?);
    let last = b.sub(groups, b.u32(1));
    Ok(b.min(ctx.linear_workgroup(), last))
}

/// `(index / stride, index % stride)`, skipping both operations when the
/// quotient can only ever be zero.
fn split_const(ctx: &Ctx<'_>, index: TileExpr, extent: u32, stride: u32) -> (TileExpr, TileExpr) {
    match extent <= 1 {
        true => (ctx.b.u32(0), index),
        false => ctx.b.divrem(index, ctx.b.u32(stride.max(1))),
    }
}

/// `local_tile -> (m_tile, n_tile)`, walked in super-blocks of `group` M
/// lines M-fastest so a wavefront shares one B column slab. A bijection on
/// `[0, tiles_m * tiles_n)`; `group == 1` is plain row-major.
fn swizzle_tile(
    ctx: &Ctx<'_>,
    local_tile: TileExpr,
    tiles_m: u32,
    tiles_n: u32,
    group: u32,
) -> (TileExpr, TileExpr) {
    let b = &ctx.b;
    if group <= 1 || tiles_m <= 1 || tiles_n <= 1 {
        return b.divrem(local_tile, b.u32(tiles_n.max(1)));
    }
    let full = (tiles_m / group).saturating_mul(group);
    let tail = tiles_m - full;
    let threshold = full.saturating_mul(tiles_n);

    let group_e = b.u32(group);
    let span = b.u32(group.saturating_mul(tiles_n));
    let (super_block, within) = b.divrem(local_tile.clone(), span);
    let (n_in, m_in) = b.divrem(within, group_e.clone());
    let m_full = b.add(b.mul(super_block, group_e), m_in);
    if tail == 0 {
        return (m_full, n_in);
    }

    // The ragged tail, selected branchlessly: nothing wants divergence around
    // the block a cooperative store is about to write.
    let threshold_e = b.u32(threshold);
    let rest = b.sub(local_tile.clone(), threshold_e.clone());
    let (n_tail, m_off) = b.divrem(rest, b.u32(tail));
    let m_tail = b.add(b.u32(full), m_off);
    let in_full = b.lt(local_tile, threshold_e);
    (
        b.select(in_full.clone(), m_full, m_tail),
        b.select(in_full, n_in, n_tail),
    )
}

/// The coordinate vector one contraction side hands its `pre`: `(row, col)`
/// split back into the operand's axes at `split`. Built only when `pre` names
/// a coordinate.
struct SideCoords {
    extents: Vec<Dim>,
    split: usize,
}

impl SideCoords {
    fn for_side(ctx: &Ctx<'_>, side: &ContractSide, rows: Dim, cols: Dim) -> Result<Option<Self>> {
        if !side.pre.reads_index_of() {
            return Ok(None);
        }
        let layout = &side.primary().layout;
        let product = |dims: &[Dim]| dims.iter().copied().fold(Dim::ONE, |a, b| a * b);
        let split = match (0..=layout.rank()).rev().find(|&split| {
            product(&layout.shape()[..split]).known_eq(rows)
                && product(&layout.shape()[split..]).known_eq(cols)
        }) {
            Some(split) => split,
            None => crate::lower::matrix_split_for(
                layout,
                &ctx.binding,
                ctx.binding.require(rows)?,
                ctx.binding.require(cols)?,
            )?,
        };
        let extents = layout.shape().to_vec();
        Ok(Some(Self { extents, split }))
    }

    /// The per-axis coordinates at `(row, col)`, innermost axis of each group
    /// varying fastest.
    fn at(&self, ctx: &Ctx<'_>, row: &TileExpr, col: &TileExpr) -> Result<Vec<TileExpr>> {
        let mut out = vec![ctx.b.u32(0); self.extents.len()];
        for (flat, axes) in [(row, 0..self.split), (col, self.split..self.extents.len())] {
            let mut rest = flat.clone();
            for i in axes.rev() {
                let e = ctx.dim_expr(positive(self.extents[i]))?;
                out[i] = ctx.b.rem(rest.clone(), e.clone());
                rest = ctx.b.div(rest, e);
            }
        }
        Ok(out)
    }
}

/// The element a staging load yields: the buffer's own, or f32 for a decode.
/// Must agree with `Kernel::load`.
fn source_element(src: &Source) -> ScalarElement {
    match src {
        Source::Storage(v) => match v.buffer.element {
            ElementType::Scalar(e) => e,
            _ => ScalarElement::F32,
        },
        Source::Quantized(_) => ScalarElement::F32,
    }
}

/// One load per buffer a side reads, all at the same `(row, col)`; `pre` reads
/// `Arg(0..sources.len())`. Each fill takes its own source's element type.
fn load_staged(
    ctx: &Ctx<'_>,
    sources: &[StagedSource],
    row: &TileExpr,
    col: &TileExpr,
    mask: &TileExpr,
    batch: &TileExpr,
) -> Result<Vec<TileExpr>> {
    let b = &ctx.b;
    sources
        .iter()
        .map(|source| match source {
            StagedSource::Const(lit) => Ok(lit.clone()),
            StagedSource::Mem(source) => {
                let addr = Addr::Rc2 {
                    row: row.clone(),
                    col: col.clone(),
                };
                let fill = b.zero(source_element(source));
                Ok(b.load(source.clone(), addr, mask.clone(), fill))
            }
            StagedSource::Indexed {
                operand,
                cols,
                elements,
                axes,
                rows_per_batch,
            } => {
                let value = if let Some((row_axis, col_axis)) = axes {
                    let base = b.mul(batch.clone(), ctx.dim_expr(*rows_per_batch)?);
                    let local_row = b.sub(row.clone(), base);
                    let layout = &operand.layout;
                    let mut address = ctx.dim_expr(layout.offset())?;
                    for (coordinate, axes) in [
                        (batch, 0..*row_axis),
                        (&local_row, *row_axis..*col_axis),
                        (col, *col_axis..layout.rank()),
                    ] {
                        let term = ctx.strided_address(
                            coordinate.clone(),
                            &layout.shape()[axes.clone()],
                            &layout.strides()[axes],
                        )?;
                        address = b.add(address, term);
                    }
                    ctx.load_operand(operand, address)?
                } else {
                    let flat = b.add(b.mul(row.clone(), ctx.dim_expr(*cols)?), col.clone());
                    ctx.load_mapped(operand, flat, *elements)?
                };
                let fill = b.zero_of(value.element());
                Ok(b.select(mask.clone(), value, fill))
            }
        })
        .collect()
}

/// The lanes of a cooperative workgroup copying operand windows into their
/// staging tiles.
struct StageNest<'e> {
    lane: &'e TileExpr,
    batch: &'e TileExpr,
    block: u32,
    elem: ScalarElement,
}

impl StageNest<'_> {
    /// Copy a `rows x cols` window at `origin` of one side into `tile` from
    /// `tile_base`, `block` elements per pass, applying `pre` on the way in. A
    /// quantized source decodes to f32; past `limit` the tile holds a zero.
    #[allow(clippy::too_many_arguments)]
    fn copy(
        &self,
        ctx: &Ctx<'_>,
        body: &mut Vec<Stmt>,
        side: &Side<'_>,
        tile: &Tile,
        tile_base: u32,
        [row_base, col_base]: [TileExpr; 2],
        [row_limit, col_limit]: [TileExpr; 2],
        [rows, cols]: [u32; 2],
    ) -> Result<()> {
        let b = &ctx.b;
        let total = rows.saturating_mul(cols).max(1);
        let lanes = self.block.max(1);
        let column_major = rows > 1
            && side
                .sources
                .iter()
                .any(|s| matches!(s, StagedSource::Mem(_)))
            && side.sources.iter().all(|source| {
                let layout = match source {
                    StagedSource::Const(_) => return true,
                    StagedSource::Indexed { .. } => return false,
                    StagedSource::Mem(Source::Storage(view)) => &view.layout,
                    StagedSource::Mem(Source::Quantized(view)) => &view.data.layout,
                };
                matches!(layout.indexing.groups.as_slice(), [row, col]
                    if matches!(row.sub_axes.as_slice(), [r] if r.stride == 1)
                        && matches!(col.sub_axes.as_slice(), [c] if c.stride > 1))
            });
        for pass in 0..total.div_ceil(lanes) {
            let flat = match pass {
                0 => self.lane.clone(),
                _ => b.add(self.lane.clone(), b.u32(pass.saturating_mul(lanes))),
            };
            // Lane order follows the source; the shared tile keeps its
            // row-major layout.
            let inner = b.u32(if column_major { rows } else { cols }.max(1));
            let (major, minor) = b.divrem(flat.clone(), inner);
            let (local_row, local_col, tile_index) = match column_major {
                true => (
                    minor.clone(),
                    major.clone(),
                    b.add(b.mul(minor, b.u32(cols)), major),
                ),
                false => (major, minor, flat.clone()),
            };
            let row = b.add(row_base.clone(), local_row);
            let col = b.add(col_base.clone(), local_col);
            let in_row = b.lt(row.clone(), row_limit.clone());
            let mut active = b.and(in_row, b.lt(col.clone(), col_limit.clone()));
            let within = ((pass + 1).saturating_mul(lanes) > total).then(|| {
                let w = b.lt(flat.clone(), b.u32(total));
                active = b.and(active.clone(), w.clone());
                w
            });
            let value = side.value(ctx, &row, &col, &active, self.batch, self.elem)?;
            let store = Stmt::StoreTile {
                dst: tile.clone(),
                index: match tile_base {
                    0 => tile_index,
                    _ => b.add(b.u32(tile_base), tile_index),
                },
                value: b.select(active, value, b.zero(self.elem)),
            };
            body.push(match within {
                Some(w) => Stmt::If {
                    condition: w,
                    accept: vec![store],
                    reject: Vec::new(),
                },
                None => store,
            });
        }
        Ok(())
    }
}

/// Walk a `rows x cols` block, `lanes` elements per step, handing the builder
/// `(flat, local_row, local_col, active)`. A counted loop, not unrolled: the
/// emitter's block-scoped memo would otherwise merge two identical `LoadTile`s
/// across a barrier.
fn per_lane_block(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    lane: &TileExpr,
    rows: u32,
    cols: u32,
    lanes: u32,
    build: impl FnOnce(&mut Vec<Stmt>, TileExpr, TileExpr, TileExpr, TileExpr) -> Result<()>,
) -> Result<()> {
    let b = &ctx.b;
    let total = rows.saturating_mul(cols).max(1);
    let lanes = lanes.max(1);
    let index = b.local(ScalarElement::U32.element());
    let flat = b.add(
        b.mul(b.load_local(index.clone()), b.u32(lanes)),
        lane.clone(),
    );
    let (local_row, local_col) = b.divrem(flat.clone(), b.u32(cols.max(1)));
    // Never constant-true: the final step may be partial, and a load with a
    // constant-true mask has to be *provably* in range for `check_loads`.
    let active = b.lt(flat.clone(), b.u32(total));
    let mut inner: Vec<Stmt> = Vec::new();
    build(&mut inner, flat, local_row, local_col, active)?;
    body.push(Stmt::Loop {
        count: Some(b.u32(total.div_ceil(lanes).max(1))),
        index: Some(index),
        accumulators: Vec::new(),
        body: inner,
    });
    Ok(())
}

/// SGEMM with a per-thread `tn`-wide register accumulator: one lane owns `tn`
/// adjacent output columns of one row and reuses the A element across them.
/// The output is contiguous and every store masked, so the address is
/// `row * n + col`.
fn lower_sgemm(ctx: Ctx<'_>, c: &Contract<'_>, p: SgemmParams) -> Result<KernelIr> {
    let shape = c.shape;
    let b = &ctx.b;
    let max_lanes = ctx.caps.limits.max_compute_invocations_per_workgroup.max(1);
    let block = ctx.block(((p.bm / p.tm.max(1)) * (p.bn / p.tn.max(1))).clamp(1, max_lanes));
    let tn = p.tn.max(1);
    let tn = tn.clamp(
        1,
        shape
            .n
            .as_const()
            .unwrap_or(u64::from(tn))
            .min(u64::from(tn)) as u32,
    );
    let out = ctx.linear_view(ctx.output()?)?;
    let rows = c.rows();
    let grid = crate::lower::tiled_grid_for(
        &IndexSpace::new([Dim::Const(u64::from(rows)), shape.n]),
        block,
        tn,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let index = ctx.global_index(block);
    let n_tiles = ceil_extent(&ctx, shape.n, tn)?;
    let (row, col_tile) = b.divrem(index.clone(), n_tiles.clone());
    let live = b.lt(index, b.mul(b.u32(rows), n_tiles));
    let (batch, b_row_base) = c.batch_of(&ctx, &row)?;

    let k_index = b.local(ScalarElement::U32.element());
    let kk = b.load_local(k_index.clone());
    let av = c.a.value(&ctx, &row, &kk, &live, &batch, c.acc)?;
    let b_row = b.add(b_row_base, kk);
    let col0 = b.mul(col_tile, b.u32(tn));
    let mut accs = Vec::with_capacity(tn as usize);
    let mut cols = Vec::with_capacity(tn as usize);
    for j in 0..tn {
        let col = b.add(col0.clone(), b.u32(j));
        let ok = b.and(live.clone(), b.lt(col.clone(), ctx.dim_expr(shape.n)?));
        let bv = c.b.value(&ctx, &b_row, &col, &ok, &batch, c.acc)?;
        let local = b.local(c.acc.element());
        let read = b.load_local(local.clone());
        accs.push(Accumulator {
            local,
            init: b.zero(c.acc),
            update: b.fma(av.clone(), bv, read),
        });
        cols.push((col, ok));
    }

    let locals: Vec<_> = accs.iter().map(|a| a.local.clone()).collect();
    let mut body = vec![Stmt::Loop {
        count: Some(ctx.dim_expr(shape.k)?),
        index: Some(k_index),
        accumulators: accs,
        body: Vec::new(),
    }];
    let n_e = ctx.dim_expr(shape.n)?;
    for (local, (col, ok)) in locals.into_iter().zip(cols) {
        // `row * n + col` is the flat row-major index of `[batch.., m.., n..]`.
        let addr = Addr::Linear(b.add(b.mul(row.clone(), n_e.clone()), col.clone()));
        let total = b.load_local(local);
        body.push(c.store_post(&ctx, &out, total, row.clone(), col, addr, ok)?);
    }
    Ok(ctx.finish("sgemm", grid, block, body))
}

/// Full vector passes then the remaining contiguous K slices; only the final
/// slice of a runtime extent is masked. `pass_at(step, masked, from, vector,
/// contiguous)` is one pass's updates.
fn gemv_partials(
    ctx: &Ctx<'_>,
    body: &mut Vec<Stmt>,
    k: Dim,
    lanes: u32,
    vector: u32,
    mut locals: Vec<Local>,
    pass_at: impl Fn(TileExpr, bool, &[Local], u32, bool) -> Result<Vec<TileExpr>>,
) -> Result<Vec<Local>> {
    let b = &ctx.b;
    let pass = (lanes * vector).max(1);
    let known = k
        .as_const()
        .map(u32::try_from)
        .transpose()
        .map_err(|_| Error::Plan("contraction extent exceeds a u32".into()))?;
    let full = match known {
        Some(k) => b.u32(k / pass),
        None => b.div(ctx.dim_expr(k)?, b.u32(pass)),
    };
    if known.is_none_or(|k| k >= pass) {
        let index = b.local(ScalarElement::U32.element());
        let step = b.mul(b.load_local(index.clone()), b.u32(pass));
        let updates = pass_at(step, false, &locals, vector, false)?;
        let accumulators = locals
            .iter()
            .zip(updates)
            .map(|(local, update)| Accumulator {
                local: local.clone(),
                init: b.zero_of(local.element),
                update,
            })
            .collect();
        body.push(Stmt::Loop {
            count: Some(full.clone()),
            index: Some(index),
            accumulators,
            body: Vec::new(),
        });
    } else {
        for local in &locals {
            body.push(Stmt::StoreLocal {
                dst: local.clone(),
                value: b.zero_of(local.element),
            });
        }
    }

    let mut tail = |start: TileExpr,
                    count: TileExpr,
                    vector: u32,
                    masked: bool,
                    stride: Option<u32>|
     -> Result<()> {
        let index = stride.map(|_| b.local(ScalarElement::U32.element()));
        let step = match (stride, &index) {
            (Some(stride), Some(index)) => {
                b.add(start, b.mul(b.load_local(index.clone()), b.u32(stride)))
            }
            _ => start,
        };
        let next: Vec<_> = locals.iter().map(|local| b.local(local.element)).collect();
        let updates = pass_at(step, masked, &next, vector, true)?;
        let accumulators = next
            .iter()
            .zip(&locals)
            .zip(updates)
            .map(|((local, from), update)| Accumulator {
                local: local.clone(),
                init: b.load_local(from.clone()),
                update,
            })
            .collect();
        body.push(Stmt::Loop {
            count: Some(count),
            index,
            accumulators,
            body: Vec::new(),
        });
        locals = next;
        Ok(())
    };
    if let Some(k) = known {
        let rem = k % pass;
        let mut at = k - rem;
        for (count, masked) in [(rem / lanes, false), (rem % lanes, true)] {
            if count == 0 {
                continue;
            }
            let vector = if masked { 1 } else { count };
            tail(b.u32(at), b.u32(1), vector, masked, None)?;
            at += if masked { count } else { count * lanes };
        }
    } else {
        let rounds = ceil_extent(ctx, k, lanes)?;
        let count = b.sub(rounds, b.mul(full.clone(), b.u32(vector)));
        tail(b.mul(full, b.u32(pass)), count, 1, true, Some(lanes))?;
    }
    Ok(locals)
}

/// `(row, column)` of workgroup `wg`, the row clamped onto the last one.
fn row_col(ctx: &Ctx<'_>, wg: TileExpr, per_row: TileExpr, rows: u32) -> (TileExpr, TileExpr) {
    let (row, col) = ctx.b.divrem(wg, per_row);
    (ctx.b.min(row, ctx.b.u32(rows - 1)), col)
}

/// Vector contraction with `vector` elements of K per lane per iteration,
/// one workgroup per output element.
fn lower_sgemv(ctx: Ctx<'_>, c: &Contract<'_>, p: SgemvParams) -> Result<KernelIr> {
    let shape = c.shape;
    let b = &ctx.b;
    let width = ctx.caps.subgroup_width();
    let block = (p.subgroups.max(1) * width)
        .min(ctx.caps.limits.max_compute_invocations_per_workgroup)
        .max(1);
    let out = ctx.linear_view(ctx.output()?)?;
    let lane = b.builtin(Builtin::Lane);
    // One workgroup per output element, linearized against the dispatch grid
    // (`distribute_workgroups` may fold it onto a second slab).
    let rows = c.rows();
    let grid = crate::lower::grid_for(
        &IndexSpace::new([Dim::Const(u64::from(rows)), shape.n]),
        1,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let wg = column_workgroup_index(&ctx, grid, rows, shape.n, 1)?;
    // `wg` enumerates `[batch, m, n]` row-major; B's row is the batch's k block
    // plus the loop's k.
    let (row, col) = row_col(&ctx, wg.clone(), ctx.dim_expr(shape.n)?, rows);
    let (batch, b_row_base) = c.batch_of(&ctx, &row)?;
    let locals = vec![b.local(c.acc.element())];

    // One k-loop pass from `step`, continuing the lane's accumulator. Only the
    // tail pass is masked: a mask's clamp defeats the aligned-window decode.
    // Each lane owns `vector` consecutive k elements so a quantized operand
    // amortizes its block decode.
    let pass_at = |step: TileExpr,
                   masked: bool,
                   from: &[Local],
                   vector: u32,
                   _: bool|
     -> Result<Vec<TileExpr>> {
        let lane_base = b.add(step, b.mul(lane.clone(), b.u32(vector)));
        let mut partial = b.load_local(from[0].clone());
        for v in 0..vector {
            let k = b.add(lane_base.clone(), b.u32(v));
            let mask = match masked {
                true => b.lt(k.clone(), ctx.dim_expr(shape.k)?),
                false => b.bool(true),
            };
            let av = c.a.value(&ctx, &row, &k, &mask, &batch, c.acc)?;
            let b_row = b.add(b_row_base.clone(), k);
            let bv = c.b.value(&ctx, &b_row, &col, &mask, &batch, c.acc)?;
            partial = b.fma(
                zero_unless(&ctx, masked, &mask, av, c.acc),
                zero_unless(&ctx, masked, &mask, bv, c.acc),
                partial,
            );
        }
        Ok(vec![partial])
    };
    let mut body: Vec<Stmt> = Vec::new();
    let vector = p.vector.max(1);
    let partials = gemv_partials(&ctx, &mut body, shape.k, block, vector, locals, pass_at)?;

    let lane_partial = b.load_local(partials[0].clone());
    let fixed_subgroup = ctx.caps.subgroups.is_some_and(|s| s.is_fixed()) && block == width;
    let kind = match fixed_subgroup {
        true => ReduceKind::Subgroup,
        false => ReduceKind::Workgroup {
            scratch: b.tile("sgemv_scratch", c.acc.element(), &[block]),
            group_size: block,
        },
    };
    let total = b.reduce(TileReduceOp::Sum, kind, lane_partial);
    let mask = b.eq(lane, b.u32(0));
    body.push(c.store_post(&ctx, &out, total, row, col, Addr::Linear(wg), mask)?);
    Ok(ctx.finish("sgemv", grid, block, body))
}

/// The `cols > 1` SGEMV: `p.cols` output columns per workgroup, each subgroup
/// owning `cols / subgroups` of them. A pass covers `width * vector` k, the
/// activation window hash-conses to one evaluation shared across columns, and
/// the reduction is a subgroup sum. The grid is decomposed as
/// `(row, column group)` so `row` is workgroup-uniform.
fn lower_sgemv_subgroup_cols(ctx: Ctx<'_>, c: &Contract<'_>, p: SgemvParams) -> Result<KernelIr> {
    let shape = c.shape;
    let b = &ctx.b;
    let width = ctx.caps.subgroup_width();
    // The domain generates exactly the points this structure tiles; one
    // reaching here from anywhere else is a plan error.
    if !fusor_tile::domains::sgemv::cols_structure_legal(&p, ctx.caps) {
        return Err(Error::Plan(format!(
            "sgemv {p:?} does not tile whole fixed-width subgroups at width {width}"
        )));
    }
    let block = p.subgroups * width;
    let cps = p.cols / p.subgroups;
    let parts = p.parts.max(1);
    let run = p.run().max(1);
    let out = ctx.linear_view(ctx.output()?)?;
    let rows = c.rows();
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
    let sg_lane = b.builtin(Builtin::SubgroupLane);
    let col_base = b.add(
        b.mul(col_group, b.u32(p.cols)),
        b.mul(b.builtin(Builtin::SubgroupId), b.u32(cps)),
    );
    let col_exact = shape
        .n
        .as_const()
        .is_some_and(|n| n.is_multiple_of(u64::from(p.cols)));
    let n_e = ctx.dim_expr(shape.n)?;

    // One accumulator per owned column, all advanced by the same loop.
    let cols_of_subgroup: Vec<(TileExpr, TileExpr)> = (0..cps)
        .map(|j| {
            let col = b.add(col_base.clone(), b.u32(j));
            let ok = match col_exact {
                true => b.bool(true),
                false => b.lt(col.clone(), n_e.clone()),
            };
            (col, ok)
        })
        .collect();
    let locals: Vec<_> = (0..cps).map(|_| b.local(c.acc.element())).collect();

    // Full passes are unmasked so aligned quantized loads share decoded
    // words. Only the final partial pass needs bounds checks.
    let pass_at = |step: TileExpr,
                   masked: bool,
                   from: &[Local],
                   vector: u32,
                   contiguous: bool|
     -> Result<Vec<TileExpr>> {
        // Each lane owns `vector` elements of the pass: consecutive at `parts == 1`,
        // else `parts` runs of `run` spaced `gap` apart, still covering exactly
        // `width * vector` consecutive k so packed-word loads hash-cons.
        let split = !contiguous && parts > 1;
        let lane_base = match split {
            false => b.add(step, b.mul(sg_lane.clone(), b.u32(vector))),
            true => {
                let (block_idx, within) = b.divrem(sg_lane.clone(), b.u32(p.gap / run));
                let local = b.add(
                    b.mul(block_idx, b.u32(p.gap * parts)),
                    b.mul(within, b.u32(run)),
                );
                b.add(step, local)
            }
        };

        // The pass's activation window, shared by every column this subgroup owns;
        // element `v` sits `(v / run) * gap + v % run` from the lane base.
        let mut a_vals: Vec<(TileExpr, TileExpr, TileExpr)> = Vec::with_capacity(vector as usize);
        for v in 0..vector {
            let off = if split {
                (v / run) * p.gap + (v % run)
            } else {
                v
            };
            let k = b.add(lane_base.clone(), b.u32(off));
            let mask = match masked {
                true => b.lt(k.clone(), ctx.dim_expr(shape.k)?),
                false => b.bool(true),
            };
            let av = c.a.value(&ctx, &row, &k, &mask, &batch, c.acc)?;
            let av = zero_unless(&ctx, masked, &mask, av, c.acc);
            a_vals.push((k, mask, av));
        }

        let mut partials = Vec::with_capacity(cps as usize);
        for (local, (col, col_ok)) in from.iter().zip(&cols_of_subgroup) {
            let mut partial = b.load_local(local.clone());
            for (k, mask, av) in &a_vals {
                let load_mask = match col_exact {
                    true => mask.clone(),
                    false => b.and(mask.clone(), col_ok.clone()),
                };
                let b_row = b.add(b_row_base.clone(), k.clone());
                let bv = c.b.value(&ctx, &b_row, col, &load_mask, &batch, c.acc)?;
                let bv = zero_unless(&ctx, masked, mask, bv, c.acc);
                partial = b.fma(av.clone(), bv, partial);
            }
            partials.push(partial);
        }
        Ok(partials)
    };

    let mut body: Vec<Stmt> = Vec::new();
    let vector = p.vector.max(1);
    let locals = gemv_partials(&ctx, &mut body, shape.k, width, vector, locals, pass_at)?;

    // Each column's partials never leave its subgroup, so the close is a
    // subgroup sum and the store is that subgroup's lane 0.
    let lane0 = b.eq(sg_lane, b.u32(0));
    for (local, (col, col_ok)) in locals.into_iter().zip(cols_of_subgroup) {
        let total = b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, b.load_local(local));
        let addr = Addr::Linear(b.add(b.mul(row.clone(), n_e.clone()), col.clone()));
        let mask = b.and(lane0.clone(), col_ok);
        body.push(c.store_post(&ctx, &out, total, row.clone(), col, addr, mask)?);
    }
    Ok(ctx.finish("sgemv_cols", grid, block, body))
}
