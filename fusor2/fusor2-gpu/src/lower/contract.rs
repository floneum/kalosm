//! Dense contraction: the cooperative-matrix, SGEMM, SGEMV and generic-fold
//! bodies.
//!
//! All four families coexist in one e-class, so nothing here routes: the arm
//! that runs is the one extraction selected. `pre_a`/`pre_b`/`post` fuse into
//! the k-loop prologue and epilogue; when the device cannot host a
//! mixed-precision cooperative store the accumulator stages through a
//! workgroup tile with a per-lane cast — footprint, never correctness.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
use fusor2_ir::ir::launch::{
    ContractSide, CoopGeom, Family, Launch, SchedPoint, SgemmParams, SgemvParams,
};
use fusor2_ir::ir::kernel::{
    Accumulator, Addr, Builtin, CoopMatrixRole, CoopSrc, ElementType, KernelIr, ReduceKind,
    ScalarElement, Source, Stmt, StorageView, Tile, TileBinaryOp, TileCompareOp, TileExpr,
    TileReduceOp, WorkgroupAxis, cooperative_store_layout_supported,
};
use fusor2_ir::scalar::{ScalarExpr, ScalarKind};
use fusor2_ir::shape::Dim;

use crate::lower::{Ctx, StagedSource, distribute_workgroups, scalar_element};

/// Dispatch on the selected family. The family is a property of *this
/// lowering*, never of the Logical node, so a gemv-shaped contraction cannot pick
/// Coop, have a tile scorer decline, and silently run a third path.
pub(crate) fn lower_contract(
    ctx: Ctx<'_>,
    op: &Launch,
    family: Family,
    theta: SchedPoint,
) -> Result<Vec<KernelIr>> {
    // Which family and which point extraction actually resolved; the answer
    // is not in the graph, it is in `theta`.
    if std::env::var_os("FUSOR2_DUMP_CONTRACT").is_some() {
        let s = shape_of(&ctx, op);
        eprintln!(
            "CONTRACT family={family:?} theta={theta:?} shape={:?}",
            s.map(|s| (s.m, s.n, s.k, s.batch))
        );
    }
    match (family, theta) {
        (
            Family::Coop,
            SchedPoint::Coop {
                geom,
                splits,
                staging,
            },
        ) => lower_coop(ctx, op, geom, splits, staging),
        (Family::Sgemm, SchedPoint::Sgemm(p)) => lower_sgemm(ctx, op, p).map(|k| vec![k]),
        (Family::Sgemv, SchedPoint::Sgemv(p)) => lower_sgemv(ctx, op, p).map(|k| vec![k]),
        (f, t) => Err(Error::Plan(format!(
            "family {f:?} cannot run at schedule point {t:?}"
        ))),
    }
}

struct Shape {
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
}

fn shape_of(ctx: &Ctx<'_>, op: &Launch) -> Result<Shape> {
    let Launch::Contract { m, n, k, batch, .. } = op else {
        return Err(Error::Plan(
            "contract lowering on a non-Contract node".into(),
        ));
    };
    let get = |d: Dim| -> Result<u32> {
        let v = ctx.binding.require(d)?;
        u32::try_from(v).map_err(|_| Error::Plan(format!("contraction extent {v} exceeds a u32")))
    };
    Ok(Shape {
        m: get(*m)?,
        n: get(*n)?,
        k: get(*k)?,
        batch: get(*batch)?.max(1),
    })
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
    /// The divisibility `CoopGeom::legal` asserts, spelled as an error rather
    /// than a silent truncation: a geometry whose fragment grid does not tile
    /// its block exactly cannot be lowered, because the k loop would drop the
    /// remainder rows instead of computing them.
    fn of(geom: CoopGeom, width: u32) -> Result<Self> {
        let dim = CoopGeom::COOP_DIM;
        let n_passes = geom.n_passes.max(1);
        let rg = geom.rg.max(1);
        let cg = geom.cg.max(1);
        let ok = geom.bn % n_passes == 0
            && geom.bk % dim == 0
            && geom.bm % (dim * rg) == 0
            && (geom.bn / n_passes) % (dim * cg) == 0;
        if !ok {
            return Err(Error::Plan(format!(
                "coop geometry {geom:?} does not tile into {dim}-sided fragments"
            )));
        }
        let bn_pass = geom.bn / n_passes;
        let sg_rows = geom.bm / rg;
        let sg_cols = bn_pass / cg;
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
/// One workgroup per `(split, batch, m_block, n_block)`, `n_passes` column
/// sub-passes, a `frags_m x frags_n` accumulator grid per subgroup, and a k
/// loop that stages `staging` K tiles per iteration between two barriers.
/// `pre_a` / `pre_b` fuse into the staging copy and are forced to zero past
/// the logical extents, so `pre(0)` cannot leak into an edge tile. `post`
/// fuses into the epilogue: into the per-lane pass when the accumulator
/// stages through a tile, and otherwise into an in-place pass over this
/// workgroup's own output block, which is disjoint from every other
/// workgroup's.
///
/// `splits > 1` is refused, not lowered: a split contraction is two launches
/// — `splits` partial slices, then a combine — and `GpuTarget` builds exactly
/// one artifact per plan launch, so the combine would never run and the
/// partial would be returned. Supporting it needs `GpuTarget::build_one` to
/// launch every kernel `lower_node` returns, in order, sharing one buffer
/// binding, plus a combine that applies `post` exactly once.
pub(crate) fn lower_coop(
    mut ctx: Ctx<'_>,
    op: &Launch,
    geom: CoopGeom,
    splits: u32,
    staging: u8,
) -> Result<Vec<KernelIr>> {
    let Launch::Contract {
        post, acc, a, b, ..
    } = op
    else {
        return Err(Error::Plan("lower_coop on a non-Contract node".into()));
    };
    let shape = shape_of(&ctx, op)?;
    let width = ctx.caps.subgroup_width();
    if !geom.legal(width, ctx.caps.limits.max_compute_invocations_per_workgroup) {
        return Err(Error::Plan(format!(
            "coop geometry {geom:?} is illegal at subgroup width {width}"
        )));
    }
    let cs = CoopShape::of(geom, width)?;
    let splits = splits.max(1);
    let depth = u32::from(staging.max(1));
    let dim = CoopGeom::COOP_DIM;

    if splits > 1 {
        // The target launches one kernel per plan launch, so the combine pass
        // a split needs cannot run.
        return Err(Error::Plan(format!(
            "split-K coop wants {splits} partials and a combine launch; GpuTarget builds one \
             kernel per launch, so the combine would be dropped and the partial returned"
        )));
    }

    let acc_elem = scalar_element(*acc);
    let operand_elem = scalar_element(ctx.plan_dtype(a.primary().src)?);

    // The operands as 2-D matrices in their own strides: A is `[batch * m, k]`
    // and B is `[batch * k, n]`, whatever ranks those extents are spread
    // across. A transposed rhs is a stride swap, never a copy.
    let a_rows = shape.batch.saturating_mul(shape.m).max(1);
    let b_rows = shape.batch.saturating_mul(shape.k).max(1);

    // The output. `plan::buffer_layout_for` pads `m` to `bm` and `n` to `bn`
    // at this schedule point so the whole-block cooperative store needs no
    // per-element mask — the store is subgroup-collective and cannot take
    // one. Padding lives in the strides, never in the shape. If the plan did
    // not deliver that padding the block store would run off the buffer, so
    // the strides are verified here: offset 0, and walking axes right to
    // left, `n` strides 1, `m` strides `n_padded`, and every batch axis
    // strides whole padded `m_padded * n_padded` blocks.
    let out = ctx.output()?;
    let out_layout = ctx.plan_layout(out)?.clone();
    let tiles_m = shape.m.max(1).div_ceil(geom.bm.max(1)).max(1);
    let tiles_n = shape.n.max(1).div_ceil(geom.bn.max(1)).max(1);
    let m_padded = tiles_m.saturating_mul(geom.bm);
    let n_padded = tiles_n.saturating_mul(geom.bn);
    let rank = out_layout.rank();
    if rank < 2 || !out_layout.offset().known_eq(Dim::Const(0)) {
        return Err(Error::Plan(format!(
            "coop needs a rank >= 2, offset-0 output layout; the plan laid out \
             rank {rank} at offset {}",
            out_layout.offset()
        )));
    }
    let mut expected = 1u64;
    for axis in (0..rank).rev() {
        let got = ctx.binding.require(out_layout.strides()[axis])?;
        if got != expected {
            return Err(Error::Plan(format!(
                "coop needs its output padded to whole {}x{} blocks in the strides: \
                 axis {axis} strides {got} where {expected} was required",
                geom.bm, geom.bn
            )));
        }
        // The next axis out strides one whole run of this one: its padded
        // extent for `n` and `m`, its logical extent for the batch axes,
        // which are never padded.
        expected = expected.saturating_mul(if axis == rank - 1 {
            u64::from(n_padded.max(1))
        } else if axis == rank - 2 {
            u64::from(m_padded.max(1))
        } else {
            ctx.binding.require(out_layout.shape()[axis])?.max(1)
        });
    }
    let want_rows = shape.batch.saturating_mul(m_padded).max(1);
    let out_view = StorageView {
        buffer: ctx.buffer(out)?,
        offset: 0,
        layout: fusor2_ir::ir::kernel::TileLayout::contiguous(
            fusor2_ir::ir::kernel::MemoryLevel::Storage,
            &[want_rows, n_padded],
        ),
    };
    let out_elem = out_view.buffer.element;

    // Operand staging tiles: `staging` buffers stacked inside one
    // declaration — the footprint `verify_launch::coop_tiles` admitted the
    // geometry on (`depth * bm * bk` plus `depth * bk * bn_pass`), and what
    // the k loop's `depth`-strided addressing below is written against.
    let a_tile = ctx.b.tile(
        "coop_a",
        ElementType::Scalar(operand_elem),
        &[depth.saturating_mul(geom.bm), geom.bk],
    );
    let b_tile = ctx.b.tile(
        "coop_b",
        ElementType::Scalar(operand_elem),
        &[depth.saturating_mul(geom.bk), cs.bn_pass],
    );

    // The staging sources, one per buffer each side reads. A side that has
    // absorbed a producer brings several, all loaded at the same `(row, col)`
    // the staging fill already computes, then combined by the side's `pre`.
    let a_sources = ctx.contract_side_sources(a, a_rows, shape.k.max(1))?;
    let b_sources = ctx.contract_side_sources(b, b_rows, shape.n.max(1))?;
    let a_coords = SideCoords::for_side(&ctx, a, u64::from(a_rows), u64::from(shape.k.max(1)))?;
    let b_coords = SideCoords::for_side(&ctx, b, u64::from(b_rows), u64::from(shape.n.max(1)))?;

    let block = cs.lanes;
    let groups = shape
        .batch
        .saturating_mul(tiles_m)
        .saturating_mul(tiles_n)
        .max(1);
    let max_dim = ctx.caps.limits.max_compute_workgroups_per_dimension;
    let grid = distribute_workgroups(groups, max_dim);

    let lane = ctx.b.builtin(Builtin::Lane);
    let tile_id = workgroup_index(&mut ctx, grid, groups);

    let per_batch = tiles_m.saturating_mul(tiles_n).max(1);
    let (batch_index, local_tile) = split_const(&mut ctx, tile_id, shape.batch.max(1), per_batch);
    let group_m = swizzle_group_m(geom, shape.n);
    let (m_tile, n_tile) = swizzle_tile(&mut ctx, local_tile, tiles_m, tiles_n, group_m);

    let bm_e = ctx.b.u32(geom.bm.max(1));
    let bn_e = ctx.b.u32(geom.bn.max(1));
    let row_block = ctx.b.mul(m_tile, bm_e);
    let col_block = ctx.b.mul(n_tile, bn_e);

    // Operand row origins of this batch element.
    let m_e = ctx.b.u32(shape.m.max(1));
    let k_e = ctx.b.u32(shape.k.max(1));
    let a_batch_base = ctx.b.mul(batch_index.clone(), m_e.clone());
    let b_batch_base = ctx.b.mul(batch_index.clone(), k_e.clone());
    let a_row_base = ctx.b.add(a_batch_base.clone(), row_block.clone());
    let a_row_limit = ctx.b.add(a_batch_base, m_e);
    let b_row_limit = ctx.b.add(b_batch_base.clone(), k_e);

    // Output row origin: the batch index walks the *padded* row space.
    let mp_e = ctx.b.u32(m_padded.max(1));
    let out_row_origin = ctx.b.mul(batch_index, mp_e);
    let out_row_base = ctx.b.add(out_row_origin, row_block);
    // The same row in the *logical* space, which is what a `post` body's
    // `IndexOf` reads: `row` counts `(batch, m)` and `col` counts `n`, exactly
    // as the register-tiled families pass them.
    let logical_row_base = a_row_base.clone();

    // Subgroup fragment origin inside the block.
    let sg = ctx.b.builtin(Builtin::SubgroupId);
    let cg_e = ctx.b.u32(geom.cg.max(1));
    let sg_row = ctx.b.binary(
        TileBinaryOp::Div,
        sg.clone(),
        cg_e.clone(),
        NumericContract::RELAXED,
    );
    let sg_col = ctx
        .b
        .binary(TileBinaryOp::Rem, sg, cg_e, NumericContract::RELAXED);
    let sg_rows_e = ctx.b.u32(cs.sg_rows);
    let sg_cols_e = ctx.b.u32(cs.sg_cols);
    let sg_row_base = ctx.b.mul(sg_row, sg_rows_e);
    let sg_col_base = ctx.b.mul(sg_col, sg_cols_e);

    let k_tiles = shape.k.max(1).div_ceil(geom.bk.max(1)).max(1);
    let iters = k_tiles.div_ceil(depth).max(1);

    let post_is_identity = matches!(post.kind(), ScalarKind::Arg(0));

    let needs_stage =
        !ctx.caps.mixed_precision_coop_store && ElementType::Scalar(acc_elem) != out_elem;
    let layout_ok = cooperative_store_layout_supported(&out_view.layout);
    let stage_tile: Option<Tile> = if needs_stage || !layout_ok {
        Some(ctx.b.tile(
            "coop_acc",
            ElementType::Scalar(acc_elem),
            &[geom.bm, cs.bn_pass],
        ))
    } else {
        None
    };

    let k_limit = ctx.b.u32(shape.k.max(1));
    let n_limit = ctx.b.u32(shape.n.max(1));

    let mut body: Vec<Stmt> = Vec::new();
    for pass in 0..geom.n_passes.max(1) {
        let pass_off = ctx.b.u32(pass.saturating_mul(cs.bn_pass));
        let pass_col_base = ctx.b.add(col_block.clone(), pass_off);

        // The k loop. `Stmt::Loop` runs `body` before the accumulator updates,
        // so the two barriers around the staging copy separate the previous
        // iteration's fragment reads from this iteration's tile writes and
        // then publish them.
        let mut loop_body: Vec<Stmt> = vec![Stmt::Barrier];
        let k_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
        let kk = ctx.b.load_local(k_index.clone());
        let iter_span = ctx.b.u32(depth.saturating_mul(geom.bk));
        let iter_base = ctx.b.mul(kk, iter_span);
        for d in 0..depth {
            let d_off = ctx.b.u32(d.saturating_mul(geom.bk));
            let k_base = ctx.b.add(iter_base.clone(), d_off);
            stage_operand_tile(
                &mut ctx,
                &mut loop_body,
                &a_tile,
                d.saturating_mul(geom.bm).saturating_mul(geom.bk),
                &a_sources,
                a_coords.as_ref(),
                &lane,
                a_row_base.clone(),
                k_base.clone(),
                a_row_limit.clone(),
                k_limit.clone(),
                geom.bm,
                geom.bk,
                cs.lanes,
                &a.pre,
                operand_elem,
            )?;
            let b_row_base = ctx.b.add(b_batch_base.clone(), k_base);
            stage_operand_tile(
                &mut ctx,
                &mut loop_body,
                &b_tile,
                d.saturating_mul(geom.bk).saturating_mul(cs.bn_pass),
                &b_sources,
                b_coords.as_ref(),
                &lane,
                b_row_base,
                pass_col_base.clone(),
                b_row_limit.clone(),
                n_limit.clone(),
                geom.bk,
                cs.bn_pass,
                cs.lanes,
                &b.pre,
                operand_elem,
            )?;
        }
        loop_body.push(Stmt::Barrier);

        // One accumulator per fragment of this subgroup's sub-block, each
        // folding every `COOP_DIM` K slice of every buffer in the iteration.
        let frags = (cs.frags_m.saturating_mul(cs.frags_n)) as usize;
        let mut accumulators = Vec::with_capacity(frags);
        let mut locals = Vec::with_capacity(frags);
        for r in 0..cs.frags_m {
            let r_off = ctx.b.u32(r.saturating_mul(dim));
            let a_row = ctx.b.add(sg_row_base.clone(), r_off);
            for c in 0..cs.frags_n {
                let c_off = ctx.b.u32(c.saturating_mul(dim));
                let b_col = ctx.b.add(sg_col_base.clone(), c_off);
                let local = ctx.b.local(ElementType::CoopMatrix {
                    scalar: acc_elem,
                    role: CoopMatrixRole::C,
                    rows: dim,
                    cols: dim,
                });
                // A fragment accumulator starts from a zero fragment: a
                // scalar zero has the wrong `ElementType`.
                let init = ctx.b.coop_zero(CoopMatrixRole::C, acc_elem, dim, dim);
                let mut update = ctx.b.load_local(local.clone());
                for d in 0..depth {
                    let a_buf = ctx.b.u32(d.saturating_mul(geom.bm));
                    let a_row_d = ctx.b.add(a_buf, a_row.clone());
                    let b_buf = d.saturating_mul(geom.bk);
                    for step in 0..cs.kk_steps {
                        let a_col = ctx.b.u32(step.saturating_mul(dim));
                        let b_row = ctx.b.u32(b_buf.saturating_add(step.saturating_mul(dim)));
                        let a_frag = ctx.b.coop_load(
                            CoopMatrixRole::A,
                            operand_elem,
                            dim,
                            dim,
                            CoopSrc::TileRegion {
                                tile: a_tile.clone(),
                                row: a_row_d.clone(),
                                col: a_col,
                                transposed: false,
                            },
                        );
                        let b_frag = ctx.b.coop_load(
                            CoopMatrixRole::B,
                            operand_elem,
                            dim,
                            dim,
                            CoopSrc::TileRegion {
                                tile: b_tile.clone(),
                                row: b_row,
                                col: b_col.clone(),
                                transposed: false,
                            },
                        );
                        update = ctx.b.coop_mma(a_frag, b_frag, update);
                    }
                }
                locals.push(local.clone());
                accumulators.push(Accumulator {
                    local,
                    init,
                    update,
                });
            }
        }

        let count = ctx.b.u32(iters);
        body.push(Stmt::Loop {
            count: Some(count),
            index: Some(k_index),
            accumulators,
            body: loop_body,
        });

        let mut taken = locals.into_iter();
        for r in 0..cs.frags_m {
            for c in 0..cs.frags_n {
                let local = taken
                    .next()
                    .ok_or_else(|| Error::Plan("coop fragment grid lost an accumulator".into()))?;
                let value = ctx.b.load_local(local);
                let r_off = ctx.b.u32(r.saturating_mul(dim));
                let c_off = ctx.b.u32(c.saturating_mul(dim));
                let frag_row = ctx.b.add(sg_row_base.clone(), r_off);
                let frag_col = ctx.b.add(sg_col_base.clone(), c_off);
                match &stage_tile {
                    Some(tile) => body.push(Stmt::CoopStoreTile {
                        acc: value,
                        tile: tile.clone(),
                        row: frag_row,
                        col: frag_col,
                    }),
                    None => {
                        let row = ctx.b.add(out_row_base.clone(), frag_row);
                        let col = ctx.b.add(pass_col_base.clone(), frag_col);
                        body.push(Stmt::CoopStore {
                            acc: value,
                            dst: out_view.clone(),
                            addr: Addr::Rc2 { row, col },
                        });
                    }
                }
            }
        }

        match &stage_tile {
            // The staged path already reads every element per lane, so a fused
            // `post` costs nothing extra there.
            Some(tile) => {
                let tile = tile.clone();
                body.push(Stmt::Barrier);
                per_lane_block(
                    &mut ctx,
                    &mut body,
                    &lane,
                    geom.bm,
                    cs.bn_pass,
                    cs.lanes,
                    |ctx, out, flat, local_row, local_col, active| {
                        let staged = ctx.b.load_tile(tile.clone(), flat);
                        let row = ctx.b.add(out_row_base.clone(), local_row.clone());
                        let col = ctx.b.add(pass_col_base.clone(), local_col);
                        let logical_row = ctx.b.add(logical_row_base.clone(), local_row);
                        let value =
                            ctx.eval_scalar(post, &[staged], &[logical_row, col.clone()])?;
                        let value = ctx.b.cast(value, out_elem);
                        out.push(Stmt::Store {
                            dst: out_view.clone(),
                            addr: Addr::Rc2 { row, col },
                            value,
                            mask: active,
                        });
                        Ok(())
                    },
                )?;
                body.push(Stmt::Barrier);
            }
            // Fragments are opaque to scalar code, so a fused `post` reads them
            // back: store, make the writes visible inside the workgroup, then
            // map `post` over this workgroup's own block in place. That block
            // is disjoint from every other workgroup's, so the
            // read-modify-write races with nothing.
            None if !post_is_identity => {
                body.push(Stmt::StorageBarrier);
                per_lane_block(
                    &mut ctx,
                    &mut body,
                    &lane,
                    geom.bm,
                    cs.bn_pass,
                    cs.lanes,
                    |ctx, out, _flat, local_row, local_col, active| {
                        let row = ctx.b.add(out_row_base.clone(), local_row.clone());
                        let col = ctx.b.add(pass_col_base.clone(), local_col);
                        let fill = ctx.b.zero(acc_elem);
                        let stored = ctx.b.load(
                            Source::Storage(out_view.clone()),
                            Addr::Rc2 {
                                row: row.clone(),
                                col: col.clone(),
                            },
                            active.clone(),
                            fill,
                        );
                        let logical_row = ctx.b.add(logical_row_base.clone(), local_row);
                        let value =
                            ctx.eval_scalar(post, &[stored], &[logical_row, col.clone()])?;
                        let value = ctx.b.cast(value, out_elem);
                        out.push(Stmt::Store {
                            dst: out_view.clone(),
                            addr: Addr::Rc2 { row, col },
                            value,
                            mask: active,
                        });
                        Ok(())
                    },
                )?;
            }
            None => {}
        }
    }

    Ok(vec![ctx.finish("coop_matmul", grid, block, body)])
}

/// The flat workgroup index of this launch, linearized against **this** grid
/// rather than against the per-dimension cap — `distribute_workgroups` sizes
/// `x` to the slab, so the cap is not the x extent.
///
/// A cooperative op needs uniform control flow, so an overhang workgroup
/// cannot return early: it is clamped onto the last block, recomputes it and
/// stores the same values. The clamp is emitted only when the grid
/// over-covers.
fn workgroup_index(ctx: &mut Ctx<'_>, grid: [u32; 3], groups: u32) -> TileExpr {
    let gx = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
    let gy = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::Y));
    let gz = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::Z));
    // Linearization constants from `@builtin(num_workgroups)`, never baked:
    // a baked grid puts the dispatch size into the body and recompiles the
    // pipeline whenever a symbolic extent moves the grid.
    let x_e = ctx.b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::X));
    let y_only = ctx.b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::Y));
    let xy_e = ctx.b.mul(x_e.clone(), y_only);
    let y_off = ctx.b.mul(gy, x_e);
    let z_off = ctx.b.mul(gz, xy_e);
    let id = ctx.b.add(gx, y_off);
    let id = ctx.b.add(id, z_off);
    let covered = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
    if covered > u64::from(groups) {
        let last = ctx.b.u32(groups.saturating_sub(1));
        ctx.b
            .binary(TileBinaryOp::Min, id, last, NumericContract::RELAXED)
    } else {
        id
    }
}

/// `(index / stride, index % stride)`, skipping both operations when the
/// quotient can only ever be zero.
fn split_const(
    ctx: &mut Ctx<'_>,
    index: TileExpr,
    extent: u32,
    stride: u32,
) -> (TileExpr, TileExpr) {
    if extent <= 1 {
        let zero = ctx.b.u32(0);
        return (zero, index);
    }
    let stride_e = ctx.b.u32(stride.max(1));
    let quotient = ctx.b.binary(
        TileBinaryOp::Div,
        index.clone(),
        stride_e.clone(),
        NumericContract::RELAXED,
    );
    let rest = ctx
        .b
        .binary(TileBinaryOp::Rem, index, stride_e, NumericContract::RELAXED);
    (quotient, rest)
}

/// `local_tile -> (m_tile, n_tile)`, walked in super-blocks of `group` M lines
/// M-fastest, so a resident wavefront shares one B column slab while touching
/// only `group` A row slabs. A bijection on `[0, tiles_m * tiles_n)`: the
/// ragged tail of `tiles_m % group` M lines walks in the same order.
/// Degenerate grids keep the plain row-major decomposition, which is also what
/// `group == 1` reduces to.
fn swizzle_tile(
    ctx: &mut Ctx<'_>,
    local_tile: TileExpr,
    tiles_m: u32,
    tiles_n: u32,
    group: u32,
) -> (TileExpr, TileExpr) {
    if group <= 1 || tiles_m <= 1 || tiles_n <= 1 {
        let n_e = ctx.b.u32(tiles_n.max(1));
        let m_tile = ctx.b.binary(
            TileBinaryOp::Div,
            local_tile.clone(),
            n_e.clone(),
            NumericContract::RELAXED,
        );
        let n_tile = ctx
            .b
            .binary(TileBinaryOp::Rem, local_tile, n_e, NumericContract::RELAXED);
        return (m_tile, n_tile);
    }
    let full = (tiles_m / group).saturating_mul(group);
    let tail = tiles_m - full;
    let threshold = full.saturating_mul(tiles_n);

    let span = ctx.b.u32(group.saturating_mul(tiles_n));
    let group_e = ctx.b.u32(group);
    let super_block = ctx.b.binary(
        TileBinaryOp::Div,
        local_tile.clone(),
        span.clone(),
        NumericContract::RELAXED,
    );
    let within = ctx.b.binary(
        TileBinaryOp::Rem,
        local_tile.clone(),
        span,
        NumericContract::RELAXED,
    );
    let m_in = ctx.b.binary(
        TileBinaryOp::Rem,
        within.clone(),
        group_e.clone(),
        NumericContract::RELAXED,
    );
    let n_in = ctx
        .b
        .binary(TileBinaryOp::Div, within, group_e.clone(), NumericContract::RELAXED);
    let m_base = ctx.b.mul(super_block, group_e);
    let m_full = ctx.b.add(m_base, m_in);
    if tail == 0 {
        return (m_full, n_in);
    }

    // The ragged tail, selected branchlessly: nothing wants divergence around
    // the block a cooperative store is about to write.
    let threshold_e = ctx.b.u32(threshold);
    let rest = ctx.b.sub(local_tile.clone(), threshold_e.clone());
    let tail_e = ctx.b.u32(tail);
    let m_off = ctx.b.binary(
        TileBinaryOp::Rem,
        rest.clone(),
        tail_e.clone(),
        NumericContract::RELAXED,
    );
    let full_e = ctx.b.u32(full);
    let m_tail = ctx.b.add(full_e, m_off);
    let n_tail = ctx
        .b
        .binary(TileBinaryOp::Div, rest, tail_e, NumericContract::RELAXED);
    let in_full = ctx.b.compare(TileCompareOp::Lt, local_tile, threshold_e);
    let m_tile = ctx.b.select(in_full.clone(), m_full, m_tail);
    let n_tile = ctx.b.select(in_full, n_in, n_tail);
    (m_tile, n_tile)
}

/// Copy a `rows x cols` window of a 2-D operand into a workgroup tile,
/// `lanes` elements per pass, applying `pre` on the way in.
///
/// The source may be block-quantized: a [`Source::Quantized`] runs the
/// format's decode program at `(row, col)` and yields f32, so the staging
/// tile holds decoded values and everything downstream is identical to the
/// dense path.
///
/// Past `row_limit` / `col_limit` the tile holds a zero, not `pre(0)`: an
/// edge tile's padding must not enter the contraction, and `pre` is an
/// arbitrary scalar body whose value at zero is arbitrary.
#[allow(clippy::too_many_arguments)]
/// The coordinate vector one contraction side hands its `pre`.
///
/// An absorbed producer's body may read its own loop coordinates, and after
/// absorption those axes name the operand's axes. The side's staging loops
/// know only the flattened `(row, col)` pair, so this splits it back: `row`
/// enumerates the leading `split` axes row-major and `col` the rest — the
/// factorization `matrix_split_for` proved exists.
///
/// Built only when the side's `pre` names a coordinate.
struct SideCoords {
    extents: Vec<u64>,
    split: usize,
}

impl SideCoords {
    fn for_side(
        ctx: &Ctx<'_>,
        side: &ContractSide,
        rows: u64,
        cols: u64,
    ) -> Result<Option<Self>> {
        if !side.pre.reads_index_of() {
            return Ok(None);
        }
        let layout = &side.primary().layout;
        let split = crate::lower::matrix_split_for(layout, &ctx.binding, rows, cols)?;
        let extents = layout
            .shape()
            .iter()
            .map(|d| ctx.binding.require(*d))
            .collect::<Result<Vec<u64>>>()?;
        Ok(Some(Self { extents, split }))
    }

    /// The per-axis coordinates at `(row, col)`, innermost axis of each group
    /// varying fastest.
    fn at(&self, ctx: &mut Ctx<'_>, row: &TileExpr, col: &TileExpr) -> Vec<TileExpr> {
        let mut out = vec![ctx.b.u32(0); self.extents.len()];
        let decompose = |ctx: &mut Ctx<'_>, flat: &TileExpr, axes: std::ops::Range<usize>, out: &mut Vec<TileExpr>| {
            let mut rest = flat.clone();
            for i in axes.rev() {
                let e = ctx.b.u32(u32::try_from(self.extents[i]).unwrap_or(u32::MAX).max(1));
                out[i] = ctx.b.binary(
                    TileBinaryOp::Rem,
                    rest.clone(),
                    e.clone(),
                    NumericContract::RELAXED,
                );
                rest = ctx.b.binary(TileBinaryOp::Div, rest, e, NumericContract::RELAXED);
            }
        };
        decompose(ctx, row, 0..self.split, &mut out);
        decompose(ctx, col, self.split..self.extents.len(), &mut out);
        out
    }
}

/// The element a staging load of this source yields — the buffer's own for a
/// plain storage read, f32 for a decode. Must agree with `Builder::load`,
/// which types the resulting expression the same way.
fn source_element(src: &Source) -> ScalarElement {
    match src {
        Source::Storage(v) => match v.buffer.element {
            ElementType::Scalar(e) => e,
            _ => ScalarElement::F32,
        },
        Source::Quantized(_) => ScalarElement::F32,
    }
}

fn stage_operand_tile(
    ctx: &mut Ctx<'_>,
    body: &mut Vec<Stmt>,
    tile: &Tile,
    tile_base: u32,
    srcs: &[StagedSource],
    coords: Option<&SideCoords>,
    lane: &TileExpr,
    row_base: TileExpr,
    col_base: TileExpr,
    row_limit: TileExpr,
    col_limit: TileExpr,
    rows: u32,
    cols: u32,
    lanes: u32,
    pre: &ScalarExpr,
    elem: ScalarElement,
) -> Result<()> {
    let total = rows.saturating_mul(cols).max(1);
    let lanes = lanes.max(1);
    for pass in 0..total.div_ceil(lanes) {
        let flat = if pass == 0 {
            lane.clone()
        } else {
            let off = ctx.b.u32(pass.saturating_mul(lanes));
            ctx.b.add(lane.clone(), off)
        };
        let cols_e = ctx.b.u32(cols.max(1));
        let local_row = ctx.b.binary(
            TileBinaryOp::Div,
            flat.clone(),
            cols_e.clone(),
            NumericContract::RELAXED,
        );
        let local_col = ctx.b.binary(
            TileBinaryOp::Rem,
            flat.clone(),
            cols_e,
            NumericContract::RELAXED,
        );
        let row = ctx.b.add(row_base.clone(), local_row);
        let col = ctx.b.add(col_base.clone(), local_col);
        let in_row = ctx
            .b
            .compare(TileCompareOp::Lt, row.clone(), row_limit.clone());
        let in_col = ctx
            .b
            .compare(TileCompareOp::Lt, col.clone(), col_limit.clone());
        let mut active = ctx.b.and(in_row, in_col);
        let within = if (pass + 1).saturating_mul(lanes) > total {
            let total_e = ctx.b.u32(total);
            let w = ctx.b.compare(TileCompareOp::Lt, flat.clone(), total_e);
            active = ctx.b.and(active, w.clone());
            Some(w)
        } else {
            None
        };
        let fill = ctx.b.zero(elem);
        // One load per buffer this side reads, all at the same coordinate;
        // `pre` is written over `Arg(0..srcs.len())` in operand order.
        // Each out-of-range fill takes its own source's element type, not the
        // staging tile's: a decode reads `u32` words and only becomes `elem`
        // after `pre` has run.
        let mut raws: Vec<TileExpr> = Vec::with_capacity(srcs.len());
        for src in srcs {
            let src = match src {
                StagedSource::Const(lit) => {
                    raws.push(lit.clone());
                    continue;
                }
                StagedSource::Mem(s) => s,
            };
            let src_fill = ctx.b.zero(source_element(src));
            raws.push(ctx.b.load(
                src.clone(),
                Addr::Rc2 {
                    row: row.clone(),
                    col: col.clone(),
                },
                active.clone(),
                src_fill,
            ));
        }
        let coord_exprs = match coords {
            Some(c) => c.at(ctx, &row, &col),
            None => Vec::new(),
        };
        let value = ctx.eval_scalar(pre, &raws, &coord_exprs)?;
        let value = ctx.b.cast(value, ElementType::Scalar(elem));
        let value = ctx.b.select(active, value, fill);
        let index = if tile_base == 0 {
            flat
        } else {
            let base = ctx.b.u32(tile_base);
            ctx.b.add(base, flat)
        };
        let store = Stmt::StoreTile {
            dst: tile.clone(),
            index,
            value,
        };
        match within {
            Some(w) => body.push(Stmt::If {
                condition: w,
                accept: vec![store],
                reject: Vec::new(),
            }),
            None => body.push(store),
        }
    }
    Ok(())
}

/// Walk a `rows x cols` block one workgroup owns, `lanes` elements per step,
/// handing the builder `(flat, local_row, local_col, active)`.
///
/// A counted loop, not an unrolled sequence, and that is load-bearing:
/// `Emitter`'s hash-cons memo is scoped to a block and is not invalidated by
/// a barrier, so an identical `LoadTile(tile, flat)` emitted twice at the top
/// level of one kernel resolves to the first one's SSA value. A loop body is
/// a nested scope and its index is a fresh identity-bearing `Local`, so each
/// call mints its own read.
fn per_lane_block<'a>(
    ctx: &mut Ctx<'a>,
    body: &mut Vec<Stmt>,
    lane: &TileExpr,
    rows: u32,
    cols: u32,
    lanes: u32,
    build: impl FnOnce(
        &mut Ctx<'a>,
        &mut Vec<Stmt>,
        TileExpr,
        TileExpr,
        TileExpr,
        TileExpr,
    ) -> Result<()>,
) -> Result<()> {
    let total = rows.saturating_mul(cols).max(1);
    let lanes = lanes.max(1);
    let index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let step = ctx.b.load_local(index.clone());
    let lanes_e = ctx.b.u32(lanes);
    let base = ctx.b.mul(step, lanes_e);
    let flat = ctx.b.add(base, lane.clone());
    let cols_e = ctx.b.u32(cols.max(1));
    let local_row = ctx.b.binary(
        TileBinaryOp::Div,
        flat.clone(),
        cols_e.clone(),
        NumericContract::RELAXED,
    );
    let local_col = ctx
        .b
        .binary(TileBinaryOp::Rem, flat.clone(), cols_e, NumericContract::RELAXED);
    // Never constant-true: the final step may be partial, and a load with a
    // constant-true mask has to be *provably* in range for `check_loads`.
    let total_e = ctx.b.u32(total);
    let active = ctx.b.compare(TileCompareOp::Lt, flat.clone(), total_e);

    let mut inner: Vec<Stmt> = Vec::new();
    build(ctx, &mut inner, flat, local_row, local_col, active)?;
    let count = ctx.b.u32(total.div_ceil(lanes).max(1));
    body.push(Stmt::Loop {
        count: Some(count),
        index: Some(index),
        accumulators: Vec::new(),
        body: inner,
    });
    Ok(())
}


/// SGEMM with a per-thread `tn`-wide register accumulator: one lane owns `tn`
/// adjacent output columns of one output row and reuses the A element across
/// them, which is the register-reuse term `SgemmParams` prices. `p.legal`
/// gates the point on the storage a staged form would need, but the emitted
/// kernel reads A and B straight from storage.
pub(crate) fn lower_sgemm(ctx: Ctx<'_>, op: &Launch, p: SgemmParams) -> Result<KernelIr> {
    let Launch::Contract {
        post, acc, a, b, ..
    } = op
    else {
        return Err(Error::Plan("lower_sgemm on a non-Contract node".into()));
    };
    let shape = shape_of(&ctx, op)?;
    let acc_elem = scalar_element(*acc);
    let operand_elem = scalar_element(ctx.plan_dtype(a.primary().src)?);
    if !p.legal(
        operand_elem.byte_size() as u32,
        ctx.caps.limits.max_compute_workgroup_storage_size,
        ctx.caps.limits.max_compute_invocations_per_workgroup,
    ) {
        return Err(Error::Plan(format!("sgemm params {p:?} are illegal here")));
    }
    contract_rows(
        ctx,
        SchedPoint::Sgemm(p),
        post,
        acc_elem,
        a,
        b,
        &shape,
        "sgemm",
    )
}

/// `(output columns one lane owns, lanes per workgroup)` of a row-tiled
/// contraction, read off the schedule point.
///
/// `SgemmParams` names both: `tn` is the register tile width and
/// `(bm / tm) * (bn / tn)` is the thread block that tile implies. Every other
/// point that reaches the always-legal generic body is one output element per
/// lane, at the widest workgroup the device admits. Deriving both from
/// `theta` stops a caller pairing one point's `tn` with another point's block
/// width.
fn row_tiling(theta: SchedPoint, caps: &Caps) -> (u32, u32) {
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup.max(1);
    match theta {
        SchedPoint::Sgemm(p) => (
            p.tn.max(1),
            ((p.bm / p.tm.max(1)) * (p.bn / p.tn.max(1))).clamp(1, max_lanes),
        ),
        _ => (1, max_lanes),
    }
}

/// The body of [`lower_sgemm`]: one lane per
/// `(row, column tile)` of the output, a k loop, `tn` accumulators.
///
/// The operands are read through **2-D views built from the contraction's own
/// extents** — A is `[batch * m, k]` and B is `[batch * k, n]` in their own
/// strides, whatever ranks those extents are spread across — and the batch
/// index is recovered from the output row rather than from a third grid axis.
/// A transposed rhs is a stride swap, never a copy.
///
/// The output is contiguous: `plan::buffer_layout_for` pads nothing at these
/// schedule points, and every store below is masked, so the address is the
/// flat row-major index `row * n + col` of `[batch.., m.., n..]`.
#[allow(clippy::too_many_arguments)]
fn contract_rows(
    mut ctx: Ctx<'_>,
    theta: SchedPoint,
    post: &fusor2_ir::scalar::ScalarExpr,
    acc_elem: ScalarElement,
    a: &ContractSide,
    b: &ContractSide,
    shape: &Shape,
    name: &'static str,
) -> Result<KernelIr> {
    let (tn, block) = row_tiling(theta, ctx.caps);
    // One source per buffer each side reads, all indexed by that side's own
    // `(row, col)`. A side that absorbed a producer simply has more of them.
    let a_rows = shape.batch.saturating_mul(shape.m).max(1);
    let b_rows = shape.batch.saturating_mul(shape.k).max(1);
    let a_sources = ctx.contract_side_sources(a, a_rows, shape.k.max(1))?;
    let b_sources = ctx.contract_side_sources(b, b_rows, shape.n.max(1))?;
    let a_coords = SideCoords::for_side(&ctx, a, u64::from(a_rows), u64::from(shape.k.max(1)))?;
    let b_coords = SideCoords::for_side(&ctx, b, u64::from(b_rows), u64::from(shape.n.max(1)))?;
    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let limits = ctx.caps.limits;

    let n = shape.n.max(1);
    let tn = tn.clamp(1, n);
    let n_tiles = n.div_ceil(tn).max(1);
    let rows = shape.m.saturating_mul(shape.batch).max(1);
    let tiles = rows.saturating_mul(n_tiles);

    let grid = distribute_workgroups(
        tiles.div_ceil(block.max(1)).max(1),
        limits.max_compute_workgroups_per_dimension,
    );
    let index = ctx.global_index(block, grid);
    let n_tiles_e = ctx.b.u32(n_tiles);
    let row = ctx.b.binary(
        TileBinaryOp::Div,
        index.clone(),
        n_tiles_e.clone(),
        NumericContract::RELAXED,
    );
    let col_tile = ctx.b.binary(
        TileBinaryOp::Rem,
        index.clone(),
        n_tiles_e,
        NumericContract::RELAXED,
    );
    let tiles_e = ctx.b.u32(tiles);
    let live = ctx.b.compare(TileCompareOp::Lt, index, tiles_e);

    // B's own row is `batch_index * k + kk`; the batch index rides in `row`.
    let m_e = ctx.b.u32(shape.m.max(1));
    let batch_index = ctx.b.binary(
        TileBinaryOp::Div,
        row.clone(),
        m_e,
        NumericContract::RELAXED,
    );
    let k_e = ctx.b.u32(shape.k.max(1));
    let b_row_base = ctx.b.mul(batch_index, k_e);

    let k_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let kk = ctx.b.load_local(k_index.clone());
    let mut avs = Vec::with_capacity(a_sources.len());
    for src in &a_sources {
        let src = match src {
            StagedSource::Const(lit) => {
                avs.push(lit.clone());
                continue;
            }
            StagedSource::Mem(s) => s,
        };
        let fill = ctx.b.zero(source_element(src));
        avs.push(ctx.b.load(
            src.clone(),
            Addr::Rc2 {
                row: row.clone(),
                col: kk.clone(),
            },
            live.clone(),
            fill,
        ));
    }
    let a_coord_exprs = match &a_coords {
        Some(c) => c.at(&mut ctx, &row, &kk),
        None => Vec::new(),
    };
    let av = ctx.eval_scalar(&a.pre, &avs, &a_coord_exprs)?;
    let av = ctx.b.cast(av, ElementType::Scalar(acc_elem));
    let b_row = ctx.b.add(b_row_base, kk);

    let tn_e = ctx.b.u32(tn);
    let col0 = ctx.b.mul(col_tile, tn_e);
    let mut accs = Vec::with_capacity(tn as usize);
    let mut cols = Vec::with_capacity(tn as usize);
    for j in 0..tn {
        let off = ctx.b.u32(j);
        let col = ctx.b.add(col0.clone(), off);
        let n_bound = ctx.b.u32(n);
        let in_n = ctx.b.compare(TileCompareOp::Lt, col.clone(), n_bound);
        let ok = ctx.b.and(live.clone(), in_n);
        let mut bvs = Vec::with_capacity(b_sources.len());
        for src in &b_sources {
            let src = match src {
                StagedSource::Const(lit) => {
                    bvs.push(lit.clone());
                    continue;
                }
                StagedSource::Mem(s) => s,
            };
            let fill = ctx.b.zero(source_element(src));
            bvs.push(ctx.b.load(
                src.clone(),
                Addr::Rc2 {
                    row: b_row.clone(),
                    col: col.clone(),
                },
                ok.clone(),
                fill,
            ));
        }
        let b_coord_exprs = match &b_coords {
            Some(c) => c.at(&mut ctx, &b_row, &col),
            None => Vec::new(),
        };
        let bv = ctx.eval_scalar(&b.pre, &bvs, &b_coord_exprs)?;
        let bv = ctx.b.cast(bv, ElementType::Scalar(acc_elem));
        let local = ctx.b.local(ElementType::Scalar(acc_elem));
        let read = ctx.b.load_local(local.clone());
        let update = ctx.b.fma(av.clone(), bv, read);
        let init = ctx.b.zero(acc_elem);
        accs.push(Accumulator {
            local,
            init,
            update,
        });
        cols.push((col, ok));
    }

    let count = ctx.b.u32(shape.k.max(1));
    let locals: Vec<_> = accs.iter().map(|a| a.local.clone()).collect();
    let mut body = vec![Stmt::Loop {
        count: Some(count),
        index: Some(k_index),
        accumulators: accs,
        body: Vec::new(),
    }];

    let n_e = ctx.b.u32(n);
    for (local, (col, ok)) in locals.into_iter().zip(cols) {
        let total = ctx.b.load_local(local);
        let value = ctx.eval_scalar(post, &[total], &[row.clone(), col.clone()])?;
        let value = ctx.b.cast(value, out_elem);
        // `row` counts `(batch, m)` and `col` counts `n`, so `row * n + col` is
        // the flat row-major index of a `[batch.., m.., n..]` output — for any
        // number of axes on each side.
        let addr = {
            let scaled = ctx.b.mul(row.clone(), n_e.clone());
            ctx.b.add(scaled, col)
        };
        body.push(Stmt::Store {
            dst: out_view.clone(),
            addr: Addr::Linear(addr),
            value,
            mask: ok,
        });
    }

    Ok(ctx.finish(name, grid, block, body))
}

/// Vector-family contraction: `subgroups` lane groups per row, each summing a
/// `chunk`-long, `vector`-wide slice of K.
pub(crate) fn lower_sgemv(mut ctx: Ctx<'_>, op: &Launch, p: SgemvParams) -> Result<KernelIr> {
    let Launch::Contract {
        post, acc, a, b, ..
    } = op
    else {
        return Err(Error::Plan("lower_sgemv on a non-Contract node".into()));
    };
    let shape = shape_of(&ctx, op)?;
    let acc_elem = scalar_element(*acc);
    let width = ctx.caps.subgroup_width();
    let block = (p.subgroups.max(1) * width)
        .min(ctx.caps.limits.max_compute_invocations_per_workgroup)
        .max(1);

    // One view per buffer each side reads; a side that has absorbed a
    // producer contributes one view per edge and the side's `pre` combines
    // them. Staged sources at the same matrix split the other families use:
    // A is `[batch * m, k]` and B is `[batch * k, n]`, whatever ranks those
    // extents are spread across. A quantized operand decodes at the same
    // coordinates through `contract_stage_source`.
    let a_rows = shape.batch.saturating_mul(shape.m).max(1);
    let b_rows = shape.batch.saturating_mul(shape.k).max(1);
    let stage = |ctx: &mut Ctx<'_>,
                 side: &ContractSide,
                 rows: u32,
                 cols: u32|
     -> Result<Vec<StagedSource>> {
        side.ops
            .iter()
            .map(|o| {
                if let Some(lit) = ctx.const_operand(o.src) {
                    return Ok(StagedSource::Const(lit));
                }
                let view = ctx.contract_operand_view(o, rows, cols)?;
                Ok(StagedSource::Mem(ctx.contract_stage_source(o, &view)?))
            })
            .collect()
    };
    let a_views = stage(&mut ctx, a, a_rows, shape.k.max(1))?;
    let b_views = stage(&mut ctx, b, b_rows, shape.n.max(1))?;
    let a_coords = SideCoords::for_side(
        &ctx,
        a,
        u64::from(shape.m.saturating_mul(shape.batch).max(1)),
        u64::from(shape.k.max(1)),
    )?;
    let b_coords = SideCoords::for_side(
        &ctx,
        b,
        u64::from(shape.batch.saturating_mul(shape.k).max(1)),
        u64::from(shape.n.max(1)),
    )?;
    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;

    if p.cols > 1 {
        return lower_sgemv_subgroup_cols(
            ctx, op, p, &shape, &a_views, &b_views, &a_coords, &b_coords, out_view,
        );
    }

    let mut body: Vec<Stmt> = Vec::new();
    let lane = ctx.b.builtin(Builtin::Lane);
    // One workgroup per output element, not per output row: the grid covers
    // `rows * n` and B is addressed at `(k, col)`. The flat workgroup index
    // is linearized against the dispatch grid — never raw `ProgramId(X)`,
    // because past the per-dimension cap `distribute_workgroups` folds the
    // dispatch onto a second slab.
    let groups = u32::try_from(
        u64::from(shape.m.saturating_mul(shape.batch).max(1))
            .saturating_mul(u64::from(shape.n.max(1)))
            .min(u64::from(u32::MAX)),
    )
    .expect("min'd to u32::MAX");
    let grid = distribute_workgroups(
        groups,
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    let wg = workgroup_index(&mut ctx, grid, groups);
    let n_e = ctx.b.u32(shape.n.max(1));
    // `wg` enumerates `[batch, m, n]` row-major: `row` is the A matrix row
    // (`batch * m + m_idx` — exactly the flat `wg / n`), and B's row is the
    // batch's k block plus the loop's own k.
    let row = ctx.b.binary(TileBinaryOp::Div, wg.clone(), n_e.clone(), NumericContract::RELAXED);
    let col = ctx.b.binary(TileBinaryOp::Rem, wg.clone(), n_e.clone(), NumericContract::RELAXED);
    let m_e = ctx.b.u32(shape.m.max(1));
    let batch_idx = ctx.b.binary(TileBinaryOp::Div, row.clone(), m_e, NumericContract::RELAXED);
    let k_e = ctx.b.u32(shape.k.max(1));
    let b_row_base = ctx.b.mul(batch_idx, k_e);

    let acc_local = ctx.b.local(ElementType::Scalar(acc_elem));
    let acc_read = ctx.b.load_local(acc_local.clone());
    let init = ctx.b.zero(acc_elem);
    let k_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let kk = ctx.b.load_local(k_index.clone());

    let vector = p.vector.max(1);
    let stride = ctx.b.u32(block * vector);
    let step = ctx.b.mul(kk, stride);
    // Each lane owns `vector` consecutive elements of k. Overlapping lanes
    // and vector offsets would double-count the interior of the window, and
    // contiguous ownership lets a quantized operand amortize its block
    // decode: the `vector` elements of one iteration land in one block, so
    // their scale subexpressions hash-cons to a single evaluation.
    let lane_base = {
        let v_e = ctx.b.u32(vector);
        let scaled = ctx.b.mul(lane.clone(), v_e);
        ctx.b.add(step.clone(), scaled)
    };
    // When the loop's stride divides k exactly, every index the body ever
    // forms is in range. A constant-true mask routes dense loads to the
    // unclamped straight-line path and quantized loads to the direct decode —
    // the clamp `Min` a mask forces is opaque to the aligned-window algebra.
    // Inexact shapes keep the per-element bound check.
    let exact = shape.k.max(1) % (block * vector).max(1) == 0;
    let mut partial = acc_read;
    for v in 0..vector {
        let v_off = ctx.b.u32(v);
        let k = ctx.b.add(lane_base.clone(), v_off);
        let mask = if exact {
            ctx.b.bool(true)
        } else {
            let k_bound = ctx.b.u32(shape.k.max(1));
            ctx.b.compare(TileCompareOp::Lt, k.clone(), k_bound)
        };
        let mut avs = Vec::with_capacity(a_views.len());
        for src in &a_views {
            let src = match src {
                StagedSource::Const(lit) => {
                    avs.push(lit.clone());
                    continue;
                }
                StagedSource::Mem(s) => s,
            };
            let fill = ctx.b.zero(source_element(src));
            avs.push(ctx.b.load(
                src.clone(),
                Addr::Rc2 {
                    row: row.clone(),
                    col: k.clone(),
                },
                mask.clone(),
                fill,
            ));
        }
        let mut bvs = Vec::with_capacity(b_views.len());
        for src in &b_views {
            let src = match src {
                StagedSource::Const(lit) => {
                    bvs.push(lit.clone());
                    continue;
                }
                StagedSource::Mem(s) => s,
            };
            let fill = ctx.b.zero(source_element(src));
            let b_row = ctx.b.add(b_row_base.clone(), k.clone());
            bvs.push(ctx.b.load(
                src.clone(),
                Addr::Rc2 {
                    row: b_row,
                    col: col.clone(),
                },
                mask.clone(),
                fill,
            ));
        }
        let a_coord_exprs = match &a_coords {
            Some(c) => c.at(&mut ctx, &row, &k),
            None => Vec::new(),
        };
        let b_coord_exprs = match &b_coords {
            Some(c) => {
                let b_row = ctx.b.add(b_row_base.clone(), k.clone());
                c.at(&mut ctx, &b_row, &col)
            }
            None => Vec::new(),
        };
        let av = ctx.eval_scalar(&a.pre, &avs, &a_coord_exprs)?;
        let bv = ctx.eval_scalar(&b.pre, &bvs, &b_coord_exprs)?;
        let mut av = ctx.b.cast(av, ElementType::Scalar(acc_elem));
        let mut bv = ctx.b.cast(bv, ElementType::Scalar(acc_elem));
        // A masked-out k lane contributes a zero, not `pre(0)`. The loads
        // above fill 0, but `pre` is an arbitrary scalar program over them:
        // `exp(s*scale - m) / l` turns an all-zero fill into `inf`, and
        // `fma(inf, 0, acc)` is NaN into the whole k-sum.
        if !exact {
            let zero = ctx.b.zero(acc_elem);
            av = ctx.b.select(mask.clone(), av, zero.clone());
            bv = ctx.b.select(mask.clone(), bv, zero);
        }
        partial = ctx.b.fma(av, bv, partial);
    }

    // The loop advances `block * vector` elements of k per iteration — the
    // stride the body actually indexes with — so that is what the count
    // divides by. `chunk` stays a schedule knob in name only until a kernel
    // gives it a meaning the body honors.
    let chunks = shape.k.div_ceil((block * vector).max(1)).max(1);
    let count = ctx.b.u32(chunks);
    body.push(Stmt::Loop {
        count: Some(count),
        index: Some(k_index),
        accumulators: vec![Accumulator {
            local: acc_local.clone(),
            init,
            update: partial,
        }],
        body: Vec::new(),
    });

    let lane_partial = ctx.b.load_local(acc_local);
    let fixed_subgroup = ctx.caps.subgroups.is_some_and(|s| s.is_fixed()) && block == width;
    let total = if fixed_subgroup {
        ctx.b
            .reduce(TileReduceOp::Sum, ReduceKind::Subgroup, lane_partial)
    } else {
        let scratch = ctx
            .b
            .tile("sgemv_scratch", ElementType::Scalar(acc_elem), &[block]);
        ctx.b.reduce(
            TileReduceOp::Sum,
            ReduceKind::Workgroup {
                scratch,
                group_size: block,
            },
            lane_partial,
        )
    };
    let value = ctx.eval_scalar(post, &[total], &[row.clone(), col.clone()])?;
    let value = ctx.b.cast(value, out_elem);
    let zero_u = ctx.b.u32(0);
    let mask = ctx.b.compare(TileCompareOp::Eq, lane, zero_u);
    body.push(Stmt::Store {
        dst: out_view,
        addr: Addr::Linear(wg),
        value,
        mask,
    });

    Ok(ctx.finish("sgemv", grid, block, body))
}

/// The `cols > 1` SGEMV structure: `p.cols` output columns per workgroup,
/// each subgroup owning `cols / subgroups` of them end-to-end.
///
/// All `width` lanes of one subgroup cooperate on each of its columns, so a
/// pass covers `width * vector` consecutive k elements — at `vector = 8` on a
/// 32-lane device that is exactly one 256-element quant super-block. The
/// pass's activation window is evaluated once per lane and reused across the
/// subgroup's columns (the A loads never mention the column, so they
/// hash-cons to a single evaluation; only the B loads and FMAs repeat), and
/// the reduction is a subgroup sum — no workgroup scratch, no barrier.
///
/// The grid is `rows * ceil(n / cols)` workgroups decomposed as
/// `(row, column group)` so `row` is uniform across the workgroup — a flat
/// `element / cols` split would straddle row boundaries and give each column
/// its own `row` expression, forfeiting the shared activation window.
#[allow(clippy::too_many_arguments)]
fn lower_sgemv_subgroup_cols(
    mut ctx: Ctx<'_>,
    op: &Launch,
    p: SgemvParams,
    shape: &Shape,
    a_views: &[StagedSource],
    b_views: &[StagedSource],
    a_coords: &Option<SideCoords>,
    b_coords: &Option<SideCoords>,
    out_view: fusor2_ir::ir::kernel::StorageView,
) -> Result<KernelIr> {
    let Launch::Contract {
        post, acc, a, b, ..
    } = op
    else {
        return Err(Error::Plan("lower_sgemv on a non-Contract node".into()));
    };
    let acc_elem = scalar_element(*acc);
    let out_elem = out_view.buffer.element;
    let width = ctx.caps.subgroup_width();
    let subgroups = p.subgroups.max(1);
    // The domain only generates these points on a fixed-subgroup device with
    // `cols % subgroups == 0` and the block within the invocation limit;
    // verify_launch rejects uneven spreads. A device where either fails has no
    // such point to select, so reaching here with one is a plan error.
    let block = subgroups * width;
    if p.cols % subgroups != 0
        || block > ctx.caps.limits.max_compute_invocations_per_workgroup
        || !ctx.caps.subgroups.is_some_and(|s| s.is_fixed())
    {
        return Err(Error::Plan(format!(
            "sgemv cols={} needs {subgroups} whole fixed-width subgroups",
            p.cols
        )));
    }
    let cps = p.cols / subgroups;
    let parts = p.parts.max(1);
    let run = p.run().max(1);
    if parts > 1
        && (p.vector % parts != 0
            || p.gap % run != 0
            || p.gap <= run
            || (width * run) % p.gap != 0)
    {
        return Err(Error::Plan(format!(
            "sgemv parts={parts} gap={} does not tile a width-{width} pass",
            p.gap
        )));
    }
    let n = shape.n.max(1);
    let rows = shape.batch.saturating_mul(shape.m).max(1);
    let groups_per_row = n.div_ceil(p.cols);
    let groups = u32::try_from(
        u64::from(rows)
            .saturating_mul(u64::from(groups_per_row))
            .min(u64::from(u32::MAX)),
    )
    .expect("min'd to u32::MAX");
    let grid = distribute_workgroups(
        groups,
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    let mut body: Vec<Stmt> = Vec::new();
    let wg = workgroup_index(&mut ctx, grid, groups);
    let gpr_e = ctx.b.u32(groups_per_row);
    let row = ctx.b.binary(
        TileBinaryOp::Div,
        wg.clone(),
        gpr_e.clone(),
        NumericContract::RELAXED,
    );
    let col_group = ctx.b.binary(TileBinaryOp::Rem, wg, gpr_e, NumericContract::RELAXED);
    let m_e = ctx.b.u32(shape.m.max(1));
    let batch_idx = ctx.b.binary(TileBinaryOp::Div, row.clone(), m_e, NumericContract::RELAXED);
    let k_e = ctx.b.u32(shape.k.max(1));
    let b_row_base = ctx.b.mul(batch_idx, k_e);

    let sg = ctx.b.builtin(Builtin::SubgroupId);
    let sg_lane = ctx.b.builtin(Builtin::SubgroupLane);
    let col_base = {
        let cols_e = ctx.b.u32(p.cols);
        let cps_e = ctx.b.u32(cps);
        let g = ctx.b.mul(col_group, cols_e);
        let s = ctx.b.mul(sg, cps_e);
        ctx.b.add(g, s)
    };

    let k_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let kk = ctx.b.load_local(k_index.clone());
    let vector = p.vector.max(1);
    let stride = ctx.b.u32(width * vector);
    let step = ctx.b.mul(kk, stride);
    // Each lane owns `vector` elements of the subgroup's pass. At
    // `parts == 1` they are consecutive — the same contiguous-ownership
    // contract as the whole-workgroup path. At `parts > 1` the window is
    // `parts` runs of `run` consecutive elements spaced `gap` apart:
    // `gap / run` adjacent lanes pack their runs into each gap, and a lane's
    // runs interleave across `parts` gaps, so the pass still covers exactly
    // `width * vector` consecutive k. A window that revisits a bit-packed
    // word at each of its k offsets makes the word loads hash-cons to one
    // evaluation instead of one per run.
    let lane_base = if parts <= 1 {
        let v_e = ctx.b.u32(vector);
        let scaled = ctx.b.mul(sg_lane.clone(), v_e);
        ctx.b.add(step, scaled)
    } else {
        let lanes_per_gap = p.gap / run;
        let lpg_e = ctx.b.u32(lanes_per_gap);
        let block_idx = ctx.b.binary(
            TileBinaryOp::Div,
            sg_lane.clone(),
            lpg_e.clone(),
            NumericContract::RELAXED,
        );
        let within = ctx.b.binary(
            TileBinaryOp::Rem,
            sg_lane.clone(),
            lpg_e,
            NumericContract::RELAXED,
        );
        let span_e = ctx.b.u32(p.gap * parts);
        let run_e = ctx.b.u32(run);
        let blk = ctx.b.mul(block_idx, span_e);
        let packed = ctx.b.mul(within, run_e);
        let local = ctx.b.add(blk, packed);
        ctx.b.add(step, local)
    };
    // Same constant-true routing as the whole-workgroup path: an exact k
    // keeps the unclamped straight-line loads and the aligned-window word
    // sharing; an exact n keeps the column loads and stores unmasked.
    let exact = shape.k.max(1) % (width * vector).max(1) == 0;
    let col_exact = n % p.cols == 0;
    let n_e = ctx.b.u32(n);

    // The pass's activation window, evaluated once and reused by every
    // column this subgroup owns.
    let mut a_vals: Vec<(TileExpr, TileExpr, TileExpr)> = Vec::with_capacity(vector as usize);
    for v in 0..vector {
        // Constant offset of window element `v` from the lane base: run
        // `v / run` sits `gap`-multiples out, position `v % run` within it.
        // At `parts == 1` this folds back to `v`.
        let off = if parts <= 1 {
            v
        } else {
            (v / run) * p.gap + (v % run)
        };
        let v_off = ctx.b.u32(off);
        let k = ctx.b.add(lane_base.clone(), v_off);
        let mask = if exact {
            ctx.b.bool(true)
        } else {
            let k_bound = ctx.b.u32(shape.k.max(1));
            ctx.b.compare(TileCompareOp::Lt, k.clone(), k_bound)
        };
        let mut avs = Vec::with_capacity(a_views.len());
        for src in a_views {
            let src = match src {
                StagedSource::Const(lit) => {
                    avs.push(lit.clone());
                    continue;
                }
                StagedSource::Mem(s) => s,
            };
            let fill = ctx.b.zero(source_element(src));
            avs.push(ctx.b.load(
                src.clone(),
                Addr::Rc2 {
                    row: row.clone(),
                    col: k.clone(),
                },
                mask.clone(),
                fill,
            ));
        }
        let a_coord_exprs = match a_coords {
            Some(c) => c.at(&mut ctx, &row, &k),
            None => Vec::new(),
        };
        let av = ctx.eval_scalar(&a.pre, &avs, &a_coord_exprs)?;
        let mut av = ctx.b.cast(av, ElementType::Scalar(acc_elem));
        // A masked-out k lane contributes a zero, not `pre(0)` — see the
        // whole-workgroup path for why the load's own zero fill is not
        // enough once `pre` is an arbitrary scalar program.
        if !exact {
            let zero = ctx.b.zero(acc_elem);
            av = ctx.b.select(mask.clone(), av, zero);
        }
        a_vals.push((k, mask, av));
    }

    // One accumulator per owned column, all advanced by the same loop.
    let cols_of_subgroup: Vec<(TileExpr, TileExpr)> = (0..cps)
        .map(|j| {
            let j_e = ctx.b.u32(j);
            let col = ctx.b.add(col_base.clone(), j_e);
            let ok = if col_exact {
                ctx.b.bool(true)
            } else {
                ctx.b.compare(TileCompareOp::Lt, col.clone(), n_e.clone())
            };
            (col, ok)
        })
        .collect();
    let mut accs = Vec::with_capacity(cps as usize);
    for (col, col_ok) in &cols_of_subgroup {
        let acc_local = ctx.b.local(ElementType::Scalar(acc_elem));
        let mut partial = ctx.b.load_local(acc_local.clone());
        for (k, mask, av) in &a_vals {
            let load_mask = if col_exact {
                mask.clone()
            } else {
                ctx.b.and(mask.clone(), col_ok.clone())
            };
            let mut bvs = Vec::with_capacity(b_views.len());
            for src in b_views {
                let src = match src {
                    StagedSource::Const(lit) => {
                        bvs.push(lit.clone());
                        continue;
                    }
                    StagedSource::Mem(s) => s,
                };
                let fill = ctx.b.zero(source_element(src));
                let b_row = ctx.b.add(b_row_base.clone(), k.clone());
                bvs.push(ctx.b.load(
                    src.clone(),
                    Addr::Rc2 {
                        row: b_row,
                        col: col.clone(),
                    },
                    load_mask.clone(),
                    fill,
                ));
            }
            let b_coord_exprs = match b_coords {
                Some(c) => {
                    let b_row = ctx.b.add(b_row_base.clone(), k.clone());
                    c.at(&mut ctx, &b_row, col)
                }
                None => Vec::new(),
            };
            let bv = ctx.eval_scalar(&b.pre, &bvs, &b_coord_exprs)?;
            let mut bv = ctx.b.cast(bv, ElementType::Scalar(acc_elem));
            if !exact {
                let zero = ctx.b.zero(acc_elem);
                bv = ctx.b.select(mask.clone(), bv, zero);
            }
            partial = ctx.b.fma(av.clone(), bv, partial);
        }
        let init = ctx.b.zero(acc_elem);
        accs.push(Accumulator {
            local: acc_local,
            init,
            update: partial,
        });
    }

    let chunks = shape.k.div_ceil((width * vector).max(1)).max(1);
    let count = ctx.b.u32(chunks);
    let locals: Vec<_> = accs.iter().map(|a| a.local.clone()).collect();
    body.push(Stmt::Loop {
        count: Some(count),
        index: Some(k_index),
        accumulators: accs,
        body: Vec::new(),
    });

    // Each column's partials never leave its subgroup, so the close is a
    // subgroup sum and the store is that subgroup's lane 0.
    let zero_u = ctx.b.u32(0);
    let lane0 = ctx.b.compare(TileCompareOp::Eq, sg_lane, zero_u);
    for (local, (col, col_ok)) in locals.into_iter().zip(cols_of_subgroup) {
        let lane_partial = ctx.b.load_local(local);
        let total = ctx
            .b
            .reduce(TileReduceOp::Sum, ReduceKind::Subgroup, lane_partial);
        let value = ctx.eval_scalar(post, &[total], &[row.clone(), col.clone()])?;
        let value = ctx.b.cast(value, out_elem);
        let addr = {
            let scaled = ctx.b.mul(row.clone(), n_e.clone());
            ctx.b.add(scaled, col)
        };
        let mask = ctx.b.and(lane0.clone(), col_ok);
        body.push(Stmt::Store {
            dst: out_view.clone(),
            addr: Addr::Linear(addr),
            value,
            mask,
        });
    }

    Ok(ctx.finish("sgemv_cols", grid, block, body))
}
