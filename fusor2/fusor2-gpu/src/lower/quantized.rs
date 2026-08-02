//! `KQContract`: six formats x two `QLayout` x two `QAct` over five geometry
//! shapes.
//!
//! **Block decode is never inlined here.** Every format's bytes are decoded by
//! W11's `BlockSpec::decode` program, so adding Q4_1 is a table row and not a
//! kernel. `QAct::F32` emits `Dequantize` + `LaneOf` + FMA; `QAct::Q8Dp4a`
//! emits `QuantizedDot`, which decodes the block scale once and is provably
//! not expressible as dequantize-then-dot.
//!
//! Owned by W9.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::{NumericContract, QAct, QFmt, QLayout};
use fusor2_ir::error::Error;
use fusor2_ir::ir::Node;
use fusor2_ir::ir::level1::{L1, SchedPoint};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, QuantizedView, ScalarElement, Source,
    Stmt, TileBinaryOp, TileCompareOp, TileExpr, WorkgroupAxis,
};
use fusor2_ir::shape::Dim;
use fusor2_ir::target::LowerCtx;

use crate::lower::{Ctx, DimBinding, distribute_workgroups, scalar_element};

/// The five geometry shapes a quantized contraction can take. The reference's
/// nine-arm first-match selector is gone: extraction picks one of these, and
/// a divisibility near-miss degrades continuously on cost instead of falling
/// several rules down a list.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum QGeom {
    /// A full workgroup per output tile; the general case.
    Workgroup,
    /// One row per subgroup, one output column per lane.
    SingleRow,
    /// `SingleRow` narrowed for Q5's 22/24-byte blocks, whose scale read
    /// straddles a word boundary in the native layout.
    Q5SmallSingleRow,
    /// Q8's wide path: four columns per lane through `Dot4I8Packed`.
    Q8Wide,
    /// A cooperative or register tile over both M and N.
    Tile { bm: u32, bn: u32 },
}

impl QGeom {
    /// Structural legality only. Profitability is the cost model's.
    pub fn legal(self, fmt: QFmt, act: QAct, caps: &Caps) -> bool {
        match self {
            Self::Workgroup => true,
            Self::SingleRow => caps.subgroups.is_some(),
            Self::Q5SmallSingleRow => {
                matches!(fmt, QFmt::Q5_0 | QFmt::Q5K) && caps.subgroups.is_some()
            }
            Self::Q8Wide => fmt == QFmt::Q8_0 && act == QAct::Q8Dp4a,
            Self::Tile { bm, bn } => {
                bm > 0
                    && bn > 0
                    && bm * bn <= caps.limits.max_compute_invocations_per_workgroup * 16
            }
        }
    }

    /// Workgroup width this shape wants.
    pub fn block(self, caps: &Caps) -> u32 {
        let width = caps.subgroup_width().max(1);
        let cap = caps.limits.max_compute_invocations_per_workgroup;
        match self {
            Self::Workgroup => 256.min(cap),
            Self::SingleRow => width.min(cap),
            Self::Q5SmallSingleRow => (width / 2).max(1).min(cap),
            Self::Q8Wide => (width * 2).min(cap),
            Self::Tile { bm, bn } => (bm * bn).clamp(1, cap),
        }
    }

    /// Output columns one lane produces.
    pub fn cols_per_lane(self) -> u32 {
        match self {
            Self::Q8Wide => 4,
            Self::Tile { bn, .. } => bn.min(4).max(1),
            _ => 1,
        }
    }
}

/// Choose the geometry from the resolved schedule point.
///
/// Mapping is total: every `SchedPoint` a quantized contraction can carry
/// names a shape, so there is no unrouted case.
pub fn geom_for(theta: SchedPoint, fmt: QFmt, act: QAct, caps: &Caps) -> QGeom {
    let candidate = match theta {
        SchedPoint::Sgemv(p) if p.vector >= 4 => QGeom::Q8Wide,
        SchedPoint::Sgemv(_) => QGeom::SingleRow,
        SchedPoint::Sgemm(p) => QGeom::Tile {
            bm: p.bm.max(1),
            bn: p.bn.max(1),
        },
        SchedPoint::Coop { geom, .. } => QGeom::Tile {
            bm: geom.bm,
            bn: geom.bn,
        },
        _ => QGeom::Workgroup,
    };
    if candidate.legal(fmt, act, caps) {
        candidate
    } else if QGeom::Q5SmallSingleRow.legal(fmt, act, caps) {
        QGeom::Q5SmallSingleRow
    } else if QGeom::SingleRow.legal(fmt, act, caps) {
        QGeom::SingleRow
    } else {
        QGeom::Workgroup
    }
}

/// Contract-shaped entry point (see CONTRACTS.md §4.10).
pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let fusor2_ir::ir::Op::L1(op) = &node.op else {
        return Err(Error::Plan("quantized got a foreign node".into()));
    };
    let ctx = Ctx::new(caps, cx, DimBinding::new())?;
    lower_kqcontract(ctx, op, theta)
}

/// Lower a quantized contraction.
pub fn lower_kqcontract(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KQContract {
        fmt,
        layout,
        act,
        m,
        n,
        k,
        acc,
        post,
        a,
        b,
        ..
    } = op
    else {
        return Err(Error::Plan(
            "lower_kqcontract on a non-KQContract node".into(),
        ));
    };
    let get = |d: Dim| -> Result<u32> {
        u32::try_from(ctx.binding.require(d)?)
            .map_err(|_| Error::Plan("quantized extent exceeds a u32".into()))
    };
    let (m, n, k) = (get(*m)?, get(*n)?, get(*k)?);
    let geom = geom_for(theta, *fmt, *act, ctx.caps);
    let block = geom.block(ctx.caps);
    let cols = geom.cols_per_lane();
    let acc_elem = scalar_element(*acc);

    // Both `QLayout`s are legal inputs everywhere; the plan says which one this
    // weight carries, so a device that never runs the alignment-sensitive
    // kernel does not pay for its requirement.
    let weights = QuantizedView {
        data: ctx.linear_view(b.src)?,
        fmt: *fmt,
        layout: *layout,
        rows: n,
        cols: k,
    };
    let spec = fusor2_gguf::block_spec(*fmt, *layout);
    if !spec.activation.contains(act) {
        return Err(Error::Plan(format!(
            "{fmt:?}/{layout:?} does not support activation packing {act:?}"
        )));
    }
    let block_elems = u32::from(spec.elements).max(1);
    let blocks_per_row = k.div_ceil(block_elems).max(1);

    let a_view = ctx.matrix_view(a, 1)?;
    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;

    let mut body: Vec<Stmt> = Vec::new();
    let lane = ctx.b.builtin(Builtin::Lane);
    let gx = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::X));

    // Row/column identity per geometry shape.
    let (row, col_base) = match geom {
        QGeom::Workgroup | QGeom::Tile { .. } => {
            let n_e = ctx.b.u32(n.max(1));
            let flat = {
                let bl = ctx.b.u32(block);
                let base = ctx.b.mul(gx, bl);
                ctx.b.add(base, lane.clone())
            };
            let r = ctx.b.binary(
                TileBinaryOp::Div,
                flat.clone(),
                n_e.clone(),
                NumericContract::RELAXED,
            );
            let c = ctx
                .b
                .binary(TileBinaryOp::Rem, flat, n_e, NumericContract::RELAXED);
            (r, c)
        }
        QGeom::SingleRow | QGeom::Q5SmallSingleRow | QGeom::Q8Wide => {
            let cpl = ctx.b.u32(cols);
            let c = ctx.b.mul(lane.clone(), cpl);
            (gx.clone(), c)
        }
    };
    let m_e = ctx.b.u32(m.max(1));
    let n_e = ctx.b.u32(n.max(1));
    let row_live = ctx.b.compare(TileCompareOp::Lt, row.clone(), m_e);
    let col_live = ctx
        .b
        .compare(TileCompareOp::Lt, col_base.clone(), n_e.clone());
    let live = ctx.b.and(row_live, col_live);

    // One accumulator per output column this lane owns.
    let mut accs: Vec<Accumulator> = Vec::with_capacity(cols as usize);
    let mut locals = Vec::with_capacity(cols as usize);

    let blk_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let blk = ctx.b.load_local(blk_index.clone());
    let be = ctx.b.u32(block_elems);
    let k_base = ctx.b.mul(blk.clone(), be.clone());

    // Activations for this block, read once and shared by every column.
    let mut activations: Vec<TileExpr> = Vec::with_capacity(block_elems as usize);
    for e in 0..block_elems {
        let e_e = ctx.b.u32(e);
        let kk = ctx.b.add(k_base.clone(), e_e);
        let k_e = ctx.b.u32(k.max(1));
        let in_k = ctx.b.compare(TileCompareOp::Lt, kk.clone(), k_e);
        let zero = ctx.b.f32(0.0);
        let v = ctx.b.load(
            Source::Storage(a_view.clone()),
            Addr::Rc2 {
                row: row.clone(),
                col: kk,
            },
            in_k,
            zero,
        );
        activations.push(ctx.b.cast(v, ElementType::Scalar(ScalarElement::F32)));
    }

    for c in 0..cols {
        let local = ctx.b.local(ElementType::Scalar(acc_elem));
        let init = ctx.b.zero(acc_elem);
        let read = ctx.b.load_local(local.clone());
        let c_e = ctx.b.u32(c);
        let col = ctx.b.add(col_base.clone(), c_e);
        let fill = ctx.b.f32(0.0);
        // The decode program's flat element index is `k_base + col + lane`
        // (`fusor2-gguf/src/decode.rs`, "Addressing convention"): the *caller*
        // folds the row stride into `col`. Passing the bare weight-row index
        // read `k` elements into row 0 instead of the start of row `col`, so
        // output column 0 was right and every other column decoded the wrong
        // block.
        let row_start = {
            let k_e = ctx.b.u32(k.max(1));
            ctx.b.mul(col.clone(), k_e)
        };

        let contribution = match act {
            // Dequantize the block into registers, then FMA. One decode is
            // projected per lane with `LaneOf`, so the scale is not re-decoded
            // per element.
            QAct::F32 => {
                let decoded = ctx.b.dequantize(
                    weights.clone(),
                    k_base.clone(),
                    row_start,
                    live.clone(),
                    fill,
                    block_elems,
                );
                let mut sum = ctx.b.zero(acc_elem);
                for (e, a_v) in activations.iter().enumerate() {
                    let w = ctx.b.lane_of(decoded.clone(), e as u32);
                    let w = ctx.b.cast(w, ElementType::Scalar(acc_elem));
                    let a_v = ctx.b.cast(a_v.clone(), ElementType::Scalar(acc_elem));
                    sum = ctx.b.fma(a_v, w, sum);
                }
                sum
            }
            // `Pack4xI8Clamp` the activations and `Dot4I8Packed` against
            // still-quantized weights. This decodes the block scale exactly
            // once and is not expressible as dequantize-then-dot.
            QAct::Q8Dp4a => {
                let d = ctx.b.quantized_dot(
                    weights.clone(),
                    QAct::Q8Dp4a,
                    activations.clone(),
                    k_base.clone(),
                    row_start,
                    live.clone(),
                    fill,
                );
                ctx.b.cast(d, ElementType::Scalar(acc_elem))
            }
        };
        let update = ctx.b.add(read, contribution);
        locals.push(local.clone());
        accs.push(Accumulator {
            local,
            init,
            update,
        });
    }

    let count = ctx.b.u32(blocks_per_row);
    body.push(Stmt::Loop {
        count: Some(count),
        index: Some(blk_index),
        accumulators: accs,
        body: Vec::new(),
    });

    // **No cross-lane reduction.** Every shape here gives one lane its own
    // `(row, col)` and loops that lane over *all* `blocks_per_row` blocks, so
    // the accumulator is already the whole dot product. Summing across the
    // group — which `Workgroup` did over 256 lanes and the single-row shapes
    // did over a subgroup — added every *other* output element of the
    // workgroup into this one. Splitting k across lanes would need `k_base` to
    // carry the lane index, and then the reduce would belong.
    for (c, local) in locals.into_iter().enumerate() {
        let total = ctx.b.load_local(local);
        let value = ctx.eval_scalar(post, &[total], &[row.clone()])?;
        let value = ctx.b.cast(value, out_elem);
        let c_e = ctx.b.u32(c as u32);
        let col = ctx.b.add(col_base.clone(), c_e);
        let addr = {
            let base = ctx.b.mul(row.clone(), n_e.clone());
            ctx.b.add(base, col.clone())
        };
        let in_n = ctx.b.compare(TileCompareOp::Lt, col, n_e.clone());
        let mask = ctx.b.and(live.clone(), in_n);
        body.push(Stmt::Store {
            dst: out_view.clone(),
            addr: Addr::Linear(addr),
            value,
            mask,
        });
    }

    let groups = match geom {
        QGeom::Workgroup | QGeom::Tile { .. } => {
            m.saturating_mul(n).div_ceil(block.max(1)).max(1)
        }
        _ => m.max(1),
    };
    let grid = distribute_workgroups(
        groups,
        ctx.caps.limits.max_compute_workgroups_per_dimension,
    );
    Ok(ctx.finish("qcontract", grid, block, body))
}

/// The 24 bodies this module covers: six formats x two layouts x two
/// activation packings. Exposed so conformance can enumerate them without
/// re-deriving the product.
pub fn all_variants() -> Vec<(QFmt, QLayout, QAct)> {
    let mut out = Vec::with_capacity(24);
    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            for act in [QAct::F32, QAct::Q8Dp4a] {
                out.push((fmt, layout, act));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::device::{DeviceKind, Limits, SubgroupWidths};

    fn caps(subgroups: bool) -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: subgroups.then_some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: false,
            coop: Default::default(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    #[test]
    fn the_variant_product_is_twenty_four() {
        assert_eq!(all_variants().len(), 24);
    }

    #[test]
    fn every_variant_resolves_to_a_legal_geometry() {
        let c = caps(true);
        for (fmt, _layout, act) in all_variants() {
            let g = geom_for(SchedPoint::Point, fmt, act, &c);
            assert!(g.legal(fmt, act, &c), "{fmt:?}/{act:?} -> {g:?} is illegal");
            assert!(g.block(&c) >= 1);
        }
    }

    #[test]
    fn q8_wide_needs_the_dp4a_packing() {
        let c = caps(true);
        assert!(QGeom::Q8Wide.legal(QFmt::Q8_0, QAct::Q8Dp4a, &c));
        assert!(!QGeom::Q8Wide.legal(QFmt::Q8_0, QAct::F32, &c));
        assert!(!QGeom::Q8Wide.legal(QFmt::Q4K, QAct::Q8Dp4a, &c));
    }

    #[test]
    fn q5_small_single_row_is_only_for_q5() {
        let c = caps(true);
        assert!(QGeom::Q5SmallSingleRow.legal(QFmt::Q5_0, QAct::F32, &c));
        assert!(QGeom::Q5SmallSingleRow.legal(QFmt::Q5K, QAct::F32, &c));
        assert!(!QGeom::Q5SmallSingleRow.legal(QFmt::Q4_0, QAct::F32, &c));
    }

    /// Without subgroups every shape degrades to `Workgroup`, which is always
    /// legal — a fallback in geometry, never in correctness.
    #[test]
    fn no_subgroups_degrades_to_workgroup() {
        let c = caps(false);
        for (fmt, _l, act) in all_variants() {
            let g = geom_for(SchedPoint::Point, fmt, act, &c);
            assert_eq!(g, QGeom::Workgroup, "{fmt:?}/{act:?}");
        }
    }

    #[test]
    fn q8_wide_produces_four_columns_per_lane() {
        assert_eq!(QGeom::Q8Wide.cols_per_lane(), 4);
        assert_eq!(QGeom::SingleRow.cols_per_lane(), 1);
    }

    /// The decode program addresses `k_base + col + lane` in **elements**, so
    /// the weight-row stride belongs in `col`. Passing the bare row index
    /// decoded output column 0 correctly and every other column from row 0.
    #[test]
    fn a_weight_rows_decode_base_is_the_row_index_times_k() {
        for fmt in QFmt::ALL {
            let k = u32::from(fmt.block_elements());
            for row in 0u32..4 {
                let flat = row * k;
                assert_eq!(flat / k, row, "{fmt:?}: row {row} must select block {row}");
                assert_eq!(flat % k, 0, "{fmt:?}: a row starts at its block's element 0");
            }
        }
    }
}
