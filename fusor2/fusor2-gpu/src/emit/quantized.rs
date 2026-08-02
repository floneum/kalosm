//! Quantized loads: `Dequantize`, `LaneOf` and the fused `QuantizedDot`.
//!
//! Formats are **data**: the per-format decode is the `BlockProgram` that
//! `fusor2-gguf` supplies as a `BlockEmitFn`, invoked here with a
//! `BlockDecodeArgs`. Adding Q4_1 is a table row over there, not a kernel here.
//!
//! `QuantizedDot` decodes the block **once** and reuses it across every
//! activation lane; `Dequantize + Dot` re-decodes per lane, and the `Q8Dp4a`
//! packing is an integer dot that dequantize-then-dot cannot express at all.
//!
//! Owned by W8.

use fusor2_ir::dtype::{BlockDecodeArgs, BlockProgram, QAct};
use fusor2_ir::ir::level2::{
    ElementType, QuantizedView, ScalarElement, TileExpr, TileExprKind, TileLiteral,
};
use fusor2_ir::target::EmitError;
use naga::{BinaryOperator, Block, Expression, Handle, MathFunction, ScalarKind};

use super::Emitter;

/// One quad of int8-packed activations: the per-quad scale, the packed word,
/// and the integer lane sum (the term a min-carrying weight format subtracts).
#[derive(Clone, Debug)]
pub struct Q8Packs {
    pub scales: Vec<Handle<Expression>>,
    pub packs: Vec<Handle<Expression>>,
    pub sums: Vec<Handle<Expression>>,
}

/// int8 saturation bound.
const QMAX: f32 = 127.0;
/// Floor on a quad's dynamic range, so an all-zero quad does not divide by 0.
const SCALE_EPSILON: f32 = 1.0e-8;

fn f32_element() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

/// `k_base + offset` as an L2 expression, folded when both are literals.
fn offset_u32(k_base: &TileExpr, offset: u32) -> TileExpr {
    if offset == 0 {
        return k_base.clone();
    }
    let u32e = ElementType::Scalar(ScalarElement::U32);
    if let TileExprKind::Literal(TileLiteral::U32(base)) = k_base.kind() {
        return TileExpr::new(TileExprKind::Literal(TileLiteral::U32(base + offset)), u32e);
    }
    TileExpr::new(
        TileExprKind::Binary {
            op: fusor2_ir::scalar::BinOp::Add,
            left: k_base.clone(),
            right: TileExpr::new(TileExprKind::Literal(TileLiteral::U32(offset)), u32e),
            numeric: fusor2_ir::dtype::NumericContract::RELAXED,
        },
        u32e,
    )
}

impl Emitter<'_> {
    /// The decode program for one `(format, layout)` pair. Formats are data:
    /// the row lives in `fusor2-gguf`, never here.
    ///
    /// A test or a conformance oracle may substitute a reference decode with
    /// [`Self::with_block_program`]; nothing in the shipped path does.
    fn block_program(&self, view: &QuantizedView) -> BlockProgram {
        self.block_program_override
            .unwrap_or_else(|| fusor2_gguf::block_spec(view.fmt, view.layout).decode)
    }

    /// Substitute the decode program for every quantized node in this kernel.
    pub fn with_block_program(mut self, program: BlockProgram) -> Self {
        self.block_program_override = Some(program);
        self
    }

    /// Run the format's `BlockProgram` and return one handle per decoded lane.
    ///
    /// naga vectors cap at four lanes, so a wider block is decoded in groups
    /// of four with `k_base` advanced per group. The block scale is therefore
    /// decoded once per group rather than once per block; a `BlockSpec` that
    /// returned a lane array instead of a vector would remove even that.
    #[allow(clippy::too_many_arguments)]
    fn run_block_program(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        k_base: &TileExpr,
        col: &TileExpr,
        mask: &TileExpr,
        fill: &TileExpr,
        lanes: u32,
        program: BlockProgram,
    ) -> Result<Vec<Handle<Expression>>, EmitError> {
        const GROUP: u32 = 4;
        let mut handles = Vec::with_capacity(lanes as usize);
        let mut start = 0u32;
        while start < lanes {
            let width = (lanes - start).min(GROUP);
            let base = offset_u32(k_base, start);
            handles.extend(self.run_block_group(out, src, &base, col, mask, fill, width, program)?);
            start += width;
        }
        Ok(handles)
    }

    fn run_block_group(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        k_base: &TileExpr,
        col: &TileExpr,
        mask: &TileExpr,
        fill: &TileExpr,
        lanes: u32,
        program: BlockProgram,
    ) -> Result<Vec<Handle<Expression>>, EmitError> {
        let args = BlockDecodeArgs {
            src: &src.data,
            layout: src.layout,
            k_base: k_base.clone(),
            col: col.clone(),
            mask: mask.clone(),
            fill: fill.clone(),
            lanes,
        };
        let decoded = (program.emit)(&args)
            .map_err(|e| EmitError::Unsupported(format!("{} decode: {e}", program.name)))?;
        let handle = self.expr(&decoded, out)?;
        match decoded.element() {
            ElementType::Scalar(ScalarElement::F32) => Ok(vec![handle]),
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: n,
            } => {
                let mut out_handles = Vec::with_capacity(n as usize);
                for i in 0..n {
                    out_handles.push(self.emit_expr(
                        out,
                        Expression::AccessIndex {
                            base: handle,
                            index: i,
                        },
                    ));
                }
                Ok(out_handles)
            }
            other => Err(EmitError::Unsupported(format!(
                "{} decode returned {other:?}, expected f32 lanes",
                program.name
            ))),
        }
    }

    /// Run the format's `BlockProgram` and project every lane, memoized on the
    /// hash-consed `Dequantize` node so N `LaneOf`s share one decode.
    pub(crate) fn dequantize_lanes(
        &mut self,
        expr: &TileExpr,
        out: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, EmitError> {
        if let Some((handles, stamp)) = self.dequant_memo.get(expr)
            && self.stamp_is_current(expr.mem_reads(), stamp)
        {
            return Ok(handles.clone());
        }
        let TileExprKind::Dequantize {
            src,
            k_base,
            col,
            mask,
            fill,
            lanes,
        } = expr.kind()
        else {
            return Err(EmitError::Unsupported(
                "LaneOf expects a Dequantize block".into(),
            ));
        };
        let program = self.block_program(src);
        let handles = self.run_block_program(
            out,
            &src.clone(),
            &k_base.clone(),
            &col.clone(),
            &mask.clone(),
            &fill.clone(),
            *lanes,
            program,
        )?;
        let stamp = self.mem_epoch;
        self.dequant_memo
            .insert(expr.clone(), (handles.clone(), stamp));
        Ok(handles)
    }

    /// Decode exactly one element to f32. The scaffold's `dequantize` entry
    /// point, kept for callers that hold the pieces rather than a node.
    pub fn dequantize(
        &mut self,
        src: &QuantizedView,
        k_base: &TileExpr,
        col: &TileExpr,
        lanes: u32,
        out: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        let program = self.block_program(src);
        let mask = TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(true)),
            ElementType::Scalar(ScalarElement::Bool),
        );
        let fill = TileExpr::new(
            TileExprKind::Literal(TileLiteral::F32(0f32.to_bits())),
            f32_element(),
        );
        let handles =
            self.run_block_program(out, src, k_base, col, &mask, &fill, lanes, program)?;
        handles
            .into_iter()
            .next()
            .ok_or_else(|| EmitError::Unsupported("empty dequantized block".into()))
    }

    /// One decoded element, addressed by L2 expressions.
    pub(crate) fn decode_one(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        row: &TileExpr,
        col: &TileExpr,
    ) -> Result<Handle<Expression>, EmitError> {
        self.dequantize(src, row, col, 1, out)
    }

    /// One decoded element, addressed by already-lowered handles. Used by the
    /// collective quantized tile fill, whose coordinates are lane arithmetic.
    pub(crate) fn decode_one_handles(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        row: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        // The block program takes L2 expressions, so wrap the two handles in
        // opaque leaves the memo resolves straight back to them.
        let row_expr = self.opaque_u32(row);
        let col_expr = self.opaque_u32(col);
        self.decode_one(out, src, &row_expr, &col_expr)
    }

    /// Mint a fresh `TileExpr` leaf already bound to `handle` in the memo, so
    /// lowering it is the identity. The literal is a placeholder the memo
    /// shadows; a fresh `Builtin` would collide across calls, so the value is
    /// keyed by the handle's index.
    fn opaque_u32(&mut self, handle: Handle<Expression>) -> TileExpr {
        let tag = handle.index() as u32;
        let expr = TileExpr::new(
            TileExprKind::Bitcast {
                value: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::U32(tag)),
                    ElementType::Scalar(ScalarElement::U32),
                ),
                to: ElementType::Scalar(ScalarElement::U32),
            },
            ElementType::Scalar(ScalarElement::U32),
        );
        // The key is a synthetic pure node standing for an SSA handle that is
        // already defined, so the stamp is never consulted: an SSA value does
        // not go stale.
        let stamp = self.mem_epoch;
        self.memo.insert(expr.clone(), (handle, stamp));
        expr
    }

    /// The fused per-column quantized dot.
    pub fn quantized_dot(
        &mut self,
        expr: &TileExpr,
        out: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        let TileExprKind::QuantizedDot {
            src,
            packing,
            activations,
            k_base,
            col,
            mask,
            fill,
        } = expr.kind()
        else {
            return Err(EmitError::Unsupported(
                "quantized_dot expects a QuantizedDot node".into(),
            ));
        };
        let (src, packing) = (src.clone(), *packing);
        let (activations, k_base, col) = (activations.clone(), k_base.clone(), col.clone());
        let (mask, fill) = (mask.clone(), fill.clone());

        if mask.is_constant_true() {
            return self.quantized_dot_body(out, &src, packing, &activations, &k_base, &col);
        }
        let fill_source = fill.element();
        let fill_h = self.expr(&fill, out)?;
        let fill_h = self.cast_tile_value(out, fill_h, fill_source, f32_element())?;
        let mask_ty = mask.element();
        let mask_h = self.expr(&mask, out)?;
        let mask_h = self.condition_value(out, mask_h, mask_ty)?;
        self.masked_value(out, f32_element(), fill_h, mask_h, move |em, accept| {
            em.quantized_dot_body(accept, &src, packing, &activations, &k_base, &col)
        })
    }

    fn quantized_dot_body(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        packing: QAct,
        activations: &[TileExpr],
        k_base: &TileExpr,
        col: &TileExpr,
    ) -> Result<Handle<Expression>, EmitError> {
        if activations.is_empty() {
            return Ok(self.f32_lit(0.0));
        }
        let lanes = activations.len() as u32;
        let program = self.block_program(src);
        let always = TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(true)),
            ElementType::Scalar(ScalarElement::Bool),
        );
        let zero = TileExpr::new(
            TileExprKind::Literal(TileLiteral::F32(0f32.to_bits())),
            f32_element(),
        );
        // One decode for the whole block: the scale is read once, not per lane.
        let weights =
            self.run_block_program(out, src, k_base, col, &always, &zero, lanes, program)?;
        if weights.len() < activations.len() {
            return Err(EmitError::Unsupported(format!(
                "{} decoded {} lanes for {} activations",
                program.name,
                weights.len(),
                activations.len()
            )));
        }
        let mut acts = Vec::with_capacity(activations.len());
        for a in activations {
            acts.push(self.expr(a, out)?);
        }

        match packing {
            QAct::F32 => {
                let mut total = self.f32_lit(0.0);
                for (a, w) in acts.iter().zip(&weights) {
                    total = self.math3(out, MathFunction::Fma, *a, *w, total);
                }
                Ok(total)
            }
            QAct::Q8Dp4a => self.dp4a_dot(out, &acts, &weights),
        }
    }

    /// The DP4a path: quantize both sides to int8 per quad, `Pack4xI8Clamp`
    /// them, `Dot4I8Packed`, then rescale.
    ///
    /// The activation packs are cached per store/quant scope, so N columns
    /// against one activation set pack once.
    ///
    /// **Deviation, reported:** the weight side is packed from the decoded f32
    /// lanes rather than from the format's own integer quants, because
    /// `BlockSpec` exposes only an f32 decode program. The integer dot is real
    /// — both `Pack4xI8Clamp` and `Dot4I8Packed` are emitted — and the extra
    /// per-quad requantization stays inside the 2e-2 relative tolerance, but a
    /// `BlockSpec::quant_packs` emit function would remove it.
    fn dp4a_dot(
        &mut self,
        out: &mut Block,
        activations: &[Handle<Expression>],
        weights: &[Handle<Expression>],
    ) -> Result<Handle<Expression>, EmitError> {
        if !activations.len().is_multiple_of(4) {
            return Err(EmitError::Unsupported(format!(
                "int8 packing needs a multiple of 4 activations, got {}",
                activations.len()
            )));
        }
        let acts = self.q8_packs(out, activations)?;
        let quads = activations.len() / 4;
        let weight_quads: Vec<&[Handle<Expression>]> =
            weights[..activations.len()].chunks(4).collect();

        let mut total = self.f32_lit(0.0);
        for q in 0..quads {
            let (w_scale, w_pack, _) = self.pack_quad(out, weight_quads[q])?;
            let dot = self.math2(out, MathFunction::Dot4I8Packed, acts.packs[q], w_pack);
            let dot_f = self.cast_as(out, dot, ScalarKind::Float, Some(4));
            let scale = self.bin(out, BinaryOperator::Multiply, acts.scales[q], w_scale);
            total = self.math3(out, MathFunction::Fma, dot_f, scale, total);
        }
        Ok(total)
    }

    /// Cached int8 packing of one activation set.
    fn q8_packs(
        &mut self,
        out: &mut Block,
        activations: &[Handle<Expression>],
    ) -> Result<Q8Packs, EmitError> {
        let cache_key = activations.to_vec();
        if let Some(packs) = self.q8_cache.get(&cache_key) {
            return Ok(packs.clone());
        }
        let mut packs = Q8Packs {
            scales: Vec::new(),
            packs: Vec::new(),
            sums: Vec::new(),
        };
        for quad in activations.chunks(4) {
            let (scale, pack, sum) = self.pack_quad(out, quad)?;
            packs.scales.push(scale);
            packs.packs.push(pack);
            packs.sums.push(sum);
        }
        self.q8_cache.insert(cache_key, packs.clone());
        Ok(packs)
    }

    /// `(scale, packed_word, integer_sum)` for four f32 values.
    fn pack_quad(
        &mut self,
        out: &mut Block,
        quad: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Handle<Expression>, Handle<Expression>), EmitError> {
        if quad.len() != 4 {
            return Err(EmitError::Unsupported(
                "int8 packing works on quads of four".into(),
            ));
        }
        let qmax = self.f32_lit(QMAX);
        let mut max_abs = self.f32_lit(0.0);
        for value in quad {
            let abs = self.math1(out, MathFunction::Abs, *value);
            max_abs = self.math2(out, MathFunction::Max, max_abs, abs);
        }
        let eps = self.f32_lit(SCALE_EPSILON);
        max_abs = self.math2(out, MathFunction::Max, max_abs, eps);
        let inv_scale = self.bin(out, BinaryOperator::Divide, qmax, max_abs);
        let scale = self.bin(out, BinaryOperator::Divide, max_abs, qmax);

        let lo = self.f32_lit(-QMAX);
        let hi = self.f32_lit(QMAX);
        let mut sum = self.i32_lit(0);
        let mut components = Vec::with_capacity(4);
        for value in quad {
            let scaled = self.bin(out, BinaryOperator::Multiply, *value, inv_scale);
            let rounded = self.math1(out, MathFunction::Round, scaled);
            let clamped = self.math2(out, MathFunction::Min, rounded, hi);
            let clamped = self.math2(out, MathFunction::Max, clamped, lo);
            let as_i32 = self.cast_as(out, clamped, ScalarKind::Sint, Some(4));
            sum = self.bin(out, BinaryOperator::Add, sum, as_i32);
            components.push(as_i32);
        }
        let ty = self.vector_type(ScalarElement::I32, 4)?;
        let vector = self.emit_expr(out, Expression::Compose { ty, components });
        let pack = self.math1(out, MathFunction::Pack4xI8Clamp, vector);
        Ok((scale, pack, sum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::testkit::{self, *};
    use fusor2_ir::dtype::{QFmt, QLayout};
    use fusor2_ir::error::Result as IrResult;
    use fusor2_ir::ir::level2::{Addr, KernelIr, Source, Stmt, StorageView};

    /// A reference decode: lane `i` of the block at `(k_base, col)` is the
    /// f32 whose bits sit at `col * 8 + k_base + i` in the data buffer.
    ///
    /// The real programs live in `fusor2-gguf`; this one exists so the dot
    /// lowering is testable on its own numbers.
    fn reference_decode(args: &BlockDecodeArgs<'_>) -> IrResult<TileExpr> {
        let u32e = ElementType::Scalar(ScalarElement::U32);
        let f32e = f32_element();
        let base = TileExpr::new(
            TileExprKind::Binary {
                op: fusor2_ir::scalar::BinOp::Add,
                left: TileExpr::new(
                    TileExprKind::Binary {
                        op: fusor2_ir::scalar::BinOp::Mul,
                        left: args.col.clone(),
                        right: TileExpr::new(TileExprKind::Literal(TileLiteral::U32(8)), u32e),
                        numeric: fusor2_ir::dtype::NumericContract::RELAXED,
                    },
                    u32e,
                ),
                right: args.k_base.clone(),
                numeric: fusor2_ir::dtype::NumericContract::RELAXED,
            },
            u32e,
        );
        let mut parts = Vec::with_capacity(args.lanes as usize);
        for i in 0..args.lanes {
            let index = TileExpr::new(
                TileExprKind::Binary {
                    op: fusor2_ir::scalar::BinOp::Add,
                    left: base.clone(),
                    right: TileExpr::new(TileExprKind::Literal(TileLiteral::U32(i)), u32e),
                    numeric: fusor2_ir::dtype::NumericContract::RELAXED,
                },
                u32e,
            );
            let word = TileExpr::new(
                TileExprKind::Load {
                    src: Source::Storage(args.src.clone()),
                    addr: Box::new(Addr::Linear(index)),
                    mask: args.mask.clone(),
                    fill: TileExpr::new(TileExprKind::Literal(TileLiteral::U32(0)), u32e),
                },
                u32e,
            );
            parts.push(TileExpr::new(
                TileExprKind::Bitcast {
                    value: word,
                    to: f32e,
                },
                f32e,
            ));
        }
        if args.lanes == 1 {
            return Ok(parts.remove(0));
        }
        Ok(TileExpr::new(
            TileExprKind::Vec {
                scalar: ScalarElement::F32,
                lanes: args.lanes,
                parts,
            },
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: args.lanes,
            },
        ))
    }

    const REFERENCE: BlockProgram = BlockProgram {
        name: "reference",
        emit: reference_decode,
    };

    const LANES: u32 = 8;
    const COLS: u32 = 4;

    fn qview(data: &fusor2_ir::ir::level2::Buffer) -> QuantizedView {
        QuantizedView {
            data: StorageView {
                buffer: data.clone(),
                offset: 0,
                layout: fusor2_ir::ir::level2::TileLayout::contiguous(
                    fusor2_ir::ir::level2::MemoryLevel::Storage,
                    &[COLS * LANES],
                ),
            },
            fmt: QFmt::Q8_0,
            layout: QLayout::Native,
            rows: LANES,
            cols: COLS,
        }
    }

    fn activations() -> Vec<TileExpr> {
        (0..LANES).map(|i| lit_f32(0.25 + i as f32 * 0.5)).collect()
    }

    /// `out[lane] = dot(activations, weights[:, lane])`, three ways.
    fn dot_kernel(kind: DotKind) -> KernelIr {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let data = testkit::buffer(1, u32e(), COLS * LANES, false);
        let dst = testkit::buffer(2, f32e(), COLS, true);
        let dv = view(&dst, &[COLS]);
        let q = qview(&data);
        let acts = activations();
        let value = match kind {
            DotKind::Fused(packing) => TileExpr::new(
                TileExprKind::QuantizedDot {
                    src: q,
                    packing,
                    activations: acts,
                    k_base: lit_u32(0),
                    col: lane(),
                    mask: tru(),
                    fill: lit_f32(0.0),
                },
                f32e(),
            ),
            DotKind::DequantizeThenDot => {
                let block = TileExpr::new(
                    TileExprKind::Dequantize {
                        src: q,
                        k_base: lit_u32(0),
                        col: lane(),
                        mask: tru(),
                        fill: lit_f32(0.0),
                        lanes: LANES,
                    },
                    ElementType::Vector {
                        scalar: ScalarElement::F32,
                        lanes: LANES,
                    },
                );
                let mut total = lit_f32(0.0);
                for (i, a) in acts.into_iter().enumerate() {
                    let w = TileExpr::new(
                        TileExprKind::LaneOf {
                            block: block.clone(),
                            lane: i as u32,
                        },
                        f32_element(),
                    );
                    let product = TileExpr::new(
                        TileExprKind::Binary {
                            op: fusor2_ir::scalar::BinOp::Mul,
                            left: a,
                            right: w,
                            numeric: fusor2_ir::dtype::NumericContract::STRICT,
                        },
                        f32_element(),
                    );
                    total = TileExpr::new(
                        TileExprKind::Binary {
                            op: fusor2_ir::scalar::BinOp::Add,
                            left: total,
                            right: product,
                            numeric: fusor2_ir::dtype::NumericContract::STRICT,
                        },
                        f32_element(),
                    );
                }
                total
            }
        };
        KernelIr {
            buffers: vec![uni, data, dst],
            grid: [1, 1, 1],
            block: COLS,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lane()),
                value,
                mask: tru(),
            }],
            byte_arena: None,
            name: "qdot",
        }
    }

    #[derive(Copy, Clone)]
    enum DotKind {
        Fused(QAct),
        DequantizeThenDot,
    }

    fn weight_bytes() -> Vec<u8> {
        // Column-major-ish: column c, lane i -> 0.1 * (c + 1) * (i + 1).
        let mut out = Vec::new();
        for c in 0..COLS {
            for i in 0..LANES {
                let w = 0.1 * (c + 1) as f32 * (i + 1) as f32;
                out.extend_from_slice(&w.to_bits().to_le_bytes());
            }
        }
        out
    }

    fn expected() -> Vec<f32> {
        let acts: Vec<f32> = (0..LANES).map(|i| 0.25 + i as f32 * 0.5).collect();
        (0..COLS)
            .map(|c| {
                (0..LANES)
                    .map(|i| acts[i as usize] * 0.1 * (c + 1) as f32 * (i + 1) as f32)
                    .sum()
            })
            .collect()
    }

    /// Test 11 — the fused dot and dequantize-then-dot agree, and the DP4a
    /// path is a real integer dot.
    #[test]
    fn quantized_dot_matches_dequant_then_dot() {
        let caps = caps(false, true);
        let fused = emit_with_program(
            &dot_kernel(DotKind::Fused(QAct::F32)),
            &caps,
            &no_plan(),
            REFERENCE,
        )
        .expect("fused emit");
        let composed = emit_with_program(
            &dot_kernel(DotKind::DequantizeThenDot),
            &caps,
            &no_plan(),
            REFERENCE,
        )
        .expect("composed emit");
        let packed = emit_with_program(
            &dot_kernel(DotKind::Fused(QAct::Q8Dp4a)),
            &caps,
            &no_plan(),
            REFERENCE,
        )
        .expect("dp4a emit");

        // The DP4a path is an integer dot, not a dequantize-then-dot.
        let text = format!("{:#?}", packed.module);
        assert!(text.contains("Pack4xI8Clamp"), "missing Pack4xI8Clamp");
        assert!(text.contains("Dot4I8Packed"), "missing Dot4I8Packed");
        // ... and the f32 path is not.
        let f32_text = format!("{:#?}", fused.module);
        assert!(!f32_text.contains("Dot4I8Packed"));

        let Some(gpu) = gpu() else {
            eprintln!("no wgpu adapter; skipping the numeric half");
            return;
        };
        let inputs = vec![uniforms(), weight_bytes(), bytes_of(&[0.0; COLS as usize])];
        let want = expected();
        let a = f32s(&run_emitted(
            &gpu,
            &dot_kernel(DotKind::Fused(QAct::F32)),
            fused,
            &inputs,
            2,
        ));
        let b = f32s(&run_emitted(
            &gpu,
            &dot_kernel(DotKind::DequantizeThenDot),
            composed,
            &inputs,
            2,
        ));
        let c = f32s(&run_emitted(
            &gpu,
            &dot_kernel(DotKind::Fused(QAct::Q8Dp4a)),
            packed,
            &inputs,
            2,
        ));
        for i in 0..COLS as usize {
            let rel = |x: f32| ((x - want[i]) / want[i]).abs();
            assert!(rel(a[i]) < 1e-6, "fused {} vs {}", a[i], want[i]);
            assert!(rel(b[i]) < 1e-6, "composed {} vs {}", b[i], want[i]);
            assert!(rel(c[i]) < 2e-2, "dp4a {} vs {}", c[i], want[i]);
        }
    }

    /// One decode serves every lane: N `LaneOf`s share one block program run.
    #[test]
    fn lanes_share_one_decode() {
        let caps = caps(false, true);
        let composed = emit_with_program(
            &dot_kernel(DotKind::DequantizeThenDot),
            &caps,
            &no_plan(),
            REFERENCE,
        )
        .expect("emit");
        // The reference program emits one `Load` per lane; eight lanes sharing
        // one decode means exactly eight loads, not sixty-four.
        let loads = count_exprs(&composed.module, |e| {
            matches!(e, naga::Expression::Load { .. })
        });
        assert_eq!(loads, LANES as usize, "one decode, {LANES} loads");
    }

    /// A masked quantized load takes the fill without touching storage.
    #[test]
    fn masked_quantized_load_lowers() {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let data = testkit::buffer(1, u32e(), COLS * LANES, false);
        let dst = testkit::buffer(2, f32e(), COLS, true);
        let dv = view(&dst, &[COLS]);
        let value = TileExpr::new(
            TileExprKind::Load {
                src: Source::Quantized(qview(&data)),
                addr: Box::new(Addr::Rc2 {
                    row: lit_u32(0),
                    col: lane(),
                }),
                mask: TileExpr::new(
                    TileExprKind::Compare {
                        op: fusor2_ir::scalar::CmpOp::Lt,
                        left: lane(),
                        right: lit_u32(2),
                    },
                    boole(),
                ),
                fill: lit_f32(-1.0),
            },
            f32_element(),
        );
        let ir = KernelIr {
            buffers: vec![uni, data, dst],
            grid: [1, 1, 1],
            block: COLS,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lane()),
                value,
                mask: tru(),
            }],
            byte_arena: None,
            name: "masked_q",
        };
        let emitted =
            emit_with_program(&ir, &caps(false, true), &no_plan(), REFERENCE).expect("emit");
        let Some(gpu) = gpu() else {
            eprintln!("no wgpu adapter; skipping the numeric half");
            return;
        };
        let inputs = vec![uniforms(), weight_bytes(), bytes_of(&[0.0; COLS as usize])];
        let out = f32s(&run_emitted(&gpu, &ir, emitted, &inputs, 2));
        assert!((out[0] - 0.1).abs() < 1e-6);
        assert!((out[1] - 0.2).abs() < 1e-6);
        assert_eq!(out[2], -1.0);
        assert_eq!(out[3], -1.0);
    }
}
