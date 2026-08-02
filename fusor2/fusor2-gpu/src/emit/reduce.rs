//! Cross-lane reductions: subgroup collectives, shared-memory trees, and the
//! loop-then-tree hybrid. The strategy is a parameter on the node, so it stays
//! a late capability-driven choice rather than a construction-time one.
//!
//! Owned by W8.

use fusor2_ir::ir::level2::{ElementType, ReduceKind, ScalarElement, Tile, TileExpr, TileReduceOp};
use fusor2_ir::target::EmitError;
use naga::{
    Barrier, BinaryOperator, Block, CollectiveOperation, Expression, Handle, MathFunction, Span,
    Statement, SubgroupOperation,
};

use super::expr::{binary_math, binary_operator};
use super::{Emitter, ScratchKind};

impl Emitter<'_> {
    pub fn reduce(
        &mut self,
        op: TileReduceOp,
        kind: &ReduceKind,
        value: &TileExpr,
        out: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        match kind {
            ReduceKind::Subgroup => {
                let element = value.element();
                let v = self.expr(value, out)?;
                self.subgroup_reduce(out, v, op, element)
            }
            ReduceKind::Workgroup {
                scratch,
                group_size,
            } => {
                let v = self.expr(value, out)?;
                self.tree_reduce(out, scratch, v, op, *group_size)
            }
            ReduceKind::Loop {
                iterations,
                index,
                scratch,
                group_size,
            } => {
                let acc = self.loop_reduce(out, op, value, *iterations, index)?;
                self.tree_reduce(out, scratch, acc, op, *group_size)
            }
        }
    }

    /// `Subgroup` — one collective, rejecting the operand shapes a collective
    /// cannot take.
    fn subgroup_reduce(
        &mut self,
        out: &mut Block,
        value: Handle<Expression>,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        let subgroup_op = match op {
            TileReduceOp::Sum => SubgroupOperation::Add,
            TileReduceOp::Product => SubgroupOperation::Mul,
            TileReduceOp::Max => SubgroupOperation::Max,
            TileReduceOp::Min => SubgroupOperation::Min,
        };
        let scalar = match element {
            ElementType::Scalar(ScalarElement::Bool) => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on bool is undefined".into(),
                ));
            }
            ElementType::Scalar(s) => s,
            ElementType::Vector { .. } => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on a vector is unsupported".into(),
                ));
            }
            ElementType::CoopMatrix { .. } => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on a cooperative fragment is unsupported".into(),
                ));
            }
        };
        let ty = self.element_type(ElementType::Scalar(scalar))?;
        let result = self.append(Expression::SubgroupOperationResult { ty });
        out.push(
            Statement::SubgroupCollectiveOperation {
                op: subgroup_op,
                collective_op: CollectiveOperation::Reduce,
                argument: value,
                result,
            },
            Span::default(),
        );
        Ok(result)
    }

    /// Per-lane accumulation into a spill local seeded with the reduce
    /// identity, then the tree.
    fn loop_reduce(
        &mut self,
        out: &mut Block,
        op: TileReduceOp,
        value: &TileExpr,
        iterations: u32,
        index: &fusor2_ir::ir::level2::Local,
    ) -> Result<Handle<Expression>, EmitError> {
        let element = value.element();
        let depth = self.depth;
        let acc = self.scratch_local(ScratchKind::Spill, element, depth)?;
        let identity = self.reduce_identity(op, element)?;
        self.store_local(out, acc, identity);

        let iter_local = self.private_local(index)?;
        let count = self.u32_lit(iterations);
        let value = value.clone();
        self.dynamic_loop(out, count, move |em, loop_body, loop_index| {
            em.store_local(loop_body, iter_local, loop_index);
            let v = em.expr(&value, loop_body)?;
            let current = em.load_local_handle(loop_body, acc);
            let combined = em.combine(loop_body, op, current, v);
            em.store_local(loop_body, acc, combined);
            Ok(())
        })?;
        Ok(self.load_local_handle(out, acc))
    }

    /// `Workgroup` — **barrier, seed `scratch[lane]`, barrier, then the halving
    /// tree with a barrier after each stride**.
    ///
    /// The leading barrier is load-bearing: when the scratch tile is reused
    /// inside one kernel, a lane could otherwise overwrite `scratch[lane]`
    /// while another lane still reads the previous reduction's value.
    fn tree_reduce(
        &mut self,
        out: &mut Block,
        scratch: &Tile,
        value: Handle<Expression>,
        op: TileReduceOp,
        group_size: u32,
    ) -> Result<Handle<Expression>, EmitError> {
        let block = self.workgroup_invocations;
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > block
            || !block.is_multiple_of(group_size)
        {
            return Err(EmitError::Unsupported(format!(
                "tree reduce needs a power-of-two group size dividing the block, got \
                 {group_size} with block {block}"
            )));
        }

        let lane = self.lane();
        let lane_ptr = self.tile_dynamic_pointer(out, scratch, lane)?;
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        self.store_tile_value(out, scratch, lane_ptr, value)?;
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == block {
            let zero = self.u32_lit(0);
            (lane, zero)
        } else {
            let group_offset = self.mod_literal_u32(out, lane, group_size);
            let group_base = self.bin(out, BinaryOperator::Subtract, lane, group_offset);
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit = self.u32_lit(stride);
            let participates = self.bin(out, BinaryOperator::Less, compare_index, limit);
            let scratch_c = scratch.clone();
            let (accept, ()) = self.nested(move |em, accept| {
                let rhs_index = em.add_literal_u32(accept, lane, stride);
                let lhs_ptr = em.tile_dynamic_pointer(accept, &scratch_c, lane)?;
                let rhs_ptr = em.tile_dynamic_pointer(accept, &scratch_c, rhs_index)?;
                let lhs = em.load_tile_value(accept, &scratch_c, lhs_ptr)?;
                let rhs = em.load_tile_value(accept, &scratch_c, rhs_ptr)?;
                let reduced = em.combine(accept, op, lhs, rhs);
                em.store_tile_value(accept, &scratch_c, lhs_ptr, reduced)
            })?;
            out.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            out.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let result_ptr = self.tile_dynamic_pointer(out, scratch, result_index)?;
        self.load_tile_value(out, scratch, result_ptr)
    }

    /// The **N-ary** reduction: an explicit log-tree over `lanes * block`
    /// scratch, evaluating the carrier's `merge` at every level.
    ///
    /// There is no hardware collective for a multi-lane merge — `subgroupAdd`
    /// folds one value with one operator — so `Subgroup` is refused rather than
    /// silently reducing lane 0 and dropping the rest. `Loop` is refused too:
    /// per-lane accumulation needs the carrier's identities to seed from, which
    /// live on `L1::KFold`, so the lowering builds that with `Stmt::Loop` and
    /// closes with the tree.
    ///
    /// Every `merge` expression reads only its formals, so all `lanes` merges
    /// are evaluated before any is written back and no level can read a slot its
    /// sibling has already overwritten.
    pub fn reduce_n(
        &mut self,
        kind: &ReduceKind,
        values: &[TileExpr],
        merge: &fusor2_ir::ir::level2::MergeBody,
        scratch: &[Tile],
        outs: &[fusor2_ir::ir::level2::Local],
        out: &mut Block,
    ) -> Result<(), EmitError> {
        let group_size = match kind {
            ReduceKind::Workgroup { group_size, .. } => *group_size,
            ReduceKind::Subgroup => {
                return Err(EmitError::Unsupported(
                    "a multi-lane merge has no subgroup collective: one value, one operator is \
                     all the hardware offers"
                        .into(),
                ));
            }
            ReduceKind::Loop { .. } => {
                return Err(EmitError::Unsupported(
                    "a multi-lane loop reduction seeds from the carrier's identities, which the \
                     lowering carries: build the per-lane loop with Stmt::Loop and close with \
                     ReduceKind::Workgroup"
                        .into(),
                ));
            }
        };
        let n = values.len();
        let block = self.workgroup_invocations;
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > block
            || !block.is_multiple_of(group_size)
        {
            return Err(EmitError::Unsupported(format!(
                "tree reduce needs a power-of-two group size dividing the block, got \
                 {group_size} with block {block}"
            )));
        }

        let staged: Vec<Handle<Expression>> = values
            .iter()
            .map(|v| self.expr(v, out))
            .collect::<Result<_, _>>()?;
        let lane = self.lane();
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        for (tile, value) in scratch.iter().zip(&staged) {
            let ptr = self.tile_dynamic_pointer(out, tile, lane)?;
            self.store_tile_value(out, tile, ptr, *value)?;
        }
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == block {
            let zero = self.u32_lit(0);
            (lane, zero)
        } else {
            let group_offset = self.mod_literal_u32(out, lane, group_size);
            let group_base = self.bin(out, BinaryOperator::Subtract, lane, group_offset);
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit = self.u32_lit(stride);
            let participates = self.bin(out, BinaryOperator::Less, compare_index, limit);
            let tiles: Vec<Tile> = scratch.to_vec();
            let merge = merge.clone();
            let (accept, ()) = self.nested(move |em, accept| {
                let rhs_index = em.add_literal_u32(accept, lane, stride);
                // Both partials into the formals first: a merge reads only its
                // formals, so nothing below can observe a half-written level.
                for (i, tile) in tiles.iter().enumerate() {
                    let lhs_ptr = em.tile_dynamic_pointer(accept, tile, lane)?;
                    let value = em.load_tile_value(accept, tile, lhs_ptr)?;
                    let local = em.private_local(&merge.lhs[i])?;
                    em.store_local(accept, local, value);
                    let rhs_ptr = em.tile_dynamic_pointer(accept, tile, rhs_index)?;
                    let value = em.load_tile_value(accept, tile, rhs_ptr)?;
                    let local = em.private_local(&merge.rhs[i])?;
                    em.store_local(accept, local, value);
                }
                let merged: Vec<Handle<Expression>> = merge
                    .body
                    .iter()
                    .map(|e| em.expr(e, accept))
                    .collect::<Result<_, _>>()?;
                for (tile, value) in tiles.iter().zip(merged) {
                    let ptr = em.tile_dynamic_pointer(accept, tile, lane)?;
                    em.store_tile_value(accept, tile, ptr, value)?;
                }
                Ok(())
            })?;
            out.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            out.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        for i in 0..n {
            let ptr = self.tile_dynamic_pointer(out, &scratch[i], result_index)?;
            let value = self.load_tile_value(out, &scratch[i], ptr)?;
            let local = self.private_local(&outs[i])?;
            self.store_local(out, local, value);
        }
        Ok(())
    }

    /// The binary the reduction folds with.
    pub(crate) fn combine(
        &mut self,
        body: &mut Block,
        op: TileReduceOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        let binop = op.binary();
        match binary_operator(binop) {
            Some(naga_op) => self.bin(body, naga_op, left, right),
            None => {
                let fun: MathFunction =
                    binary_math(binop).expect("min/max are the only math reductions");
                self.math2(body, fun, left, right)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::emit_module;
    use crate::emit::testkit::{self, *};
    use fusor2_ir::ir::level2::{Addr, KernelIr, Stmt, TileExprKind};

    /// `out[0] = reduce(in[lane])`, written by lane 0.
    fn sum_kernel(block: u32, kind: ReduceKind, name: &'static str) -> KernelIr {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), block, false);
        let dst = testkit::buffer(2, f32e(), 1, true);
        let sv = view(&src, &[block]);
        let dv = view(&dst, &[1]);
        let total = TileExpr::new(
            TileExprKind::Reduce {
                op: TileReduceOp::Sum,
                kind: Box::new(kind),
                value: load(&sv, lane()),
            },
            f32e(),
        );
        let lane_is_zero = TileExpr::new(
            TileExprKind::Compare {
                op: fusor2_ir::scalar::CmpOp::Eq,
                left: lane(),
                right: lit_u32(0),
            },
            boole(),
        );
        KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lit_u32(0)),
                value: total,
                mask: lane_is_zero,
            }],
            byte_arena: None,
            name,
        }
    }

    /// Test 9 — the `WgTree` fallback and the subgroup collective agree
    /// bit-for-bit, and the 256-lane tree is exact.
    ///
    /// A subgroup collective reduces within one subgroup, so the two
    /// strategies are compared at the subgroup width; the 256-lane total is
    /// checked against the tree, whose exactness over `[0, 256)` is the point.
    #[test]
    fn wgtree_reduce_matches_subgroup() {
        let Some(gpu) = gpu() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let width = gpu.caps().subgroup_width();

        // 256-lane tree over [0..256) is exactly 32640.
        let values: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let scratch = testkit::tile(f32e(), &[256]);
        let ir = sum_kernel(
            256,
            ReduceKind::Workgroup {
                scratch,
                group_size: 256,
            },
            "wgtree256",
        );
        let inputs = vec![uniforms(), bytes_of(&values), bytes_of(&[0.0])];
        let out = f32s(&run(&gpu, &ir, &no_plan(), &inputs, 2));
        assert_eq!(out[0], 32640.0);

        if gpu.caps().subgroups.is_none() {
            eprintln!("no subgroups; the WgTree fallback is the only path");
            return;
        }
        // At the subgroup width the two strategies must be bit-identical.
        let narrow: Vec<f32> = (0..width).map(|i| i as f32).collect();
        let scratch = testkit::tile(f32e(), &[width]);
        let tree = sum_kernel(
            width,
            ReduceKind::Workgroup {
                scratch,
                group_size: width,
            },
            "wgtree_narrow",
        );
        let collective = sum_kernel(width, ReduceKind::Subgroup, "subgroup_narrow");
        let inputs = vec![uniforms(), bytes_of(&narrow), bytes_of(&[0.0])];
        let a = f32s(&run(&gpu, &tree, &no_plan(), &inputs, 2));
        let b = f32s(&run(&gpu, &collective, &no_plan(), &inputs, 2));
        let expected = (width * (width - 1) / 2) as f32;
        assert_eq!(a[0], expected);
        assert_eq!(
            a[0].to_bits(),
            b[0].to_bits(),
            "tree {a:?} vs collective {b:?}"
        );
    }

    /// A group size that does not divide the block, is not a power of two, or
    /// exceeds it is refused rather than mis-lowered.
    #[test]
    fn illegal_group_sizes_are_refused() {
        for group_size in [0u32, 3, 512] {
            let scratch = testkit::tile(f32e(), &[256]);
            let ir = sum_kernel(
                256,
                ReduceKind::Workgroup {
                    scratch,
                    group_size,
                },
                "bad_group",
            );
            assert!(
                emit_module(&ir, &caps(false, true), &no_plan()).is_err(),
                "group_size {group_size} must be refused"
            );
        }
    }

    /// The emitted WGSL for a plain single-slot fold, as text.
    ///
    /// Set `FUSOR2_WGSL_DUMP=<dir>` to write the four shaders out; that is how
    /// [`single_slot_reduce_wgsl_is_unchanged`]'s goldens were recorded from the
    /// tree *before* the N-ary reduction landed.
    fn reduce_wgsl(name: &'static str, ir: &KernelIr) -> String {
        let emitted = emit_module(ir, &caps(false, true), &no_plan()).expect("emits");
        let mut flags = naga::back::wgsl::WriterFlags::empty();
        flags.set(naga::back::wgsl::WriterFlags::EXPLICIT_TYPES, true);
        let text = naga::back::wgsl::write_string(&emitted.module, &emitted.info, flags)
            .expect("wgsl");
        if let Ok(dir) = std::env::var("FUSOR2_WGSL_DUMP") {
            let _ = std::fs::write(format!("{dir}/{name}.wgsl"), &text);
        }
        text
    }

    /// **The fast path is byte-identical.** `TileReduceOp::{Sum,Max}` at one
    /// scalar slot must keep emitting the same subgroup collective and the same
    /// shared-memory tree it emitted before `Stmt::Reduce` existed. The assert is
    /// textual equality of the shader, not numeric agreement: every one of the
    /// passing folds in the suite goes down this path, and a diff here means the
    /// N-ary form was built *in place of* the collective rather than beside it.
    ///
    /// The goldens are FNV-1a hashes of the exact shader text plus its length, so
    /// a deliberate change is re-recorded by copying one line, and the failure
    /// message prints the text.
    #[test]
    fn single_slot_reduce_wgsl_is_unchanged() {
        let cases: [(&'static str, u64, usize); 4] = [
            ("sum_subgroup", 0x3b02_cd5a_329c_469b, 495),
            ("sum_wgtree", 0x32f0_985c_9f9e_51b3, 1861),
            ("max_subgroup", 0x7b04_6317_1c0c_11d6, 495),
            ("max_wgtree", 0x3564_68e0_a1c7_22bf, 1873),
        ];
        // Every shader is emitted before any is asserted, so one dump run
        // records all four.
        let texts: Vec<(&'static str, String)> = cases
            .iter()
            .map(|(name, _, _)| {
                let op = if name.starts_with("max") {
                    TileReduceOp::Max
                } else {
                    TileReduceOp::Sum
                };
                let kind = if name.ends_with("wgtree") {
                    ReduceKind::Workgroup {
                        scratch: testkit::tile(f32e(), &[64]),
                        group_size: 64,
                    }
                } else {
                    ReduceKind::Subgroup
                };
                (*name, reduce_wgsl(name, &op_kernel(64, op, kind, name)))
            })
            .collect();
        for ((name, want_hash, want_len), (_, text)) in cases.iter().zip(&texts) {
            let hash = fnv1a(text.as_bytes());
            assert_eq!(
                (hash, text.len()),
                (*want_hash, *want_len),
                "{name} shader moved:\n{text}"
            );
        }
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// [`sum_kernel`] at an arbitrary operator.
    fn op_kernel(block: u32, op: TileReduceOp, kind: ReduceKind, name: &'static str) -> KernelIr {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), block, false);
        let dst = testkit::buffer(2, f32e(), 1, true);
        let sv = view(&src, &[block]);
        let dv = view(&dst, &[1]);
        let total = TileExpr::new(
            TileExprKind::Reduce {
                op,
                kind: Box::new(kind),
                value: load(&sv, lane()),
            },
            f32e(),
        );
        let lane_is_zero = TileExpr::new(
            TileExprKind::Compare {
                op: fusor2_ir::scalar::CmpOp::Eq,
                left: lane(),
                right: lit_u32(0),
            },
            boole(),
        );
        KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lit_u32(0)),
                value: total,
                mask: lane_is_zero,
            }],
            byte_arena: None,
            name,
        }
    }

    /// `Loop` accumulates per lane into a spill local seeded with the
    /// identity, then runs the tree.
    #[test]
    fn loop_then_tree_lowers() {
        let index = testkit::local(u32e());
        let scratch = testkit::tile(f32e(), &[64]);
        let ir = sum_kernel(
            64,
            ReduceKind::Loop {
                iterations: 4,
                index,
                scratch,
                group_size: 64,
            },
            "loop_tree",
        );
        emit_module(&ir, &caps(false, true), &no_plan()).expect("loop reduce lowers");
    }
    /// **A hand-built one-lane `Stmt::Reduce` still emits the collective.** The
    /// statement form is the general merge, but its fast field is the same
    /// decision the expression form makes, so a node that reaches the emitter
    /// this way emits `subgroupAdd` rather than a generic merge loop.
    #[test]
    fn a_one_lane_reduce_statement_emits_the_subgroup_collective() {
        use fusor2_ir::ir::level2::{Local, LocalDecl, MergeBody};
        use fusor2_ir::scalar::BinOp;
        use std::sync::Arc;

        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), 64, false);
        let dst = testkit::buffer(2, f32e(), 1, true);
        let sv = view(&src, &[64]);
        let dv = view(&dst, &[1]);
        let lhs: Local = Arc::new(LocalDecl::new(f32e()));
        let rhs: Local = Arc::new(LocalDecl::new(f32e()));
        let out: Local = Arc::new(LocalDecl::new(f32e()));
        let load = |l: &Local| TileExpr::new(TileExprKind::LoadLocal(Arc::clone(l)), f32e());
        let body = TileExpr::new(
            TileExprKind::Binary {
                op: BinOp::Add,
                left: load(&lhs),
                right: load(&rhs),
                numeric: fusor2_ir::dtype::NumericContract::RELAXED,
            },
            f32e(),
        );
        let ir = KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 64,
            body: vec![
                Stmt::Reduce {
                    kind: Box::new(ReduceKind::Subgroup),
                    values: smallvec::smallvec![load_src(&sv)],
                    merge: Box::new(MergeBody {
                        lhs: smallvec::smallvec![lhs],
                        rhs: smallvec::smallvec![rhs],
                        body: smallvec::smallvec![body],
                    }),
                    fast: Some(TileReduceOp::Sum),
                    outs: smallvec::smallvec![Arc::clone(&out)],
                    scratch: Default::default(),
                },
                Stmt::Store {
                    dst: dv,
                    addr: Addr::Linear(lit_u32(0)),
                    value: load(&out),
                    mask: testkit::tru(),
                },
            ],
            byte_arena: None,
            name: "fast_stmt",
        };
        let text = reduce_wgsl("fast_stmt", &ir);
        assert!(
            text.contains("subgroupAdd"),
            "the fast field must reach the collective, got:\n{text}"
        );
        assert!(
            !text.contains("workgroupBarrier"),
            "a subgroup collective stages nothing"
        );
    }

    fn load_src(sv: &fusor2_ir::ir::level2::StorageView) -> TileExpr {
        load(sv, lane())
    }
}
