#![allow(unused_imports)]
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    BlockDequantId, BufferAccess, BufferDecl, BufferRef, CoopAccDecl, CoopAccId, CoopFragmentId,
    CoopOperandRole, DynamicOffset, F32Bits, F32Vec4, Im2ColNhwcMap, KernelIr, Layout, LocalDecl,
    LocalRef, LoopFoldGroup, LoopFoldGroupId, MemoryLevel, Numeric, Op, PinId,
    QuantizedVecDotKind, Shape, StorageIndexMap, StorageView, TileBinaryOp, TileCompareOp,
    TileDecl, TileExpr, TileIndexExpr, TileIndexedStoreStmt, TileLevel, TileLinearLoadExpr,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr,
    TileReduceOp, TileRef, TileScalarExpr, TileStmt, TileStoreStmt, TileUnaryOp, TileVec4LoadExpr,
    WorkgroupAxis, WorkgroupOffset, F32, U32,
};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};
use super::*;
use super::types::{matrix_shape, cooperative_store_layout_supported};
use super::grid::{qgemv_grid, store_qgemv_sums, q4k_ggml_activations, dot4_sum};

macro_rules! tile_reduce_entrypoints {
    ($(($reduce:ident, $loop_reduce:ident, $group_reduce:ident, $subgroup_reduce:ident, $op:ident)),+ $(,)?) => {
        $(
            pub fn $reduce(&mut self, value: Tile<BLOCK>) -> Scalar {
                self.reduce(TileReduceOp::$op, value)
            }

            pub fn $loop_reduce(&mut self, iterations: u32, value: Tile<BLOCK>) -> Scalar {
                self.loop_reduce(TileReduceOp::$op, iterations, value)
            }

            pub fn $group_reduce<const GROUP: usize>(&mut self, value: Tile<BLOCK>) -> Tile<BLOCK> {
                self.group_reduce::<GROUP>(TileReduceOp::$op, value)
            }

            pub fn $subgroup_reduce(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
                self.subgroup_reduce(TileReduceOp::$op, value)
            }
        )+
    };
}

macro_rules! quantized_vec_dot_entrypoint {
    ($name:ident, $kind:ident, [$($n:literal),+ $(,)?], $msg:literal) => {
        pub fn $name<const N: usize>(
            &self,
            a: [Tile<BLOCK>; N],
            matrix: &QuantizedMatrix,
            k_base: impl IntoIndex<BLOCK>,
            col: impl IntoIndex<BLOCK>,
            mask: Mask<BLOCK>,
            fill: f32,
        ) -> Tile<BLOCK> {
            assert!($(N == $n)||+, $msg);
            Tile {
                expr: TileExpr::QuantizedVecDot {
                    kind: QuantizedVecDotKind::$kind,
                    a: a.into_iter().map(|value| Box::new(value.expr)).collect(),
                    src: matrix.clone(),
                    k_base: k_base.into_index(),
                    col: col.into_index(),
                    mask: mask.expr,
                    fill: F32Bits::new(fill),
                    block_n: N as u32,
                },
            }
        }
    };
}

pub struct TileBlock<'a, const BLOCK: usize> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) body: Vec<TileStmt>,
    /// Stack of nested statement builders. The innermost frame collects
    /// statements emitted inside `while_true` closures; popped into
    /// `WhileTrue` on closure exit.
    pub(super) stmt_stack: Vec<Vec<TileStmt>>,
}


impl<const BLOCK: usize> TileBlock<'_, BLOCK> {
    pub fn program_id(&self, axis: WorkgroupAxis) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::ProgramId(axis),
        }
    }

    pub fn subgroup_id(&self) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::SubgroupId,
        }
    }

    pub fn subgroup_lane(&self) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::SubgroupLane,
        }
    }

    pub fn subgroup_size(&self) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::SubgroupSize,
        }
    }

    pub fn num_subgroups(&self) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::NumSubgroups,
        }
    }

    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn arange(&self) -> Range<BLOCK> {
        Range {
            expr: TileIndexExpr::Lane,
        }
    }

    pub fn lane_tile_2d<const ROWS: usize, const COLS: usize>(
        &self,
    ) -> LaneTile2d<ROWS, COLS, BLOCK> {
        assert!(
            ROWS > 0 && COLS > 0 && ROWS * COLS == BLOCK,
            "2D lane tile shape must match the tile program block size"
        );
        let lane = self.arange();
        LaneTile2d {
            row: lane.clone() / COLS as u32,
            col: lane % COLS as u32,
        }
    }

    pub fn loop_index(&self) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::LoopIndex,
        }
    }

    pub fn load<T>(&self, address: Address<T, BLOCK>, mask: Mask<BLOCK>, fill: f32) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Load(TileLoadExpr {
                src: address.view,
                row: address.row,
                col: address.col,
                mask: mask.expr,
                fill: TileLiteral::F32(F32Bits::new(fill)),
            }),
        }
    }

    pub fn load_linear<T: Numeric>(
        &self,
        address: LinearAddress<T, BLOCK>,
        mask: Mask<BLOCK>,
        fill: TileLiteral,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::LoadLinear(TileLinearLoadExpr {
                src: address.view,
                index: address.index,
                mask: mask.expr,
                fill,
            }),
        }
    }

    pub fn load_vec4(
        &self,
        address: LinearAddress<F32Vec4, BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::LoadVec4(TileVec4LoadExpr {
                src: address.view,
                index: address.index,
                mask: mask.expr,
                fill: F32Bits::new(fill),
            }),
        }
    }

    pub fn load_erased(
        &self,
        address: ErasedAddress<BLOCK>,
        mask: Mask<BLOCK>,
        fill: TileLiteral,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Load(TileLoadExpr {
                src: address.view,
                row: address.row,
                col: address.col,
                mask: mask.expr,
                fill,
            }),
        }
    }

    pub fn load_quantized(
        &self,
        matrix: &QuantizedMatrix,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::QuantizedLoad(TileQuantizedLoadExpr {
                src: matrix.clone(),
                row: row.into_index(),
                col: col.into_index(),
                mask: mask.expr,
                fill: F32Bits::new(fill),
            }),
        }
    }

    /// Load N consecutive dequantized values from one column of a packed
    /// quantized matrix. The lowerer emits a format-specific helper when one
    /// exists, otherwise it lowers the same block-shaped request as N scalar
    /// dequantizations. Each lane is bound to a private local that subsequent
    /// references load. `k_base` must be aligned to N so the values cover one
    /// scale block.
    pub fn load_quantized_block<const N: usize>(
        &mut self,
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> [Tile<BLOCK>; N] {
        assert!(
            N == 8 || N == 16,
            "load_quantized_block currently supports N == 8 or N == 16"
        );
        let id = self.program.next_block_dequant_id();
        let k_base = k_base.into_index();
        let col = col.into_index();
        let mask_expr = mask.expr;
        let fill_bits = F32Bits::new(fill);
        std::array::from_fn(|lane| Tile {
            expr: TileExpr::QuantizedBlockLane {
                id,
                src: matrix.clone(),
                k_base: k_base.clone(),
                col: col.clone(),
                mask: mask_expr.clone(),
                fill: fill_bits,
                block_n: N as u32,
                lane: lane as u32,
            },
        })
    }

    /// Bind a subexpression to a private local so subsequent references reuse
    /// the value without re-emitting its computation. Returns N references that
    /// all evaluate to the same value within the same scope.
    pub fn pin(&mut self, value: Tile<BLOCK>) -> Pinned<BLOCK> {
        let id = self.program.next_pin_id(value.expr);
        Pinned {
            id,
            _block: PhantomData,
        }
    }

    /// Build a left-associated sum from a flat value list without creating a
    /// deep nested binary expression.
    pub fn sum(&self, values: impl IntoIterator<Item = Tile<BLOCK>>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Sum {
                values: values
                    .into_iter()
                    .map(|value| Box::new(value.expr))
                    .collect(),
            },
        }
    }

    /// Run one K-loop with N parallel reductions. The body closure runs once
    /// at IR-build time and produces N tile expressions that all share the
    /// same loop scope; the lowerer materializes a single Naga loop with N
    /// accumulator locals so common subexpressions across the N outputs are
    /// emitted only once per iteration (when bound via `pin`).
    pub fn loop_fold_n<const N: usize, F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initials: [TileLiteral; N],
        body: F,
    ) -> [Tile<BLOCK>; N]
    where
        F: FnOnce(&mut Self) -> [Tile<BLOCK>; N],
    {
        assert!(iterations > 0, "loop_fold_n iterations must be non-zero");
        assert!(N > 0, "loop_fold_n must have at least one accumulator");
        let bodies = body(self);
        let group = self.program.next_loop_fold_group_id(LoopFoldGroup {
            iterations,
            op,
            initials: initials.to_vec(),
            bodies: bodies.into_iter().map(|t| t.expr).collect(),
        });
        std::array::from_fn(|lane| Tile {
            expr: TileExpr::LoopFoldGroupOutput {
                group,
                lane: lane as u32,
            },
        })
    }

    /// Fused 4-way dot product: `a[0]*b[0] + .. + a[3]*b[3]` in a single
    /// `Math::Dot` over `vec4<f32>` operands. Lowers to the same instruction
    /// sequence the qgemv accelerator emits.
    pub fn dot4(&self, a: [Tile<BLOCK>; 4], b: [Tile<BLOCK>; 4]) -> Tile<BLOCK> {
        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        Tile {
            expr: TileExpr::Dot4 {
                a: [
                    Box::new(a0.expr),
                    Box::new(a1.expr),
                    Box::new(a2.expr),
                    Box::new(a3.expr),
                ],
                b: [
                    Box::new(b0.expr),
                    Box::new(b1.expr),
                    Box::new(b2.expr),
                    Box::new(b3.expr),
                ],
            },
        }
    }

    pub fn vec4_dot(&self, left: Tile<BLOCK>, right: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Vec4Dot {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
        }
    }

    pub fn vec4_splat(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Vec4Splat {
                value: Box::new(value.expr),
            },
        }
    }

    pub fn quantized_q8_0_dot8(
        &self,
        a: [Tile<BLOCK>; 8],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        let a = a.map(|value| Box::new(value.expr));
        Tile {
            expr: TileExpr::QuantizedQ8_0Dot8 {
                a,
                src: matrix.clone(),
                k_base: k_base.into_index(),
                col: col.into_index(),
                mask: mask.expr,
                fill: F32Bits::new(fill),
            },
        }
    }

    quantized_vec_dot_entrypoint!(
        quantized_q8_activation_dot,
        Q8Activation,
        [8, 16],
        "q8 activation dot currently supports N == 8 or N == 16"
    );

    quantized_vec_dot_entrypoint!(
        quantized_q4k_f32_dot,
        Q4KF32,
        [8, 16, 32],
        "q4k f32 dot currently supports N == 8, N == 16, or N == 32"
    );

    #[allow(clippy::too_many_arguments)]
    pub fn quantized_q4k_ggml_dot(
        &self,
        a_low: [Tile<BLOCK>; 16],
        a_high: [Tile<BLOCK>; 16],
        sums: [Tile<BLOCK>; 4],
        matrix: &QuantizedMatrix,
        block: impl IntoIndex<BLOCK>,
        iq: impl IntoIndex<BLOCK>,
        ir: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::QuantizedQ4KGgmlDot {
                a_low: a_low
                    .into_iter()
                    .map(|value| Box::new(value.expr))
                    .collect(),
                a_high: a_high
                    .into_iter()
                    .map(|value| Box::new(value.expr))
                    .collect(),
                sums: sums.into_iter().map(|value| Box::new(value.expr)).collect(),
                src: matrix.clone(),
                block: block.into_index(),
                iq: iq.into_index(),
                ir: ir.into_index(),
                col: col.into_index(),
                mask: mask.expr,
                fill: F32Bits::new(fill),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantized_q6k_ggml_dot(
        &self,
        a: [Tile<BLOCK>; 16],
        matrix: &QuantizedMatrix,
        block: impl IntoIndex<BLOCK>,
        ip: impl IntoIndex<BLOCK>,
        il: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::QuantizedQ6KGgmlDot {
                a: a.into_iter().map(|value| Box::new(value.expr)).collect(),
                src: matrix.clone(),
                block: block.into_index(),
                ip: ip.into_index(),
                il: il.into_index(),
                col: col.into_index(),
                mask: mask.expr,
                fill: F32Bits::new(fill),
            },
        }
    }

    pub fn full(&self, value: f32) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Full(F32Bits::new(value)),
        }
    }

    pub fn literal(&self, value: TileLiteral) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Literal(value),
        }
    }

    pub fn f32(&self, value: f32) -> Tile<BLOCK> {
        self.literal(TileLiteral::F32(F32Bits::new(value)))
    }

    pub fn u32(&self, value: u32) -> Tile<BLOCK> {
        self.literal(TileLiteral::U32(value))
    }

    pub fn bool(&self, value: bool) -> Tile<BLOCK> {
        self.literal(TileLiteral::Bool(value))
    }

    pub fn index(&self, value: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Index(value.into_index()),
        }
    }

    pub fn exp(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::Unary {
                op: TileUnaryOp::Exp,
                value: Box::new(value.expr),
            },
        }
    }

    tile_reduce_entrypoints!(
        (
            reduce_sum,
            loop_reduce_sum,
            group_reduce_sum,
            subgroup_reduce_sum,
            Sum
        ),
        (
            reduce_max,
            loop_reduce_max,
            group_reduce_max,
            subgroup_reduce_max,
            Max
        ),
        (
            reduce_min,
            loop_reduce_min,
            group_reduce_min,
            subgroup_reduce_min,
            Min
        ),
    );

    pub fn loop_fold(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        value: Tile<BLOCK>,
        initial: TileLiteral,
    ) -> Tile<BLOCK> {
        assert!(iterations > 0, "loop fold iterations must be non-zero");
        Tile {
            expr: TileExpr::LoopFold {
                op,
                iterations,
                value: Box::new(value.expr),
                initial,
            },
        }
    }

    fn subgroup_reduce(&self, op: TileReduceOp, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::SubgroupReduce {
                op,
                value: Box::new(value.expr),
            },
        }
    }

    fn group_reduce<const GROUP: usize>(
        &mut self,
        op: TileReduceOp,
        value: Tile<BLOCK>,
    ) -> Tile<BLOCK> {
        assert!(
            GROUP > 0 && GROUP <= BLOCK && GROUP.is_power_of_two() && BLOCK % GROUP == 0,
            "tile group reduction size must be a power-of-two divisor of the block"
        );
        let scratch = self.program.alloc_tile::<F32>(
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([BLOCK as u32])),
            TileLevel::Workgroup,
        );
        Tile {
            expr: TileExpr::GroupReduce {
                op,
                value: Box::new(value.expr),
                scratch,
                group_size: GROUP as u32,
            },
        }
    }

    fn reduce(&mut self, op: TileReduceOp, value: Tile<BLOCK>) -> Scalar {
        let scratch = self.program.alloc_tile::<F32>(
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([BLOCK as u32])),
            TileLevel::Workgroup,
        );
        Scalar {
            expr: TileScalarExpr::Reduce {
                op,
                value: Box::new(value.expr),
                scratch,
            },
        }
    }

    fn loop_reduce(&mut self, op: TileReduceOp, iterations: u32, value: Tile<BLOCK>) -> Scalar {
        assert!(iterations > 0, "loop reduce iterations must be non-zero");
        let scratch = self.program.alloc_tile::<F32>(
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([BLOCK as u32])),
            TileLevel::Workgroup,
        );
        Scalar {
            expr: TileScalarExpr::LoopReduce {
                op,
                iterations,
                value: Box::new(value.expr),
                scratch,
            },
        }
    }

    /// Allocate an 8x8 f32 cooperative-matrix accumulator local. Returned
    /// handle is consumed by `zero_coop_acc`, `mma_from_tiles`, and
    /// `coop_store`.
    pub fn alloc_coop_acc(&mut self) -> CoopAcc {
        let id = CoopAccId(self.program.ir.coop_accs.len() as u32);
        self.program.ir.coop_accs.push(CoopAccDecl {
            id,
            rows: 8,
            cols: 8,
        });
        CoopAcc { id }
    }

    pub fn zero_coop_acc(&mut self, acc: &CoopAcc) {
        self.push_stmt(TileStmt::ZeroCoopAcc { id: acc.id });
    }

    /// Stage a workgroup-tile region of dense `src` into the workgroup-tile
    /// `dst`. Used for the A operand in qmatmul. The lowerer emits a flat
    /// per-invocation loop.
    pub fn copy_storage_to_tile(
        &mut self,
        dst_tile: TileRef,
        src: &Storage<F32, 2>,
        row_offset: impl IntoIndex<BLOCK>,
        col_offset: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::CopyToWorkgroupTile {
            dst: dst_tile,
            src: src.view.clone(),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    /// Stage a workgroup-tile region of quantized `src` into the f32
    /// workgroup-tile `dst`, dequantizing on the fly. Used for the B operand
    /// in qmatmul.
    pub fn copy_quant_to_tile(
        &mut self,
        dst_tile: TileRef,
        src: &QuantizedMatrix,
        row_offset: impl IntoIndex<BLOCK>,
        col_offset: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::CopyQuantToWorkgroupTile {
            dst: dst_tile,
            src: src.clone(),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    pub fn workgroup_barrier(&mut self) {
        self.push_stmt(TileStmt::Barrier);
    }

    pub fn private<T: Numeric>(&mut self) -> Local<T, BLOCK> {
        Local {
            local: self.program.alloc_local::<T>(),
            _ty: PhantomData,
        }
    }

    pub fn load_local<T>(&self, local: &Local<T, BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::LoadLocal(local.local),
        }
    }

    pub fn store_local<T>(&mut self, local: &Local<T, BLOCK>, value: Tile<BLOCK>) {
        self.push_stmt(TileStmt::StoreLocal {
            dst: local.local,
            value: value.expr,
        });
    }

    pub fn emit(&mut self, value: Tile<BLOCK>) {
        self.push_stmt(TileStmt::Emit { value: value.expr });
    }

    pub fn load_workgroup(&self, tile: TileRef, index: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::LoadWorkgroup {
                src: tile,
                index: index.into_index(),
            },
        }
    }

    pub fn store_workgroup(
        &mut self,
        tile: TileRef,
        index: impl IntoIndex<BLOCK>,
        value: Tile<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreWorkgroup {
            dst: tile,
            index: index.into_index(),
            value: value.expr,
        });
    }

    pub fn if_then(&mut self, condition: Tile<BLOCK>, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }

    pub fn if_else(
        &mut self,
        condition: Tile<BLOCK>,
        accept: impl FnOnce(&mut Self),
        reject: impl FnOnce(&mut Self),
    ) {
        self.stmt_stack.push(Vec::new());
        accept(self);
        let accept = self.stmt_stack.pop().expect("if accept frame missing");
        self.stmt_stack.push(Vec::new());
        reject(self);
        let reject = self.stmt_stack.pop().expect("if reject frame missing");
        self.push_stmt(TileStmt::If {
            condition: condition.expr,
            accept,
            reject,
        });
    }

    pub fn loop_forever(&mut self, body: impl FnOnce(&mut Self)) {
        self.stmt_stack.push(Vec::new());
        body(self);
        let body = self.stmt_stack.pop().expect("loop frame missing");
        self.push_stmt(TileStmt::Loop { body });
    }

    pub fn break_loop(&mut self) {
        self.push_stmt(TileStmt::Break);
    }

    pub fn return_(&mut self) {
        self.push_stmt(TileStmt::Return);
    }

    /// `acc += coop_load_a(a_tile, ar, ak) * coop_load_b(b_tile, bk, bc)`.
    /// Convenience wrapper that emits `coop_load_a`, `coop_load_b`, then
    /// `coop_mma`. For MMAs that share an A or B operand across the inner row ×
    /// col grid, prefer the explicit calls so fragment handles can be reused.
    pub fn mma_from_tiles(
        &mut self,
        acc: &CoopAcc,
        a_tile: TileRef,
        a_row: impl IntoIndex<BLOCK>,
        a_col: impl IntoIndex<BLOCK>,
        b_tile: TileRef,
        b_row: impl IntoIndex<BLOCK>,
        b_col: impl IntoIndex<BLOCK>,
    ) {
        let a = self.coop_load_a(a_tile, a_row, a_col);
        let b = self.coop_load_b(b_tile, b_row, b_col);
        self.coop_mma(acc, &a, &b);
    }

    /// Cooperatively load an 8x8 A fragment from a workgroup tile. The
    /// returned handle's SSA value is bound at the load site and reused
    /// wherever the handle is consumed by `coop_mma` in the same scope.
    pub fn coop_load_a(
        &mut self,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        self.coop_load(CoopOperandRole::A, tile, row, col)
    }

    /// Cooperatively load an 8x8 B fragment.
    pub fn coop_load_b(
        &mut self,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        self.coop_load(CoopOperandRole::B, tile, row, col)
    }

    fn coop_load(
        &mut self,
        role: CoopOperandRole,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        let id = self.program.next_coop_fragment_id();
        self.push_stmt(TileStmt::LoadCoop {
            id,
            role,
            tile,
            row: row.into_index(),
            col: col.into_index(),
        });
        CoopFragment { id, role }
    }

    /// `acc += a * b` where `a`/`b` are fragments previously loaded via
    /// `coop_load_a`/`coop_load_b`.
    pub fn coop_mma(&mut self, acc: &CoopAcc, a: &CoopFragment, b: &CoopFragment) {
        assert_eq!(
            a.role,
            CoopOperandRole::A,
            "coop_mma A operand must be an A-role fragment"
        );
        assert_eq!(
            b.role,
            CoopOperandRole::B,
            "coop_mma B operand must be a B-role fragment"
        );
        self.push_stmt(TileStmt::Mma {
            acc: acc.id,
            a: a.id,
            b: b.id,
        });
    }

    /// Cooperatively store `acc` to `dst` at (row, col).
    pub fn coop_store(
        &mut self,
        acc: &CoopAcc,
        dst: &Storage<F32, 2>,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreCoopAcc {
            acc: acc.id,
            dst: dst.view.clone(),
            row: row.into_index(),
            col: col.into_index(),
        });
    }

    /// Emit a counted `while true` loop where `program.loop_index()` resolves
    /// to the current iteration. This is intentionally generic while the IR's
    /// loop-carried value model settles.
    pub fn while_true<F: FnOnce(&mut Self)>(&mut self, max_iterations: u32, body: F) {
        assert!(
            max_iterations > 0,
            "while_true max_iterations must be non-zero"
        );
        self.stmt_stack.push(Vec::new());
        body(self);
        let stmts = self.stmt_stack.pop().expect("while_true frame missing");
        self.push_stmt(TileStmt::WhileTrue {
            max_iterations,
            body: stmts,
        });
    }

    fn push_stmt(&mut self, stmt: TileStmt) {
        if let Some(frame) = self.stmt_stack.last_mut() {
            frame.push(stmt);
        } else {
            self.body.push(stmt);
        }
    }

    pub fn store<T>(&mut self, address: Address<T, BLOCK>, value: Tile<BLOCK>, mask: Mask<BLOCK>) {
        self.push_stmt(TileStmt::Store(TileStoreStmt {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        }));
    }

    pub fn store_linear<T: Numeric>(
        &mut self,
        address: LinearAddress<T, BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreIndexed(TileIndexedStoreStmt {
            dst: address.view,
            index: address.index,
            value: value.expr,
            mask: mask.expr,
        }));
    }

    pub fn store_vec4(
        &mut self,
        address: LinearAddress<F32Vec4, BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreIndexed(TileIndexedStoreStmt {
            dst: address.view,
            index: address.index,
            value: value.expr,
            mask: mask.expr,
        }));
    }

    pub fn store_erased(
        &mut self,
        address: ErasedAddress<BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.push_stmt(TileStmt::Store(TileStoreStmt {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        }));
    }
}
