use std::marker::PhantomData;

use super::value::{boxed_index, boxed_u32_literal};
use super::*;
use crate::ir::{
    Bool, Builtin, Expr, F16, F32, FloatElement, Numeric, ScalarMarker, TileLiteral, TileLoadExpr,
    TileStmt, TileUnaryOp, U32, Vector, WorkgroupAxis,
};
use crate::quantized::QuantizedMatrix;

pub struct TileBlock<'a> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) block: usize,
    pub(super) body: Vec<TileStmt>,
    pub(super) stmt_stack: Vec<Vec<TileStmt>>,
}

fn builtin_index(builtin: Builtin) -> Tile<U32> {
    Tile::from_expr(Expr::Builtin(builtin))
}

pub(super) fn f32_fill(value: f32) -> Box<Expr> {
    Box::new(Expr::Literal(TileLiteral::f32(value)))
}

pub(super) fn tiles_to_exprs<T: Numeric>(values: impl IntoIterator<Item = Tile<T>>) -> Vec<Expr> {
    values.into_iter().map(|t| t.expr).collect()
}

impl TileBlock<'_> {
    pub fn program_id(&self, axis: WorkgroupAxis) -> Tile<U32> {
        builtin_index(Builtin::ProgramId(axis))
    }
    pub fn subgroup_id(&self) -> Tile<U32> {
        builtin_index(Builtin::SubgroupId)
    }
    pub fn subgroup_lane(&self) -> Tile<U32> {
        builtin_index(Builtin::SubgroupLane)
    }
    pub fn subgroup_size(&self) -> Tile<U32> {
        builtin_index(Builtin::SubgroupSize)
    }
    pub fn num_subgroups(&self) -> Tile<U32> {
        builtin_index(Builtin::NumSubgroups)
    }
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }
    pub fn block_size(&self) -> usize {
        self.block
    }
    pub fn lane(&self) -> Tile<U32> {
        Tile::from_expr(Expr::Builtin(Builtin::Lane))
    }
    pub fn lane_tiles<const DIMS: usize>(&self, dims: &[u32; DIMS]) -> [Tile<U32>; DIMS] {
        let lane_count = dims.iter().try_fold(1usize, |product, &dim| {
            assert!(dim > 0, "lane tile dimensions must be non-zero");
            product.checked_mul(dim as usize)
        });
        assert_eq!(lane_count, Some(self.block));
        let lane = self.lane();
        std::array::from_fn(|axis| {
            let stride = dims[axis + 1..].iter().product::<u32>();
            let coord = if stride == 1 {
                lane.clone()
            } else {
                lane.clone() / stride
            };
            coord % dims[axis]
        })
    }
    pub fn load<T: Numeric, const R: usize>(
        &self,
        address: Address<T, R>,
        mask: impl Into<Mask>,
        fill: impl Into<TileLiteral>,
    ) -> Tile<T> {
        Tile::from_expr(address.load_expr(mask.into().expr, fill.into()))
    }
    pub fn load_quantized(
        &self,
        matrix: &QuantizedMatrix,
        row: impl Into<Tile<U32>>,
        col: impl Into<Tile<U32>>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Tile<F32> {
        Tile::from_expr(Expr::Load(TileLoadExpr {
            src: crate::ir::LoadSource::Quantized(matrix.clone()),
            row: boxed_index(row),
            col: boxed_index(col),
            mask: Box::new(mask.into().expr),
            fill: f32_fill(fill),
        }))
    }
    pub fn load_quantized_block<const N: usize>(
        &mut self,
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile<U32>>,
        col: impl Into<Tile<U32>>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> [Tile<F32>; N] {
        assert!(N == 8 || N == 16);
        let id = self.program.next_block_dequant_id();
        let k_base = boxed_index(k_base);
        let col = boxed_index(col);
        let mask = Box::new(mask.into().expr);
        let fill = f32_fill(fill);
        std::array::from_fn(|lane| {
            Tile::from_expr(Expr::QuantizedBlockLane {
                id,
                src: matrix.clone(),
                k_base: k_base.clone(),
                col: col.clone(),
                mask: mask.clone(),
                fill: fill.clone(),
                block_n: N as u32,
                lane: lane as u32,
            })
        })
    }
    pub fn load_quantized_block_vec(
        &mut self,
        lanes: u32,
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile<U32>>,
        col: impl Into<Tile<U32>>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Vec<Tile<F32>> {
        assert!(lanes == 8 || lanes == 16);
        let id = self.program.next_block_dequant_id();
        let k_base = boxed_index(k_base);
        let col = boxed_index(col);
        let mask = Box::new(mask.into().expr);
        let fill = f32_fill(fill);
        (0..lanes)
            .map(|lane| {
                Tile::from_expr(Expr::QuantizedBlockLane {
                    id,
                    src: matrix.clone(),
                    k_base: k_base.clone(),
                    col: col.clone(),
                    mask: mask.clone(),
                    fill: fill.clone(),
                    block_n: lanes,
                    lane,
                })
            })
            .collect()
    }
    pub fn bind<T: Numeric>(&mut self, value: impl Into<Tile<T>>) -> Tile<T> {
        let value = value.into();
        let local = self.program.alloc_local_element(value.expr.element());
        self.push_stmt(TileStmt::StoreLocal {
            dst: local,
            value: value.expr,
        });
        Tile::from_expr(Expr::LoadLocal(local))
    }
    pub fn vector_dot<T: FloatElement, const LANES: usize>(
        &self,
        left: Tile<Vector<T, LANES>>,
        right: Tile<Vector<T, LANES>>,
    ) -> Tile<T> {
        validate_vector_lanes(LANES, "vector_dot");
        Tile::from_expr(Expr::VectorDot {
            scalar: T::SCALAR,
            lanes: LANES as u32,
            left: Box::new(left.expr),
            right: Box::new(right.expr),
        })
    }
    pub fn vector_splat<T: ScalarMarker + Numeric, const LANES: usize>(
        &self,
        value: Tile<T>,
    ) -> Tile<Vector<T, LANES>> {
        validate_vector_lanes(LANES, "vector_splat");
        Tile::from_expr(Expr::ComposeVector {
            scalar: T::SCALAR,
            lanes: LANES as u32,
            values: (0..LANES).map(|_| value.expr.clone()).collect(),
        })
    }
    pub fn compose_vector<T: ScalarMarker + Numeric, const LANES: usize>(
        &self,
        values: [Tile<T>; LANES],
    ) -> Tile<Vector<T, LANES>> {
        validate_vector_lanes(LANES, "compose_vector");
        Tile::from_expr(Expr::ComposeVector {
            scalar: T::SCALAR,
            lanes: LANES as u32,
            values: values.into_iter().map(|value| value.expr).collect(),
        })
    }
    pub fn literal<T: Numeric>(&self, value: impl Into<TileLiteral>) -> Tile<T> {
        Tile::literal(value)
    }
    pub fn f32(&self, value: f32) -> Tile<F32> {
        Tile::literal(TileLiteral::f32(value))
    }
    pub fn f16_bits(&self, value: u16) -> Tile<F16> {
        Tile::literal(TileLiteral::F16(value))
    }
    pub fn u32(&self, value: u32) -> Tile<U32> {
        Tile::literal(TileLiteral::U32(value))
    }
    pub fn bool(&self, value: bool) -> Tile<Bool> {
        Tile::literal(TileLiteral::Bool(value))
    }
    pub fn index(&self, value: impl Into<Tile<U32>>) -> Tile<U32> {
        value.into()
    }
    pub fn exp(&self, value: Tile<F32>) -> Tile<F32> {
        value.unary(TileUnaryOp::Exp)
    }
    pub fn workgroup_barrier(&mut self) {
        self.push_stmt(TileStmt::Barrier);
    }
    pub fn private<T: Numeric>(&mut self) -> Local<T> {
        Local {
            local: self.program.alloc_local::<T>(),
            _ty: PhantomData,
        }
    }
    pub fn load_local<T: Numeric>(&self, local: &Local<T>) -> Tile<T> {
        Tile::from_expr(Expr::LoadLocal(local.local))
    }
    pub fn store_local<T: Numeric>(&mut self, local: &Local<T>, value: impl Into<Tile<T>>) {
        let value = value.into();
        self.push_stmt(TileStmt::StoreLocal {
            dst: local.local,
            value: value.expr,
        });
    }
    pub fn load_workgroup<T: Numeric>(
        &self,
        tile: Workgroup<T>,
        index: impl Into<Tile<U32>>,
    ) -> Tile<T> {
        Tile::from_expr(Expr::LoadWorkgroup {
            src: tile.tile,
            index: boxed_index(index),
        })
    }
    pub fn store_workgroup<T: Numeric>(
        &mut self,
        tile: Workgroup<T>,
        index: impl Into<Tile<U32>>,
        value: impl Into<Tile<T>>,
    ) {
        let value = value.into();
        self.push_stmt(TileStmt::StoreWorkgroup {
            dst: tile.tile,
            index: boxed_index(index),
            value: value.expr,
        });
    }
    pub fn if_then(&mut self, condition: impl Into<Mask>, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }
    pub fn if_else(
        &mut self,
        condition: impl Into<Mask>,
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
            condition: condition.into().expr,
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
    pub fn break_if(&mut self, condition: impl Into<Mask>) {
        self.if_then(condition, |program| program.break_loop());
    }
    pub fn return_(&mut self) {
        self.push_stmt(TileStmt::Return);
    }
    pub fn while_true<F>(&mut self, max_iterations: u32, body: F)
    where
        F: FnOnce(&mut Self, Tile<U32>),
    {
        assert!(max_iterations > 0);
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        body(self, Tile::from_expr(Expr::LoadLocal(iter_var_local)));
        let stmts = self.stmt_stack.pop().expect("while_true frame missing");
        self.push_stmt(TileStmt::Fold {
            count: boxed_u32_literal(max_iterations),
            iter_var: iter_var_local.id,
            body: stmts,
            accumulators: Vec::new(),
        });
    }
    pub(super) fn push_stmt(&mut self, stmt: TileStmt) {
        if let Some(frame) = self.stmt_stack.last_mut() {
            frame.push(stmt);
        } else {
            self.body.push(stmt);
        }
    }
    pub fn store<T: Numeric, const R: usize>(
        &mut self,
        address: Address<T, R>,
        value: Tile<T>,
        mask: impl Into<Mask>,
    ) {
        self.push_stmt(address.store_stmt(value.expr, mask.into().expr));
    }
}

fn validate_vector_lanes(lanes: usize, op: &str) {
    assert!((2..=4).contains(&lanes), "{op} supports 2, 3, or 4 lanes");
}
