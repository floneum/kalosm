use super::block::tiles_to_exprs;
use super::value::boxed_u32_literal;
use super::*;
use crate::ir::{
    ElementType, Expr, Layout, LocalRef, MemoryLevel, Numeric, ScalarElement, Shape, TileBinaryOp,
    TileLiteral, TileReduceOp, TileStmt, F32, U32,
};

macro_rules! tile_reduce_entrypoints {
    ($(($reduce:ident, $loop_reduce:ident, $group_reduce:ident, $subgroup_reduce:ident, $op:ident)),+ $(,)?) => {
        $(
            pub fn $reduce<T: Numeric>(&mut self, value: Tile<T>) -> Tile<T> {
                self.reduce(TileReduceOp::$op, value)
            }
            pub fn $loop_reduce<T: Numeric, F>(&mut self, iterations: u32, body: F) -> Tile<T>
            where
                F: FnOnce(&mut Self, Tile<U32>) -> Tile<T>,
            {
                self.loop_reduce(TileReduceOp::$op, iterations, body)
            }
            pub fn $group_reduce<const GROUP: usize, T: Numeric>(&mut self, value: Tile<T>) -> Tile<T> {
                self.group_reduce::<GROUP, T>(TileReduceOp::$op, value)
            }
            pub fn $subgroup_reduce<T: Numeric>(&self, value: Tile<T>) -> Tile<T> {
                self.subgroup_reduce(TileReduceOp::$op, value)
            }
        )+
    };
}

impl TileBlock<'_> {
    pub fn sum<T: Numeric>(&self, values: impl IntoIterator<Item = Tile<T>>) -> Tile<T> {
        let mut exprs = tiles_to_exprs(values);
        if exprs.is_empty() {
            return Tile::from_expr(zero_expr(T::ELEMENT));
        }
        while exprs.len() > 1 {
            let mut next = Vec::with_capacity(exprs.len().div_ceil(2));
            let mut iter = exprs.into_iter();
            while let Some(left) = iter.next() {
                match iter.next() {
                    Some(right) => next.push(Expr::Binary {
                        op: TileBinaryOp::Add,
                        left: Box::new(left),
                        right: Box::new(right),
                    }),
                    None => next.push(left),
                }
            }
            exprs = next;
        }
        Tile::from_expr(exprs.pop().expect("at least one element"))
    }

    pub fn fold<const N: usize, F>(
        &mut self,
        iter: super::FoldIter,
        initial: [Tile<F32>; N],
        body: F,
    ) -> [Tile<F32>; N]
    where
        F: FnOnce(&mut Self, Tile<U32>, [Tile<F32>; N]) -> [Tile<F32>; N],
    {
        assert!(N > 0);
        let initial_exprs = tiles_to_exprs(initial);
        let iter_var_local = self.program.alloc_local::<U32>();
        let acc_locals: [LocalRef; N] = std::array::from_fn(|_| self.program.alloc_local::<F32>());
        self.stmt_stack.push(Vec::new());
        let iter_element = Tile::from_expr(Expr::LoadLocal(iter_var_local));
        let acc_tiles: [Tile<F32>; N] =
            std::array::from_fn(|i| Tile::from_expr(Expr::LoadLocal(acc_locals[i])));
        let new_state = body(self, iter_element, acc_tiles);
        let body_stmts = self.stmt_stack.pop().expect("fold body frame missing");
        let accumulators = new_state
            .into_iter()
            .enumerate()
            .map(|(i, new)| crate::ir::FoldAccumulator {
                name: acc_locals[i].id,
                element: acc_locals[i].element,
                init: initial_exprs[i].clone(),
                update: new.expr,
            })
            .collect();
        self.push_stmt(TileStmt::Fold {
            count: iter.count,
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators,
        });
        std::array::from_fn(|i| Tile::from_expr(Expr::LoadLocal(acc_locals[i])))
    }

    pub fn loop_fold_n<const N: usize, T: Numeric, F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initials: [TileLiteral; N],
        body: F,
    ) -> [Tile<T>; N]
    where
        F: FnOnce(&mut Self, Tile<U32>) -> [Tile<T>; N],
    {
        assert!(iterations > 0);
        assert!(N > 0);
        for initial in initials {
            assert_eq!(initial.element(), T::ELEMENT);
        }
        let acc_locals: [LocalRef; N] = std::array::from_fn(|_| self.program.alloc_local::<T>());
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        let bodies = body(self, Tile::from_expr(Expr::LoadLocal(iter_var_local)));
        let body_stmts = self
            .stmt_stack
            .pop()
            .expect("loop_fold_n body frame missing");
        let binary_op = op.binary();
        let accumulators = bodies
            .into_iter()
            .enumerate()
            .map(|(i, lane_value)| crate::ir::FoldAccumulator {
                name: acc_locals[i].id,
                element: acc_locals[i].element,
                init: Expr::Literal(initials[i]),
                update: Expr::Binary {
                    op: binary_op,
                    left: Box::new(Expr::LoadLocal(acc_locals[i])),
                    right: Box::new(lane_value.expr),
                },
            })
            .collect();
        self.push_stmt(TileStmt::Fold {
            count: boxed_u32_literal(iterations),
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators,
        });
        std::array::from_fn(|i| Tile::from_expr(Expr::LoadLocal(acc_locals[i])))
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

    pub fn loop_fold<T: Numeric, F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initial: TileLiteral,
        body: F,
    ) -> Tile<T>
    where
        F: FnOnce(&mut Self, Tile<U32>) -> Tile<T>,
    {
        assert!(iterations > 0);
        assert_eq!(initial.element(), T::ELEMENT);
        let acc_local = self.program.alloc_local::<T>();
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        let value = body(self, Tile::from_expr(Expr::LoadLocal(iter_var_local)));
        let body_stmts = self.stmt_stack.pop().expect("loop_fold body frame missing");
        self.push_stmt(TileStmt::Fold {
            count: boxed_u32_literal(iterations),
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators: vec![crate::ir::FoldAccumulator {
                name: acc_local.id,
                element: acc_local.element,
                init: Expr::Literal(initial),
                update: Expr::Binary {
                    op: op.binary(),
                    left: Box::new(Expr::LoadLocal(acc_local)),
                    right: Box::new(value.expr),
                },
            }],
        });
        Tile::from_expr(Expr::LoadLocal(acc_local))
    }

    fn subgroup_reduce<T: Numeric>(&self, op: TileReduceOp, value: Tile<T>) -> Tile<T> {
        Tile::from_expr(Expr::SubgroupReduce {
            op,
            value: Box::new(value.expr),
        })
    }

    fn group_reduce<const GROUP: usize, T: Numeric>(
        &mut self,
        op: TileReduceOp,
        value: Tile<T>,
    ) -> Tile<T> {
        let block = self.block_size();
        assert!(GROUP > 0 && GROUP <= block && GROUP.is_power_of_two());
        let scratch = self.program.alloc_tile::<T>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        Tile::from_expr(Expr::Reduce {
            op,
            iterations: 1,
            iter_var: None,
            value: Box::new(value.expr),
            scratch: scratch.into(),
            group_size: GROUP as u32,
        })
    }

    fn reduce<T: Numeric>(&mut self, op: TileReduceOp, value: Tile<T>) -> Tile<T> {
        let block = self.block_size();
        let scratch = self.program.alloc_tile::<T>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        Tile::from_expr(Expr::Reduce {
            op,
            iterations: 1,
            iter_var: None,
            value: Box::new(value.expr),
            scratch: scratch.into(),
            group_size: block as u32,
        })
    }

    fn loop_reduce<T: Numeric, F>(&mut self, op: TileReduceOp, iterations: u32, body: F) -> Tile<T>
    where
        F: FnOnce(&mut Self, Tile<U32>) -> Tile<T>,
    {
        assert!(iterations > 0);
        let block = self.block_size();
        let scratch = self.program.alloc_tile::<T>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        let value = body(self, Tile::from_expr(Expr::LoadLocal(iter_var_local)));
        let leaked = self
            .stmt_stack
            .pop()
            .expect("loop_reduce body frame missing");
        assert!(leaked.is_empty());
        Tile::from_expr(Expr::Reduce {
            op,
            iterations,
            iter_var: Some(iter_var_local.id),
            value: Box::new(value.expr),
            scratch: scratch.into(),
            group_size: block as u32,
        })
    }
}

fn zero_expr(element: ElementType) -> Expr {
    match element {
        ElementType::F32 => Expr::Literal(TileLiteral::f32(0.0)),
        ElementType::F16 => Expr::Literal(TileLiteral::F16(0)),
        ElementType::U32 => Expr::Literal(TileLiteral::U32(0)),
        ElementType::Bool => Expr::Literal(TileLiteral::Bool(false)),
        ElementType::Vector { scalar, lanes } => {
            let literal = match scalar {
                ScalarElement::F32 => TileLiteral::f32(0.0),
                ScalarElement::F16 => TileLiteral::F16(0),
                ScalarElement::U32 => TileLiteral::U32(0),
                ScalarElement::Bool => TileLiteral::Bool(false),
            };
            Expr::ComposeVector {
                scalar,
                lanes,
                values: (0..lanes).map(|_| Expr::Literal(literal)).collect(),
            }
        }
        ElementType::CoopMatrix { .. } => panic!("unsupported cooperative matrix sum"),
    }
}
