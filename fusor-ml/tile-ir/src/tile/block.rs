use super::value::{boxed_index, zero_fill, Address, CoopAcc, PrivateLocal, WorkgroupTile};
use super::{Mask, Program, Storage, Tile};
use crate::ir::{
    Accumulator, Builtin, ElementType, Expr, ExprKind, Local, ScalarElement, Source, Stmt,
    TileLiteral, WorkgroupAxis,
};
use crate::quantized::QuantizedMatrix;

/// One logical lane of a tile program body. Created by
/// [`Program::program_grid`]; every expression describes the computation for a
/// single workgroup invocation.
pub struct TileBlock<'a> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) block: u32,
    pub(super) body: Vec<Stmt>,
    pub(super) stmt_stack: Vec<Vec<Stmt>>,
}

impl TileBlock<'_> {
    // ---- builtins --------------------------------------------------------

    /// `@builtin(workgroup_id).{x|y|z}`.
    pub fn program_id(&self, axis: WorkgroupAxis) -> Tile {
        Tile::builtin(Builtin::ProgramId(axis))
    }
    /// `@builtin(subgroup_id)`.
    pub(crate) fn subgroup_id(&self) -> Tile {
        Tile::builtin(Builtin::SubgroupId)
    }
    /// `@builtin(subgroup_invocation_id)`.
    pub(crate) fn subgroup_lane(&self) -> Tile {
        Tile::builtin(Builtin::SubgroupLane)
    }
    /// `@builtin(subgroup_size)`.
    pub(crate) fn subgroup_size(&self) -> Tile {
        Tile::builtin(Builtin::SubgroupSize)
    }
    /// `@builtin(num_subgroups)`.
    pub(crate) fn num_subgroups(&self) -> Tile {
        Tile::builtin(Builtin::NumSubgroups)
    }
    /// `@builtin(local_invocation_index)` — flat lane within the workgroup.
    pub fn lane(&self) -> Tile {
        Tile::builtin(Builtin::Lane)
    }

    /// Dispatch grid passed to [`Program::program_grid`].
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }
    /// Workgroup invocation count (`block`).
    pub fn block_size(&self) -> u32 {
        self.block
    }

    // ---- literals --------------------------------------------------------

    /// A typed scalar literal.
    pub fn literal(&self, value: impl Into<TileLiteral>) -> Tile {
        Tile::literal(value)
    }
    /// An f32 literal.
    pub fn f32(&self, value: f32) -> Tile {
        Tile::f32(value)
    }
    /// A u32 literal.
    pub fn u32(&self, value: u32) -> Tile {
        Tile::u32(value)
    }
    /// A bool literal.
    pub fn bool(&self, value: bool) -> Tile {
        Tile::bool(value)
    }
    /// Coerce any index-like value into a `u32`-typed tile.
    pub fn index(&self, value: impl Into<Tile>) -> Tile {
        value.into()
    }

    // ---- loads / stores --------------------------------------------------

    /// Masked storage load. `fill` is the masked-out value.
    pub fn load(
        &self,
        address: Address,
        mask: impl Into<Mask>,
        fill: impl Into<TileLiteral>,
    ) -> Tile {
        let fill = fill.into();
        let fill_expr = Expr::new(ExprKind::Literal(fill), fill.element());
        Tile::from_expr(address.load_expr(mask.into().into_expr(), fill_expr))
    }

    /// Masked dequantizing load of one f32 value from a quantized matrix at a
    /// `(row, col)` coordinate.
    pub fn load_quantized(
        &self,
        matrix: &QuantizedMatrix,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Tile {
        Tile::new(
            ExprKind::Load {
                src: Source::Quantized(matrix.clone()),
                addr: crate::ir::Addr::Rc2 {
                    row: boxed_index(row),
                    col: boxed_index(col),
                },
                mask: Box::new(mask.into().into_expr()),
                fill: Box::new(Expr::new(
                    ExprKind::Literal(TileLiteral::f32(fill)),
                    ElementType::F32,
                )),
            },
            ElementType::F32,
        )
    }

    /// Per-lane masked storage store (dense rank-2 or indexed rank-1).
    pub fn store(&mut self, address: Address, value: Tile, mask: impl Into<Mask>) {
        self.push_stmt(address.store_stmt(value.into_expr(), mask.into().into_expr()));
    }

    /// Per-lane masked store, casting `value` to the destination element type
    /// first.
    pub fn store_cast(&mut self, address: Address, value: Tile, mask: impl Into<Mask>) {
        let target = address.view.buffer.element;
        let value = if value.element() == target {
            value
        } else {
            value.cast(target)
        };
        self.push_stmt(address.store_stmt(value.into_expr(), mask.into().into_expr()));
    }

    // ---- bind / private locals ------------------------------------------

    /// Bind `value` to a fresh private local and return a load of it. SSA
    /// first-write helper.
    pub fn bind(&mut self, value: impl Into<Tile>) -> Tile {
        let value = value.into();
        let local = self.program.alloc_local(value.element());
        self.push_stmt(Stmt::StoreLocal {
            dst: local.decl().clone(),
            value: value.into_expr(),
        });
        load_local_expr(local.decl())
    }

    /// Allocate a fresh, mutable private local of the given element type.
    pub fn private(&mut self, element: ElementType) -> PrivateLocal {
        self.program.alloc_local(element)
    }

    /// Load the current value of a private local.
    pub fn load_local(&self, local: &PrivateLocal) -> Tile {
        load_local_expr(local.decl())
    }

    /// Store `value` into a private local.
    pub fn store_local(&mut self, local: &PrivateLocal, value: impl Into<Tile>) {
        let value = value.into();
        self.push_stmt(Stmt::StoreLocal {
            dst: local.decl().clone(),
            value: value.into_expr(),
        });
    }

    // ---- workgroup tiles -------------------------------------------------

    /// Load from a workgroup tile at a dynamic flat index.
    pub fn load_workgroup(&self, tile: &WorkgroupTile, index: impl Into<Tile>) -> Tile {
        Tile::new(
            ExprKind::LoadTile {
                tile: tile.decl().clone(),
                index: boxed_index(index),
            },
            tile.element(),
        )
    }

    /// Store to a workgroup tile at a dynamic flat index.
    pub fn store_workgroup(
        &mut self,
        tile: &WorkgroupTile,
        index: impl Into<Tile>,
        value: impl Into<Tile>,
    ) {
        self.push_stmt(Stmt::StoreTile {
            dst: tile.decl().clone(),
            index: boxed_index(index),
            value: value.into().into_expr(),
        });
    }

    /// Fill `dst` from a per-element masked storage load. The lowerer derives
    /// coords and applies the vec4 fast path internally. `row`/`col` are the
    /// tile origin in `src`.
    pub fn fill_tile(
        &mut self,
        dst: &WorkgroupTile,
        src: &Storage,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
    ) {
        let value = Expr::new(
            ExprKind::Load {
                src: Source::Storage(src.view().clone()),
                addr: crate::ir::Addr::Rc2 {
                    row: boxed_index(row),
                    col: boxed_index(col),
                },
                mask: Box::new(Tile::all().into_expr()),
                fill: Box::new(zero_fill(scalar_of(src.element()))),
            },
            scalar_of(src.element()).element(),
        );
        self.push_stmt(Stmt::FillTile {
            dst: dst.decl().clone(),
            value,
        });
    }

    /// Fill `dst` by dequantizing a quantized matrix region. `row`/`col` are
    /// the tile origin into the dense matrix.
    pub fn fill_tile_quantized(
        &mut self,
        dst: &WorkgroupTile,
        src: &QuantizedMatrix,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
    ) {
        let value = Expr::new(
            ExprKind::Load {
                src: Source::Quantized(src.clone()),
                addr: crate::ir::Addr::Rc2 {
                    row: boxed_index(row),
                    col: boxed_index(col),
                },
                mask: Box::new(Tile::all().into_expr()),
                fill: Box::new(Expr::new(
                    ExprKind::Literal(TileLiteral::f32(0.0)),
                    ElementType::F32,
                )),
            },
            ElementType::F32,
        );
        self.push_stmt(Stmt::FillTile {
            dst: dst.decl().clone(),
            value,
        });
    }

    // ---- vector composition / dot ---------------------------------------

    /// Compose `LANES` scalar tiles into one vector tile.
    pub fn compose_vector<const LANES: usize>(
        &self,
        scalar: ScalarElement,
        values: [Tile; LANES],
    ) -> Tile {
        validate_vector_lanes(LANES, "compose_vector");
        let lanes = LANES as u32;
        Tile::new(
            ExprKind::Vec {
                scalar,
                lanes,
                parts: values.into_iter().map(Tile::into_expr).collect(),
            },
            ElementType::vector(scalar, lanes),
        )
    }

    /// Compose a `LANES`-wide vector by broadcasting one scalar `value` into
    /// every lane.
    pub fn vector_splat<const LANES: usize>(&self, scalar: ScalarElement, value: Tile) -> Tile {
        validate_vector_lanes(LANES, "vector_splat");
        let lanes = LANES as u32;
        let parts = (0..LANES).map(|_| value.clone().into_expr()).collect();
        Tile::new(
            ExprKind::Vec {
                scalar,
                lanes,
                parts,
            },
            ElementType::vector(scalar, lanes),
        )
    }

    /// Dot product between two vector tiles. Produces the scalar element type.
    pub fn vector_dot(&self, left: Tile, right: Tile) -> Tile {
        let scalar = match left.element() {
            ElementType::Vector { scalar, .. } => scalar,
            _ => panic!("vector_dot requires vector operands"),
        };
        Tile::new(
            ExprKind::Dot {
                left: Box::new(left.into_expr()),
                right: Box::new(right.into_expr()),
            },
            scalar.element(),
        )
    }

    // ---- control flow ----------------------------------------------------

    /// Workgroup-scope memory barrier.
    pub fn workgroup_barrier(&mut self) {
        self.push_stmt(Stmt::Barrier);
    }

    /// Conditional block (no else).
    pub fn if_then(&mut self, condition: impl Into<Mask>, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }

    /// Conditional block with an else.
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
        self.push_stmt(Stmt::If {
            condition: condition.into().into_expr(),
            accept,
            reject,
        });
    }

    /// Unstructured loop with a data-dependent exit. Use `break_loop` /
    /// `break_if` inside to exit. Retained verbatim (ARBOR_DESIGN.md §5).
    pub fn loop_forever(&mut self, body: impl FnOnce(&mut Self)) {
        self.stmt_stack.push(Vec::new());
        body(self);
        let body = self.stmt_stack.pop().expect("loop frame missing");
        self.push_stmt(Stmt::Loop {
            count: None,
            index: None,
            accumulators: Vec::new(),
            body,
        });
    }

    /// Break out of the innermost loop.
    pub fn break_loop(&mut self) {
        self.push_stmt(Stmt::Break);
    }

    /// Break out of the innermost loop when `condition` is true. Sugar for
    /// `if_then(cond, |b| b.break_loop())`.
    pub fn break_if(&mut self, condition: impl Into<Mask>) {
        self.if_then(condition, |program| program.break_loop());
    }

    /// Return from the kernel entry point.
    pub fn return_(&mut self) {
        self.push_stmt(Stmt::Return);
    }

    /// Counted loop over `0..count` with no carried accumulators. The body
    /// receives the loop index. Replaces the old `while_true`.
    pub fn loop_range(&mut self, count: u32, body: impl FnOnce(&mut Self, Tile)) {
        assert!(count > 0, "loop_range count must be non-zero");
        let index = self.program.alloc_local(ElementType::U32);
        self.stmt_stack.push(Vec::new());
        body(self, load_local_expr(index.decl()));
        let body = self.stmt_stack.pop().expect("loop_range frame missing");
        self.push_stmt(Stmt::Loop {
            count: Some(Expr::new(
                ExprKind::Literal(TileLiteral::U32(count)),
                ElementType::U32,
            )),
            index: Some(index.decl().clone()),
            accumulators: Vec::new(),
            body,
        });
    }

    // ---- shared helpers --------------------------------------------------

    pub(super) fn push_stmt(&mut self, stmt: Stmt) {
        if let Some(frame) = self.stmt_stack.last_mut() {
            frame.push(stmt);
        } else {
            self.body.push(stmt);
        }
    }

    /// Build a counted `Stmt::Loop` from a body-built statement vec and the
    /// carried accumulators. Shared by `fold`/the reduce helpers.
    pub(super) fn push_counted_loop(
        &mut self,
        count: Expr,
        index: Option<Local>,
        accumulators: Vec<Accumulator>,
        body: Vec<Stmt>,
    ) {
        self.push_stmt(Stmt::Loop {
            count: Some(count),
            index,
            accumulators,
            body,
        });
    }

    /// Open a fresh statement frame; the matching [`close_frame`] pops it.
    pub(super) fn open_frame(&mut self) {
        self.stmt_stack.push(Vec::new());
    }

    pub(super) fn close_frame(&mut self) -> Vec<Stmt> {
        self.stmt_stack.pop().expect("statement frame missing")
    }
}

/// `LoadLocal(local)` as a runtime tile (its element type is the local's).
pub(super) fn load_local_expr(local: &Local) -> Tile {
    Tile::new(ExprKind::LoadLocal(local.clone()), local.element)
}

/// Coop accumulator load — used by the coop value-threading helpers.
pub(super) fn load_coop_acc(acc: &CoopAcc) -> Tile {
    load_local_expr(acc.decl())
}

fn validate_vector_lanes(lanes: usize, op: &str) {
    assert!((2..=4).contains(&lanes), "{op} supports 2, 3, or 4 lanes");
}

/// The scalar component of a (possibly vector) element type.
pub(super) fn scalar_of(element: ElementType) -> ScalarElement {
    match element {
        ElementType::F32 => ScalarElement::F32,
        ElementType::F16 => ScalarElement::F16,
        ElementType::U32 => ScalarElement::U32,
        ElementType::Bool => ScalarElement::Bool,
        ElementType::Vector { scalar, .. } => scalar,
        ElementType::CoopMatrix { scalar, .. } => scalar,
    }
}
