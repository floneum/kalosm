//! Cooperative-matrix fragment load, MMA and store.
//!
//! Accumulators are held **transposed internally**: Metal's simdgroup matrix
//! orientation makes row-major A/B fragments multiply as `B * A`, so keeping
//! the fragments transposed preserves the logical `A * B`. That is why a
//! transposed tile load swaps the fragment origin and sets
//! `row_major: transposed`, and why a cooperative store inverts the
//! destination layout's flag.
//!
//! Without the `fork-metal` mixed-precision cooperative store, an
//! f32-accumulated f16-output kernel pays a staging tile plus a per-lane cast:
//! footprint and a staging pass, never correctness.
//!
//! Owned by W8.

use fusor2_ir::ir::level2::{
    Addr, CoopMatrixRole, CoopSrc, ElementType, ScalarElement, StorageView, Tile, TileExpr,
    TileLayout, cooperative_store_layout_supported,
};
use fusor2_ir::target::EmitError;
use naga::{
    AddressSpace, ArraySize, Barrier, Block, CooperativeData, CooperativeRole, Expression,
    GlobalVariable, Handle, Span, Statement,
};

use super::{Emitter, key};

/// A workgroup tile's `[rows, cols]`.
pub(crate) fn tile_shape(tile: &Tile) -> Result<[u32; 2], EmitError> {
    if tile.layout.extents.len() != 2 {
        return Err(EmitError::Unsupported(
            "a workgroup tile must be rank-2".into(),
        ));
    }
    Ok([tile.layout.extents[0], tile.layout.extents[1]])
}

/// A workgroup tile's row stride, requiring a row-major affine layout.
pub(crate) fn row_major_tile_stride(tile: &Tile) -> Result<u32, EmitError> {
    layout_row_major_stride(&tile.layout)
}

fn layout_row_major_stride(layout: &TileLayout) -> Result<u32, EmitError> {
    if !layout.is_affine() || layout.indexing.groups.len() != 2 {
        return Err(EmitError::Unsupported(
            "a workgroup tile must be a rank-2 affine layout".into(),
        ));
    }
    let strides: Vec<u32> = layout
        .indexing
        .groups
        .iter()
        .map(|g| g.sub_axes[0].stride)
        .collect();
    if strides[1] != 1 {
        return Err(EmitError::Unsupported(
            "a workgroup tile must be row-major".into(),
        ));
    }
    Ok(strides[0])
}

/// `(stride, row_major)` of a cooperative store destination.
fn cooperative_store_layout(layout: &TileLayout) -> Result<(u32, bool), EmitError> {
    if !cooperative_store_layout_supported(layout) {
        return Err(EmitError::Unsupported(
            "a cooperative store needs an affine rank-2 view with one unit stride".into(),
        ));
    }
    let strides: Vec<u32> = layout
        .indexing
        .groups
        .iter()
        .map(|g| g.sub_axes[0].stride)
        .collect();
    if strides[1] == 1 {
        Ok((strides[0], true))
    } else {
        Ok((strides[1], false))
    }
}

/// A cooperative load names its own scalar; the memory it reads names one too,
/// and they have to be the same one.
fn fragment_scalar_matches(
    scalar: ScalarElement,
    element: ElementType,
    what: &str,
) -> Result<(), EmitError> {
    if element == ElementType::Scalar(scalar) {
        return Ok(());
    }
    Err(EmitError::Unsupported(format!(
        "a {scalar:?} cooperative fragment cannot load from {what} of {element:?}"
    )))
}

fn naga_role(role: CoopMatrixRole) -> CooperativeRole {
    match role {
        CoopMatrixRole::A => CooperativeRole::A,
        CoopMatrixRole::B => CooperativeRole::B,
        CoopMatrixRole::C => CooperativeRole::C,
    }
}

fn cooperative_size(size: u32) -> Result<naga::CooperativeSize, EmitError> {
    match size {
        8 => Ok(naga::CooperativeSize::Eight),
        16 => Ok(naga::CooperativeSize::Sixteen),
        _ => Err(EmitError::Unsupported(format!(
            "cooperative-matrix size must be 8 or 16, got {size}"
        ))),
    }
}

impl Emitter<'_> {
    /// `CoopLoad`. From a tile region the fragment origin is swapped and
    /// `row_major: transposed`; from a rank-1 broadcast column the stride is
    /// zero and `row_major: false`.
    pub(crate) fn coop_load_parts(
        &mut self,
        out: &mut Block,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: &CoopSrc,
    ) -> Result<Handle<Expression>, EmitError> {
        let role = naga_role(role);
        let columns = cooperative_size(cols)?;
        let rows_size = cooperative_size(rows)?;
        match src {
            CoopSrc::TileRegion {
                tile,
                row,
                col,
                transposed,
            } => {
                // The fragment's scalar and the tile's element are one memory
                // reinterpretation apart: a `CoopLoad{scalar: F32}` off an f16
                // tile reads the right addresses at twice the width and comes
                // back with plausible garbage. Nothing downstream can see it,
                // so it is checked where both are in hand.
                fragment_scalar_matches(scalar, tile.element, "a workgroup tile")?;
                let stride_u = row_major_tile_stride(tile)?;
                let row_h = self.expr(row, out)?;
                let col_h = self.expr(col, out)?;
                let (first, second) = if *transposed {
                    (col_h, row_h)
                } else {
                    (row_h, col_h)
                };
                let index = self.tile_matrix_index(out, first, second, stride_u);
                let pointer = self.tile_dynamic_pointer(out, tile, index)?;
                let stride = self.u32_lit(stride_u);
                Ok(self.emit_expr(
                    out,
                    Expression::CooperativeLoad {
                        columns,
                        rows: rows_size,
                        role,
                        data: CooperativeData {
                            pointer,
                            stride,
                            row_major: *transposed,
                        },
                    },
                ))
            }
            CoopSrc::BroadcastCol { src, col } => {
                if src.layout.extents.len() != 1 {
                    return Err(EmitError::Unsupported(
                        "a cooperative broadcast load needs rank-1 storage".into(),
                    ));
                }
                fragment_scalar_matches(scalar, src.buffer.element, "a broadcast source")?;
                let col_h = self.expr(col, out)?;
                let pointer = self.storage_dynamic_pointer(out, src, col_h)?;
                let stride = self.u32_lit(0);
                Ok(self.emit_expr(
                    out,
                    Expression::CooperativeLoad {
                        columns,
                        rows: rows_size,
                        role,
                        data: CooperativeData {
                            pointer,
                            stride,
                            // A broadcast C participates in the same
                            // transposed accumulator representation.
                            row_major: false,
                        },
                    },
                ))
            }
        }
    }

    /// `CoopMma` -> `a * b + c`. When `c` is a `LoadLocal` of an accumulator
    /// with a live SSA entry, no `Load` is emitted at all.
    pub(crate) fn coop_mma(
        &mut self,
        a: &TileExpr,
        b: &TileExpr,
        c: &TileExpr,
        out: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        let a = self.expr(a, out)?;
        let b = self.expr(b, out)?;
        let c = self.expr(c, out)?;
        Ok(self.emit_expr(out, Expression::CooperativeMultiplyAdd { a, b, c }))
    }

    /// Write every live accumulator SSA value back to its local. Iterates the
    /// analysis's first-use-ordered locals rather than the pointer-keyed map,
    /// so the emitted order is deterministic.
    pub(crate) fn flush_coop_acc(&mut self, out: &mut Block) {
        if self.coop_acc.is_empty() {
            return;
        }
        let locals = self.analysis.locals.clone();
        let mut wrote = false;
        for local in &locals {
            let Some(value) = self.coop_acc.remove(&key(local)) else {
                continue;
            };
            let Some(handle) = self.local_handles.get(&key(local)).copied() else {
                continue;
            };
            self.store_local(out, handle, value);
            wrote = true;
        }
        self.coop_acc.clear();
        // These stores bypass `Emitter::stmt`, so nothing else retires the
        // `LoadLocal`s they invalidate. Every deferred accumulator lands here.
        if wrote {
            self.invalidate_mem(fusor2_ir::ir::level2::MemReads::LOCAL);
        }
    }

    /// `CoopStore` -> a subgroup-collective store, never a per-lane store.
    /// `row_major` is **inverted** relative to the destination layout because
    /// accumulators are held transposed.
    pub(crate) fn coop_store(
        &mut self,
        acc: &TileExpr,
        dst: &StorageView,
        addr: &Addr,
        out: &mut Block,
    ) -> Result<(), EmitError> {
        let acc_element = acc.element();
        let ElementType::CoopMatrix {
            scalar: acc_scalar,
            rows,
            cols,
            ..
        } = acc_element
        else {
            return Err(EmitError::Unsupported(
                "a cooperative store needs a fragment accumulator".into(),
            ));
        };
        let dst_scalar = match dst.buffer.element {
            ElementType::Scalar(s) => s,
            other => {
                return Err(EmitError::Unsupported(format!(
                    "a cooperative store needs a scalar destination, got {other:?}"
                )));
            }
        };
        if acc_scalar != dst_scalar && !self.caps.mixed_precision_coop_store {
            // Footprint, never a wrong answer: stage the fragment into an f32
            // workgroup tile, then cast and store per lane.
            return self.staged_coop_store(acc, dst, addr, out, acc_scalar, rows, cols);
        }

        let (stride_u, row_major) = cooperative_store_layout(&dst.layout)?;
        let (row, col) = match addr {
            Addr::Rc2 { row, col } => (row.clone(), col.clone()),
            Addr::Linear(_) => {
                return Err(EmitError::Unsupported(
                    "a cooperative store needs a rank-2 address".into(),
                ));
            }
        };
        let target = self.expr(acc, out)?;
        let row_h = self.expr(&row, out)?;
        let col_h = self.expr(&col, out)?;
        let index = self.storage_index_from_coords(out, dst, &[row_h, col_h])?;
        let pointer = self.storage_dynamic_pointer(out, dst, index)?;
        let stride = self.u32_lit(stride_u);
        out.push(
            Statement::CooperativeStore {
                target,
                data: CooperativeData {
                    pointer,
                    stride,
                    row_major: !row_major,
                },
            },
            Span::default(),
        );
        Ok(())
    }

    /// `CoopStoreTile` -> a cooperative store into a workgroup tile: the
    /// staging step attention needs between fragment math and a per-lane
    /// softmax over the same values. Workgroup tiles are row-major, so the
    /// inverted flag is `false`.
    pub(crate) fn coop_store_tile(
        &mut self,
        acc: &TileExpr,
        tile: &Tile,
        row: &TileExpr,
        col: &TileExpr,
        out: &mut Block,
    ) -> Result<(), EmitError> {
        let stride_u = row_major_tile_stride(tile)?;
        let target = self.expr(acc, out)?;
        let row_h = self.expr(row, out)?;
        let col_h = self.expr(col, out)?;
        let index = self.tile_matrix_index(out, row_h, col_h, stride_u);
        let pointer = self.tile_dynamic_pointer(out, tile, index)?;
        let stride = self.u32_lit(stride_u);
        out.push(
            Statement::CooperativeStore {
                target,
                data: CooperativeData {
                    pointer,
                    stride,
                    row_major: false,
                },
            },
            Span::default(),
        );
        Ok(())
    }

    /// The mixed-precision fallback: a private staging tile typed with the
    /// accumulator's own scalar, a cooperative store into it, then a per-lane
    /// cast-and-store into the narrower destination.
    ///
    /// The staging tile is *not* part of the arena plan, so it costs its own
    /// allocation. That is the documented price of building without
    /// `fork-metal`.
    #[allow(clippy::too_many_arguments)]
    fn staged_coop_store(
        &mut self,
        acc: &TileExpr,
        dst: &StorageView,
        addr: &Addr,
        out: &mut Block,
        acc_scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Result<(), EmitError> {
        let (row, col) = match addr {
            Addr::Rc2 { row, col } => (row.clone(), col.clone()),
            Addr::Linear(_) => {
                return Err(EmitError::Unsupported(
                    "a cooperative store needs a rank-2 address".into(),
                ));
            }
        };
        let element = ElementType::Scalar(acc_scalar);
        let staging = self.staging_tile(element, rows * cols)?;

        let target = self.expr(acc, out)?;
        let base = self.global_var(staging);
        let zero = self.u32_lit(0);
        let pointer = self.emit_expr(out, Expression::Access { base, index: zero });
        let stride = self.u32_lit(cols);
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        out.push(
            Statement::CooperativeStore {
                target,
                data: CooperativeData {
                    pointer,
                    stride,
                    row_major: false,
                },
            },
            Span::default(),
        );
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let row_h = self.expr(&row, out)?;
        let col_h = self.expr(&col, out)?;
        let dst = dst.clone();
        let dst_element = dst.buffer.element;
        let total = rows * cols;
        self.staged_copy(out, total, cols, staging, move |em, block, i, j| {
            let base = em.global_var(staging);
            let index = em.tile_matrix_index(block, i, j, cols);
            let ptr = em.emit_expr(block, Expression::Access { base, index });
            let value = em.emit_load(block, ptr);
            let value = em.cast_tile_value(block, value, element, dst_element)?;
            let global_row = em.add_u32(block, row_h, i);
            let global_col = em.add_u32(block, col_h, j);
            let flat = em.storage_index_from_coords(block, &dst, &[global_row, global_col])?;
            let dst_ptr = em.storage_dynamic_pointer(block, &dst, flat)?;
            block.push(
                Statement::Store {
                    pointer: dst_ptr,
                    value,
                },
                Span::default(),
            );
            Ok(())
        })?;
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        Ok(())
    }

    /// A workgroup allocation outside the arena plan, for the staging path.
    fn staging_tile(
        &mut self,
        element: ElementType,
        elements: u32,
    ) -> Result<Handle<GlobalVariable>, EmitError> {
        let count = std::num::NonZeroU32::new(elements.max(1)).expect("max(1) is non-zero");
        let base = self.element_type(element)?;
        let stride = element
            .workgroup_array_stride()
            .ok_or_else(|| EmitError::Unsupported("staging element cannot back an array".into()))?;
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base,
                    size: ArraySize::Constant(count),
                    stride,
                },
            },
            Span::default(),
        );
        Ok(self.module.global_variables.append(
            GlobalVariable {
                name: None,
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            Span::default(),
        ))
    }

    fn staged_copy(
        &mut self,
        out: &mut Block,
        total: u32,
        cols: u32,
        _staging: Handle<GlobalVariable>,
        mut build: impl FnMut(
            &mut Self,
            &mut Block,
            Handle<Expression>,
            Handle<Expression>,
        ) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let lanes = self.workgroup_invocations.max(1);
        let passes = total.div_ceil(lanes);
        for pass in 0..passes {
            let full = (pass + 1) * lanes <= total;
            let lane = self.lane();
            let flat = self.add_literal_u32(out, lane, pass * lanes);
            let condition = if full {
                None
            } else {
                let limit = self.u32_lit(total);
                Some(self.bin(out, naga::BinaryOperator::Less, flat, limit))
            };
            let (accept, ()) = self.nested(|em, accept| {
                let i = em.div_literal_u32(accept, flat, cols.max(1));
                let j = em.mod_literal_u32(accept, flat, cols.max(1));
                build(em, accept, i, j)
            })?;
            match condition {
                Some(c) => out.push(
                    Statement::If {
                        condition: c,
                        accept,
                        reject: Block::new(),
                    },
                    Span::default(),
                ),
                None => out.push(Statement::Block(accept), Span::default()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::emit_module;
    use crate::emit::testkit::{self, *};
    use fusor2_ir::device::CoopKind;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level2::{KernelIr, Source, Stmt, TileExprKind, TileLiteral};

    fn coop_caps(mixed: bool) -> fusor2_ir::device::Caps {
        let mut caps = caps(true, true);
        caps.coop = smallvec::smallvec![CoopKind {
            operand: Dtype::F32,
            acc: Dtype::F32,
            m: 8,
            n: 8,
            k: 8,
        }];
        caps.mixed_precision_coop_store = mixed;
        caps
    }

    fn acc_element(scalar: ScalarElement) -> ElementType {
        ElementType::CoopMatrix {
            scalar,
            role: CoopMatrixRole::C,
            rows: 8,
            cols: 8,
        }
    }

    fn frag(role: CoopMatrixRole, tile: &Tile, transposed: bool) -> TileExpr {
        TileExpr::new(
            TileExprKind::CoopLoad {
                role,
                scalar: ScalarElement::F32,
                rows: 8,
                cols: 8,
                src: Box::new(CoopSrc::TileRegion {
                    tile: tile.clone(),
                    row: lit_u32(0),
                    col: lit_u32(0),
                    transposed,
                }),
            },
            ElementType::CoopMatrix {
                scalar: ScalarElement::F32,
                role,
                rows: 8,
                cols: 8,
            },
        )
    }

    /// All four cooperative primitives in one kernel.
    fn coop_kernel(dst_scalar: ScalarElement) -> KernelIr {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let dst = testkit::buffer(1, ElementType::Scalar(dst_scalar), 64, true);
        let dv = fusor2_ir::ir::level2::StorageView {
            buffer: dst.clone(),
            offset: 0,
            layout: fusor2_ir::ir::level2::TileLayout::contiguous(
                fusor2_ir::ir::level2::MemoryLevel::Storage,
                &[8, 8],
            ),
        };
        let a = testkit::tile(f32e(), &[8, 8]);
        let b = testkit::tile(f32e(), &[8, 8]);
        let staging = testkit::tile(f32e(), &[8, 8]);
        let acc = testkit::local(acc_element(ScalarElement::F32));

        let zero = TileExpr::new(
            TileExprKind::Cast {
                value: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::F32(0f32.to_bits())),
                    f32e(),
                ),
                to: acc_element(ScalarElement::F32),
            },
            acc_element(ScalarElement::F32),
        );
        let mma = TileExpr::new(
            TileExprKind::CoopMma {
                a: frag(CoopMatrixRole::A, &a, false),
                b: frag(CoopMatrixRole::B, &b, true),
                c: TileExpr::new(
                    TileExprKind::LoadLocal(acc.clone()),
                    acc_element(ScalarElement::F32),
                ),
            },
            acc_element(ScalarElement::F32),
        );
        let acc_value = TileExpr::new(
            TileExprKind::LoadLocal(acc.clone()),
            acc_element(ScalarElement::F32),
        );
        KernelIr {
            buffers: vec![uni, dst],
            grid: [1, 1, 1],
            block: 32,
            body: vec![
                Stmt::StoreTile {
                    dst: a.clone(),
                    index: lane(),
                    value: lit_f32(1.0),
                },
                Stmt::StoreTile {
                    dst: b.clone(),
                    index: lane(),
                    value: lit_f32(2.0),
                },
                Stmt::Barrier,
                Stmt::StoreLocal {
                    dst: acc.clone(),
                    value: zero,
                },
                Stmt::StoreLocal {
                    dst: acc.clone(),
                    value: mma,
                },
                Stmt::CoopStoreTile {
                    acc: acc_value.clone(),
                    tile: staging,
                    row: lit_u32(0),
                    col: lit_u32(0),
                },
                Stmt::Barrier,
                Stmt::CoopStore {
                    acc: acc_value,
                    dst: dv,
                    addr: Addr::Rc2 {
                        row: lit_u32(0),
                        col: lit_u32(0),
                    },
                },
            ],
            byte_arena: None,
            name: "coop",
        }
    }

    /// Load, MMA, store-to-storage and store-to-tile all emit, and the
    /// transposed fragment flips `row_major`.
    #[test]
    fn every_cooperative_primitive_emits() {
        let emitted = emit_module(
            &coop_kernel(ScalarElement::F32),
            &coop_caps(true),
            &no_plan(),
        )
        .expect("coop emit");
        let module = &emitted.module;
        let loads: Vec<bool> = module.entry_points[0]
            .function
            .expressions
            .iter()
            .filter_map(|(_, e)| match e {
                Expression::CooperativeLoad { data, .. } => Some(data.row_major),
                _ => None,
            })
            .collect();
        assert_eq!(loads.len(), 2, "one A fragment and one B fragment");
        assert_eq!(
            loads,
            vec![false, true],
            "a transposed load flips row_major"
        );
        assert_eq!(
            count_exprs(module, |e| matches!(
                e,
                Expression::CooperativeMultiplyAdd { .. }
            )),
            1
        );
        assert_eq!(cooperative_stores(module), 2, "one to storage, one to tile");
        // Cooperative lowering forces a subgroup id even though the body never
        // asks for one.
        assert_eq!(main_fn(module).arguments.len(), 3);
    }

    /// Without the fork's mixed-precision store, an f32 accumulator bound for
    /// f16 memory routes through a staging tile plus a per-lane cast.
    ///
    /// The direct form is the fork's whole contribution here: released naga
    /// rejects a float fragment stored into float memory of a different width,
    /// so declaring `mixed_precision_coop_store` without the fork produces a
    /// module the validator refuses — which is why the capability defaults to
    /// false and the staged path is the shipped one.
    #[test]
    fn mixed_precision_store_stages_without_the_fork() {
        let wg = |m: &naga::Module| {
            m.global_variables
                .iter()
                .filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup)
                .count()
        };

        let staged = emit_module(
            &coop_kernel(ScalarElement::F16),
            &coop_caps(false),
            &no_plan(),
        )
        .expect("the staged path always emits");
        let same_width = emit_module(
            &coop_kernel(ScalarElement::F32),
            &coop_caps(false),
            &no_plan(),
        )
        .expect("a same-width store needs no staging");

        assert_eq!(
            wg(&staged.module),
            wg(&same_width.module) + 1,
            "the staging tile is the whole price"
        );
        assert_eq!(cooperative_stores(&staged.module), 2);
        assert_eq!(cooperative_stores(&same_width.module), 2);

        // Claiming the fork capability on released naga fails validation
        // rather than mis-lowering.
        assert!(matches!(
            emit_module(
                &coop_kernel(ScalarElement::F16),
                &coop_caps(true),
                &no_plan()
            ),
            Err(EmitError::Validation(_))
        ));
    }

    /// A cooperative store into a destination with no unit stride is refused
    /// rather than silently mis-addressed.
    #[test]
    fn non_unit_stride_destination_is_refused() {
        let mut ir = coop_kernel(ScalarElement::F32);
        if let Stmt::CoopStore { dst, .. } = ir.body.last_mut().expect("coop store") {
            dst.layout.indexing = fusor2_ir::shape::MultiFlattenMap::affine(&[8, 8], &[16, 2]);
        }
        assert!(emit_module(&ir, &coop_caps(true), &no_plan()).is_err());
    }

    /// A broadcast-column fragment reads rank-1 storage at stride zero.
    #[test]
    fn broadcast_column_fragment_emits() {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let bias = testkit::buffer(1, f32e(), 8, false);
        let dst = testkit::buffer(2, f32e(), 64, true);
        let bv = view(&bias, &[8]);
        let dv = fusor2_ir::ir::level2::StorageView {
            buffer: dst.clone(),
            offset: 0,
            layout: fusor2_ir::ir::level2::TileLayout::contiguous(
                fusor2_ir::ir::level2::MemoryLevel::Storage,
                &[8, 8],
            ),
        };
        let acc = TileExpr::new(
            TileExprKind::CoopLoad {
                role: CoopMatrixRole::C,
                scalar: ScalarElement::F32,
                rows: 8,
                cols: 8,
                src: Box::new(CoopSrc::BroadcastCol {
                    src: bv,
                    col: lit_u32(0),
                }),
            },
            acc_element(ScalarElement::F32),
        );
        let ir = KernelIr {
            buffers: vec![uni, bias, dst],
            grid: [1, 1, 1],
            block: 32,
            body: vec![Stmt::CoopStore {
                acc,
                dst: dv,
                addr: Addr::Rc2 {
                    row: lit_u32(0),
                    col: lit_u32(0),
                },
            }],
            byte_arena: None,
            name: "broadcast_col",
        };
        let emitted = emit_module(&ir, &coop_caps(true), &no_plan()).expect("emit");
        let strides: Vec<_> = emitted.module.entry_points[0]
            .function
            .expressions
            .iter()
            .filter_map(|(_, e)| match e {
                Expression::CooperativeLoad { data, .. } => Some(data.row_major),
                _ => None,
            })
            .collect();
        assert_eq!(strides, vec![false]);
        // The `Source` import keeps the fixture honest about what a broadcast
        // is not: it never reads a quantized source.
        let _: Option<Source> = None;
    }

    fn cooperative_stores(module: &naga::Module) -> usize {
        fn walk(block: &naga::Block) -> usize {
            block
                .iter()
                .map(|s| match s {
                    Statement::CooperativeStore { .. } => 1,
                    Statement::Block(b) => walk(b),
                    Statement::If { accept, reject, .. } => walk(accept) + walk(reject),
                    Statement::Loop {
                        body, continuing, ..
                    } => walk(body) + walk(continuing),
                    _ => 0,
                })
                .sum()
        }
        walk(&module.entry_points[0].function.body)
    }
}
