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

use fusor_ir::ir::kernel::{
    Addr, CoopMatrixRole, CoopSrc, ElementType, ScalarElement, StorageView, Tile, TileExpr,
    TileLayout, cooperative_store_layout_supported,
};
use fusor_ir::target::EmitError;
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
                // A `CoopLoad{scalar: F32}` off an f16 tile reads the right
                // addresses at twice the width and comes back with plausible
                // garbage, so the scalars are checked here where both are in
                // hand.
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
            self.invalidate_mem(fusor_ir::ir::kernel::MemReads::LOCAL);
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
