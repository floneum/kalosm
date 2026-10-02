//! Cooperative-matrix fragment load, MMA and store.
//!
//! Accumulators are held transposed: Metal's simdgroup orientation multiplies
//! row-major fragments as `B * A`. A transposed tile load therefore swaps the
//! fragment origin, and a cooperative store inverts the layout flag.

use fusor_ir::ir::kernel::{
    Addr, CoopMatrixRole, CoopSrc, ElementType, ScalarElement, StorageView, Tile, TileExpr,
    TileLayout, cooperative_store_layout_supported,
};
use fusor_ir::target::EmitError;
use naga::{ArraySize, Block, CooperativeData, Expression, GlobalVariable, Handle, Statement};

use super::expr::{barrier, push};
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
    let layout = &tile.layout;
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

/// A cooperative store's row and column, which only a rank-2 address has.
fn rc2(addr: &Addr) -> Result<(&TileExpr, &TileExpr), EmitError> {
    match addr {
        Addr::Rc2 { row, col } => Ok((row, col)),
        Addr::Linear(_) => Err(EmitError::Unsupported(
            "a cooperative store needs a rank-2 address".into(),
        )),
    }
}

impl Emitter<'_> {
    /// `CoopLoad`. From a tile region the fragment origin is swapped and
    /// `row_major: transposed`.
    pub(crate) fn coop_load_parts(
        &mut self,
        out: &mut Block,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: &CoopSrc,
    ) -> Result<Handle<Expression>, EmitError> {
        let role = super::types::naga_role(role);
        let columns = super::types::cooperative_size(cols)?;
        let rows_size = super::types::cooperative_size(rows)?;
        let CoopSrc {
            tile,
            row,
            col,
            transposed,
        } = src;
        // An f32 fragment off an f16 tile reads plausible garbage, so the scalars
        // are checked here.
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

    /// Write every live accumulator SSA value back to its local, in the
    /// analysis's deterministic first-use order.
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

    /// `CoopStore` -> a subgroup-collective store; `row_major` is inverted since
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
        if (acc_scalar != dst_scalar && !self.caps.mixed_precision_coop_store)
            || self.buffer_element(&dst.buffer) != dst.buffer.element
            || self.analysis.atomic_buffers.contains(&dst.buffer.binding)
        {
            // Footprint, never a wrong answer: stage the fragment into an f32
            // workgroup tile, then cast and store per lane.
            return self.staged_coop_store(acc, dst, addr, out, acc_scalar, rows, cols);
        }

        let (stride_u, row_major) = cooperative_store_layout(&dst.layout)?;
        let (row, col) = rc2(addr)?;
        let target = self.expr(acc, out)?;
        let row_h = self.expr(row, out)?;
        let col_h = self.expr(col, out)?;
        let index = self.storage_index_from_coords(out, dst, &[row_h, col_h])?;
        let pointer = self.storage_dynamic_pointer(out, dst, index)?;
        self.push_coop_store(out, target, pointer, stride_u, !row_major);
        Ok(())
    }

    /// `coopStore(target, pointer, stride)`; the stride literal is appended here.
    fn push_coop_store(
        &mut self,
        out: &mut Block,
        target: Handle<Expression>,
        pointer: Handle<Expression>,
        stride: u32,
        row_major: bool,
    ) {
        let stride = self.u32_lit(stride);
        let data = CooperativeData {
            pointer,
            stride,
            row_major,
        };
        push(out, Statement::CooperativeStore { target, data });
    }

    /// `CoopStoreTile` -> a cooperative store into a row-major workgroup tile
    /// (inverted flag `false`).
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
        self.push_coop_store(out, target, pointer, stride_u, false);
        Ok(())
    }

    /// The mixed-precision fallback without `fork-metal`: cooperative store into a
    /// private staging tile of the accumulator's scalar (outside the arena plan),
    /// then a per-lane cast-and-store.
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
        let (row, col) = rc2(addr)?;
        let element = ElementType::Scalar(acc_scalar);
        let (staging, offset) = self.subgroup_staging(out, element, rows * cols)?;

        let target = self.expr(acc, out)?;
        let base = self.global_var(staging);
        let pointer = self.emit_expr(
            out,
            Expression::Access {
                base,
                index: offset,
            },
        );
        let stride = self.u32_lit(cols);
        barrier(out);
        let data = CooperativeData {
            pointer,
            stride,
            row_major: false,
        };
        push(out, Statement::CooperativeStore { target, data });
        barrier(out);

        let row_h = self.expr(row, out)?;
        let col_h = self.expr(col, out)?;
        let dst = dst.clone();
        let dst_element = dst.buffer.element;
        let total = rows * cols;
        let lanes = self.caps.subgroup_width();
        let lane_arg = self.subgroup_args[1].expect("cooperative subgroup lane");
        self.copy_passes(out, total, lanes, lane_arg, move |em, block, flat| {
            let i = em.div_literal_u32(block, flat, cols.max(1));
            let j = em.mod_literal_u32(block, flat, cols.max(1));
            let base = em.global_var(staging);
            let index = em.tile_matrix_index(block, i, j, cols);
            let index = em.add_u32(block, offset, index);
            let ptr = em.emit_expr(block, Expression::Access { base, index });
            let value = em.emit_load(block, ptr);
            let value = em.cast_tile_value(block, value, element, dst_element)?;
            let global_row = em.add_u32(block, row_h, i);
            let global_col = em.add_u32(block, col_h, j);
            let flat = em.storage_index_from_coords(block, &dst, &[global_row, global_col])?;
            em.store_storage_value(block, &dst, flat, value)
        })?;
        barrier(out);
        Ok(())
    }

    /// Each subgroup owns a fragment and needs a disjoint staging region.
    fn subgroup_staging(
        &mut self,
        out: &mut Block,
        element: ElementType,
        elements: u32,
    ) -> Result<(Handle<GlobalVariable>, Handle<Expression>), EmitError> {
        let width = self.caps.subgroup_width();
        let groups = self.workgroup_invocations.div_ceil(width);
        let staging = self.staging_tile(element, elements * groups)?;
        let subgroup = self.function_arg(self.subgroup_args[0].expect("cooperative subgroup id"));
        let offset = self.mul_literal_u32(out, subgroup, elements);
        Ok((staging, offset))
    }

    /// A workgroup allocation outside the arena plan, for the staging path.
    fn staging_tile(
        &mut self,
        element: ElementType,
        elements: u32,
    ) -> Result<Handle<GlobalVariable>, EmitError> {
        let count = std::num::NonZeroU32::new(elements.max(1)).expect("max(1) is non-zero");
        let size = ArraySize::Constant(count);
        let ty = super::types::array_type(&mut self.module, element, size)?;
        Ok(super::types::workgroup_var(&mut self.module, ty))
    }
}
