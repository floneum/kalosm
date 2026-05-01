use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::{
    BufferAccess, BufferDecl, BufferRef, DynamicOffset, F32Bits, GgmlQuantFormat, Im2ColNhwcMap,
    KernelIr, Layout, MemoryLevel, Numeric, Op, QuantizedMatrix, Shape, StorageIndexMap,
    StorageView, TileBinaryOp, TileCompareOp, TileDecl, TileExpr, TileIndexExpr, TileLevel,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramAccelerator, TileProgramOp,
    TileQGemvProgramOp, TileQMatmulProgramOp, TileQuantizedLoadExpr, TileReduceOp, TileRef,
    TileScalarExpr, TileStoreProgramOp, TileUnaryOp, WorkgroupAxis, WorkgroupOffset, F32, U32,
};

/// Build a Triton-like source tile IR.
pub fn build(f: impl FnOnce(&mut Program)) -> KernelIr {
    let mut program = Program {
        ir: KernelIr::default(),
    };
    f(&mut program);
    program.ir
}

/// Source tile program builder.
pub struct Program {
    ir: KernelIr,
}

impl Program {
    pub fn storage_read<T: Numeric, const R: usize>(&mut self, shape: Shape) -> Storage<T, R> {
        self.storage_with_layout_and_access(
            Layout::contiguous(MemoryLevel::Storage, shape),
            0,
            None,
            BufferAccess::Read,
        )
    }

    pub fn storage_write<T: Numeric, const R: usize>(&mut self, shape: Shape) -> Storage<T, R> {
        self.storage_with_layout_and_access(
            Layout::contiguous(MemoryLevel::Storage, shape),
            0,
            None,
            BufferAccess::ReadWrite,
        )
    }

    pub fn storage_read_with_layout<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(layout, 0, None, BufferAccess::Read)
    }

    pub fn storage_write_with_layout<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(layout, 0, None, BufferAccess::ReadWrite)
    }

    pub fn storage_read_with_layout_offset<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(layout, offset, None, BufferAccess::Read)
    }

    pub fn storage_write_with_layout_offset<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(layout, offset, None, BufferAccess::ReadWrite)
    }

    pub fn storage_read_with_layout_offset_and_index_map<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
        index_map: StorageIndexMap,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(layout, offset, Some(index_map), BufferAccess::Read)
    }

    pub fn storage_write_with_layout_offset_and_index_map<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
        index_map: StorageIndexMap,
    ) -> Storage<T, R> {
        self.storage_with_layout_and_access(
            layout,
            offset,
            Some(index_map),
            BufferAccess::ReadWrite,
        )
    }

    fn storage_with_layout_and_access<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
        index_map: Option<StorageIndexMap>,
        access: BufferAccess,
    ) -> Storage<T, R> {
        let view = self.storage_view_with_layout_and_access::<R>(
            T::ELEMENT,
            layout,
            offset,
            index_map,
            access,
        );
        Storage {
            view,
            _ty: PhantomData,
        }
    }

    pub fn storage_read_element_with_layout_offset<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
    ) -> ErasedStorage<R> {
        ErasedStorage {
            view: self.storage_view_with_layout_and_access::<R>(
                element,
                layout,
                offset,
                None,
                BufferAccess::Read,
            ),
        }
    }

    pub fn storage_write_element_with_layout_offset<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
    ) -> ErasedStorage<R> {
        ErasedStorage {
            view: self.storage_view_with_layout_and_access::<R>(
                element,
                layout,
                offset,
                None,
                BufferAccess::ReadWrite,
            ),
        }
    }

    fn storage_view_with_layout_and_access<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
        index_map: Option<StorageIndexMap>,
        access: BufferAccess,
    ) -> StorageView {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage tensors must use MemoryLevel::Storage"
        );
        assert_eq!(layout.shape().rank(), R, "storage rank mismatch");
        let buffer = self.alloc_buffer_element(element, layout.clone(), access);
        StorageView {
            buffer,
            offset,
            dynamic_offsets: vec![None; layout.shape().rank()],
            layout,
            index_map,
        }
    }

    pub fn quantized_matrix(
        &mut self,
        format: GgmlQuantFormat,
        rows: u32,
        cols: u32,
    ) -> QuantizedMatrix {
        assert!(
            rows > 0 && cols > 0,
            "quantized matrix shape must be non-zero"
        );
        assert_eq!(
            rows % format.block_elements(),
            0,
            "quantized rows/K dimension must be a multiple of the format block size"
        );
        let blocks_per_col = rows / format.block_elements();
        let words = blocks_per_col
            .checked_mul(cols)
            .and_then(|blocks| blocks.checked_mul(format.block_words()))
            .expect("quantized matrix word count overflow");
        let data = self.storage_read::<U32, 1>(Shape::new([words]));
        QuantizedMatrix {
            data: data.view,
            format,
            rows,
            cols,
        }
    }

    pub fn qmatmul<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
    ) {
        self.qmatmul_options::<BM, BN, BK>(a, b, y, vector_width, true, 1);
    }

    pub fn qgemv<const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
        workgroups_x: u32,
    ) {
        self.qmatmul_options::<1, BN, BK>(a, b, y, vector_width, true, workgroups_x);
    }

    pub fn qmatmul_options<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
        use_qgemv: bool,
        workgroups_x: u32,
    ) {
        assert!(
            BM > 0 && BN > 0 && BK > 0,
            "qmatmul tile shape must be non-zero"
        );
        assert!(vector_width > 0, "qmatmul vector width must be non-zero");
        assert!(workgroups_x > 0, "qmatmul workgroups_x must be non-zero");
        let [m, k] = matrix_shape(&a.view.layout);
        let [y_m, y_n] = matrix_shape(&y.view.layout);
        assert_eq!(k, b.rows, "qmatmul K dimensions must match");
        assert_eq!(m, y_m, "qmatmul output row count must match A");
        assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

        if m == 1 && use_qgemv {
            self.qgemv_tile::<BN, BK>(a, b, y, workgroups_x);
        } else {
            self.qmatmul_tile::<BM, BN, BK>(a, b, y);
        }
    }

    fn qgemv_tile<const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        workgroups_x: u32,
    ) {
        let [m, k] = matrix_shape(&a.view.layout);
        assert_eq!(m, 1, "qgemv requires a single input row");
        let large_q5_0_gemv = b.format == GgmlQuantFormat::Q5_0
            && b.rows
                .checked_mul(b.cols)
                .is_some_and(|elements| elements >= 4 * 1024 * 1024);
        let wide_q8_0_gemv = b.format == GgmlQuantFormat::Q8_0 && b.rows <= 1024 && b.cols >= 8192;
        if b.format == GgmlQuantFormat::Q4K || large_q5_0_gemv || wide_q8_0_gemv {
            let cols_per_workgroup = b.format.qgemv_cols_per_workgroup();
            let total_workgroups = b.cols.div_ceil(cols_per_workgroup);
            let dispatch_y = total_workgroups.div_ceil(workgroups_x);
            self.ir.body.push(Op::TileProgram(TileProgramOp {
                grid: [workgroups_x, dispatch_y, 1],
                block: b.format.qgemv_subgroups_per_workgroup() * 32,
                stores: Vec::new(),
                accelerator: Some(TileProgramAccelerator::QGemv(TileQGemvProgramOp {
                    a: a.view.clone(),
                    b: b.clone(),
                    y: y.view.clone(),
                    workgroups_x,
                })),
            }));
            return;
        }

        const LANES: usize = 256;
        assert!(
            BN > 0 && BK > 0 && BN * BK == LANES && BN.is_power_of_two() && BK.is_power_of_two(),
            "qgemv expects BN * BK == 1024 with power-of-two column and K lane groups"
        );
        let total_workgroups = b.cols.div_ceil(BN as u32);
        let dispatch_y = total_workgroups.div_ceil(workgroups_x);
        let k_iterations = k.div_ceil(BK as u32);
        self.program_grid::<LANES>([workgroups_x, dispatch_y, 1], |program| {
            let tile = program.program_id(WorkgroupAxis::X)
                + program.program_id(WorkgroupAxis::Y) * workgroups_x;
            let lane = program.arange();
            let col_lane = lane.clone() / BK as u32;
            let k_lane = lane % BK as u32;
            let col = tile * BN as u32 + col_lane;
            let loop_index = program.loop_index();
            let k_index = loop_index * BK as u32 + k_lane.clone();
            let mask = col.lt(b.cols).and(k_index.lt(k));
            let a_value = program.load(a.at(0, &k_index), mask.clone(), 0.0);
            let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                a_value * b_value,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let sum = program.group_reduce_sum::<BK>(partial);
            let store_mask = k_lane.eq(0).and(col.lt(b.cols));
            program.store(y.at(0, col), sum, store_mask);
        });
    }

    fn qmatmul_tile<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
    ) {
        const LANES: usize = 256;
        assert!(
            BM > 0 && BN > 0 && BK > 0,
            "qmatmul tile shape must be non-zero"
        );
        let [m, k] = matrix_shape(&a.view.layout);
        let coop_subgroups = qmatmul_coop_subgroups(BM as u32, BN as u32);
        if let Some(subgroups) = coop_subgroups
            .filter(|_| BK.is_multiple_of(8))
            .filter(|_| BK == 16 || BK == 32)
            .filter(|_| m.is_multiple_of(BM as u32))
            .filter(|_| b.cols.is_multiple_of(BN as u32))
            .filter(|_| k.is_multiple_of(BK as u32))
            .filter(|_| cooperative_store_layout_supported(&y.view.layout))
        {
            let a_tile = self.alloc_tile::<F32>(
                Layout::contiguous(MemoryLevel::Workgroup, Shape::new([BM as u32, BK as u32])),
                TileLevel::Workgroup,
            );
            let b_tile = self.alloc_tile::<F32>(
                Layout::contiguous(MemoryLevel::Workgroup, Shape::new([BK as u32, BN as u32])),
                TileLevel::Workgroup,
            );
            self.ir.body.push(Op::TileProgram(TileProgramOp {
                grid: [b.cols.div_ceil(BN as u32), m.div_ceil(BM as u32), 1],
                block: subgroups * 32,
                stores: Vec::new(),
                accelerator: Some(TileProgramAccelerator::QMatmul(TileQMatmulProgramOp {
                    a: a.view.clone(),
                    b: b.clone(),
                    y: y.view.clone(),
                    a_tile,
                    b_tile,
                    tile_m: BM as u32,
                    tile_n: BN as u32,
                    tile_k: BK as u32,
                })),
            }));
            return;
        }
        if BM * BN * BK != LANES || !BK.is_power_of_two() {
            self.qmatmul_tile::<8, 4, 8>(a, b, y);
            return;
        }
        let k_iterations = k.div_ceil(BK as u32);
        self.program_grid::<LANES>(
            [b.cols.div_ceil(BN as u32), m.div_ceil(BM as u32), 1],
            |program| {
                let lane = program.arange();
                let k_lane = lane.clone() % BK as u32;
                let output_lane = lane / BK as u32;
                let row_lane = output_lane.clone() / BN as u32;
                let col_lane = output_lane % BN as u32;
                let row = program.program_id(WorkgroupAxis::Y) * BM as u32 + row_lane;
                let col = program.program_id(WorkgroupAxis::X) * BN as u32 + col_lane;
                let loop_index = program.loop_index();
                let k_index = loop_index * BK as u32 + k_lane.clone();
                let mask = row.lt(m).and(col.lt(b.cols)).and(k_index.lt(k));
                let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
                let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                let partial = program.loop_fold(
                    TileReduceOp::Sum,
                    k_iterations,
                    a_value * b_value,
                    TileLiteral::F32(F32Bits::new(0.0)),
                );
                let sum = program.group_reduce_sum::<BK>(partial);
                let store_mask = k_lane.eq(0).and(row.lt(m)).and(col.lt(b.cols));
                program.store(y.at(row, col), sum, store_mask);
            },
        );
    }

    pub fn matmul<const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &Storage<F32, 2>,
        y: &Storage<F32, 2>,
    ) {
        const TILE_ROWS: usize = 16;
        const TILE_COLS: usize = 16;
        const LANES: usize = TILE_ROWS * TILE_COLS;
        assert!(BK > 0, "matmul K tile shape must be non-zero");
        let [m, k] = matrix_shape(&a.view.layout);
        let [b_k, n] = matrix_shape(&b.view.layout);
        let [y_m, y_n] = matrix_shape(&y.view.layout);
        assert_eq!(k, b_k, "matmul K dimensions must match");
        assert_eq!(m, y_m, "matmul output row count must match A");
        assert_eq!(n, y_n, "matmul output column count must match B");

        self.program_grid::<LANES>(
            [
                n.div_ceil(TILE_COLS as u32),
                m.div_ceil(TILE_ROWS as u32),
                1,
            ],
            |program| {
                let lane_tile = program.lane_tile_2d::<TILE_ROWS, TILE_COLS>();
                let row = program.program_id(WorkgroupAxis::Y) * TILE_ROWS as u32 + lane_tile.row();
                let col = program.program_id(WorkgroupAxis::X) * TILE_COLS as u32 + lane_tile.col();
                let mask = row.lt(m).and(col.lt(n));
                let k_index = program.loop_index();
                let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
                let b_value = program.load(b.at(&k_index, &col), mask.clone(), 0.0);
                let sum = program.loop_fold(
                    TileReduceOp::Sum,
                    k,
                    a_value * b_value,
                    TileLiteral::F32(F32Bits::new(0.0)),
                );
                program.store(y.at(row, col), sum, mask);
            },
        );
    }

    pub fn qdequantize(&mut self, b: &QuantizedMatrix, y: &Storage<F32, 1>, workgroups_x: u32) {
        const BLOCK: usize = 256;
        assert!(
            workgroups_x > 0,
            "qdequantize workgroups_x must be non-zero"
        );
        assert_eq!(
            y.view.layout.element_count().get(),
            b.rows
                .checked_mul(b.cols)
                .expect("qdequantize output element count overflow"),
            "qdequantize output must contain one dense f32 per quantized element"
        );
        assert!(
            y.view.layout.is_row_major(),
            "qdequantize output must be row-major"
        );

        let total = b
            .rows
            .checked_mul(b.cols)
            .expect("qdequantize output element count overflow");
        let workgroups = total.div_ceil(BLOCK as u32);
        let dispatch_y = workgroups.div_ceil(workgroups_x);
        let y = Storage::<F32, 2> {
            view: StorageView {
                buffer: y.view.buffer,
                offset: y.view.offset,
                dynamic_offsets: vec![None, None],
                layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([1, total])),
                index_map: None,
            },
            _ty: PhantomData,
        };
        self.program_grid::<BLOCK>([workgroups_x, dispatch_y, 1], |program| {
            let lane = program.arange();
            let linear_group = program.program_id(WorkgroupAxis::X)
                + program.program_id(WorkgroupAxis::Y) * workgroups_x;
            let flat = linear_group * BLOCK as u32 + lane;
            let mask = flat.lt(total);
            let value = program.load_quantized(
                b,
                flat.clone() % b.rows,
                flat.clone() / b.rows,
                mask.clone(),
                0.0,
            );
            program.store(y.at(0, flat), value, mask);
        });
    }

    pub fn program_grid<const BLOCK: usize>(
        &mut self,
        grid: [u32; 3],
        body: impl FnOnce(&mut TileBlock<'_, BLOCK>),
    ) {
        assert!(BLOCK > 0, "tile block size must be non-zero");
        assert!(
            BLOCK <= 1024 && BLOCK.is_power_of_two(),
            "tile block size must be a power of two at most 1024"
        );
        let mut block = TileBlock {
            program: self,
            grid,
            stores: Vec::new(),
        };
        body(&mut block);
        block.program.ir.body.push(Op::TileProgram(TileProgramOp {
            grid,
            block: BLOCK as u32,
            stores: block.stores,
            accelerator: None,
        }));
    }

    fn alloc_buffer_element(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        access: BufferAccess,
    ) -> BufferRef {
        let id = crate::BufferId(self.ir.next_buffer);
        self.ir.next_buffer += 1;
        let buffer = BufferRef::new(id, element);
        self.ir.buffers.push(BufferDecl {
            id,
            element,
            layout,
            access,
        });
        buffer
    }

    fn alloc_tile<T: Numeric>(&mut self, layout: Layout, level: TileLevel) -> TileRef {
        let id = crate::TileId(self.ir.next_tile);
        self.ir.next_tile += 1;
        let tile = TileRef::new(id, T::ELEMENT);
        self.ir.tiles.push(TileDecl {
            id,
            element: T::ELEMENT,
            layout,
            level,
            origin: TileOrigin::Allocation,
        });
        tile
    }
}

/// A storage tensor handle in the source tile IR.
#[derive(Clone)]
pub struct Storage<T, const R: usize> {
    pub(crate) view: StorageView,
    _ty: PhantomData<T>,
}

/// A storage tensor whose element type is known at runtime.
#[derive(Clone)]
pub struct ErasedStorage<const R: usize> {
    pub(crate) view: StorageView,
}

impl<const R: usize> ErasedStorage<R> {
    pub fn view(&self) -> &StorageView {
        &self.view
    }
}

impl<T> Storage<T, 2> {
    pub fn at<const N: usize>(
        &self,
        row: impl IntoIndex<N>,
        col: impl IntoIndex<N>,
    ) -> Address<T, N> {
        Address {
            view: self.view.clone(),
            row: row.into_index(),
            col: col.into_index(),
            _ty: PhantomData,
        }
    }

    pub fn dynamic_tile_2d(
        &self,
        shape: Shape,
        row_offset: Option<DynamicOffset>,
        col_offset: Option<DynamicOffset>,
    ) -> Self {
        assert_eq!(self.view.layout.shape().rank(), 2, "parent view must be 2D");
        assert_eq!(shape.rank(), 2, "tile view must be 2D");
        assert!(
            self.view.dynamic_offsets.iter().all(Option::is_none),
            "nested dynamic storage views are not supported"
        );
        assert!(
            self.view.index_map.is_none(),
            "nested mapped storage views are not supported"
        );
        let layout = Layout::strided(
            MemoryLevel::Storage,
            shape,
            self.view.layout.strides().clone(),
        );
        Self {
            view: StorageView {
                buffer: self.view.buffer,
                offset: self.view.offset,
                layout,
                dynamic_offsets: vec![row_offset, col_offset],
                index_map: None,
            },
            _ty: PhantomData,
        }
    }

    pub fn workgroup_tile_2d(
        &self,
        shape: Shape,
        row_offset: Option<WorkgroupOffset>,
        col_offset: Option<WorkgroupOffset>,
    ) -> Self {
        self.dynamic_tile_2d(
            shape,
            row_offset.map(DynamicOffset::Workgroup),
            col_offset.map(DynamicOffset::Workgroup),
        )
    }
}

impl ErasedStorage<2> {
    pub fn at<const N: usize>(
        &self,
        row: impl IntoIndex<N>,
        col: impl IntoIndex<N>,
    ) -> ErasedAddress<N> {
        ErasedAddress {
            view: self.view.clone(),
            row: row.into_index(),
            col: col.into_index(),
        }
    }
}

impl<T> Storage<T, 4> {
    /// Create a rank-2 im2col matrix view over a rank-4 NHWC tensor.
    pub fn im2col_nhwc(
        &self,
        output_hw: [u32; 2],
        kernel_hw: [u32; 2],
        stride_hw: [u32; 2],
        dilation_hw: [u32; 2],
    ) -> Storage<T, 2> {
        assert_eq!(
            self.view.layout.shape().rank(),
            4,
            "NHWC input must be rank-4"
        );
        assert!(
            self.view.dynamic_offsets.iter().all(Option::is_none),
            "im2col views do not support dynamic offsets"
        );
        assert!(
            self.view.index_map.is_none(),
            "nested mapped storage views are not supported"
        );
        let input_dims = self.view.layout.shape().dims();
        let batch = input_dims[0].get();
        let input_h = input_dims[1].get();
        let input_w = input_dims[2].get();
        let channels = input_dims[3].get();
        let [out_h, out_w] = output_hw;
        let [kernel_h, kernel_w] = kernel_hw;
        let [stride_h, stride_w] = stride_hw;
        let [dilation_h, dilation_w] = dilation_hw;
        assert!(
            out_h > 0 && out_w > 0,
            "im2col output shape must be non-zero"
        );
        assert!(
            kernel_h > 0 && kernel_w > 0,
            "im2col kernel shape must be non-zero"
        );
        assert!(
            stride_h > 0 && stride_w > 0,
            "im2col stride must be non-zero"
        );
        assert!(
            dilation_h > 0 && dilation_w > 0,
            "im2col dilation must be non-zero"
        );
        let used_h = out_h
            .checked_sub(1)
            .and_then(|value| value.checked_mul(stride_h))
            .and_then(|value| {
                kernel_h
                    .checked_sub(1)
                    .and_then(|kernel| kernel.checked_mul(dilation_h))
                    .and_then(|kernel| value.checked_add(kernel))
            })
            .and_then(|value| value.checked_add(1))
            .expect("im2col height extent overflow");
        let used_w = out_w
            .checked_sub(1)
            .and_then(|value| value.checked_mul(stride_w))
            .and_then(|value| {
                kernel_w
                    .checked_sub(1)
                    .and_then(|kernel| kernel.checked_mul(dilation_w))
                    .and_then(|kernel| value.checked_add(kernel))
            })
            .and_then(|value| value.checked_add(1))
            .expect("im2col width extent overflow");
        assert!(used_h <= input_h, "im2col view exceeds input height");
        assert!(used_w <= input_w, "im2col view exceeds input width");
        let shape = Shape::new([
            batch
                .checked_mul(out_h)
                .and_then(|value| value.checked_mul(out_w))
                .expect("im2col M dimension overflow"),
            kernel_h
                .checked_mul(kernel_w)
                .and_then(|value| value.checked_mul(channels))
                .expect("im2col K dimension overflow"),
        ]);
        let strides = self.view.layout.strides().values();
        let map = Im2ColNhwcMap {
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            channels,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            batch_stride: strides[0],
            row_stride: strides[1],
            col_stride: strides[2],
            channel_stride: strides[3],
        };
        Storage {
            view: StorageView {
                buffer: self.view.buffer,
                offset: self.view.offset,
                layout: Layout::contiguous(MemoryLevel::Storage, shape),
                dynamic_offsets: vec![None, None],
                index_map: Some(StorageIndexMap::Im2ColNhwc(map)),
            },
            _ty: PhantomData,
        }
    }
}

impl<T, const R: usize> Storage<T, R> {
    pub fn view(&self) -> &StorageView {
        &self.view
    }
}

pub struct TileBlock<'a, const BLOCK: usize> {
    program: &'a mut Program,
    grid: [u32; 3],
    stores: Vec<TileStoreProgramOp>,
}

impl<const BLOCK: usize> TileBlock<'_, BLOCK> {
    pub fn program_id(&self, axis: WorkgroupAxis) -> ScalarIndex {
        ScalarIndex {
            expr: TileIndexExpr::ProgramId(axis),
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

    pub fn reduce_sum(&mut self, value: Tile<BLOCK>) -> Scalar {
        self.reduce(TileReduceOp::Sum, value)
    }

    pub fn reduce_max(&mut self, value: Tile<BLOCK>) -> Scalar {
        self.reduce(TileReduceOp::Max, value)
    }

    pub fn reduce_min(&mut self, value: Tile<BLOCK>) -> Scalar {
        self.reduce(TileReduceOp::Min, value)
    }

    pub fn loop_reduce_sum(&mut self, iterations: u32, value: Tile<BLOCK>) -> Scalar {
        self.loop_reduce(TileReduceOp::Sum, iterations, value)
    }

    pub fn loop_reduce_max(&mut self, iterations: u32, value: Tile<BLOCK>) -> Scalar {
        self.loop_reduce(TileReduceOp::Max, iterations, value)
    }

    pub fn loop_reduce_min(&mut self, iterations: u32, value: Tile<BLOCK>) -> Scalar {
        self.loop_reduce(TileReduceOp::Min, iterations, value)
    }

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

    pub fn group_reduce_sum<const GROUP: usize>(&mut self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        self.group_reduce::<GROUP>(TileReduceOp::Sum, value)
    }

    pub fn group_reduce_max<const GROUP: usize>(&mut self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        self.group_reduce::<GROUP>(TileReduceOp::Max, value)
    }

    pub fn group_reduce_min<const GROUP: usize>(&mut self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        self.group_reduce::<GROUP>(TileReduceOp::Min, value)
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

    pub fn store<T>(&mut self, address: Address<T, BLOCK>, value: Tile<BLOCK>, mask: Mask<BLOCK>) {
        self.stores.push(TileStoreProgramOp {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        });
    }

    pub fn store_erased(
        &mut self,
        address: ErasedAddress<BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.stores.push(TileStoreProgramOp {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        });
    }
}

pub struct Address<T, const N: usize> {
    view: StorageView,
    row: TileIndexExpr,
    col: TileIndexExpr,
    _ty: PhantomData<T>,
}

pub struct ErasedAddress<const N: usize> {
    view: StorageView,
    row: TileIndexExpr,
    col: TileIndexExpr,
}

#[derive(Clone)]
pub struct LaneTile2d<const ROWS: usize, const COLS: usize, const N: usize> {
    row: Range<N>,
    col: Range<N>,
}

impl<const ROWS: usize, const COLS: usize, const N: usize> LaneTile2d<ROWS, COLS, N> {
    pub fn row(&self) -> Range<N> {
        self.row.clone()
    }

    pub fn col(&self) -> Range<N> {
        self.col.clone()
    }
}

pub trait IntoIndex<const N: usize> {
    fn into_index(self) -> TileIndexExpr;
}

#[derive(Clone)]
pub struct ScalarIndex {
    expr: TileIndexExpr,
}

#[derive(Clone)]
pub struct Range<const N: usize> {
    expr: TileIndexExpr,
}

impl<const N: usize> IntoIndex<N> for ScalarIndex {
    fn into_index(self) -> TileIndexExpr {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &ScalarIndex {
    fn into_index(self) -> TileIndexExpr {
        self.expr.clone()
    }
}

impl<const N: usize> IntoIndex<N> for Range<N> {
    fn into_index(self) -> TileIndexExpr {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &Range<N> {
    fn into_index(self) -> TileIndexExpr {
        self.expr.clone()
    }
}

impl<const N: usize> IntoIndex<N> for u32 {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Literal(self)
    }
}

impl<const N: usize> IntoIndex<N> for Tile<N> {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Value(Box::new(self.expr))
    }
}

impl<const N: usize> IntoIndex<N> for &Tile<N> {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Value(Box::new(self.expr.clone()))
    }
}

impl<const N: usize> Range<N> {
    pub fn lt(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Lt, value)
    }

    pub fn le(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Le, value)
    }

    pub fn gt(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Gt, value)
    }

    pub fn ge(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Ge, value)
    }

    pub fn eq(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Eq, value)
    }

    fn compare(&self, op: TileCompareOp, value: u32) -> Mask<N> {
        Mask {
            expr: TileMaskExpr::Compare {
                op,
                left: self.expr.clone(),
                right: TileIndexExpr::Literal(value),
            },
        }
    }
}

impl ScalarIndex {
    pub fn lt<const N: usize>(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Lt, value)
    }

    pub fn le<const N: usize>(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Le, value)
    }

    pub fn gt<const N: usize>(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Gt, value)
    }

    pub fn ge<const N: usize>(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Ge, value)
    }

    pub fn eq<const N: usize>(&self, value: u32) -> Mask<N> {
        self.compare(TileCompareOp::Eq, value)
    }

    fn compare<const N: usize>(&self, op: TileCompareOp, value: u32) -> Mask<N> {
        Mask {
            expr: TileMaskExpr::Compare {
                op,
                left: self.expr.clone(),
                right: TileIndexExpr::Literal(value),
            },
        }
    }
}

impl Add<u32> for ScalarIndex {
    type Output = ScalarIndex;

    fn add(self, rhs: u32) -> Self::Output {
        ScalarIndex {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(TileIndexExpr::Literal(rhs))),
        }
    }
}

impl Add<ScalarIndex> for ScalarIndex {
    type Output = ScalarIndex;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        ScalarIndex {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl Mul<u32> for ScalarIndex {
    type Output = ScalarIndex;

    fn mul(self, rhs: u32) -> Self::Output {
        ScalarIndex {
            expr: TileIndexExpr::Mul(Box::new(self.expr), rhs),
        }
    }
}

impl<const N: usize> Add<u32> for Range<N> {
    type Output = Range<N>;

    fn add(self, rhs: u32) -> Self::Output {
        Range {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(TileIndexExpr::Literal(rhs))),
        }
    }
}

impl<const N: usize> Mul<u32> for Range<N> {
    type Output = Range<N>;

    fn mul(self, rhs: u32) -> Self::Output {
        Range {
            expr: TileIndexExpr::Mul(Box::new(self.expr), rhs),
        }
    }
}

impl<const N: usize> Div<u32> for Range<N> {
    type Output = Range<N>;

    fn div(self, rhs: u32) -> Self::Output {
        assert!(rhs > 0, "tile index divisor must be non-zero");
        Range {
            expr: TileIndexExpr::Div(Box::new(self.expr), rhs),
        }
    }
}

impl<const N: usize> Rem<u32> for Range<N> {
    type Output = Range<N>;

    fn rem(self, rhs: u32) -> Self::Output {
        assert!(rhs > 0, "tile index modulus must be non-zero");
        Range {
            expr: TileIndexExpr::Mod(Box::new(self.expr), rhs),
        }
    }
}

impl<const N: usize> Add<Range<N>> for ScalarIndex {
    type Output = Range<N>;

    fn add(self, rhs: Range<N>) -> Self::Output {
        Range {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl<const N: usize> Add<ScalarIndex> for Range<N> {
    type Output = Range<N>;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        Range {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

#[derive(Clone)]
pub struct Mask<const N: usize> {
    expr: TileMaskExpr,
}

impl<const N: usize> Mask<N> {
    pub fn all() -> Self {
        Self {
            expr: TileMaskExpr::True,
        }
    }

    pub fn and(self, rhs: Self) -> Self {
        Self {
            expr: TileMaskExpr::And(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

#[derive(Clone)]
pub struct Scalar {
    expr: TileScalarExpr,
}

impl Scalar {
    pub fn literal(value: f32) -> Self {
        Self {
            expr: TileScalarExpr::Literal(TileLiteral::F32(F32Bits::new(value))),
        }
    }
}

#[derive(Clone)]
pub struct Tile<const N: usize> {
    expr: TileExpr,
}

impl<const N: usize> Tile<N> {
    pub fn literal(value: TileLiteral) -> Self {
        Self {
            expr: TileExpr::Literal(value),
        }
    }

    pub fn from_index(index: impl IntoIndex<N>) -> Self {
        Self {
            expr: TileExpr::Index(index.into_index()),
        }
    }

    pub fn exp(self) -> Self {
        self.unary(TileUnaryOp::Exp)
    }

    pub fn unary(self, op: TileUnaryOp) -> Self {
        Self {
            expr: TileExpr::Unary {
                op,
                value: Box::new(self.expr),
            },
        }
    }

    pub fn exp2(self) -> Self {
        self.unary(TileUnaryOp::Exp2)
    }

    pub fn cast(self, to: crate::ElementType) -> Self {
        Self {
            expr: TileExpr::Cast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    pub fn select(condition: Self, accept: Self, reject: Self) -> Self {
        Self {
            expr: TileExpr::Select {
                condition: Box::new(condition.expr),
                accept: Box::new(accept.expr),
                reject: Box::new(reject.expr),
            },
        }
    }

    pub fn compare(op: TileCompareOp, left: Self, right: Self, output: crate::ElementType) -> Self {
        Self {
            expr: TileExpr::Compare {
                op,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                output,
            },
        }
    }

    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        Tile {
            expr: TileExpr::Binary {
                op,
                left: Box::new(self.expr),
                right: Box::new(rhs.expr),
            },
        }
    }

    pub fn max(self, rhs: Self) -> Self {
        self.binary(TileBinaryOp::Max, rhs)
    }

    pub fn min(self, rhs: Self) -> Self {
        self.binary(TileBinaryOp::Min, rhs)
    }
}

impl<const N: usize> From<Scalar> for Tile<N> {
    fn from(value: Scalar) -> Self {
        Self {
            expr: TileExpr::Scalar(value.expr),
        }
    }
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<const N: usize> $trait for Tile<N> {
            type Output = Tile<N>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.binary($op, rhs)
            }
        }

        impl<const N: usize> $trait<Scalar> for Tile<N> {
            type Output = Tile<N>;

            fn $method(self, rhs: Scalar) -> Self::Output {
                Tile {
                    expr: TileExpr::Binary {
                        op: $op,
                        left: Box::new(self.expr),
                        right: Box::new(TileExpr::Scalar(rhs.expr)),
                    },
                }
            }
        }
    };
}

impl_tile_binary!(Add, add, TileBinaryOp::Add);
impl_tile_binary!(Sub, sub, TileBinaryOp::Sub);
impl_tile_binary!(Mul, mul, TileBinaryOp::Mul);
impl_tile_binary!(Div, div, TileBinaryOp::Div);
impl_tile_binary!(Rem, rem, TileBinaryOp::Rem);

fn matrix_shape(layout: &Layout) -> [u32; 2] {
    assert_eq!(layout.shape().rank(), 2, "matrix operands must be rank-2");
    [
        layout.shape().dims()[0].get(),
        layout.shape().dims()[1].get(),
    ]
}

fn qmatmul_coop_subgroups(m: u32, n: u32) -> Option<u32> {
    const COOP: u32 = 8;
    if m == 0 || n == 0 || !m.is_multiple_of(COOP) || !n.is_multiple_of(COOP) {
        return None;
    }
    let tile_rows = m / COOP;
    let tile_cols = n / COOP;
    if tile_rows * tile_cols <= 16 {
        return Some(1);
    }
    if m == 32 && n.is_multiple_of(32) {
        let subgroups = n / 32;
        if (2..=8).contains(&subgroups) {
            return Some(subgroups);
        }
    }
    if n >= m && m <= 64 && n.is_multiple_of(16) {
        let subgroups = n / 16;
        if (2..=8).contains(&subgroups) {
            return Some(subgroups);
        }
    }
    if n <= 64 && m.is_multiple_of(16) {
        let subgroups = m / 16;
        if (2..=8).contains(&subgroups) {
            return Some(subgroups);
        }
    }
    if m == 128 && n == 128 {
        return Some(16);
    }
    if m.is_multiple_of(32) && n.is_multiple_of(32) {
        let subgroups = (m / 32).checked_mul(n / 32)?;
        if (2..=16).contains(&subgroups) {
            return Some(subgroups);
        }
    }
    None
}

fn cooperative_store_layout_supported(layout: &Layout) -> bool {
    layout.shape().rank() == 2
        && layout.strides().rank() == 2
        && (layout.strides().values()[0] == 1 || layout.strides().values()[1] == 1)
}
