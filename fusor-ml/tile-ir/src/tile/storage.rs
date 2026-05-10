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

pub struct Storage<T, const R: usize> {
    pub(crate) view: StorageView,
    pub(super) _ty: PhantomData<T>,
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

impl<T> Storage<T, 1> {
    pub fn at<const N: usize>(&self, index: impl IntoIndex<N>) -> LinearAddress<T, N> {
        LinearAddress {
            view: self.view.clone(),
            index: index.into_index(),
            _ty: PhantomData,
        }
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
