use std::{
    fmt::{Debug, Display},
    ops::Range,
};

#[cfg(feature = "graphvis")]
use tabbycat::Graph;
use wgpu::COPY_BUFFER_ALIGNMENT;

use crate::{
    Device, FlashAttentionInputs, FlashAttentionOperation, MatMulOperation, MatMulParams,
    ReduceFunction, ReduceOperation,
    compute_graph::NodeIndex,
    map_layout::MapLayoutOperation,
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryOperation, NaryScalar},
    quantized::QMatrix,
    quantized::matmul::{ElementwiseEpilogue, QMatMulOperation},
    resize::ResizeOperation,
    rms_norm::RmsNormOperation,
    slice_assign::SliceAssignOperation,
};

pub use fusor_types::TensorSlice;

mod eager_data;
mod layout_info;
mod lazy_data;
mod sampling;
mod traits;

pub use traits::{DataType, DataTypeEnum, FloatDataType};

pub(crate) use eager_data::TensorData;
pub(crate) use layout_info::{TensorInfo, TensorLayoutInfo};
pub(crate) use lazy_data::LazyTensorData;

pub struct Tensor {
    pub(crate) data: LazyTensorData,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {:?}", self.datatype(), self.shape())
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({} x {:?})", self.datatype(), self.shape())
    }
}

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Self {
            data: LazyTensorData::new(value),
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl Tensor {
    /// Resolve the current tensor value on device and return a fresh leaf tensor
    /// that no longer carries the original compute graph history.
    pub fn detach(&self) -> Self {
        let (data, _) = self.data.materialize();
        Self {
            data: LazyTensorData::new(data),
        }
    }
}

impl Tensor {
    pub fn q_mat_mul_add2(&self, other: &QMatrix, first: &Self, second: &Self) -> Self {
        // When M is unaligned, pad the activation, the two residuals, and
        // the matmul output back to a multiple of 64/128 so the matmul
        // kernel takes the coop-tile fast path. The slice at the end
        // narrows everything back to the caller's shape.
        if self.rank() >= 2 {
            let in_shape = self.shape();
            let m_axis = self.rank() - 2;
            let m = in_shape[m_axis];
            let n = other.shape()[0];
            if let Some(padded_m) = crate::quantized::matmul::qmatmul_m_pad_target_pub(m, n) {
                let mut padded_in_shape = in_shape.to_vec();
                padded_in_shape[m_axis] = padded_m;
                let padded_self = self.resize(padded_in_shape);

                // first / second are shaped like the matmul output:
                // replace dim R-2 with padded_m, keep last dim = N.
                let first_shape = first.shape();
                let mut padded_out_shape = first_shape.to_vec();
                padded_out_shape[m_axis] = padded_m;
                let padded_first = first.resize(&padded_out_shape);
                let padded_second = second.resize(&padded_out_shape);

                let padded_result =
                    padded_self.q_mat_mul_add2(other, &padded_first, &padded_second);

                // Narrow result back to original M via a layout view.
                let result_shape = padded_result.shape();
                let specs: Vec<crate::StrideSpec> = (0..padded_result.rank())
                    .map(|i| {
                        if i == m_axis {
                            crate::StrideSpec::dim(i, m)
                        } else {
                            crate::StrideSpec::dim(i, result_shape[i])
                        }
                    })
                    .collect();
                return padded_result.restride(specs);
            }
        }

        let mut operation = QMatMulOperation::new(
            DataTypeEnum::F32,
            self.shape(),
            self.data.key,
            other.clone(),
        );

        assert_eq!(
            operation.out_shape.as_ref(),
            first.shape(),
            "first residual shape must match q_mat_mul output shape"
        );
        assert_eq!(
            operation.out_shape.as_ref(),
            second.shape(),
            "second residual shape must match q_mat_mul output shape"
        );

        let dtype = DataTypeEnum::F32;
        let rank = self.rank();
        let matmul = NaryExpr::input(0, rank);
        let first_residual = NaryExpr::input(1, rank);
        let second_residual = NaryExpr::input(2, rank);
        let sum = NaryExpr::Op {
            children: vec![matmul, first_residual],
            function: NaryFunction::binary(
                Some("add".to_string()),
                NaryOp::Add,
                dtype,
                dtype,
                dtype,
            ),
        };
        let expression = NaryExpr::Op {
            children: vec![sum, second_residual],
            function: NaryFunction::binary(
                Some("add".to_string()),
                NaryOp::Add,
                dtype,
                dtype,
                dtype,
            ),
        };
        operation.post_element_wise_expr = Some(ElementwiseEpilogue {
            expression,
            extras: vec![first.data.key, second.data.key],
            input_datatype: dtype,
            output_datatype: dtype,
        });

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    pub fn q_mat_mul_paired_silu_product(&self, other: &QMatrix) -> Self {
        assert_eq!(
            other.shape().len(),
            2,
            "paired q_mat_mul requires 2D weight tensor, got {}D",
            other.shape().len()
        );
        assert!(
            other.shape()[0].is_multiple_of(2),
            "paired q_mat_mul requires an even output dimension"
        );
        // Pad activation M to unlock the coop-tile matmul path; narrow
        // the output back to the original M with a layout view.
        if self.rank() >= 2 {
            let in_shape = self.shape();
            let m_axis = self.rank() - 2;
            let m = in_shape[m_axis];
            let n = other.shape()[0];
            if let Some(padded_m) = crate::quantized::matmul::qmatmul_m_pad_target_pub(m, n) {
                let mut padded_in_shape = in_shape.to_vec();
                padded_in_shape[m_axis] = padded_m;
                let padded_self = self.resize(padded_in_shape);
                let padded_result = padded_self.q_mat_mul_paired_silu_product(other);
                let result_shape = padded_result.shape();
                let specs: Vec<crate::StrideSpec> = (0..padded_result.rank())
                    .map(|i| {
                        if i == m_axis {
                            crate::StrideSpec::dim(i, m)
                        } else {
                            crate::StrideSpec::dim(i, result_shape[i])
                        }
                    })
                    .collect();
                return padded_result.restride(specs);
            }
        }
        let pair_len = other.shape()[0] / 2;
        let dtype = DataTypeEnum::F32;
        let gate = NaryExpr::input(0, 1);
        let up = NaryExpr::input(1, 1);
        let neg_gate = NaryExpr::Op {
            children: vec![gate.clone()],
            function: NaryFunction::unary(Some("neg".to_string()), NaryOp::Neg, dtype, dtype),
        };
        let exp_neg_gate = NaryExpr::Op {
            children: vec![neg_gate],
            function: NaryFunction::unary(Some("exp".to_string()), NaryOp::Exp, dtype, dtype),
        };
        let one_plus_exp = NaryExpr::Op {
            children: vec![exp_neg_gate],
            function: NaryFunction::unary(
                Some("add_const".to_string()),
                NaryOp::AddConst(NaryScalar::F32(1.0)),
                dtype,
                dtype,
            ),
        };
        let silu = NaryExpr::Op {
            children: vec![gate, one_plus_exp],
            function: NaryFunction::binary(
                Some("div".to_string()),
                NaryOp::Div,
                dtype,
                dtype,
                dtype,
            ),
        };
        let expression = NaryExpr::Op {
            children: vec![silu, up],
            function: NaryFunction::binary(
                Some("mul".to_string()),
                NaryOp::Mul,
                dtype,
                dtype,
                dtype,
            ),
        };
        let epilogue =
            fusor_tile_ir_kernels::PairedEpilogue::with_extras("silu_mul", 0, move |tiles| {
                let inputs = [
                    (tiles[0].clone(), DataTypeEnum::F32),
                    (tiles[1].clone(), DataTypeEnum::F32),
                ];
                crate::nary_direct::eval_nary_expr_on_tiles(&expression, &inputs).0
            });
        let operation = QMatMulOperation::new_paired(
            DataTypeEnum::F32,
            self.shape(),
            self.data.key,
            other.clone(),
            pair_len,
            epilogue,
            Vec::new(),
        );

        Self::from_parts(self.data.q_mat_mul(operation))
    }
}

impl<const R: usize, D, T> fusor_types::FromArray<R, D, T, Device> for Tensor
where
    D: DataType,
    T: fusor_types::IntoFlatArray<D, R>,
{
    fn from_array(data: T, device: &Device) -> Self {
        let flat = data.into_flat_array();
        Tensor::new_inner(device, flat.data.iter(), flat.shape)
    }
}

impl Tensor {
    pub fn new<D: DataType, const R: usize, T>(device: &Device, data: T) -> Self
    where
        Self: fusor_types::FromArray<R, D, T, Device>,
    {
        fusor_types::FromArray::from_array(data, device)
    }

    pub fn from_slice<D: DataType>(
        device: &Device,
        shape: impl AsRef<[usize]>,
        data: &[D],
    ) -> Self {
        let shape = shape.as_ref();
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        Tensor::new_inner(device, data.iter(), shape)
    }

    pub fn splat<D: DataType>(device: &Device, value: D, shape: impl AsRef<[usize]>) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_splat(
            device,
            shape.as_ref(),
            value,
        )))
    }

    /// Alias for [`Tensor::splat`]
    pub fn full<D: DataType>(device: &Device, value: D, shape: impl AsRef<[usize]>) -> Self {
        Self::splat(device, value, shape)
    }

    pub(crate) fn from_parts(data: LazyTensorData) -> Self {
        Self { data }
    }

    fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: impl AsRef<[usize]>,
    ) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_inner(
            device,
            data,
            shape.as_ref(),
        )))
    }

    pub(crate) async fn as_slice_from_tensor_data<const R: usize, D: DataType>(
        tensor: &TensorData,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        assert_eq!(tensor.datatype(), D::DATA_TYPE);
        assert_eq!(tensor.layout().shape().len(), R);
        let buffer = tensor.buffer();
        let device = tensor.device.wgpu_device();
        let queue = tensor.device.wgpu_queue();
        let size = buffer.size();

        // Create a staging buffer for reading
        let download = device.create_buffer(&wgpu::BufferDescriptor {
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: None,
        });

        // Copy data to staging buffer
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &download, 0, size);
        queue.submit(Some(encoder.finish()));

        // Map the staging buffer using map_async which correctly uses WasmNotSend
        let (sender, receiver) = futures_channel::oneshot::channel();
        download
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                _ = sender.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        tensor.device.poll_wait();

        receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        // Get the mapped view
        let view = download.slice(..).get_mapped_range();
        Ok(TensorSlice::new(
            MappedBuffer { view },
            tensor.layout().clone(),
        ))
    }

    /// Synchronously dispatch and wait for GPU completion using device.poll().
    /// More efficient than the async version for benchmarking since it avoids
    /// the on_submitted_work_done callback overhead.
    pub fn materialize_sync(&self) {
        self.data.materialize();
        self.device().poll_wait();
    }

    #[track_caller]
    pub fn materialize(&self) -> impl Future<Output = ()> + 'static {
        let data = self.data.clone();
        let device = self.device().clone();
        #[cfg(feature = "extra_assertions")]
        let caller = std::panic::Location::caller();
        async move {
            #[cfg_attr(not(feature = "extra_assertions"), allow(unused_variables))]
            let (data, _) = data.materialize();
            #[cfg(not(target_arch = "wasm32"))]
            device.poll_wait();
            #[cfg(target_arch = "wasm32")]
            {
                let (sender, receiver) = futures_channel::oneshot::channel();
                device.wgpu_queue().on_submitted_work_done(|| {
                    _ = sender.send(());
                });
                let _ = receiver.await;
            }
            #[cfg(feature = "extra_assertions")]
            {
                let mut contains_non_finite = false;
                if data.datatype() == DataTypeEnum::F32 && data.layout().rank() == 1 {
                    let data: TensorSlice<1, f32, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                } else if data.datatype() == DataTypeEnum::F16 && data.layout().rank() == 1 {
                    let data: TensorSlice<1, half::f16, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                }

                if contains_non_finite {
                    tracing::warn!(
                        "Tensor materialized at {} contains non-finite values. This may lead to unexpected behavior.",
                        caller
                    );
                }
            }
        }
    }

    /// How many kernel calls are needed to fully resolve this tensor
    pub fn count_kernels_to_resolve(&self) -> usize {
        let (data, count) = self.data.materialize();
        #[cfg(not(target_arch = "wasm32"))]
        data.device().poll_wait();
        count
    }

    pub async fn as_slice<const R: usize, D: DataType>(
        &self,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        self.assert_rank::<R>();
        self.assert_datatype::<D>();
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let (tensor, _) = self.data.materialize();
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Materialized tensor in {:?}", start_time.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let out = Self::as_slice_from_tensor_data(&tensor).await;
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Downloaded tensor in {:?}", start_time.elapsed());
        out
    }

    pub async fn to_scalar<D: DataType>(&self) -> Result<D, wgpu::BufferAsyncError> {
        let slice = self.as_slice::<0, D>().await?;
        Ok(slice.as_scalar())
    }

    pub fn debug_assert_real(self) -> Self {
        #[cfg(debug_assertions)]
        {
            use pollster::FutureExt as _;
            if self.rank() == 1 {
                match self.datatype() {
                    DataTypeEnum::F32 => {
                        let as_slice = self.as_slice::<1, f32>().block_on().unwrap();
                        for item in as_slice.as_slice() {
                            assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
                        }
                    }
                    DataTypeEnum::F16 => {
                        let as_slice = self.as_slice::<1, half::f16>().block_on().unwrap();
                        for item in as_slice.as_slice() {
                            assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
                        }
                    }
                    DataTypeEnum::U32 => {}
                }
            }
        }
        self
    }

    pub(crate) fn unary_nary<D2: DataType>(&self, function: NaryFunction) -> Tensor {
        Tensor::from_parts(self.data.unary_nary(function))
    }

    pub(crate) fn unary_nary_dtype(&self, function: NaryFunction) -> Tensor {
        Tensor::from_parts(self.data.unary_nary(function))
    }

    pub(crate) fn binary_nary(&self, other: &Self, function: NaryFunction) -> Self {
        // Keep one storage input while preserving the binary expression.
        if self.data.key == other.data.key {
            let device = self.device().clone();
            let mut info = self.data.info.clone();
            info.datatype = function.output_type;
            let rank = self.shape().len();
            let nary = NaryOperation {
                inputs: vec![self.data.key],
                expression: NaryExpr::Op {
                    children: vec![NaryExpr::input(0, rank), NaryExpr::input(0, rank)],
                    function,
                },
                shape: self.shape().into(),
                output_datatype: info.datatype,
            };
            let key = device.compute_graph().create_nary(nary);
            return Self::from_parts(LazyTensorData::from_parts(device, info, key));
        }

        assert_eq!(self.shape(), other.shape());
        Self::from_parts(
            self.data
                .binary_nary(other.data.key, function, self.shape()),
        )
    }

    pub(crate) fn add_mat_mul(&self, other: &Self, parameters: Option<MatMulParams>) -> Self {
        let operation = MatMulOperation::new(
            self.datatype(),
            self.data.key,
            other.data.key,
            self.shape(),
            other.shape(),
            parameters,
            &self.data.device,
        );

        Self::from_parts(self.data.mat_mul(operation))
    }

    pub(crate) fn add_q_mat_mul(&self, other: &QMatrix) -> Self {
        let operation =
            QMatMulOperation::new(self.datatype(), self.shape(), self.data.key, other.clone());

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    pub(crate) fn add_resize(&self, op: ResizeOperation) -> Tensor {
        Tensor::from_parts(self.data.resize(op))
    }

    pub(crate) fn add_slice_assign(
        &self,
        other: &Self,
        slices: impl Into<Box<[Range<usize>]>>,
    ) -> Self {
        let input_shape: Box<[usize]> = self.shape().to_vec().into_boxed_slice();
        let op =
            SliceAssignOperation::new(self.data.key, other.data.key, slices.into(), input_shape);
        Self::from_parts(self.data.slice_assign(op))
    }

    #[doc(hidden)]
    pub fn slice_assign_in_place(
        &self,
        slices: impl Into<Box<[Range<usize>]>>,
        value: &Self,
    ) -> Self {
        let input_shape: Box<[usize]> = self.shape().to_vec().into_boxed_slice();
        let op = SliceAssignOperation::new_in_place(
            self.data.key,
            value.data.key,
            slices.into(),
            input_shape,
        );
        Self::from_parts(self.data.slice_assign(op))
    }

    pub(crate) fn reduce(&self, function: ReduceFunction, dim: usize) -> Tensor {
        Tensor::from_parts(self.data.reduce(ReduceOperation::new(
            self.data.key,
            function,
            dim,
            self.shape(),
        )))
    }

    pub(crate) fn add_map_layout(&self, op: MapLayoutOperation) -> Tensor {
        Tensor::from_parts(self.data.map_layout(op))
    }

    /// Return the compute-graph node index for this tensor.
    pub fn key(&self) -> NodeIndex {
        self.data.key
    }

    pub fn shape(&self) -> &[usize] {
        self.data.info.shape()
    }

    pub fn shape_array<const R: usize>(&self) -> &[usize; R] {
        self.shape().try_into().unwrap_or_else(|_| {
            panic!(
                "Expected a tensor of rank {R}, found shape: {:?}",
                self.shape()
            )
        })
    }

    pub fn assert_rank<const R: usize>(&self) {
        assert_eq!(self.rank(), R, "unexpected tensor rank");
    }

    pub fn assert_datatype<D: DataType>(&self) {
        assert_eq!(self.datatype(), D::DATA_TYPE, "unexpected tensor dtype");
    }

    pub fn rank(&self) -> usize {
        self.data.info.rank()
    }

    pub fn datatype(&self) -> DataTypeEnum {
        self.data.info.datatype()
    }

    pub(crate) fn try_rms_norm_direct(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Option<Self> {
        if self.datatype() != DataTypeEnum::F32 {
            return None;
        }
        let operation = RmsNormOperation::new(
            self.data.key,
            weight.data.key,
            bias.map(|bias| bias.data.key),
            self.shape(),
            eps,
        );
        Some(Self::from_parts(self.data.rms_norm(operation)))
    }

    pub(crate) fn try_rms_norm_residual_direct(
        &self,
        residual: &Self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Option<Self> {
        if self.datatype() != DataTypeEnum::F32 || self.shape() != residual.shape() {
            return None;
        }
        let operation = RmsNormOperation::new_with_residual(
            self.data.key,
            residual.data.key,
            weight.data.key,
            bias.map(|bias| bias.data.key),
            self.shape(),
            eps,
        );
        Some(Self::from_parts(self.data.rms_norm(operation)))
    }

    pub(crate) fn try_flash_attention_direct(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor>,
    ) -> Option<Self> {
        self.try_flash_attention_direct_inner(k, v, scale, mask, false)
    }

    pub(crate) fn try_flash_attention_direct_causal(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
    ) -> Option<Self> {
        self.try_flash_attention_direct_inner(k, v, scale, None, true)
    }

    fn try_flash_attention_direct_inner(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor>,
        causal: bool,
    ) -> Option<Self> {
        if self.rank() != 4 || !matches!(self.datatype(), DataTypeEnum::F32 | DataTypeEnum::F16) {
            return None;
        }
        if causal && mask.is_some() {
            return None;
        }
        // The streaming flash attention kernel emits a separate
        // monomorphization per hardware subgroup width and relies on
        // `subgroup_reduce_*`, so it can only target devices where we know the
        // effective subgroup width.
        self.data.device.fixed_width_subgroup_size()?;
        let q_shape = self.shape();
        let k_shape = k.shape();
        const MIN_DECODE_KV_SEQ: usize = 32;
        let is_decode_candidate = q_shape[2] == 1
            && q_shape[3] == 128
            && mask.is_none()
            && !causal
            && self.datatype() == DataTypeEnum::F32;
        if is_decode_candidate && k_shape[2] < MIN_DECODE_KV_SEQ {
            return None;
        }
        let v_shape = v.shape();
        if q_shape[0] != k_shape[0]
            || q_shape[0] != v_shape[0]
            || k_shape[1] != v_shape[1]
            || k_shape[2] != v_shape[2]
            || q_shape[3] != k_shape[3]
            || q_shape[3] != v_shape[3]
            || q_shape[0] == 0
            || q_shape[1] == 0
            || q_shape[2] == 0
            || k_shape[1] == 0
            || !q_shape[1].is_multiple_of(k_shape[1])
            || q_shape[3] == 0
            || k_shape[2] == 0
        {
            return None;
        }
        if let Some(mask) = mask
            && mask.shape() != [q_shape[2], k_shape[2]]
        {
            return None;
        }
        if causal && q_shape[2] != k_shape[2] {
            // Causal optimisation only kicks in for self-attention prefill
            // where q_seq_len == kv_seq_len. Other shapes (e.g. cached decode)
            // fall back to the masked path.
            return None;
        }
        let batch = u32::try_from(q_shape[0]).ok()?;
        let num_heads = u32::try_from(q_shape[1]).ok()?;
        let q_seq_len = u32::try_from(q_shape[2]).ok()?;
        let head_dim = u32::try_from(q_shape[3]).ok()?;
        let row_dispatch = batch.checked_mul(num_heads)?.checked_mul(q_seq_len)?;
        let x_dispatch = head_dim.div_ceil(8);
        let max_dispatch = self
            .data
            .device
            .limits()
            .max_compute_workgroups_per_dimension;
        if x_dispatch > max_dispatch || row_dispatch > max_dispatch {
            return None;
        }

        let operation = FlashAttentionOperation::new(FlashAttentionInputs {
            q: self.data.key,
            k: k.data.key,
            v: v.data.key,
            mask: mask.map(|mask| mask.data.key),
            q_shape,
            k_shape,
            v_shape,
            scale,
            input_dtype: self.datatype(),
            causal,
        });
        Some(Self::from_parts(self.data.flash_attention(operation)))
    }

    pub fn device(&self) -> &Device {
        &self.data.device
    }

    #[cfg(feature = "graphvis")]
    pub fn graphvis(&self) -> Graph {
        self.data.graphvis()
    }

    pub(crate) fn data(&self) -> &LazyTensorData {
        &self.data
    }
}

/// A buffer that has been mapped for reading. Wraps a wgpu BufferView and provides
/// access to its mapped contents.
pub struct MappedBuffer {
    view: wgpu::BufferView,
}

impl std::ops::Deref for MappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.view.as_ref()
    }
}

pub(crate) fn padded_tensor_size(size: u64) -> u64 {
    // Valid vulkan usage is
    // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
    // 2. buffer size must be greater than 0.
    // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;

    ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}
