use std::{any::Any, hash::Hash, sync::OnceLock};

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::{FxHashMap, FxHasher};

use crate::{
    DataTypeEnum, Device, Layout,
    compute_graph::{ComputeGraphInner, GraphOperation, NodeIndex},
    kernel_selection::KernelDeviceCaps,
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::{TensorData, TensorLayoutInfo},
    visit_tiled::distribute_workgroups,
};

const SOFTMAX_BLOCKS: [u32; 3] = [128, 512, 1024];
const SOFTMAX_MODULE_CACHE_SIZE: usize = 128;

fn softmax_module_cache() -> &'static kernel_backend::ModuleCache {
    static CACHE: OnceLock<kernel_backend::ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| kernel_backend::module_cache(SOFTMAX_MODULE_CACHE_SIZE))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum SoftmaxKernelVariant {
    Single,
    Partials,
    Reduce,
    Write,
}

struct SoftmaxDirectKernelVariant;

fn block_supported(block: u32, caps: KernelDeviceCaps) -> bool {
    block <= caps.max_compute_invocations_per_workgroup
        && block <= caps.max_compute_workgroup_size_x
}

fn choose_softmax_block(axis_len: u32, caps: KernelDeviceCaps) -> Option<u32> {
    for block in SOFTMAX_BLOCKS {
        if axis_len <= block && block_supported(block, caps) {
            return Some(block);
        }
    }

    SOFTMAX_BLOCKS
        .iter()
        .rev()
        .copied()
        .find(|block| block_supported(*block, caps))
}

fn total_elements(shape: &[usize]) -> Option<u32> {
    shape
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))
}

fn tensor_meta(tensor: &TensorData) -> Option<tile_ir_kernels::TensorMeta> {
    let strides = tensor
        .layout()
        .strides()
        .iter()
        .copied()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let offset = tensor.layout().offset().try_into().ok()?;
    Some(tile_ir_kernels::TensorMeta::new(strides, offset))
}

#[derive(Clone, Debug)]
pub(crate) struct SoftmaxOperation {
    input: NodeIndex,
    shape: Box<[usize]>,
    axis: usize,
    datatype: DataTypeEnum,
}

impl SoftmaxOperation {
    pub(crate) fn new(
        input: NodeIndex,
        shape: &[usize],
        axis: usize,
        datatype: DataTypeEnum,
        device: &Device,
    ) -> Option<Self> {
        if axis >= shape.len() || shape.contains(&0) {
            return None;
        }
        if !matches!(datatype, DataTypeEnum::F32 | DataTypeEnum::F16) {
            return None;
        }
        if datatype == DataTypeEnum::F16 && !device.f16_supported() {
            return None;
        }

        let axis_len: u32 = shape[axis].try_into().ok()?;
        let _total = total_elements(shape)?;
        choose_softmax_block(axis_len, KernelDeviceCaps::from_device(device))?;

        Some(Self {
            input,
            shape: shape.into(),
            axis,
            datatype,
        })
    }

    fn meta(
        &self,
        input: &TensorData,
        output: &TensorData,
        dispatch_size: [u32; 3],
        caps: KernelDeviceCaps,
    ) -> Option<tile_ir_kernels::SoftmaxMeta> {
        let shape = self
            .shape
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let axis_len = shape[self.axis];
        let rows = total_elements(&self.shape)?.checked_div(axis_len)?;
        let block = choose_softmax_block(axis_len, caps)?;
        let split_blocks = axis_len.div_ceil(block);

        Some(tile_ir_kernels::SoftmaxMeta {
            shape,
            axis: self.axis.try_into().ok()?,
            rows,
            axis_len,
            block,
            split_blocks,
            input_meta: tensor_meta(input)?,
            output_meta: tensor_meta(output)?,
            dispatch_size,
        })
    }

    fn dispatch_for(&self, total_groups: u32, device: &Device) -> [u32; 3] {
        distribute_workgroups(
            total_groups,
            device.limits().max_compute_workgroups_per_dimension,
        )
    }

    fn split_blocks(&self, device: &Device) -> Option<u32> {
        let axis_len: u32 = self.shape[self.axis].try_into().ok()?;
        let block = choose_softmax_block(axis_len, KernelDeviceCaps::from_device(device))?;
        Some(axis_len.div_ceil(block))
    }

    fn dispatch_softmax<E: tile_ir::Numeric>(
        &self,
        device: &Device,
        input: &TensorData,
        output: &TensorData,
        meta: tile_ir_kernels::SoftmaxMeta,
    ) -> Option<DirectKernel> {
        let dispatch_size = meta.dispatch_size;
        let variant =
            kernel_backend::KernelVariantKey::with_payload::<SoftmaxDirectKernelVariant>(|state| {
                SoftmaxKernelVariant::Single.hash(state);
                meta.block.hash(state);
                meta.split_blocks.hash(state);
                self.datatype.hash(state);
            });
        let inputs = vec![input.clone().into(), output.clone().into()];
        let key = self.kernel_module_key_with_dispatch(variant, None, dispatch_size, &inputs);
        let buffers = vec![input.buffer().clone(), output.buffer().clone()];
        let layout = tile_ir_kernels::linear_storage_layout();

        kernel_backend::dynamic_kernel_from_hashed_ir(
            device.kernel_cache(),
            softmax_module_cache(),
            "softmax",
            key,
            buffers,
            dispatch_size,
            move || {
                let mut kb = tile_ir::KernelBuilder::<()>::new();
                let input_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                let output_ref = tile_ir::KernelTensorRef::new((), layout);
                tile_ir_kernels::softmax::<E, _>(&mut kb, input_ref, output_ref, meta)?;
                Some(kb.finish().0)
            },
        )
    }

    fn dispatch_split_softmax<E: tile_ir::Numeric>(
        &self,
        device: &Device,
        input: &TensorData,
        output: &TensorData,
        meta: tile_ir_kernels::SoftmaxMeta,
    ) -> Option<DirectKernel> {
        let scratch_elements = meta.rows as u64 * meta.split_blocks as u64 * 2;
        let scratch = device.create_buffer(
            scratch_elements * std::mem::size_of::<f32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let global_scratch = device.create_buffer(
            meta.rows as u64 * 2 * std::mem::size_of::<f32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let partial_dispatch_size = meta.dispatch_size;
        let reduce_dispatch_size = self.dispatch_for(meta.rows, device);
        let inputs = vec![input.clone().into(), output.clone().into()];
        let layout = tile_ir_kernels::linear_storage_layout();

        let partial_variant =
            kernel_backend::KernelVariantKey::with_payload::<SoftmaxDirectKernelVariant>(|state| {
                SoftmaxKernelVariant::Partials.hash(state);
                meta.block.hash(state);
                meta.split_blocks.hash(state);
                self.datatype.hash(state);
            });
        let partial_key = self.kernel_module_key_with_dispatch(
            partial_variant,
            None,
            partial_dispatch_size,
            &inputs,
        );
        let partial_buffers = vec![input.buffer().clone(), scratch.clone()];
        let partial_layout = layout.clone();
        let partial_meta = meta.clone();
        let partial = kernel_backend::dynamic_kernel_from_hashed_ir(
            device.kernel_cache(),
            softmax_module_cache(),
            "softmax_partials",
            partial_key,
            partial_buffers,
            partial_dispatch_size,
            move || {
                let mut kb = tile_ir::KernelBuilder::<()>::new();
                let input_ref = tile_ir::KernelTensorRef::new((), partial_layout.clone());
                let scratch_ref = tile_ir::KernelTensorRef::new((), partial_layout);
                tile_ir_kernels::softmax_partials::<E, _>(
                    &mut kb,
                    input_ref,
                    scratch_ref,
                    partial_meta,
                )?;
                Some(kb.finish().0)
            },
        )?;

        let reduce_variant =
            kernel_backend::KernelVariantKey::with_payload::<SoftmaxDirectKernelVariant>(|state| {
                SoftmaxKernelVariant::Reduce.hash(state);
                meta.block.hash(state);
                meta.split_blocks.hash(state);
            });
        let reduce_key = self.kernel_module_key_with_dispatch(
            reduce_variant,
            None,
            reduce_dispatch_size,
            &inputs,
        );
        let reduce_buffers = vec![scratch, global_scratch.clone()];
        let reduce_layout = layout.clone();
        let mut reduce_meta = meta.clone();
        reduce_meta.dispatch_size = reduce_dispatch_size;
        let reduce = kernel_backend::dynamic_kernel_from_hashed_ir(
            device.kernel_cache(),
            softmax_module_cache(),
            "softmax_reduce",
            reduce_key,
            reduce_buffers,
            reduce_dispatch_size,
            move || {
                let mut kb = tile_ir::KernelBuilder::<()>::new();
                let scratch_ref = tile_ir::KernelTensorRef::new((), reduce_layout.clone());
                let global_ref = tile_ir::KernelTensorRef::new((), reduce_layout);
                tile_ir_kernels::softmax_reduce::<_>(
                    &mut kb,
                    scratch_ref,
                    global_ref,
                    reduce_meta,
                )?;
                Some(kb.finish().0)
            },
        )?;

        let write_variant =
            kernel_backend::KernelVariantKey::with_payload::<SoftmaxDirectKernelVariant>(|state| {
                SoftmaxKernelVariant::Write.hash(state);
                meta.block.hash(state);
                meta.split_blocks.hash(state);
                self.datatype.hash(state);
            });
        let write_key = self.kernel_module_key_with_dispatch(
            write_variant,
            None,
            partial_dispatch_size,
            &inputs,
        );
        let write_buffers = vec![
            input.buffer().clone(),
            global_scratch,
            output.buffer().clone(),
        ];
        let write_layout = layout.clone();
        let write = kernel_backend::dynamic_kernel_from_hashed_ir(
            device.kernel_cache(),
            softmax_module_cache(),
            "softmax_write",
            write_key,
            write_buffers,
            partial_dispatch_size,
            move || {
                let mut kb = tile_ir::KernelBuilder::<()>::new();
                let input_ref = tile_ir::KernelTensorRef::new((), write_layout.clone());
                let scratch_ref = tile_ir::KernelTensorRef::new((), write_layout.clone());
                let output_ref = tile_ir::KernelTensorRef::new((), write_layout);
                tile_ir_kernels::softmax_write::<E, _>(
                    &mut kb,
                    input_ref,
                    scratch_ref,
                    output_ref,
                    meta,
                )?;
                Some(kb.finish().0)
            },
        )?;

        Some(DirectKernel::sequence(
            "softmax_split",
            vec![partial, reduce, write],
        ))
    }
}

impl Operation for SoftmaxOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.shape.hash(state);
        self.axis.hash(state);
        self.datatype.hash(state);
    }

    fn workgroup_shape_constraints(&self, _device: &Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let Some(input) = inputs.first().and_then(MirValue::as_tensor) else {
            return [1, 1, 1];
        };
        let Some(axis_len) = self
            .shape
            .get(self.axis)
            .and_then(|dim| u32::try_from(*dim).ok())
        else {
            return [1, 1, 1];
        };
        let Some(rows) = total_elements(&self.shape).and_then(|total| total.checked_div(axis_len))
        else {
            return [1, 1, 1];
        };
        let Some(split_blocks) = self.split_blocks(input.device()) else {
            return [1, 1, 1];
        };
        self.dispatch_for(rows.saturating_mul(split_blocks), input.device())
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let output = TensorData::new_for_shape(input.device(), &self.shape, self.datatype);
        vec![input.clone().into(), output.into()]
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[1].clone()
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let input = inputs.first()?.as_tensor()?;
        let output = inputs.get(1)?.as_tensor()?;
        if input.datatype() != self.datatype || output.datatype() != self.datatype {
            return None;
        }
        if self.datatype == DataTypeEnum::F16 && !graph.device().f16_supported() {
            return None;
        }

        let device = graph.device();
        let caps = KernelDeviceCaps::from_device(&device);
        let dispatch_size = self.dispatch_size(&WorkgroupShape::new(1, 1, 1), inputs);
        let meta = self.meta(input, output, dispatch_size, caps)?;

        match self.datatype {
            DataTypeEnum::F32 if meta.split_blocks == 1 => {
                self.dispatch_softmax::<tile_ir::F32>(&device, input, output, meta)
            }
            DataTypeEnum::F16 if meta.split_blocks == 1 => {
                self.dispatch_softmax::<tile_ir::F16>(&device, input, output, meta)
            }
            DataTypeEnum::F32 => {
                self.dispatch_split_softmax::<tile_ir::F32>(&device, input, output, meta)
            }
            DataTypeEnum::F16 => {
                self.dispatch_split_softmax::<tile_ir::F16>(&device, input, output, meta)
            }
            DataTypeEnum::U32 => None,
        }
    }

    fn name(&self) -> String {
        format!("softmax_axis_{}", self.axis)
    }
}

impl GraphOperation for SoftmaxOperation {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn category(&self) -> &'static str {
        "softmax"
    }

    fn output_layout(
        &self,
        _input_layouts: &FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> Option<TensorLayoutInfo> {
        Some(TensorLayoutInfo::new(
            Layout::contiguous(&self.shape),
            self.datatype,
        ))
    }
}
