use std::{any::Any, hash::Hash};

use fusor_tile_ir as tile_ir;
use rustc_hash::{FxHashMap, FxHasher};

use crate::{
    DataTypeEnum, Device, Layout,
    compute_graph::{ComputeGraphInner, GraphOperation, NodeIndex},
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_direct::{
        TensorMeta, ValueTile, flat_layout, layout_index, linear_group, output_dims_from_flat,
        tile_u32,
    },
    tensor::{TensorData, TensorLayoutInfo},
    visit_tiled::distribute_workgroups,
};

#[derive(Clone, Debug)]
pub(crate) struct ConvNdOperation {
    input: NodeIndex,
    weight: NodeIndex,
    bias: Option<NodeIndex>,
    input_shape: Box<[usize]>,
    weight_shape: Box<[usize]>,
    output_shape: Box<[usize]>,
    padding: Box<[u32]>,
    strides: Box<[u32]>,
    datatype: DataTypeEnum,
}

impl ConvNdOperation {
    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    pub(crate) fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        input_shape: &[usize],
        weight_shape: &[usize],
        bias_shape: Option<&[usize]>,
        padding: &[usize],
        strides: &[usize],
        datatype: DataTypeEnum,
        device: &Device,
    ) -> Option<Self> {
        let spatial_rank = padding.len();
        if spatial_rank == 0
            || strides.len() != spatial_rank
            || input_shape.len() != spatial_rank + 2
            || weight_shape.len() != spatial_rank + 2
            || !matches!(datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
            || (datatype == DataTypeEnum::F16 && !device.f16_supported())
        {
            return None;
        }

        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        if weight_shape[1] != in_channels
            || strides.contains(&0)
            || input_shape.contains(&0)
            || weight_shape.contains(&0)
        {
            return None;
        }
        if let Some(bias_shape) = bias_shape
            && bias_shape != [out_channels]
        {
            return None;
        }

        let mut output_shape = Vec::with_capacity(input_shape.len());
        output_shape.push(batch);
        output_shape.push(out_channels);
        for axis in 0..spatial_rank {
            let input_len = input_shape[axis + 2];
            let kernel_len = weight_shape[axis + 2];
            let padded_len = input_len.checked_add(padding[axis].checked_mul(2)?)?;
            let out_len = padded_len
                .checked_sub(kernel_len)?
                .checked_div(strides[axis])?
                + 1;
            if out_len == 0 {
                return None;
            }
            output_shape.push(out_len);
        }

        let kernel_volume = weight_shape[2..]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        let total_outputs = output_shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        u32::try_from(total_outputs).ok()?;
        u32::try_from(in_channels.checked_mul(kernel_volume)?).ok()?;

        Some(Self {
            input,
            weight,
            bias,
            input_shape: input_shape.into(),
            weight_shape: weight_shape.into(),
            output_shape: output_shape.into_boxed_slice(),
            padding: padding
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?
                .into_boxed_slice(),
            strides: strides
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?
                .into_boxed_slice(),
            datatype,
        })
    }

    #[inline]
    fn spatial_rank(&self) -> usize {
        self.padding.len()
    }

    #[inline]
    fn output_index(&self) -> usize {
        2 + usize::from(self.bias.is_some())
    }

    #[inline]
    fn total_outputs(&self) -> Option<u32> {
        self.output_shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))
    }

    fn storage_read<B>(
        kb: &mut tile_ir::KernelBuilder<B>,
        datatype: DataTypeEnum,
        tensor: tile_ir::KernelTensorRef<B>,
    ) -> crate::nary_direct::Storage2 {
        let element = crate::nary_direct::datatype_element(datatype);
        let storage = kb.read(element, tensor);
        match datatype {
            DataTypeEnum::F32 => crate::nary_direct::Storage2::F32(storage),
            DataTypeEnum::F16 => crate::nary_direct::Storage2::F16(storage),
            DataTypeEnum::U32 => crate::nary_direct::Storage2::U32(storage),
        }
    }

    fn storage_write<B>(
        kb: &mut tile_ir::KernelBuilder<B>,
        datatype: DataTypeEnum,
        tensor: tile_ir::KernelTensorRef<B>,
    ) -> crate::nary_direct::Storage2 {
        let element = crate::nary_direct::datatype_element(datatype);
        let storage = kb.write(element, tensor);
        match datatype {
            DataTypeEnum::F32 => crate::nary_direct::Storage2::F32(storage),
            DataTypeEnum::F16 => crate::nary_direct::Storage2::F16(storage),
            DataTypeEnum::U32 => crate::nary_direct::Storage2::U32(storage),
        }
    }
}

struct ConvNdDirectKernelVariant;

impl Operation for ConvNdOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.input_shape.hash(state);
        self.weight_shape.hash(state);
        self.output_shape.hash(state);
        self.padding.hash(state);
        self.strides.hash(state);
        self.bias.is_some().hash(state);
        self.datatype.hash(state);
    }

    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        let workgroup_size = device.limits().max_compute_workgroup_size_x.min(256);
        constraints.add_constraint(0, Constraint::equals(workgroup_size));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let Some(output) = inputs
            .get(self.output_index())
            .and_then(MirValue::as_tensor)
        else {
            return [1, 1, 1];
        };
        let Some(total_outputs) = self.total_outputs() else {
            return [1, 1, 1];
        };
        distribute_workgroups(
            total_outputs.div_ceil(workgroup_shape.x()),
            output
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let weight = nodes.get_cached_result(self.weight).unwrap();
        let mut inputs = Vec::with_capacity(3 + usize::from(self.bias.is_some()));
        inputs.push(input.clone().into());
        inputs.push(weight.clone().into());
        if let Some(bias) = self.bias {
            inputs.push(nodes.get_cached_result(bias).unwrap().clone().into());
        }
        inputs.push(
            TensorData::new_for_shape(input.device(), &self.output_shape, self.datatype).into(),
        );
        inputs
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[self.output_index()].clone()
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let input = inputs.first()?.as_tensor()?.clone();
        let weight = inputs.get(1)?.as_tensor()?.clone();
        let bias = self
            .bias
            .is_some()
            .then(|| inputs.get(2)?.as_tensor().cloned())
            .flatten();
        let output = inputs.get(self.output_index())?.as_tensor()?.clone();
        if input.datatype() != self.datatype
            || weight.datatype() != self.datatype
            || output.datatype() != self.datatype
            || bias
                .as_ref()
                .is_some_and(|bias| bias.datatype() != self.datatype)
            || (self.datatype == DataTypeEnum::F16 && !graph.device().f16_supported())
        {
            return None;
        }

        let input_meta = TensorMeta::new(&input)?;
        let weight_meta = TensorMeta::new(&weight)?;
        let bias_meta = match &bias {
            Some(bias) => Some(TensorMeta::new(bias)?),
            None => None,
        };
        let output_meta = TensorMeta::new(&output)?;
        let input_shape = to_u32_vec(&self.input_shape)?;
        let weight_shape = to_u32_vec(&self.weight_shape)?;
        let output_shape_usize = self.output_shape.to_vec();
        let kernel_shape = weight_shape[2..].to_vec();
        let kernel_volume = kernel_shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
        let reduce_len = input_shape[1].checked_mul(kernel_volume)?;
        let total_outputs = self.total_outputs()?;
        let dispatch_size = self.dispatch_size(workgroup_shape, inputs);
        let cache_key = self.kernel_cache_key_with_dispatch(
            kernel_backend::KernelVariantKey::of::<ConvNdDirectKernelVariant>(),
            Some(workgroup_shape),
            dispatch_size,
            inputs,
        );

        let input_buffer = input.buffer().clone();
        let weight_buffer = weight.buffer().clone();
        let bias_buffer = bias.as_ref().map(|bias| bias.buffer().clone());
        let output_buffer = output.buffer().clone();
        let input_layout = flat_layout(input_meta.allocation_len);
        let weight_layout = flat_layout(weight_meta.allocation_len);
        let bias_layout = bias_meta
            .as_ref()
            .map(|meta| flat_layout(meta.allocation_len));
        let output_layout = flat_layout(output_meta.allocation_len);
        let input_meta_body = input_meta.clone();
        let weight_meta_body = weight_meta.clone();
        let bias_meta_body = bias_meta.clone();
        let output_meta_body = output_meta.clone();
        let padding = self.padding.to_vec();
        let strides = self.strides.to_vec();
        let spatial_rank = self.spatial_rank();
        let datatype = self.datatype;
        let block = workgroup_shape.x();

        kernel_backend::run_kernel(
            graph.device().kernel_cache(),
            self.name(),
            cache_key,
            dispatch_size,
            move |kb| {
                let input_tensor =
                    tile_ir::KernelTensorRef::new(input_buffer.clone(), input_layout.clone());
                let weight_tensor =
                    tile_ir::KernelTensorRef::new(weight_buffer.clone(), weight_layout.clone());
                let output_tensor =
                    tile_ir::KernelTensorRef::new(output_buffer.clone(), output_layout.clone());
                let input_storage = Self::storage_read(kb, datatype, input_tensor);
                let weight_storage = Self::storage_read(kb, datatype, weight_tensor);
                let bias_storage = bias_buffer
                    .as_ref()
                    .zip(bias_layout.as_ref())
                    .zip(bias_meta_body.as_ref())
                    .map(|((buffer, layout), _)| {
                        let bias_tensor =
                            tile_ir::KernelTensorRef::new(buffer.clone(), layout.clone());
                        Self::storage_read(kb, datatype, bias_tensor)
                    });
                let output_storage = Self::storage_write(kb, datatype, output_tensor);

                kb.program().program_grid(block, dispatch_size, |program| {
                    let lane = program.lane();
                    let group = linear_group(program, dispatch_size);
                    let flat = group * block + lane;
                    let in_bounds = flat.lt(total_outputs);
                    let output_dims = output_dims_from_flat(flat.clone(), &output_shape_usize);
                    let batch = output_dims[0].clone();
                    let out_channel = output_dims[1].clone();

                    let zero = tile_ir::tile::Tile::literal(tile_ir::TileLiteral::f32(0.0));
                    let initial = if let (Some(bias_storage), Some(bias_meta)) =
                        (bias_storage.as_ref(), bias_meta_body.as_ref())
                    {
                        bias_storage
                            .load(
                                program,
                                layout_index(bias_meta, &[out_channel.clone()]),
                                in_bounds.clone(),
                            )
                            .into_f32()
                    } else {
                        zero
                    };

                    let [sum] = program.fold(
                        tile_ir::tile::range(reduce_len),
                        [initial],
                        |program, reduce_index, [acc]| {
                            let in_channel = reduce_index.clone() / tile_u32(kernel_volume);
                            let kernel_linear = reduce_index % tile_u32(kernel_volume);
                            let mut active = in_bounds.clone();
                            let mut input_coords = Vec::with_capacity(spatial_rank + 2);
                            input_coords.push(batch.clone());
                            input_coords.push(in_channel.clone());
                            let mut weight_coords = Vec::with_capacity(spatial_rank + 2);
                            weight_coords.push(out_channel.clone());
                            weight_coords.push(in_channel);

                            for axis in 0..spatial_rank {
                                let divisor = kernel_shape[axis + 1..]
                                    .iter()
                                    .fold(1u32, |acc, dim| acc.saturating_mul(*dim));
                                let quotient = if divisor == 1 {
                                    kernel_linear.clone()
                                } else {
                                    kernel_linear.clone() / tile_u32(divisor)
                                };
                                let kernel_coord = quotient % tile_u32(kernel_shape[axis]);
                                let padded_coord = output_dims[axis + 2].clone() * strides[axis]
                                    + kernel_coord.clone();
                                let source_coord = padded_coord.clone() - padding[axis];
                                let valid_coord = padded_coord
                                    .ge(padding[axis])
                                    .and(source_coord.clone().lt(input_shape[axis + 2]));
                                active = active.and(valid_coord);
                                input_coords.push(source_coord);
                                weight_coords.push(kernel_coord);
                            }

                            let input_index = layout_index(&input_meta_body, &input_coords);
                            let weight_index = layout_index(&weight_meta_body, &weight_coords);
                            let input_value = input_storage
                                .load(program, input_index, active.clone())
                                .into_f32();
                            let input_value = tile_ir::tile::Tile::select(
                                active.clone(),
                                input_value,
                                tile_ir::tile::Tile::literal(tile_ir::TileLiteral::f32(0.0)),
                            );
                            let weight_value = weight_storage
                                .load(program, weight_index, active)
                                .into_f32();
                            [acc + input_value * weight_value]
                        },
                    );

                    output_storage.store(
                        program,
                        layout_index(&output_meta_body, &output_dims),
                        ValueTile::F32(sum).cast_to(datatype),
                        in_bounds,
                    );
                });
                Some(())
            },
        )
    }

    fn name(&self) -> String {
        format!("conv{}d_direct", self.spatial_rank())
    }
}

impl GraphOperation for ConvNdOperation {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn category(&self) -> &'static str {
        "conv_nd"
    }

    fn output_layout(
        &self,
        _input_layouts: &FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> Option<TensorLayoutInfo> {
        Some(TensorLayoutInfo::new(
            Layout::contiguous(&self.output_shape),
            self.datatype,
        ))
    }
}

fn to_u32_vec(values: &[usize]) -> Option<Vec<u32>> {
    values
        .iter()
        .copied()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}
