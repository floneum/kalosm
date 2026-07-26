//! Recognized attention-pattern execution: the fused cooperative-matrix
//! kernels from `fusor-tile-ir-kernels` behind one execution-graph operation.
//!
//! Each [`AttentionKernel`] kind is one recognizable computational pattern
//! over a scaled-masked score cluster — the fused softmax·v output, the row
//! log-sum-exp, and the three probability-contraction shapes. Route
//! selection lives in pattern recognition; decode shapes (one query row)
//! stay on the attention row program, and shapes no kernel can host run the
//! composed clusters unchanged.

use std::hash::Hash;
use std::sync::Arc;

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use tile_ir_kernels::{
    FlashAttentionLayouts, FlashAttentionShape, FlashBwdLayouts, FlashKvOutputs, FlashMaskLayout,
    FlashOperandLayout, FlashRowLayout,
};

use crate::{
    DataTypeEnum, Device, TensorData,
    compute_graph::NodeIndex,
    kernel_selection::CooperativeMatrixKind,
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    row_program::AttentionInputs,
};

struct FlashAttentionDirectKernelVariant;

/// Which fused kernel a recognized attention-pattern cluster lowers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AttentionKernel {
    /// `softmax(scale·q·kᵀ [+ mask]) · v`.
    Output,
    /// Row log-sum-exp of the scores: `m + ln Σ exp(s − m)`.
    LogSumExp,
    /// `(p ∘ (grad_o·vᵀ − dsum) · scale) · k` with `p = exp(s − lse)`.
    GradQ,
    /// `(p ∘ (grad_o·vᵀ − dsum) · scale)ᵀ · q`.
    GradK,
    /// `pᵀ · grad_o`.
    GradV,
    /// Both KV-side contractions in one dispatch, landing in a tensor whose
    /// sequence axis spans `2·kv_len` (dk rows first, dv rows after) — the
    /// probability recomputation is shared between them.
    GradKV,
}

/// One recognized attention-pattern cluster bound for a fused kernel.
/// Operand layouts are read from the resolved inputs at kernel-build time,
/// so strided views (transposes, offsets) execute without materialization.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlashAttentionOperation {
    pub(crate) kind: AttentionKernel,
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: Option<NodeIndex>,
    pub(crate) grad_o: Option<NodeIndex>,
    pub(crate) lse: Option<NodeIndex>,
    pub(crate) dsum: Option<NodeIndex>,
    pub(crate) mask: Option<NodeIndex>,
    batch: u32,
    heads: u32,
    kv_heads: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    scale: f32,
    causal: bool,
    datatype: DataTypeEnum,
}

/// The node set of one matched pattern, in dependency order.
pub(crate) struct AttentionPatternNodes {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: Option<NodeIndex>,
    pub(crate) grad_o: Option<NodeIndex>,
    pub(crate) lse: Option<NodeIndex>,
    pub(crate) dsum: Option<NodeIndex>,
    pub(crate) mask: Option<NodeIndex>,
}

impl FlashAttentionOperation {
    /// The fused-output route for a recognized forward cluster; `None`
    /// falls back to the caller's other attention routes.
    pub(crate) fn try_new_output(device: &Device, inputs: &AttentionInputs<'_>) -> Option<Self> {
        let [batch, heads, q_len, head_dim] = *inputs.q_shape else {
            return None;
        };
        let [kv_batch, kv_heads, kv_len, kv_dim] = *inputs.k_shape else {
            return None;
        };
        if inputs.k_shape != inputs.v_shape || kv_batch != batch || kv_dim != head_dim {
            return None;
        }
        match (inputs.mask, inputs.mask_shape, inputs.causal) {
            (Some(_), Some(mask_shape), false) if mask_shape == [q_len, kv_len] => {}
            (None, None, _) => {}
            _ => return None,
        }
        Self::try_new(
            device,
            AttentionKernel::Output,
            AttentionPatternNodes {
                q: inputs.q,
                k: inputs.k,
                v: Some(inputs.v),
                grad_o: None,
                lse: None,
                dsum: None,
                mask: inputs.mask,
            },
            [batch, heads, kv_heads, q_len, kv_len, head_dim],
            inputs.scale,
            inputs.causal,
            inputs.input_dtype,
        )
    }

    /// Build the operation when the device and shape qualify for `kind`;
    /// `None` leaves the composed cluster in place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        device: &Device,
        kind: AttentionKernel,
        nodes: AttentionPatternNodes,
        dims: [usize; 6],
        scale: f32,
        causal: bool,
        datatype: DataTypeEnum,
    ) -> Option<Self> {
        if !matches!(datatype, DataTypeEnum::F32 | DataTypeEnum::F16) {
            return None;
        }
        let [batch, heads, kv_heads, q_len, kv_len, head_dim] = dims;
        if kv_heads == 0 || heads % kv_heads != 0 {
            return None;
        }
        let operation = Self {
            kind,
            q: nodes.q,
            k: nodes.k,
            v: nodes.v,
            grad_o: nodes.grad_o,
            lse: nodes.lse,
            dsum: nodes.dsum,
            mask: nodes.mask,
            batch: batch.try_into().ok()?,
            heads: heads.try_into().ok()?,
            kv_heads: kv_heads.try_into().ok()?,
            q_len: q_len.try_into().ok()?,
            kv_len: kv_len.try_into().ok()?,
            head_dim: head_dim.try_into().ok()?,
            scale,
            causal,
            datatype,
        };
        if causal && nodes.mask.is_some() {
            return None;
        }
        device.coop_token(CooperativeMatrixKind::F32F32M8N8K8)?;
        let subgroups = device.subgroup_config()?;
        // The staged tiles must fit the device's workgroup-storage limit
        // (16 KB WebGPU default, 32 KB Apple) or pipeline creation fails.
        // The analytic bound models the forward kernel's staging
        // (`tests/footprint.rs` pins it to the lowered IR); per-kind
        // backward footprints are not modelled yet.
        let stage = match datatype {
            DataTypeEnum::F16 => tile_ir::ScalarElement::F16,
            _ => tile_ir::ScalarElement::F32,
        };
        if tile_ir_kernels::flash_attention_workgroup_bytes(operation.head_dim, stage)
            > u64::from(device.limits().max_compute_workgroup_storage_size)
        {
            return None;
        }
        let shape = operation.flash_shape();
        let supported = match kind {
            AttentionKernel::Output => {
                tile_ir_kernels::flash_attention_supported(&shape, subgroups)
            }
            AttentionKernel::LogSumExp => {
                tile_ir_kernels::flash_attention_bwd_supported(&shape, subgroups)
            }
            // The streaming contractions recompute probabilities per tile;
            // without the causal break they currently lose to the composed
            // matmuls, so masked/unmasked shapes keep the composed cluster.
            AttentionKernel::GradQ
            | AttentionKernel::GradK
            | AttentionKernel::GradV
            | AttentionKernel::GradKV => {
                causal && tile_ir_kernels::flash_attention_bwd_supported(&shape, subgroups)
            }
        };
        supported.then_some(operation)
    }

    fn flash_shape(&self) -> FlashAttentionShape {
        FlashAttentionShape {
            batch: self.batch,
            heads: self.heads,
            kv_groups: self.heads / self.kv_heads,
            q_len: self.q_len,
            kv_len: self.kv_len,
            head_dim: self.head_dim,
            scale: self.scale,
            causal: self.causal,
        }
    }

    fn out_shape(&self) -> Vec<usize> {
        let (batch, heads) = (self.batch as usize, self.heads as usize);
        let (q_len, kv_len, d) = (
            self.q_len as usize,
            self.kv_len as usize,
            self.head_dim as usize,
        );
        match self.kind {
            AttentionKernel::Output | AttentionKernel::GradQ => vec![batch, heads, q_len, d],
            AttentionKernel::LogSumExp => vec![batch, heads, q_len],
            AttentionKernel::GradK | AttentionKernel::GradV => vec![batch, heads, kv_len, d],
            AttentionKernel::GradKV => vec![batch, heads, 2 * kv_len, d],
        }
    }
}

/// Element strides of a resolved rank-4 tensor.
fn operand_layout(layout: &crate::Layout) -> Option<FlashOperandLayout> {
    let [batch_stride, head_stride, seq_stride, dim_stride] = *layout.strides() else {
        return None;
    };
    Some(FlashOperandLayout {
        offset: layout.offset().try_into().ok()?,
        batch_stride: batch_stride.try_into().ok()?,
        head_stride: head_stride.try_into().ok()?,
        seq_stride: seq_stride.try_into().ok()?,
        dim_stride: dim_stride.try_into().ok()?,
    })
}

/// Element strides of a resolved rank-3 row statistic.
fn row_layout(layout: &crate::Layout) -> Option<FlashRowLayout> {
    let [batch_stride, head_stride, seq_stride] = *layout.strides() else {
        return None;
    };
    Some(FlashRowLayout {
        offset: layout.offset().try_into().ok()?,
        batch_stride: batch_stride.try_into().ok()?,
        head_stride: head_stride.try_into().ok()?,
        seq_stride: seq_stride.try_into().ok()?,
    })
}

/// Element strides of the resolved rank-2 mask.
fn mask_layout(layout: &crate::Layout) -> Option<FlashMaskLayout> {
    let [q_stride, kv_stride] = *layout.strides() else {
        return None;
    };
    Some(FlashMaskLayout {
        offset: layout.offset().try_into().ok()?,
        q_stride: q_stride.try_into().ok()?,
        kv_stride: kv_stride.try_into().ok()?,
    })
}

/// A whole-buffer linear view: the kernels compute element indices
/// themselves from the baked operand strides.
fn linear_ref(buffer: Arc<wgpu::Buffer>) -> Option<tile_ir::KernelTensorRef<Arc<wgpu::Buffer>>> {
    let elements: u32 = (buffer.size() / size_of::<f32>() as u64).try_into().ok()?;
    let layout = tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([elements]),
        &[1],
    );
    Some(tile_ir::KernelTensorRef::new(buffer, layout))
}

struct ResolvedTensors<'a> {
    q: &'a TensorData,
    k: &'a TensorData,
    v: Option<&'a TensorData>,
    grad_o: Option<&'a TensorData>,
    lse: Option<&'a TensorData>,
    dsum: Option<&'a TensorData>,
    mask: Option<&'a TensorData>,
    output: &'a TensorData,
}

impl FlashAttentionOperation {
    fn split_inputs<'a>(&self, inputs: &'a [MirValue]) -> Option<ResolvedTensors<'a>> {
        let mut iter = inputs.iter();
        let mut next = || iter.next().and_then(MirValue::as_tensor);
        let q = next()?;
        let k = next()?;
        let v = if self.v.is_some() {
            Some(next()?)
        } else {
            None
        };
        let grad_o = if self.grad_o.is_some() {
            Some(next()?)
        } else {
            None
        };
        let lse = if self.lse.is_some() {
            Some(next()?)
        } else {
            None
        };
        let dsum = if self.dsum.is_some() {
            Some(next()?)
        } else {
            None
        };
        let mask = if self.mask.is_some() {
            Some(next()?)
        } else {
            None
        };
        let output = next()?;
        Some(ResolvedTensors {
            q,
            k,
            v,
            grad_o,
            lse,
            dsum,
            mask,
            output,
        })
    }
}

impl Operation for FlashAttentionOperation {
    fn hash_kernel_fields(&self, state: &mut rustc_hash::FxHasher) {
        self.kind.hash(state);
        self.batch.hash(state);
        self.heads.hash(state);
        self.kv_heads.hash(state);
        self.q_len.hash(state);
        self.kv_len.hash(state);
        self.head_dim.hash(state);
        self.scale.to_bits().hash(state);
        self.causal.hash(state);
        self.v.is_some().hash(state);
        self.grad_o.is_some().hash(state);
        self.lse.is_some().hash(state);
        self.dsum.is_some().hash(state);
        self.mask.is_some().hash(state);
        self.datatype.hash(state);
    }

    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints {
        let block = device
            .subgroup_config()
            .map(|subgroups| subgroups.block_for_subgroups(4))
            .unwrap_or(128);
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(block));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let max_per_dim = inputs
            .last()
            .and_then(MirValue::as_tensor)
            .expect("attention output must be a tensor")
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        let shape = self.flash_shape();
        match self.kind {
            AttentionKernel::Output => {
                tile_ir_kernels::flash_attention_dispatch(&shape, max_per_dim)
            }
            AttentionKernel::LogSumExp => tile_ir_kernels::flash_lse_dispatch(&shape, max_per_dim),
            AttentionKernel::GradQ => tile_ir_kernels::flash_bwd_q_dispatch(&shape, max_per_dim),
            AttentionKernel::GradK | AttentionKernel::GradV | AttentionKernel::GradKV => {
                tile_ir_kernels::flash_bwd_kv_dispatch(&shape, max_per_dim)
            }
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        for node in [self.v, self.grad_o, self.lse, self.dsum, self.mask]
            .into_iter()
            .flatten()
        {
            f(node);
        }
    }

    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        f(&mut self.q);
        f(&mut self.k);
        for node in [
            &mut self.v,
            &mut self.grad_o,
            &mut self.lse,
            &mut self.dsum,
            &mut self.mask,
        ]
        .into_iter()
        .flatten()
        {
            f(node);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let resolved = |node: NodeIndex| {
            nodes
                .get_result(node)
                .expect("attention inputs must be resolved before kernel launch")
        };
        let device = nodes.device();
        let output = TensorData::new_for_shape(&device, &self.out_shape(), self.datatype);
        let mut inputs = vec![resolved(self.q).into(), resolved(self.k).into()];
        for node in [self.v, self.grad_o, self.lse, self.dsum, self.mask]
            .into_iter()
            .flatten()
        {
            inputs.push(resolved(node).into());
        }
        inputs.push(output.into());
        inputs
    }

    fn output(
        &self,
        _nodes: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> MirValue {
        inputs.last().unwrap().as_tensor().unwrap().clone().into()
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let tensors = self.split_inputs(inputs)?;
        let device = graph.device();
        let subgroups = device.subgroup_config()?;
        let coop = device.coop_token(CooperativeMatrixKind::F32F32M8N8K8)?;
        let byte_arena = device.byte_arena_token();
        let max_per_dim = device.limits().max_compute_workgroups_per_dimension;
        let shape = self.flash_shape();
        let dispatch_size = self.dispatch_size(workgroup_shape, inputs);
        let cache_key = self.kernel_cache_key_with_dispatch(
            kernel_backend::KernelVariantKey::of::<FlashAttentionDirectKernelVariant>(),
            Some(workgroup_shape),
            dispatch_size,
            inputs,
        );
        let kind = self.kind;
        let datatype = self.datatype;

        let buffer = |data: &TensorData| data.buffer().clone();
        let q_buffer = buffer(tensors.q);
        let k_buffer = buffer(tensors.k);
        let lq = operand_layout(tensors.q.layout())?;
        let lk = operand_layout(tensors.k.layout())?;
        let mask_parts = match tensors.mask {
            Some(mask) => Some((buffer(mask), mask_layout(mask.layout())?)),
            None => None,
        };
        let output_buffer = buffer(tensors.output);

        match kind {
            AttentionKernel::Output => {
                let v = tensors.v?;
                let layouts = FlashAttentionLayouts {
                    q: lq,
                    k: lk,
                    v: operand_layout(v.layout())?,
                    o: operand_layout(tensors.output.layout())?,
                };
                let v_buffer = buffer(v);
                kernel_backend::run_kernel(
                    device.kernel_cache(),
                    self.name(),
                    cache_key,
                    dispatch_size,
                    move |kb| {
                        if let Some(token) = byte_arena {
                            kb.program().enable_byte_arena(token);
                        }
                        let f32e = match datatype {
                            DataTypeEnum::F16 => tile_ir::ScalarElement::F16.element(),
                            _ => tile_ir::ScalarElement::F32.element(),
                        };
                        let q = kb.read(f32e, linear_ref(q_buffer)?);
                        let k = kb.read(f32e, linear_ref(k_buffer)?);
                        let v = kb.read(f32e, linear_ref(v_buffer)?);
                        let mask = match mask_parts {
                            Some((buffer, layout)) => {
                                Some((kb.read(f32e, linear_ref(buffer)?), layout))
                            }
                            None => None,
                        };
                        let o = kb.write(f32e, linear_ref(output_buffer)?);
                        tile_ir_kernels::flash_attention_f32(
                            kb.program(),
                            &q,
                            &k,
                            &v,
                            mask.as_ref().map(|(storage, layout)| (storage, *layout)),
                            &o,
                            &layouts,
                            shape,
                            subgroups,
                            coop,
                            max_per_dim,
                        )
                        .then_some(())
                    },
                )
            }
            AttentionKernel::LogSumExp => {
                let lse_layout = row_layout(tensors.output.layout())?;
                kernel_backend::run_kernel(
                    device.kernel_cache(),
                    self.name(),
                    cache_key,
                    dispatch_size,
                    move |kb| {
                        if let Some(token) = byte_arena {
                            kb.program().enable_byte_arena(token);
                        }
                        let f32e = match datatype {
                            DataTypeEnum::F16 => tile_ir::ScalarElement::F16.element(),
                            _ => tile_ir::ScalarElement::F32.element(),
                        };
                        let q = kb.read(f32e, linear_ref(q_buffer)?);
                        let k = kb.read(f32e, linear_ref(k_buffer)?);
                        let mask = match mask_parts {
                            Some((buffer, layout)) => {
                                Some((kb.read(f32e, linear_ref(buffer)?), layout))
                            }
                            None => None,
                        };
                        let lse = kb.write(f32e, linear_ref(output_buffer)?);
                        tile_ir_kernels::flash_lse_f32(
                            kb.program(),
                            &q,
                            &k,
                            mask.as_ref().map(|(storage, layout)| (storage, *layout)),
                            &lse,
                            lq,
                            lk,
                            lse_layout,
                            shape,
                            subgroups,
                            coop,
                            max_per_dim,
                        )
                        .then_some(())
                    },
                )
            }
            AttentionKernel::GradQ
            | AttentionKernel::GradK
            | AttentionKernel::GradV
            | AttentionKernel::GradKV => {
                let grad_o = tensors.grad_o?;
                let lse = tensors.lse?;
                let needs_dsum = kind != AttentionKernel::GradV;
                // The combined output's sequence axis spans 2*kv_len; its
                // layout already reflects that from the allocation.
                let placeholder =
                    FlashOperandLayout::contiguous(self.kv_heads, self.kv_len, self.head_dim);
                let layouts = FlashBwdLayouts {
                    q: lq,
                    k: lk,
                    v: match tensors.v {
                        Some(v) => operand_layout(v.layout())?,
                        None => placeholder,
                    },
                    grad_o: operand_layout(grad_o.layout())?,
                    lse: row_layout(lse.layout())?,
                    dsum: match tensors.dsum {
                        Some(dsum) => row_layout(dsum.layout())?,
                        None => FlashRowLayout {
                            offset: 0,
                            batch_stride: 0,
                            head_stride: 0,
                            seq_stride: 0,
                        },
                    },
                    out: operand_layout(tensors.output.layout())?,
                };
                if needs_dsum && (tensors.v.is_none() || tensors.dsum.is_none()) {
                    return None;
                }
                let v_buffer = tensors.v.map(buffer);
                let grad_o_buffer = buffer(grad_o);
                let lse_buffer = buffer(lse);
                let dsum_buffer = tensors.dsum.map(buffer);
                kernel_backend::run_kernel(
                    device.kernel_cache(),
                    self.name(),
                    cache_key,
                    dispatch_size,
                    move |kb| {
                        if let Some(token) = byte_arena {
                            kb.program().enable_byte_arena(token);
                        }
                        let f32e = match datatype {
                            DataTypeEnum::F16 => tile_ir::ScalarElement::F16.element(),
                            _ => tile_ir::ScalarElement::F32.element(),
                        };
                        let q = kb.read(f32e, linear_ref(q_buffer)?);
                        let k = kb.read(f32e, linear_ref(k_buffer)?);
                        let v = match v_buffer {
                            Some(buffer) => Some(kb.read(f32e, linear_ref(buffer)?)),
                            None => None,
                        };
                        let grad_o = kb.read(f32e, linear_ref(grad_o_buffer)?);
                        let lse = kb.read(f32e, linear_ref(lse_buffer)?);
                        let dsum = match dsum_buffer {
                            Some(buffer) => Some(kb.read(f32e, linear_ref(buffer)?)),
                            None => None,
                        };
                        let mask = match mask_parts {
                            Some((buffer, layout)) => {
                                Some((kb.read(f32e, linear_ref(buffer)?), layout))
                            }
                            None => None,
                        };
                        let out = kb.write(f32e, linear_ref(output_buffer)?);
                        let mask = mask.as_ref().map(|(storage, layout)| (storage, *layout));
                        match kind {
                            AttentionKernel::GradQ => tile_ir_kernels::flash_bwd_q_f32(
                                kb.program(),
                                &q,
                                &k,
                                v.as_ref().expect("grad_q requires values"),
                                &grad_o,
                                &lse,
                                dsum.as_ref().expect("grad_q requires dsum"),
                                mask,
                                &out,
                                &layouts,
                                shape,
                                subgroups,
                                coop,
                                max_per_dim,
                            ),
                            AttentionKernel::GradK => tile_ir_kernels::flash_bwd_kv_f32(
                                kb.program(),
                                &q,
                                &k,
                                v.as_ref(),
                                &grad_o,
                                &lse,
                                dsum.as_ref(),
                                mask,
                                &out,
                                &layouts,
                                FlashKvOutputs::Dk,
                                shape,
                                subgroups,
                                coop,
                                max_per_dim,
                            ),
                            AttentionKernel::GradKV => tile_ir_kernels::flash_bwd_kv_f32(
                                kb.program(),
                                &q,
                                &k,
                                v.as_ref(),
                                &grad_o,
                                &lse,
                                dsum.as_ref(),
                                mask,
                                &out,
                                &layouts,
                                FlashKvOutputs::Both,
                                shape,
                                subgroups,
                                coop,
                                max_per_dim,
                            ),
                            AttentionKernel::GradV => tile_ir_kernels::flash_bwd_kv_f32(
                                kb.program(),
                                &q,
                                &k,
                                None,
                                &grad_o,
                                &lse,
                                None,
                                mask,
                                &out,
                                &layouts,
                                FlashKvOutputs::Dv,
                                shape,
                                subgroups,
                                coop,
                                max_per_dim,
                            ),
                            _ => unreachable!(),
                        }
                        .then_some(())
                    },
                )
            }
        }
    }

    fn name(&self) -> String {
        let kind = match self.kind {
            AttentionKernel::Output => "out",
            AttentionKernel::LogSumExp => "lse",
            AttentionKernel::GradQ => "dq",
            AttentionKernel::GradK => "dk",
            AttentionKernel::GradV => "dv",
            AttentionKernel::GradKV => "dkv",
        };
        format!(
            "flash_attention_{kind}_{}x{}x{}x{}x{}",
            self.batch, self.heads, self.q_len, self.kv_len, self.head_dim
        )
    }
}
