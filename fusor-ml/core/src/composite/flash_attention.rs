use std::{fmt::Write, sync::Arc};

use crate::{
    DataType, DataTypeEnum, LastRank, Tensor, TensorData,
    compute_graph::NodeIndex,
    min_for_dtype,
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::distribute_workgroups,
};

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Computes flash attention with optional masking.
    ///
    /// Supports grouped-query attention (GQA) and multi-query attention (MQA) where
    /// K and V may have fewer heads than Q. The number of Q heads must be divisible
    /// by the number of K/V heads.
    ///
    /// Args:
    ///   - k: Key tensor with shape [batch, num_kv_heads, kv_seq_len, head_dim]
    ///   - v: Value tensor with shape [batch, num_kv_heads, kv_seq_len, head_dim]
    ///   - scale: Scale factor (typically 1/sqrt(head_dim))
    ///   - mask: Optional attention mask with shape [q_seq_len, kv_seq_len]
    pub fn flash_attention<const R2: usize>(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor<2, T>>,
    ) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
        T: crate::FloatDataType,
    {
        let operation = FlashAttentionOperation::new(
            self.key(),
            k.key(),
            v.key(),
            mask.map(|m| m.key()),
            self.datatype(),
            self.shape(),
            k.shape(),
            scale,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }
}

#[derive(Debug, Clone)]
struct FlashAttentionOperation {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) head_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) scale: f32,
}

impl FlashAttentionOperation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        mask: Option<NodeIndex>,
        datatype: DataTypeEnum,
        q_shape: &[usize],
        kv_shape: &[usize],
        scale: f32,
    ) -> Self {
        let num_heads = q_shape[1];
        let num_kv_heads = kv_shape[1];
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            num_heads,
            num_kv_heads
        );
        Self {
            q,
            k,
            v,
            mask,
            datatype,
            head_dim: q_shape[3],
            num_heads,
            num_kv_heads,
            scale,
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        _blocksize: u32,
        kernel: &mut GenericKernel,
        device: &crate::Device,
    ) {
        let dtype = self.datatype;
        let out_datatype = self.out_datatype();
        let has_mask = self.mask.is_some();

        // Input tensors
        let q_tensor = kernel.add_tensor_input(4, false, dtype);
        let k_tensor = kernel.add_tensor_input(4, false, dtype);
        let v_tensor = kernel.add_tensor_input(4, false, dtype);
        let mask_tensor = if has_mask {
            Some(kernel.add_tensor_input(2, false, dtype))
        } else {
            None
        };
        let output_tensor = kernel.add_tensor_input(4, true, out_datatype);

        // Dimensions - Q and K/V may have different sequence lengths (KV cache case)
        // and different number of heads (GQA/MQA case)
        let batch_size = q_tensor.shape_binding(0);
        let num_heads = q_tensor.shape_binding(1);
        let q_seq_len = q_tensor.shape_binding(2);
        let head_dim = q_tensor.shape_binding(3);
        let kv_seq_len = k_tensor.shape_binding(2);

        // For GQA/MQA: compute the group size for mapping Q heads to KV heads
        // kv_head_idx = head_idx / num_key_value_groups = head_idx * kv_num_heads / num_heads
        let num_key_value_groups = self.num_heads / self.num_kv_heads;

        // Workgroup indices
        let workgroup_index = workgroup_shape.linearized_workgroup_index(kernel);
        let workgroup_local_index = kernel.workgroup_local_index();

        let workgroup_size = workgroup_shape.x();

        // Subgroup-based implementation: each subgroup computes one output row [batch, head, seq, :]
        // Threads within the subgroup cooperatively compute Q·K and load V
        if device.subgroups_supported() {
            let subgroup_size = kernel.subgroup_size();
            let subgroup_local_id = kernel.subgroup_local_index();

            // Each workgroup handles one position [batch, head, seq]
            // workgroup_index directly maps to the position
            writeln!(kernel, "let global_position_id = {};", workgroup_index).unwrap();

            // Calculate batch, head, seq indices from global_position_id
            writeln!(kernel, "var pos_idx = global_position_id;").unwrap();
            writeln!(kernel, "let seq_idx = pos_idx % {q_seq_len};").unwrap();
            writeln!(kernel, "pos_idx /= {q_seq_len};").unwrap();
            writeln!(kernel, "let head_idx = pos_idx % {num_heads};").unwrap();
            writeln!(kernel, "pos_idx /= {num_heads};").unwrap();
            writeln!(kernel, "let batch_idx = pos_idx;").unwrap();
            // Map Q head index to KV head index for GQA/MQA
            writeln!(
                kernel,
                "let kv_head_idx = head_idx / {num_key_value_groups}u;"
            )
            .unwrap();

            // Early exit if beyond valid positions
            writeln!(
                kernel,
                "let total_positions = {batch_size} * {num_heads} * {q_seq_len};"
            )
            .unwrap();
            writeln!(kernel, "if global_position_id >= total_positions {{").unwrap();
            writeln!(kernel, "return;").unwrap();
            writeln!(kernel, "}}").unwrap();

            // Each thread handles ceil(head_dim / subgroup_size) dimensions
            writeln!(
                kernel,
                "let dims_per_thread = ({head_dim} + {subgroup_size} - 1u) / {subgroup_size};"
            )
            .unwrap();

            // Initialize online softmax variables for each dimension this thread handles
            // We use arrays to handle multiple dimensions per thread
            // Max dims per thread = ceil(head_dim / min_subgroup_size)
            // Always use f32 for accumulation to avoid overflow/underflow in f16
            let min_subgroup_size = device.min_subgroup_size() as usize;
            let max_dims_per_thread = self.head_dim.div_ceil(min_subgroup_size);
            writeln!(kernel, "var m_arr: array<f32, {max_dims_per_thread}>;").unwrap();
            writeln!(kernel, "var d_arr: array<f32, {max_dims_per_thread}>;").unwrap();
            writeln!(kernel, "var acc_arr: array<f32, {max_dims_per_thread}>;").unwrap();
            writeln!(
                kernel,
                "for (var i = 0u; i < {max_dims_per_thread}u; i++) {{"
            )
            .unwrap();
            writeln!(kernel, "m_arr[i] = f32({});", min_for_dtype(dtype)).unwrap();
            writeln!(kernel, "d_arr[i] = f32(0.0);").unwrap();
            writeln!(kernel, "acc_arr[i] = f32(0.0);").unwrap();
            writeln!(kernel, "}}").unwrap();

            // Process all K/V sequence positions for attention
            writeln!(
                kernel,
                "for (var k_seq = 0u; k_seq < {kv_seq_len}; k_seq++) {{"
            )
            .unwrap();
            {
                // COOPERATIVE Q·K COMPUTATION
                // Each thread computes partial dot product for dimensions it owns
                writeln!(kernel, "var score_partial = f32(0.0);").unwrap();
                writeln!(kernel, "for (var d = 0u; d < dims_per_thread; d++) {{").unwrap();
                {
                    writeln!(
                        kernel,
                        "let d_idx = {} + d * {subgroup_size};",
                        subgroup_local_id
                    )
                    .unwrap();
                    writeln!(kernel, "if d_idx < {head_dim} {{").unwrap();
                    {
                        // Load Q value and convert to f32
                        write!(kernel, "let q_idx = ").unwrap();
                        q_tensor
                            .strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "d_idx"]);
                        writeln!(kernel, ";").unwrap();
                        writeln!(kernel, "let q_val = f32({q_tensor}[q_idx]);").unwrap();

                        // Load K value and convert to f32
                        write!(kernel, "let k_idx = ").unwrap();
                        k_tensor
                            .strided_index(kernel, ["batch_idx", "kv_head_idx", "k_seq", "d_idx"]);
                        writeln!(kernel, ";").unwrap();
                        writeln!(kernel, "let k_val = f32({k_tensor}[k_idx]);").unwrap();

                        writeln!(kernel, "score_partial += q_val * k_val;").unwrap();
                    }
                    writeln!(kernel, "}}").unwrap();
                }
                writeln!(kernel, "}}").unwrap();

                // Reduce across subgroup to get the full attention score
                writeln!(
                    kernel,
                    "let score = subgroupAdd(score_partial) * {};",
                    self.scale
                )
                .unwrap();

                // Apply attention mask if provided (same score for all threads in subgroup)
                if let Some(mask) = &mask_tensor {
                    write!(kernel, "let mask_idx = ").unwrap();
                    mask.strided_index(kernel, ["seq_idx", "k_seq"]);
                    writeln!(kernel, ";").unwrap();
                    writeln!(kernel, "let masked_score = score + f32({mask}[mask_idx]);").unwrap();
                } else {
                    writeln!(kernel, "let masked_score = score;").unwrap();
                }

                // COOPERATIVE V LOADING & ONLINE SOFTMAX UPDATE
                // Each thread loads V values for its dimensions and updates accumulators
                writeln!(kernel, "for (var d = 0u; d < dims_per_thread; d++) {{").unwrap();
                {
                    writeln!(
                        kernel,
                        "let out_dim = {} + d * {subgroup_size};",
                        subgroup_local_id
                    )
                    .unwrap();
                    writeln!(kernel, "if out_dim < {head_dim} {{").unwrap();
                    {
                        // Load V value for this dimension and convert to f32
                        write!(kernel, "let v_idx = ").unwrap();
                        v_tensor.strided_index(
                            kernel,
                            ["batch_idx", "kv_head_idx", "k_seq", "out_dim"],
                        );
                        writeln!(kernel, ";").unwrap();
                        writeln!(kernel, "let v_val = f32({v_tensor}[v_idx]);").unwrap();

                        // Online softmax update for this dimension
                        writeln!(kernel, "let old_m = m_arr[d];").unwrap();
                        writeln!(kernel, "m_arr[d] = max(m_arr[d], masked_score);").unwrap();
                        writeln!(kernel, "let exp_old_m_diff = exp(old_m - m_arr[d]);").unwrap();
                        writeln!(kernel, "let exp_score_diff = exp(masked_score - m_arr[d]);")
                            .unwrap();
                        writeln!(
                            kernel,
                            "d_arr[d] = d_arr[d] * exp_old_m_diff + exp_score_diff;"
                        )
                        .unwrap();
                        writeln!(
                            kernel,
                            "acc_arr[d] = acc_arr[d] * exp_old_m_diff + exp_score_diff * v_val;"
                        )
                        .unwrap();
                    }
                    writeln!(kernel, "}}").unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();

            // Write output for each dimension this thread handles
            writeln!(kernel, "for (var d = 0u; d < dims_per_thread; d++) {{").unwrap();
            {
                writeln!(
                    kernel,
                    "let out_dim = {} + d * {subgroup_size};",
                    subgroup_local_id
                )
                .unwrap();
                writeln!(kernel, "if out_dim < {head_dim} {{").unwrap();
                {
                    write!(kernel, "let out_idx = ").unwrap();
                    output_tensor
                        .strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "out_dim"]);
                    writeln!(kernel, ";").unwrap();
                    // Convert result back to output dtype
                    writeln!(
                        kernel,
                        "{output_tensor}[out_idx] = {dtype}(acc_arr[d] / d_arr[d]);"
                    )
                    .unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        } else {
            // Fallback: original per-thread implementation for devices without subgroup support
            // Each thread computes one output element [batch, head, seq, dim]
            writeln!(
                kernel,
                "let global_thread_id = ({}) * {workgroup_size}u + {};",
                workgroup_index, workgroup_local_index
            )
            .unwrap();

            // Calculate output indices from global thread id
            writeln!(kernel, "var idx = global_thread_id;").unwrap();
            writeln!(kernel, "let out_dim = idx % {head_dim};").unwrap();
            writeln!(kernel, "idx /= {head_dim};").unwrap();
            writeln!(kernel, "let seq_idx = idx % {q_seq_len};").unwrap();
            writeln!(kernel, "idx /= {q_seq_len};").unwrap();
            writeln!(kernel, "let head_idx = idx % {num_heads};").unwrap();
            writeln!(kernel, "idx /= {num_heads};").unwrap();
            writeln!(kernel, "let batch_idx = idx;").unwrap();
            // Map Q head index to KV head index for GQA/MQA
            writeln!(
                kernel,
                "let kv_head_idx = head_idx / {num_key_value_groups}u;"
            )
            .unwrap();

            // Early exit if we're beyond valid elements
            writeln!(
                kernel,
                "let total_elements = {batch_size} * {num_heads} * {q_seq_len} * {head_dim};"
            )
            .unwrap();
            writeln!(kernel, "if global_thread_id >= total_elements {{").unwrap();
            writeln!(kernel, "return;").unwrap();
            writeln!(kernel, "}}").unwrap();

            // Initialize online softmax variables - use f32 for accumulation
            writeln!(kernel, "var m = f32({});", min_for_dtype(dtype)).unwrap();
            writeln!(kernel, "var d = f32(0.0);").unwrap();
            writeln!(kernel, "var acc = f32(0.0);").unwrap();

            // Process all K/V sequence positions for attention
            writeln!(
                kernel,
                "for (var k_seq = 0u; k_seq < {kv_seq_len}; k_seq++) {{"
            )
            .unwrap();
            {
                // Compute attention score as full dot product over all head dimensions
                writeln!(kernel, "var score = f32(0.0);").unwrap();
                writeln!(
                    kernel,
                    "for (var d_idx = 0u; d_idx < {head_dim}; d_idx++) {{"
                )
                .unwrap();
                {
                    // Load Q value and convert to f32
                    write!(kernel, "let q_idx = ").unwrap();
                    q_tensor.strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "d_idx"]);
                    writeln!(kernel, ";").unwrap();
                    writeln!(kernel, "let q_val = f32({q_tensor}[q_idx]);").unwrap();

                    // Load K value and convert to f32
                    write!(kernel, "let k_idx = ").unwrap();
                    k_tensor.strided_index(kernel, ["batch_idx", "kv_head_idx", "k_seq", "d_idx"]);
                    writeln!(kernel, ";").unwrap();
                    writeln!(kernel, "let k_val = f32({k_tensor}[k_idx]);").unwrap();

                    writeln!(kernel, "score += q_val * k_val;").unwrap();
                }
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "score *= {};", self.scale).unwrap();

                // Apply attention mask if provided
                if let Some(mask) = &mask_tensor {
                    write!(kernel, "let mask_idx = ").unwrap();
                    mask.strided_index(kernel, ["seq_idx", "k_seq"]);
                    writeln!(kernel, ";").unwrap();
                    writeln!(kernel, "score = score + f32({mask}[mask_idx]);").unwrap();
                }

                // Load V value for the output dimension we're computing and convert to f32
                write!(kernel, "let v_idx = ").unwrap();
                v_tensor.strided_index(kernel, ["batch_idx", "kv_head_idx", "k_seq", "out_dim"]);
                writeln!(kernel, ";").unwrap();
                writeln!(kernel, "let v_val = f32({v_tensor}[v_idx]);").unwrap();

                // Online softmax update
                writeln!(kernel, "let old_m = m;").unwrap();
                writeln!(kernel, "m = max(m, score);").unwrap();
                writeln!(kernel, "let exp_old_m_diff = exp(old_m - m);").unwrap();
                writeln!(kernel, "let exp_score_diff = exp(score - m);").unwrap();
                writeln!(kernel, "d = d * exp_old_m_diff + exp_score_diff;").unwrap();
                writeln!(
                    kernel,
                    "acc = acc * exp_old_m_diff + exp_score_diff * v_val;"
                )
                .unwrap();
            }
            writeln!(kernel, "}}").unwrap();

            // Write output - convert back to output dtype
            write!(kernel, "let out_idx = ").unwrap();
            output_tensor.strided_index(kernel, ["batch_idx", "head_idx", "seq_idx", "out_dim"]);
            writeln!(kernel, ";").unwrap();
            writeln!(kernel, "{output_tensor}[out_idx] = {dtype}(acc / d);").unwrap();
        }
    }
}

impl Operation for FlashAttentionOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        if device.subgroups_supported() {
            // For subgroup-based implementation, use subgroup size as base
            constraints.add_constraint(
                0,
                Constraint::more_than_or_equals(device.min_subgroup_size()),
            );
            constraints.add_constraint(
                0,
                Constraint::less_than_or_equals(device.max_subgroup_size()),
            );
        } else {
            // Fallback: fixed workgroup size
            constraints.add_constraint(0, Constraint::equals(256));
        }
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let q_tensor = inputs[0].as_tensor().unwrap();
        let shape = q_tensor.layout().shape();
        let device = q_tensor.device();

        let workgroup_size = workgroup_shape.x();

        if device.subgroups_supported() {
            // Subgroup-based: each workgroup (containing 1 subgroup) handles one position [batch, head, seq]
            // workgroup_size == subgroup_size, so we need total_positions workgroups
            let total_positions = (shape[0] * shape[1] * shape[2]) as u32;
            distribute_workgroups(total_positions)
        } else {
            // Fallback: each thread handles one output element [batch, head, seq, dim]
            let total_elements = (shape[0] * shape[1] * shape[2] * shape[3]) as u32;
            let total_workgroups = total_elements.div_ceil(workgroup_size);
            distribute_workgroups(total_workgroups)
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        f(self.v);
        if let Some(mask) = self.mask {
            f(mask);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let q_tensor = nodes.get_cached_result(self.q).unwrap();
        let k_tensor = nodes.get_cached_result(self.k).unwrap();
        let v_tensor = nodes.get_cached_result(self.v).unwrap();

        let shape = q_tensor.layout().shape();
        let output_type = self.out_datatype();
        let output_tensor = TensorData::new_for_shape(q_tensor.device(), shape, output_type);

        let mut inputs = vec![
            MirValue::Tensor(q_tensor.clone()),
            MirValue::Tensor(k_tensor.clone()),
            MirValue::Tensor(v_tensor.clone()),
        ];

        if let Some(mask_idx) = self.mask {
            let mask_tensor = nodes.get_cached_result(mask_idx).unwrap();
            inputs.push(MirValue::Tensor(mask_tensor.clone()));
        }

        inputs.push(MirValue::Tensor(output_tensor.clone()));
        inputs
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let max_blocksize = workgroup_shape.x();
        self.kernel(workgroup_shape, max_blocksize, kernel, &graph.device());
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        // Output is the last input (after q, k, v, and optional mask)
        let output_idx = if self.mask.is_some() { 4 } else { 3 };
        let output_tensor: TensorData = inputs[output_idx].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        if self.mask.is_some() {
            "flash_attention_masked".to_string()
        } else {
            "flash_attention".to_string()
        }
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input_layout = map.get(&self.q).unwrap();
        input_layout.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention() {
    use crate::Device;

    let device = Device::test_instance();

    // Test flash attention - 4D tensors [batch, heads, seq, dim]
    let q_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let k_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let v_data = [[[[1.0f32, 2.0], [3.0, 4.0]]]]; // [1, 1, 2, 2]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    let scale = 1.0 / (2.0_f32.sqrt());

    // Test flash attention
    let output = q.flash_attention(&k, &v, scale, None);
    let result = output.as_slice().await.unwrap();

    // Compare with standard attention (non-fused implementation)
    let scores = q.mat_mul(&k.t()) * scale;
    let attn_weights = scores.softmax_last_dim();
    let expected = attn_weights.mat_mul(&v);
    let expected_result = expected.as_slice().await.unwrap();

    // Compare flash attention output against standard attention
    let tolerance = 0.01;
    for i in 0..2 {
        for j in 0..2 {
            let flash_val = result[[0, 0, i, j]];
            let std_val = expected_result[[0, 0, i, j]];
            assert!(
                (flash_val - std_val).abs() < tolerance,
                "Mismatch at [{}, {}]: flash={}, standard={}",
                i,
                j,
                flash_val,
                std_val
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_causal_mask() {
    use crate::Device;

    let device = Device::test_instance();

    // Test flash attention with causal mask - 4D tensors [batch, heads, seq, dim]
    let q_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let k_data = [[[[1.0f32, 0.0], [0.0, 1.0]]]]; // [1, 1, 2, 2]
    let v_data = [[[[1.0f32, 2.0], [3.0, 4.0]]]]; // [1, 1, 2, 2]

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);

    // Create causal mask: lower triangular with 0s, upper triangular with -inf
    // For seq_len=2: [[0, -inf], [0, 0]]
    let neg_inf = f32::NEG_INFINITY;
    let causal_mask_data = [[0.0f32, neg_inf], [0.0, 0.0]]; // [2, 2]
    let causal_mask = Tensor::new(&device, &causal_mask_data);

    let scale = 1.0 / (2.0_f32.sqrt());

    // Test flash attention with causal mask
    let output = q.flash_attention(&k, &v, scale, Some(&causal_mask));
    let result = output.as_slice().await.unwrap();

    // Compare with standard masked attention (non-fused implementation)
    let scores = q.mat_mul(&k.t()) * scale;
    // Reshape 2D mask [seq, seq] to 4D [1, 1, seq, seq] for broadcasting
    let causal_mask_4d: Tensor<4, f32> = causal_mask.reshape([1, 1, 2, 2]);
    let masked_scores = scores + causal_mask_4d;
    let attn_weights = masked_scores.softmax_last_dim();
    let expected = attn_weights.mat_mul(&v);
    let expected_result = expected.as_slice().await.unwrap();

    // Compare flash attention output against standard masked attention
    let tolerance = 0.01;
    for i in 0..2 {
        for j in 0..2 {
            let flash_val = result[[0, 0, i, j]];
            let std_val = expected_result[[0, 0, i, j]];
            assert!(
                (flash_val - std_val).abs() < tolerance,
                "Mismatch at [{}, {}]: flash={}, standard={}",
                i,
                j,
                flash_val,
                std_val
            );
        }
    }

    // Additional check: for causal mask, first row should only attend to first position
    // So first row output should equal first row of V
    assert!(
        (result[[0, 0, 0, 0]] - v_data[0][0][0][0]).abs() < tolerance,
        "First position should attend only to itself with causal mask: got {}, expected {}",
        result[[0, 0, 0, 0]],
        v_data[0][0][0][0]
    );
    assert!(
        (result[[0, 0, 0, 1]] - v_data[0][0][0][1]).abs() < tolerance,
        "First position should attend only to itself with causal mask: got {}, expected {}",
        result[[0, 0, 0, 1]],
        v_data[0][0][0][1]
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_masked_fuzz() {
    use crate::Device;

    let device = Device::test_instance();

    // Test multiple random configurations
    for _iteration in 0..100 {
        // Random dimensions (keep small for test speed)
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        let heads = (rand::random::<u32>() as usize % 16) + 1;
        let seq_len = (rand::random::<u32>() as usize % 128) + 1;
        let head_dim = (rand::random::<u32>() as usize % 64) + 1;

        // Generate random Q, K, V tensors as nested Vecs
        let q_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let k_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let v_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Generate causal mask
        let mask_data: Vec<Vec<f32>> = (0..seq_len)
            .map(|row| {
                (0..seq_len)
                    .map(|col| {
                        if col > row {
                            // Upper triangular: masked out
                            f32::NEG_INFINITY
                        } else {
                            // Lower triangular: small random values
                            rand::random::<f32>() * 0.5 - 0.25
                        }
                    })
                    .collect()
            })
            .collect();

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let mask = Tensor::new(&device, &mask_data);

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Flash attention with mask
        let flash_output = q.flash_attention(&k, &v, scale, Some(&mask));
        let flash_result = flash_output.as_slice().await.unwrap();

        // Standard attention with mask
        let scores = q.mat_mul(&k.t()) * scale;
        let mask_4d: Tensor<4, f32> = mask.reshape([1, 1, seq_len, seq_len]);
        let mask_broadcast = mask_4d.broadcast_as([batch, heads, seq_len, seq_len]);
        let masked_scores = scores + mask_broadcast;
        let attn_weights = masked_scores.softmax_last_dim();
        let std_output = attn_weights.mat_mul(&v);
        let std_result = std_output.as_slice().await.unwrap();

        // Compare results
        let tolerance = 0.01;
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let flash_val = flash_result[[b, h, s, d]];
                        let std_val = std_result[[b, h, s, d]];

                        assert!(
                            (flash_val - std_val).abs() < tolerance,
                            "Mismatch at [{}, {}, {}, {}]: flash={}, standard={} \
                             (batch={}, heads={}, seq_len={}, head_dim={})",
                            b,
                            h,
                            s,
                            d,
                            flash_val,
                            std_val,
                            batch,
                            heads,
                            seq_len,
                            head_dim
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_kv_cache_fuzz() {
    use crate::Device;

    let device = Device::test_instance();

    // Fuzz test for KV cache scenarios with various dimension combinations
    for _iteration in 0..50 {
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        let heads = (rand::random::<u32>() as usize % 8) + 1;
        let q_seq_len = (rand::random::<u32>() as usize % 4) + 1; // Small Q sequence (1-4)
        let kv_seq_len = q_seq_len + (rand::random::<u32>() as usize % 32); // KV >= Q
        let head_dim = (rand::random::<u32>() as usize % 64) + 1;

        let q_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..q_seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let k_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..kv_seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let v_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..kv_seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Mask: [q_seq_len, kv_seq_len] - allow attending to all positions
        let mask_data: Vec<Vec<f32>> = (0..q_seq_len).map(|_| vec![0.0; kv_seq_len]).collect();

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let mask = Tensor::new(&device, &mask_data);

        let scale = 1.0 / (head_dim as f32).sqrt();

        let flash_output = q.flash_attention(&k, &v, scale, Some(&mask));
        let flash_result = flash_output.as_slice().await.unwrap();

        // Standard attention
        let scores = q.mat_mul(&k.t()) * scale;
        let mask_4d: Tensor<4, f32> = mask.reshape([1, 1, q_seq_len, kv_seq_len]);
        let mask_broadcast = mask_4d.broadcast_as([batch, heads, q_seq_len, kv_seq_len]);
        let masked_scores = scores + mask_broadcast;
        let attn_weights = masked_scores.softmax_last_dim();
        let std_output = attn_weights.mat_mul(&v);
        let std_result = std_output.as_slice().await.unwrap();

        let tolerance = 0.01;
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..q_seq_len {
                    for d in 0..head_dim {
                        let flash_val = flash_result[[b, h, s, d]];
                        let std_val = std_result[[b, h, s, d]];

                        assert!(
                            (flash_val - std_val).abs() < tolerance,
                            "KV cache fuzz mismatch at [{}, {}, {}, {}]: flash={}, standard={} \
                             (batch={}, heads={}, q_seq_len={}, kv_seq_len={}, head_dim={})",
                            b,
                            h,
                            s,
                            d,
                            flash_val,
                            std_val,
                            batch,
                            heads,
                            q_seq_len,
                            kv_seq_len,
                            head_dim
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_flash_attention_gqa_fuzz() {
    use crate::Device;

    let device = Device::test_instance();

    // Fuzz test for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
    // where K/V have fewer heads than Q
    for _iteration in 0..50 {
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        // Choose num_kv_heads first, then make num_heads a multiple of it
        let num_kv_heads = (rand::random::<u32>() as usize % 4) + 1;
        let num_key_value_groups = (rand::random::<u32>() as usize % 4) + 1;
        let num_heads = num_kv_heads * num_key_value_groups;
        let seq_len = (rand::random::<u32>() as usize % 32) + 1;
        let head_dim = (rand::random::<u32>() as usize % 64) + 1;

        // Q: [batch, num_heads, seq_len, head_dim]
        let q_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..num_heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // K/V: [batch, num_kv_heads, seq_len, head_dim]
        let k_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..num_kv_heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let v_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|_| {
                (0..num_kv_heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|_| {
                                (0..head_dim)
                                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Flash attention with GQA
        let flash_output = q.flash_attention(&k, &v, scale, None);
        let flash_result = flash_output.as_slice().await.unwrap();

        // Reference: expand K/V to match Q heads, then do standard attention
        let k_expanded: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|b| {
                (0..num_heads)
                    .map(|h| {
                        let kv_h = h / num_key_value_groups;
                        k_data[b][kv_h].clone()
                    })
                    .collect()
            })
            .collect();
        let v_expanded: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
            .map(|b| {
                (0..num_heads)
                    .map(|h| {
                        let kv_h = h / num_key_value_groups;
                        v_data[b][kv_h].clone()
                    })
                    .collect()
            })
            .collect();

        let k_exp = Tensor::new(&device, &k_expanded);
        let v_exp = Tensor::new(&device, &v_expanded);

        let scores = q.mat_mul(&k_exp.t()) * scale;
        let attn_weights = scores.softmax_last_dim();
        let std_output = attn_weights.mat_mul(&v_exp);
        let std_result = std_output.as_slice().await.unwrap();

        let tolerance = 0.01;
        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let flash_val = flash_result[[b, h, s, d]];
                        let std_val = std_result[[b, h, s, d]];

                        assert!(
                            (flash_val - std_val).abs() < tolerance,
                            "GQA fuzz mismatch at [{}, {}, {}, {}]: flash={}, standard={} \
                             (batch={}, num_heads={}, num_kv_heads={}, seq_len={}, head_dim={})",
                            b,
                            h,
                            s,
                            d,
                            flash_val,
                            std_val,
                            batch,
                            num_heads,
                            num_kv_heads,
                            seq_len,
                            head_dim
                        );
                    }
                }
            }
        }
    }
}
