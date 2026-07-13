//! The shared conformance case registry.
//!
//! Every promoted suite is listed here once as a `(name, fn)` entry. The suite
//! functions construct [`AssertionCase`] values without running them; both
//! consumers drive execution off the collected list:
//!   * `cargo test` — the `registry!` macro generates one `#[tokio::test]` per
//!     suite (see `generated_tests`) and runs the suite's assertion list, and
//!   * the in-browser WebGPU suite ([`super::webgpu::run_webgpu_kernel_suite`])
//!     iterates [`assertions`] so the browser runs the same assertion list.
//!
//! Cases run on every device returned by [`crate::available_devices`] (CPU
//! baseline + the GPU variant matrix), so the list is device-agnostic. Cases
//! that only make sense natively (timing, intentional panics) are not listed.

use crate::{AssertionCase, AssertionCases, CaseResult};

trait IntoAssertionCases {
    fn into_assertion_cases(self, name: &'static str) -> Vec<AssertionCase>;
}

impl IntoAssertionCases for AssertionCase {
    fn into_assertion_cases(self, _name: &'static str) -> Vec<AssertionCase> {
        vec![self]
    }
}

impl IntoAssertionCases for AssertionCases {
    fn into_assertion_cases(self, _name: &'static str) -> Vec<AssertionCase> {
        self.into_vec()
    }
}

fn suite_assertions(name: &'static str, cases: impl IntoAssertionCases) -> Vec<AssertionCase> {
    cases.into_assertion_cases(name)
}

#[cfg(test)]
pub(crate) fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub async fn run_cases(
    cases: impl IntoIterator<Item = AssertionCase>,
    mut progress: impl FnMut(&str),
) -> CaseResult {
    for case in cases {
        let name = case.name().to_string();
        let mut current_name = name.clone();
        progress(&name);
        {
            let mut case_progress = |variant: &str| {
                current_name = variant.to_string();
                progress(variant);
            };
            case.run_with_progress(&mut case_progress)
                .await
                .map_err(|err| -> crate::CaseError { format!("{current_name}: {err}").into() })?;
        }
    }
    Ok(())
}

pub async fn run_case(name: &str) -> CaseResult {
    let Some(case) = assertions().into_iter().find(|case| case.name() == name) else {
        return Err(format!("unknown conformance case: {name}").into());
    };
    case.run().await
}

macro_rules! registry {
    ($($module:ident :: $case:ident),* $(,)?) => {
        /// The full shared list of conformance assertions.
        pub fn assertions() -> Vec<AssertionCase> {
            let mut assertions = Vec::new();
            $(
                assertions.extend(suite_assertions(
                    concat!(stringify!($module), "::", stringify!($case)),
                    crate::suite::native::$module::$case(),
                ));
            )*
            assertions
        }

        /// Back-compat alias for callers that still refer to conformance cases.
        pub fn cases() -> Vec<AssertionCase> {
            assertions()
        }

        pub fn assertions_for_suite(name: &str) -> Option<Vec<AssertionCase>> {
            match name {
                $(
                    concat!(stringify!($module), "::", stringify!($case)) => {
                        Some(suite_assertions(
                            concat!(stringify!($module), "::", stringify!($case)),
                            crate::suite::native::$module::$case(),
                        ))
                    }
                )*
                _ => None,
            }
        }

        /// One `#[tokio::test]` per registered case, generated from the same
        /// list as [`cases`]. This is what `cargo test` runs natively: each
        /// generated test first constructs its suite's assertion list, then the
        /// shared runner executes those assertions.
        #[cfg(test)]
        mod generated_tests {
            $(
                #[allow(clippy::await_holding_lock)]
                #[tokio::test]
                async fn $case() {
                    let _gpu_guard = crate::suite::registry::gpu_test_guard();
                    let assertions = crate::suite::registry::assertions_for_suite(
                        concat!(stringify!($module), "::", stringify!($case))
                    )
                    .expect("registered conformance suite should exist");
                    crate::suite::registry::run_cases(assertions, |_| {})
                    .await
                    .unwrap();
                }
            )*
        }
    };
}

registry! {
    cache_ops::attention_mask_apply_broadcasts_to_varied_3d_and_4d_shapes,
    cache_ops::attention_mask_causal_matches_expected_on_varied_sizes,
    cache_ops::kv_cache_append_and_reset_work_across_varied_cases,
    cache_ops::mask_cache_supports_varied_offsets_and_sliding_windows,
    cache_ops::tensor_cache_append_and_reset_work_across_varied_cases,
    cache_ops::tensor_cache_gpu_lazy_appends_preserve_pending_writes,
    dtypes::f16_matmul_matches_host_reference,
    dtypes::f16_pairwise_ops_match_host_reference,
    dtypes::f16_reduce_post_abs_matches_host_reference,
    dtypes::f16_reductions_match_host_reference,
    dtypes::f16_unary_ops_match_host_reference,
    dtypes::f16_zeros_matches_expected,
    dtypes::f32_to_f16_round_trip_preserves_value,
    dtypes::u32_pairwise_add_matches_host_reference,
    elementwise_ops::activation_and_scalar_ops_match_host_reference,
    elementwise_ops::binary_ops_match_host_reference,
    elementwise_ops::comparison_and_conditionals_match_expected,
    elementwise_ops::large_tensor_binary_and_conditional_regressions,
    elementwise_ops::restricted_domain_unary_ops_match_host_reference,
    elementwise_ops::same_shape_binary_ops_match_host_reference,
    elementwise_ops::tanh_exact_saturation_at_large_magnitudes,
    elementwise_ops::unary_math_ops_match_host_reference,
    elementwise_ops::where_cond_fuzzed,
    attention_ops::attention_decode_tiled_matches_cpu_reference,
    attention_ops::attention_decode_tiled_with_transposed_q_matches_cpu_reference,
    attention_ops::attention_f16_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_f16_with_qk_mask_matches_cpu_reference,
    attention_ops::attention_gqa_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_without_subgroups_preserves_gpu_backend,
    attention_ops::attention_tiled_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_with_batch_key_mask_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_with_kv_cache_matches_cpu_reference_on_varied_shapes,
    attention_ops::attention_with_qk_mask_matches_cpu_reference_on_varied_shapes,
    fusion_behavior::gpu_attention_fuses_into_one_kernel,
    fusion_behavior::gpu_coop_matmul_fuses_pre_and_post_unary_chains,
    fusion_behavior::gpu_gelu_lowers_to_one_kernel,
    fusion_behavior::gpu_indexing_then_arithmetic_matches_cpu,
    fusion_behavior::gpu_matmul_then_unary_chain_fuses_into_one_kernel,
    fusion_behavior::gpu_nary_fusion_respects_binding_limit,
    fusion_behavior::gpu_nary_same_input_multiple_times_deduplicates_bindings,
    fusion_behavior::gpu_nary_triple_add_fuses_into_one_kernel,
    fusion_behavior::gpu_nary_unary_chain_fuses_into_one_kernel,
    fusion_behavior::gpu_nary_where_cond_fuses_into_one_kernel,
    fusion_behavior::gpu_reduce_then_gelu_uses_two_kernels,
    fusion_behavior::gpu_reduce_then_unary_chain_fuses_into_one_kernel,
    fusion_behavior::gpu_residual_rms_norm_fuses_into_one_kernel,
    fusion_behavior::gpu_unary_inputs_fuse_into_matmul_kernel,
    fusion_correctness::fused_cached_results_fuzzed,
    fusion_correctness::inplace_clone_immutability_fuzzed,
    fusion_correctness::nary_chain_then_pairwise_fuzzed,
    fusion_correctness::nary_mixed_ops_fuzzed,
    fusion_correctness::nary_nested_pairwise_fuzzed,
    fusion_correctness::nary_same_input_fuzzed,
    fusion_correctness::nary_triple_add_fuzzed,
    fusion_correctness::nary_unary_in_middle_fuzzed,
    fusion_correctness::nary_where_cond_fuzzed,
    layer_ops::conv1d_matches_cpu_reference_on_varied_shapes,
    layer_ops::conv1d_properties_match_configuration,
    layer_ops::embedding_lookup_matches_cpu_reference_on_varied_shapes,
    layer_ops::layer_norm_fused_cpu_matches_reference_on_varied_shapes,
    layer_ops::layer_norm_matches_cpu_reference_on_varied_shapes,
    layer_ops::rms_norm_matches_cpu_reference_on_varied_shapes,
    layout_ops::broadcast_as_non_contiguous_input_matches_expected_view,
    layout_ops::cat_stack_and_chunk_match_expected_views,
    layout_ops::restride_and_restride_layout_match_expected_views,
    layout_ops::shape_and_layout_ops_match_host_reference,
    layout_ops::sliding_window_then_transpose_then_reshape_matches_expected,
    layout_ops::sliding_window_with_cat_padding_matches_expected,
    layout_ops::tensor_i_op_matches_expected_views,
    layout_ops::transpose_reshape_consumed_by_elementwise_matches_expected,
    matmul_conv_pool::conv_and_pool_match_host_reference,
    matmul_conv_pool::conv2d_matches_host_reference,
    matmul_conv_pool::conv3d_matches_host_reference,
    matmul_conv_pool::f16_matmul_coop_tile_matches_host_reference,
    matmul_conv_pool::f16_matmul_multi_tile_matches_host_reference,
    matmul_conv_pool::matmul_attention_4d_matches_host_reference,
    matmul_conv_pool::matmul_batched_3d_matches_host_reference,
    matmul_conv_pool::matmul_identity_matrix,
    matmul_conv_pool::matmul_match_host_reference,
    matmul_conv_pool::matmul_non_affine_prefix_matches_host_reference,
    matmul_conv_pool::matmul_non_contiguous_input_matches_host_reference,
    matmul_conv_pool::matmul_sgemv_variants_match_host_reference,
    matmul_conv_pool::matmul_small_fixed_regression,
    matmul_conv_pool::matmul_transposed_operand_matches_host_reference,
    normalization_ops::softmax_and_normalization_match_reference_paths,
    normalization_ops::softmax_direct_boundary_lengths_match_reference,
    normalization_ops::softmax_direct_transposed_and_middle_axis_match_reference,
    normalization_ops::softmax_middle_axis_rank3_matches_reference,
    normalization_ops::softmax_slow_variants_match_reference,
    quantized_matmul::q5_0_q_mat_mul_single_row_splits_large_qgemv_dispatch,
    quantized_matmul::q8_0_dequantize_then_add_matches_cpu_reference,
    quantized_matmul::quantized_dequantize_matches_cpu_reference,
    quantized_matmul::quantized_q_mat_mul_matches_cpu_reference,
    quantized_matmul_batched::q4k_llama_decode_transpose_reshape_qmatmul_matches_one_hot_reference,
    quantized_matmul_batched::q_mat_mul_batched_layouts_match_host_reference,
    quantized_matmul_batched::q_mat_mul_batched_matches_unbatched_property,
    quantized_matmul_batched::q_mat_mul_consumes_transpose_reshape_copy_matches_cpu_reference,
    quantized_matmul_fusion::q4k_q6k_ffn_chain_matches_cpu_reference_for_decode_rows,
    quantized_matmul_fusion::q4k_qmatmul_fusion_kernels,
    quantized_matmul_fusion::q8_0_qmatmul_epilogue_tests,
    quantized_matmul_fusion::rmsnorm_post_relu_resolves_to_single_kernel,
    quantized_matmul_paired::q4k_concat_split_gated_natural_form_matches_cpu_reference,
    quantized_matmul_paired::q4k_concat_split_llama_shape_match_cpu_reference,
    quantized_matmul_paired::q4k_dynamic_paired_helper_swiglu_matches_cpu_reference_for_decode_row,
    rank_and_empty::empty_tensor_elementwise_add_returns_empty,
    rank_and_empty::empty_tensor_sum_along_zero_axis_returns_identity,
    rank_and_empty::rank4_reductions_match_reference,
    reductions_indexing::full_tensor_sum_large_fuzzed,
    reductions_indexing::index_select_embedding_width_regression,
    reductions_indexing::index_select_fuzzed,
    reductions_indexing::index_select_single_rank_and_large_regressions,
    reductions_indexing::indexing_cast_and_rank_specific_indexing_match_reference,
    reductions_indexing::middle_axis_rank3_reductions_match_host_reference,
    reductions_indexing::reductions_match_host_reference,
    rope_ops::rope_and_cache_paths_match_reference_variants,
    tensor_construction_smoke::construction_aliases_match_on_varied_shapes,
    tensor_construction_smoke::device_wrappers_and_variant_accessors_work,
}
