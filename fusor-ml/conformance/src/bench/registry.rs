//! Shared WebGPU benchmark registry.

use fusor::Device;

use super::{
    BenchmarkCase, BenchmarkConfig, BenchmarkError, BenchmarkEvent, BenchmarkReport,
    BenchmarkResult,
};

pub async fn run_cases(
    device: &Device,
    config: BenchmarkConfig,
    cases: impl IntoIterator<Item = BenchmarkCase>,
    mut progress: impl FnMut(BenchmarkEvent),
) -> BenchmarkResult<Vec<BenchmarkReport>> {
    let mut reports = Vec::new();
    for case in cases {
        let name = case.name().to_string();
        progress(BenchmarkEvent::Started(name.clone()));
        let report = case
            .run(device, config)
            .await
            .map_err(|err| -> BenchmarkError { format!("{name}: {err}").into() })?;
        progress(BenchmarkEvent::Finished(report.clone()));
        reports.push(report);
    }
    Ok(reports)
}

pub async fn run_case(
    name: &str,
    device: &Device,
    config: BenchmarkConfig,
) -> BenchmarkResult<BenchmarkReport> {
    let Some(case) = cases().into_iter().find(|case| case.name() == name) else {
        return Err(format!("unknown benchmark case: {name}").into());
    };
    case.run(device, config).await
}

macro_rules! registry {
    ($($case:ident),* $(,)?) => {
        pub fn cases() -> Vec<BenchmarkCase> {
            let mut cases = Vec::new();
            $(
                if let Some(mut burn_cases) = burn_cases_for_suite(
                    concat!("burn::", stringify!($case))
                ) {
                    cases.append(&mut burn_cases);
                }
                cases.push(crate::bench::webgpu::$case());
            )*
            cases
        }

        pub fn cases_for_suite(name: &str) -> Option<Vec<BenchmarkCase>> {
            match name {
                $(
                    concat!("webgpu::", stringify!($case)) => Some(vec![
                        crate::bench::webgpu::$case(),
                    ]),
                )*
                _ => burn_cases_for_suite(name),
            }
        }

        #[cfg(test)]
        mod generated_tests {
            use super::*;

            async fn gpu_device() -> Option<Device> {
                match Device::gpu().await {
                    Ok(device) => Some(device),
                    Err(err) => {
                        tracing::warn!("skipping WebGPU benchmark smoke test: {err}");
                        None
                    }
                }
            }

            $(
                #[allow(clippy::await_holding_lock)]
                #[tokio::test]
                async fn $case() {
                    let _gpu_guard = crate::suite::registry::gpu_test_guard();
                    let Some(device) = gpu_device().await else {
                        return;
                    };
                    let cases = crate::bench::registry::cases_for_suite(
                        concat!("webgpu::", stringify!($case))
                    )
                    .expect("registered benchmark suite should exist");
                    let reports = crate::bench::registry::run_cases(
                        &device,
                        BenchmarkConfig::smoke(),
                        cases,
                        |_| {},
                    )
                    .await
                    .unwrap();
                    for report in reports {
                        tracing::info!(
                            "{}: mean={:.3} ms median={:.3} ms stddev={:.3} ms samples={} iterations/sample={}",
                            report.name,
                            report.mean_ms,
                            report.median_ms,
                            report.stddev_ms,
                            report.samples,
                            report.iterations
                        );
                    }
                }
            )*
        }
    };
}

#[cfg(feature = "burn-bench")]
fn burn_cases_for_suite(name: &str) -> Option<Vec<BenchmarkCase>> {
    match name {
        "burn::elementwise_add_square" => Some(vec![crate::bench::burn::elementwise_add_square()]),
        "burn::elementwise_mul_rank4" => Some(vec![crate::bench::burn::elementwise_mul_rank4()]),
        "burn::unary_trig_chain" => Some(vec![crate::bench::burn::unary_trig_chain()]),
        "burn::activation_gelu" => Some(vec![crate::bench::burn::activation_gelu()]),
        "burn::broadcast_add" => Some(vec![crate::bench::burn::broadcast_add()]),
        "burn::transpose_then_elementwise" => {
            Some(vec![crate::bench::burn::transpose_then_elementwise()])
        }
        "burn::reduction_sum_last_dim" => Some(vec![crate::bench::burn::reduction_sum_last_dim()]),
        "burn::reduction_max_middle_axis" => {
            Some(vec![crate::bench::burn::reduction_max_middle_axis()])
        }
        "burn::softmax_last_dim" => Some(vec![crate::bench::burn::softmax_last_dim()]),
        "burn::softmax_middle_axis" => Some(vec![crate::bench::burn::softmax_middle_axis()]),
        "burn::layer_norm_last_dim" => Some(vec![crate::bench::burn::layer_norm_last_dim()]),
        "burn::rms_norm_fused" => Some(vec![crate::bench::burn::rms_norm_fused()]),
        "burn::dense_matmul_square" => Some(vec![crate::bench::burn::dense_matmul_square()]),
        "burn::dense_batched_matmul" => Some(vec![crate::bench::burn::dense_batched_matmul()]),
        "burn::conv1d_small" => Some(vec![crate::bench::burn::conv1d_small()]),
        "burn::top_k_large" => Some(vec![crate::bench::burn::top_k_large()]),
        "burn::top_k_qwen_vocab" => Some(vec![crate::bench::burn::top_k_qwen_vocab()]),
        "burn::q8_0_qgemv" => Some(vec![crate::bench::burn::q8_0_qgemv()]),
        "burn::q4k_qgemv" => Some(vec![crate::bench::burn::q4k_qgemv()]),
        "burn::q4k_paired_silu" => Some(vec![crate::bench::burn::q4k_paired_silu()]),
        "burn::flash_attention_small" => Some(vec![crate::bench::burn::flash_attention_small()]),
        "burn::flash_attention_causal_small" => {
            Some(vec![crate::bench::burn::flash_attention_causal_small()])
        }
        "burn::rope_fused_decode" => Some(vec![crate::bench::burn::rope_fused_decode()]),
        _ => None,
    }
}

#[cfg(not(feature = "burn-bench"))]
fn burn_cases_for_suite(_name: &str) -> Option<Vec<BenchmarkCase>> {
    None
}

#[cfg(all(test, feature = "burn-bench"))]
mod burn_generated_tests {
    use super::*;

    async fn gpu_device() -> Option<Device> {
        match Device::gpu().await {
            Ok(device) => Some(device),
            Err(err) => {
                tracing::warn!("skipping Burn benchmark smoke test: {err}");
                None
            }
        }
    }

    macro_rules! burn_tests {
        ($($case:ident),* $(,)?) => {
            $(
                #[allow(clippy::await_holding_lock)]
                #[tokio::test]
                async fn $case() {
                    let _gpu_guard = crate::suite::registry::gpu_test_guard();
                    let Some(device) = gpu_device().await else {
                        return;
                    };
                    let reports = crate::bench::registry::run_cases(
                        &device,
                        BenchmarkConfig::smoke(),
                        vec![crate::bench::burn::$case()],
                        |_| {},
                    )
                    .await
                    .unwrap();
                    for report in reports {
                        tracing::info!(
                            "{}: mean={:.3} ms median={:.3} ms stddev={:.3} ms samples={} iterations/sample={}",
                            report.name,
                            report.mean_ms,
                            report.median_ms,
                            report.stddev_ms,
                            report.samples,
                            report.iterations
                        );
                    }
                }
            )*
        };
    }

    burn_tests! {
        elementwise_add_square,
        elementwise_mul_rank4,
        unary_trig_chain,
        activation_gelu,
        broadcast_add,
        transpose_then_elementwise,
        reduction_sum_last_dim,
        reduction_max_middle_axis,
        softmax_last_dim,
        softmax_middle_axis,
        layer_norm_last_dim,
        rms_norm_fused,
        dense_matmul_square,
        dense_batched_matmul,
        conv1d_small,
        top_k_large,
        top_k_qwen_vocab,
        q8_0_qgemv,
        q4k_qgemv,
        q4k_paired_silu,
        flash_attention_small,
        flash_attention_causal_small,
        rope_fused_decode,
    }
}

registry! {
    elementwise_add_square,
    elementwise_mul_rank4,
    unary_trig_chain,
    activation_gelu,
    broadcast_add,
    transpose_then_elementwise,
    reduction_sum_last_dim,
    reduction_max_middle_axis,
    softmax_last_dim,
    softmax_middle_axis,
    layer_norm_last_dim,
    rms_norm_fused,
    dense_matmul_square,
    dense_batched_matmul,
    conv1d_small,
    top_k_large,
    top_k_qwen_vocab,
    q8_0_qgemv,
    q4k_qgemv,
    q4k_paired_silu,
    flash_attention_small,
    flash_attention_causal_small,
    rope_fused_decode,
}
