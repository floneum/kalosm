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

        #[cfg(feature = "burn-bench")]
        fn burn_cases_for_suite(name: &str) -> Option<Vec<BenchmarkCase>> {
            match name {
                $(
                    concat!("burn::", stringify!($case)) => Some(vec![
                        crate::bench::burn::$case(),
                    ]),
                )*
                _ => None,
            }
        }

        #[cfg(not(feature = "burn-bench"))]
        fn burn_cases_for_suite(_name: &str) -> Option<Vec<BenchmarkCase>> {
            None
        }

        #[cfg(all(test, not(target_arch = "wasm32")))]
        mod generated_tests {
            use super::*;

            /// One smoke run per registered case, at the smallest config: the
            /// case builds, dispatches and fences. Skipped without a GPU.
            fn smoke(suite: &str) {
                let _gpu_guard = crate::harness::gpu_test_guard();
                pollster::block_on(async {
                    let Ok(device) = Device::gpu().await else {
                        eprintln!("skipping benchmark smoke test: no GPU");
                        return;
                    };
                    let cases = crate::bench::registry::cases_for_suite(suite)
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
                        eprintln!(
                            "{}: mean={:.3} ms median={:.3} ms stddev={:.3} ms samples={} iterations/sample={}",
                            report.name,
                            report.mean_ms,
                            report.median_ms,
                            report.stddev_ms,
                            report.samples,
                            report.iterations
                        );
                    }
                });
            }

            $(
                #[test]
                fn $case() {
                    smoke(concat!("webgpu::", stringify!($case)));
                }
            )*

            #[cfg(feature = "burn-bench")]
            mod burn {
                $(
                    #[test]
                    fn $case() {
                        super::smoke(concat!("burn::", stringify!($case)));
                    }
                )*
            }
        }
    };
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
    // `top_k_large` / `top_k_qwen_vocab` are not registered: the current
    // sampling path is an `n x n` rank-by-counting sort, so 65k and 152k
    // logits are 17 GB and 92 GB of device memory. They return with a real
    // top-k kernel.
    q8_0_qgemv,
    q4k_qgemv,
    q4k_paired_silu,
    attention_small,
    attention_causal_small,
    rope_fused_decode,
}
