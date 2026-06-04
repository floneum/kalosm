//! Runs the in-browser WebGPU conformance suite natively against the GPU
//! device. This is the same `run_webgpu_kernel_suite` the kalosm-chat web app
//! executes on the `/conformance` route, exercised here so the cases can be
//! iterated quickly without a browser.
//!
//! The suite internally expands each case across the {subgroups, no subgroups}
//! × {cold pool, poisoned pool} device matrix, so the no-subgroup kernel
//! fallbacks — the only path the web build ever takes — are covered on native
//! too.
//!
//! To approximate the browser quantized storage layout on a native GPU, set:
//!   FUSOR_Q_NATIVE=0   -> forces the `GpuF32Scales` quantized storage layout
//!                         (the web build always disables `SHADER_F16`, so it
//!                         never uses the native f16-scale layout).

use fusor::Device;
use fusor_conformance::{available_devices, suite::webgpu::run_webgpu_kernel_suite};

#[tokio::test]
async fn webgpu_kernel_suite_runs_on_gpu() {
    let mut ran_on_gpu = false;
    for device in available_devices().await {
        if let Device::Gpu(_) = device {
            ran_on_gpu = true;
            run_webgpu_kernel_suite(&device)
                .await
                .expect("webgpu kernel suite should pass on the GPU device");
        }
    }
    assert!(
        ran_on_gpu,
        "no GPU device was available to run the webgpu kernel suite"
    );
}
