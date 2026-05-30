#[cfg(feature = "candle")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "candle")]
use candle_core::{
    backend::BackendStorage,
    quantized::{GgmlDType, QMatMul, QTensor},
    utils::*,
    Device, Storage, Tensor, WithDType,
};

mod cache;
pub use cache::*;

#[cfg(target_arch = "wasm32")]
pub mod opfs;
#[cfg(target_arch = "wasm32")]
pub use opfs::*;
#[cfg(feature = "candle")]
mod kv_cache;
#[cfg(feature = "candle")]
pub use kv_cache::*;
#[cfg(feature = "candle")]
mod mask;
#[cfg(feature = "candle")]
pub use mask::*;

#[cfg(feature = "candle")]
/// Create a candle device that uses any available accelerator.
pub fn accelerated_device_if_available() -> candle_core::Result<Device> {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    if let Some(device) = DEVICE.get() {
        return Ok(device.clone());
    }
    let device = if cuda_is_available() {
        Device::new_cuda(0)?
    } else if metal_is_available() {
        Device::new_metal(0)?
    } else {
        #[cfg(all(debug_assertions, target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with the metal feature enabled. If you don't have access to an accelerator make sure you are running in release mode with `--release`. Models will run extremely slowly in debug mode on the CPU");
        }
        #[cfg(not(all(debug_assertions, target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with the cuda feature enabled. If you don't have access to an accelerator make sure you are running in release mode with `--release`. Models will run extremely slowly in debug mode on the CPU");
        }
        Device::Cpu
    };
    let _ = DEVICE.set(device.clone());
    Ok(device)
}

#[cfg(feature = "candle")]
/// Wrap a closure in a release pool if the metal feature is enabled
pub fn maybe_autoreleasepool<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "metal")]
    // Adding a manual autoreleasepool here is necessary to avoid a memory leak https://github.com/huggingface/candle/issues/2271
    {
        metal::objc::rc::autoreleasepool(f)
    }
    #[cfg(not(feature = "metal"))]
    {
        f()
    }
}

#[cfg(feature = "candle")]
/// Clear a `Vec<T>` and copy the contents of a tensor into it.
pub fn copy_tensor_into_vec<T: WithDType>(
    tensor: &Tensor,
    into: &mut Vec<T>,
) -> candle_core::Result<()> {
    into.clear();
    assert_eq!(tensor.rank(), 1);
    let mut from_cpu_storage = |cpu_storage: &candle_core::CpuStorage,
                                layout: &candle_core::Layout| {
        let data = cpu_storage.as_slice()?;
        match layout.contiguous_offsets() {
            Some((o1, o2)) => into.extend_from_slice(&data[o1..o2]),
            None => {
                for logit in tensor.strided_index().map(|i| data[i]) {
                    into.push(logit);
                }
            }
        };
        candle_core::Result::Ok(())
    };
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(storage) => from_cpu_storage(storage, layout),
        Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?, layout),
        Storage::Metal(storage) => from_cpu_storage(&storage.to_cpu_storage()?, layout),
    }
}

#[cfg(feature = "candle")]
fn cuda_compatible_dequantize_f16(qtensor: &QTensor) -> candle_core::Result<Tensor> {
    let device = qtensor.device();
    qtensor
        .dequantize(&device)?
        .to_dtype(candle_core::DType::F16)?
        .to_device(&device)
}

/// Convert a QTensor to a QMatMul
#[cfg(feature = "candle")]
pub fn qmatmul_from_qtensor(qtensor: impl Into<Arc<QTensor>>) -> candle_core::Result<QMatMul> {
    let qtensor = qtensor.into();
    Ok(match qtensor.dtype() {
        GgmlDType::F32 => QMatMul::Tensor(qtensor.dequantize(&qtensor.device())?),
        GgmlDType::F16 => QMatMul::TensorF16(cuda_compatible_dequantize_f16(&qtensor)?),
        _ => QMatMul::QTensor(qtensor),
    })
}
