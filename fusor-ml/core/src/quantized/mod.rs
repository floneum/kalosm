use std::{num::NonZeroUsize, sync::Arc};

use crate::Device;
use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock,
    GgufMetadata, GgufReadError, GgufTensorMetadata,
};
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;

pub(crate) mod dequantize;
pub(crate) mod matmul;

const QMATRIX_DIRECT_PIPELINE_CACHE_SIZE: usize = 16;

#[derive(Clone)]
pub struct QMatrix {
    device: Device,
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
    direct_pipeline_cache: Arc<RwLock<LruCache<String, wgpu::ComputePipeline, FxBuildHasher>>>,
}

impl std::fmt::Debug for QMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QMatrix")
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("buffer", &self.buffer)
            .field("datatype", &self.datatype)
            .finish_non_exhaustive()
    }
}

impl PartialEq for QMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.datatype == other.datatype && self.buffer == other.buffer
    }
}

impl QMatrix {
    pub fn read_from_file<R: std::io::Read + std::io::Seek>(
        device: &Device,
        metadata: &GgufMetadata,
        reader: &mut R,
        key: &str,
    ) -> Result<Option<Self>, GgufReadError> {
        Ok(match metadata.tensor_infos.get(key) {
            Some(tensor) => Some(QMatrix::read(
                device,
                tensor,
                reader,
                metadata.tensor_data_offset,
            )?),
            None => None,
        })
    }

    pub fn read<R: std::io::Read + std::io::Seek>(
        device: &Device,
        metadata: &GgufTensorMetadata,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<Self, GgufReadError> {
        let bytes = metadata.read_tensor_bytes(reader, tensor_data_offset)?;
        let shape = metadata.shape.iter().map(|x| *x as usize).collect();
        QMatrix::from_parts(device, &bytes, shape, metadata.ty)
    }

    /// Create a QMatrix from raw quantized bytes.
    ///
    /// The primary buffer stores the block layout consumed by the typed qmatmul
    /// prototype. Explicit dequantize is lowered separately and does not keep a
    /// dense f32 backing here.
    pub fn from_parts(
        device: &Device,
        bytes: &[u8],
        shape: Box<[usize]>,
        ty: GgmlType,
    ) -> Result<Self, GgufReadError> {
        let storage_bytes: Box<[u8]> = match ty {
            GgmlType::Q4_0 => bytemuck::cast_slice::<_, BlockQ4_0>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ4_0::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::Q5_0 => bytemuck::cast_slice::<_, BlockQ5_0>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ5_0::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::Q8_0 => bytemuck::cast_slice::<_, BlockQ8_0>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ8_0::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::Q4K => bytemuck::cast_slice::<_, BlockQ4K>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ4K::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::Q5K => bytemuck::cast_slice::<_, BlockQ5K>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ5K::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::Q6K => bytemuck::cast_slice::<_, BlockQ6K>(bytes)
                .iter()
                .copied()
                .flat_map(BlockQ6K::into_gpu_storage_bytes_f32)
                .collect(),
            GgmlType::F16 => bytemuck::cast_slice::<_, half::f16>(bytes)
                .iter()
                .flat_map(|value| value.to_f32().to_le_bytes())
                .collect(),
            GgmlType::F32 => bytes.into(),
            unsupported => return Err(GgufReadError::UnsupportedDType(unsupported as u32)),
        };
        let datatype = if ty == GgmlType::F16 {
            GgmlType::F32
        } else {
            ty
        };
        let buffer = device.create_buffer_init(
            &storage_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Ok(QMatrix {
            device: device.clone(),
            shape,
            buffer,
            datatype,
            direct_pipeline_cache: Arc::new(RwLock::new(LruCache::with_hasher(
                NonZeroUsize::new(QMATRIX_DIRECT_PIPELINE_CACHE_SIZE).unwrap(),
                Default::default(),
            ))),
        })
    }

    pub(crate) fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub(crate) fn direct_pipeline_cache(
        &self,
    ) -> &RwLock<LruCache<String, wgpu::ComputePipeline, FxBuildHasher>> {
        &self.direct_pipeline_cache
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn datatype(&self) -> GgmlType {
        self.datatype
    }
}
