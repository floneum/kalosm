use std::{borrow::Cow, mem::size_of, num::NonZeroUsize, sync::Arc};

use crate::{Device, Layout};
use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock,
    GgufMetadata, GgufReadError, GgufTensorMetadata,
};
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;

pub(crate) mod dequantize;
pub(crate) mod embedding;
pub(crate) mod matmul;

const QMATRIX_DIRECT_PIPELINE_CACHE_SIZE: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct QMatMulDirectPipelineKey {
    format: u8,
    storage_layout: QMatrixStorageLayout,
    m: u32,
    k: u32,
    n: u32,
    // Structural hash of attached epilogues and lowering-only choices.
    // Zero for plain qmatmul; non-zero values disambiguate kernels whose
    // bindings/dispatch are otherwise identical.
    epilogue_identity: u64,
    dispatch_size: [u32; 3],
    input_layout: QMatMulDirectLayoutKey,
    output_layout: QMatMulDirectLayoutKey,
}

#[derive(Clone, Copy)]
pub(crate) struct QMatMulShape {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl QMatMulDirectPipelineKey {
    pub(crate) fn new(
        format: GgmlType,
        storage_layout: QMatrixStorageLayout,
        shape: QMatMulShape,
        dispatch_size: [u32; 3],
        input_layout: &Layout,
        output_layout: &Layout,
    ) -> Self {
        Self::new_with_epilogue(
            format,
            storage_layout,
            shape,
            0,
            dispatch_size,
            input_layout,
            output_layout,
        )
    }

    pub(crate) fn new_with_epilogue(
        format: GgmlType,
        storage_layout: QMatrixStorageLayout,
        shape: QMatMulShape,
        epilogue_identity: u64,
        dispatch_size: [u32; 3],
        input_layout: &Layout,
        output_layout: &Layout,
    ) -> Self {
        let QMatMulShape { m, k, n } = shape;
        Self {
            format: format as u8,
            storage_layout,
            m,
            k,
            n,
            epilogue_identity,
            dispatch_size,
            input_layout: QMatMulDirectLayoutKey::new(input_layout),
            output_layout: QMatMulDirectLayoutKey::new(output_layout),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum QMatMulDirectLayoutKey {
    Rank2 {
        offset: usize,
        shape: [usize; 2],
        strides: [usize; 2],
    },
    General {
        offset: usize,
        shape: Box<[usize]>,
        strides: Box<[usize]>,
    },
}

impl QMatMulDirectLayoutKey {
    fn new(layout: &Layout) -> Self {
        if layout.shape().len() == 2 && layout.strides().len() == 2 {
            Self::Rank2 {
                offset: layout.offset(),
                shape: [layout.shape()[0], layout.shape()[1]],
                strides: [layout.strides()[0], layout.strides()[1]],
            }
        } else {
            Self::General {
                offset: layout.offset(),
                shape: layout.shape().into(),
                strides: layout.strides().into(),
            }
        }
    }
}

fn padded_copy_size(size: u64) -> u64 {
    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    ((size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
}

fn quantized_storage_size<B: GgufBlock>(element_count: usize) -> Option<u64> {
    if !element_count.is_multiple_of(B::BLOCK_SIZE) {
        return None;
    }

    let blocks = element_count / B::BLOCK_SIZE;
    blocks
        .checked_mul(size_of::<B::BytesF32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
}

fn quantized_native_storage_size<B: GgufBlock>(element_count: usize) -> Option<u64> {
    if !element_count.is_multiple_of(B::BLOCK_SIZE) {
        return None;
    }

    let blocks = element_count / B::BLOCK_SIZE;
    blocks
        .checked_mul(size_of::<B::Bytes>())
        .and_then(|bytes| u64::try_from(bytes).ok())
}

fn matrix_storage_size(
    shape: &[usize],
    datatype: GgmlType,
    storage_layout: QMatrixStorageLayout,
) -> Option<u64> {
    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;

    match (datatype, storage_layout) {
        (GgmlType::Q4_0, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ4_0>(element_count)
        }
        (GgmlType::Q4_0, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ4_0>(element_count)
        }
        (GgmlType::Q5_0, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ5_0>(element_count)
        }
        (GgmlType::Q5_0, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ5_0>(element_count)
        }
        (GgmlType::Q8_0, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ8_0>(element_count)
        }
        (GgmlType::Q8_0, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ8_0>(element_count)
        }
        (GgmlType::Q4K, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ4K>(element_count)
        }
        (GgmlType::Q4K, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ4K>(element_count)
        }
        (GgmlType::Q5K, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ5K>(element_count)
        }
        (GgmlType::Q5K, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ5K>(element_count)
        }
        (GgmlType::Q6K, QMatrixStorageLayout::Native) => {
            quantized_native_storage_size::<BlockQ6K>(element_count)
        }
        (GgmlType::Q6K, QMatrixStorageLayout::GpuF32Scales) => {
            quantized_storage_size::<BlockQ6K>(element_count)
        }
        (GgmlType::F32, _) => element_count
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok()),
        (GgmlType::F16, _) => element_count
            .checked_mul(size_of::<half::f16>())
            .and_then(|bytes| u64::try_from(bytes).ok()),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum QMatrixStorageLayout {
    Native,
    GpuF32Scales,
}

fn env_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "native" => Some(true),
        "0" | "false" | "no" | "off" | "expanded" | "f32" => Some(false),
        _ => None,
    }
}

fn native_half_scale_storage_enabled(
    shader_f16_supported: bool,
    env_override: Option<&str>,
) -> bool {
    env_override
        .and_then(env_bool)
        .unwrap_or(shader_f16_supported)
}

fn qmatrix_storage_layout_for_parts(
    ty: GgmlType,
    shader_f16_supported: bool,
) -> QMatrixStorageLayout {
    qmatrix_storage_layout_for_parts_with_env(
        ty,
        shader_f16_supported,
        std::env::var("FUSOR_Q_NATIVE")
            .ok()
            .or_else(|| std::env::var("FUSOR_Q4K_NATIVE").ok())
            .as_deref(),
    )
}

fn qmatrix_storage_layout_for_parts_with_env(
    ty: GgmlType,
    shader_f16_supported: bool,
    env_override: Option<&str>,
) -> QMatrixStorageLayout {
    if matches!(
        ty,
        GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    ) && native_half_scale_storage_enabled(shader_f16_supported, env_override)
    {
        QMatrixStorageLayout::Native
    } else if matches!(
        ty,
        GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    ) {
        QMatrixStorageLayout::GpuF32Scales
    } else {
        QMatrixStorageLayout::Native
    }
}

#[derive(Clone)]
pub struct QMatrix {
    device: Device,
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
    storage_layout: QMatrixStorageLayout,
    direct_pipeline_cache:
        Arc<RwLock<LruCache<QMatMulDirectPipelineKey, wgpu::ComputePipeline, FxBuildHasher>>>,
}

impl std::fmt::Debug for QMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QMatrix")
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("buffer", &self.buffer)
            .field("datatype", &self.datatype)
            .field("storage_layout", &self.storage_layout)
            .finish_non_exhaustive()
    }
}

impl PartialEq for QMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.datatype == other.datatype
            && self.storage_layout == other.storage_layout
            && self.buffer == other.buffer
    }
}

impl QMatrix {
    pub fn concat_rows(matrices: &[&Self]) -> Option<Self> {
        let first = matrices.first().copied()?;
        if matrices.len() == 1 {
            return Some(first.clone());
        }
        if first.shape.len() != 2 {
            return None;
        }

        let datatype = first.datatype;
        let storage_layout = first.storage_layout;
        let device = first.device.clone();
        let columns = first.shape[1];
        let mut rows = 0usize;
        let mut storage_sizes = Vec::with_capacity(matrices.len());
        let mut total_storage_size = 0u64;

        for matrix in matrices {
            if matrix.shape.len() != 2
                || matrix.shape[1] != columns
                || matrix.datatype != datatype
                || matrix.storage_layout != storage_layout
                || !matrix.device.is_same_device(&device)
            {
                return None;
            }

            let storage_size =
                matrix_storage_size(&matrix.shape, matrix.datatype, matrix.storage_layout)?;
            if storage_size > matrix.buffer.size() {
                return None;
            }

            rows = rows.checked_add(matrix.shape[0])?;
            total_storage_size = total_storage_size.checked_add(storage_size)?;
            storage_sizes.push(storage_size);
        }

        let buffer = device.create_buffer(
            padded_copy_size(total_storage_size),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut command_encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("QMatrix row concat"),
                });
        let mut destination_offset = 0u64;
        for (matrix, storage_size) in matrices.iter().zip(storage_sizes) {
            command_encoder.copy_buffer_to_buffer(
                &matrix.buffer,
                0,
                &buffer,
                destination_offset,
                storage_size,
            );
            destination_offset += storage_size;
        }
        device.wgpu_queue().submit(Some(command_encoder.finish()));

        Some(QMatrix {
            device,
            shape: Box::new([rows, columns]),
            buffer,
            datatype,
            storage_layout,
            direct_pipeline_cache: Arc::new(RwLock::new(LruCache::with_hasher(
                NonZeroUsize::new(QMATRIX_DIRECT_PIPELINE_CACHE_SIZE).unwrap(),
                Default::default(),
            ))),
        })
    }

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
    /// The primary buffer stores the block layout consumed by the tiled qmatmul
    /// path. Explicit dequantize is lowered separately and does not keep a
    /// dense f32 backing here.
    pub fn from_parts(
        device: &Device,
        bytes: &[u8],
        shape: Box<[usize]>,
        ty: GgmlType,
    ) -> Result<Self, GgufReadError> {
        let storage_layout = qmatrix_storage_layout_for_parts(ty, device.f16_supported());
        let storage_bytes: Cow<'_, [u8]> = match (ty, storage_layout) {
            (GgmlType::Q4_0, QMatrixStorageLayout::Native) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ4_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ4_0::into_gpu_storage_bytes)
                    .collect(),
            ),
            // Q4_K/Q5_K raw GGUF bytes already match the native shader block
            // layout, so avoid the load-time block repack entirely.
            (GgmlType::Q4K, QMatrixStorageLayout::Native) => Cow::Borrowed(bytes),
            (GgmlType::Q5_0, QMatrixStorageLayout::Native) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ5_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ5_0::into_gpu_storage_bytes)
                    .collect(),
            ),
            (GgmlType::Q8_0, QMatrixStorageLayout::Native) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ8_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ8_0::into_gpu_storage_bytes)
                    .collect(),
            ),
            (GgmlType::Q5K, QMatrixStorageLayout::Native) => Cow::Borrowed(bytes),
            (GgmlType::Q6K, QMatrixStorageLayout::Native) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ6K>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ6K::into_gpu_storage_bytes)
                    .collect(),
            ),
            (GgmlType::Q4_0, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ4_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ4_0::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::Q5_0, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ5_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ5_0::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::Q8_0, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ8_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ8_0::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::Q4K, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ4K>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ4K::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::Q5K, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ5K>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ5K::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::Q6K, QMatrixStorageLayout::GpuF32Scales) => Cow::Owned(
                bytemuck::cast_slice::<_, BlockQ6K>(bytes)
                    .iter()
                    .copied()
                    .flat_map(BlockQ6K::into_gpu_storage_bytes_f32)
                    .collect(),
            ),
            (GgmlType::F16, _) => Cow::Borrowed(bytes),
            (GgmlType::F32, _) => Cow::Borrowed(bytes),
            (unsupported, _) => return Err(GgufReadError::UnsupportedDType(unsupported as u32)),
        };
        let datatype = ty;
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
            storage_layout,
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
    ) -> &RwLock<LruCache<QMatMulDirectPipelineKey, wgpu::ComputePipeline, FxBuildHasher>> {
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

    pub(crate) fn storage_layout(&self) -> QMatrixStorageLayout {
        self.storage_layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_capable_quant_storage_defaults_to_native_when_shader_f16_is_supported() {
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q5_0,
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ] {
            assert_eq!(
                qmatrix_storage_layout_for_parts_with_env(ty, true, None),
                QMatrixStorageLayout::Native
            );
            assert_eq!(
                qmatrix_storage_layout_for_parts_with_env(ty, false, None),
                QMatrixStorageLayout::GpuF32Scales
            );
        }
    }

    #[test]
    fn q4k_storage_env_can_force_native_or_f32_expanded() {
        assert_eq!(
            qmatrix_storage_layout_for_parts_with_env(GgmlType::Q4K, false, Some("1")),
            QMatrixStorageLayout::Native
        );
        assert_eq!(
            qmatrix_storage_layout_for_parts_with_env(GgmlType::Q4K, true, Some("0")),
            QMatrixStorageLayout::GpuF32Scales
        );
    }

    #[test]
    fn concat_rows_combines_f32_gpu_matrices() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let first_raw: Vec<u8> = (1..=8)
                .map(|value| value as f32)
                .flat_map(f32::to_le_bytes)
                .collect();
            let second_raw: Vec<u8> = (9..=12)
                .map(|value| value as f32)
                .flat_map(f32::to_le_bytes)
                .collect();
            let first =
                QMatrix::from_parts(&device, &first_raw, Box::new([2, 4]), GgmlType::F32).unwrap();
            let second =
                QMatrix::from_parts(&device, &second_raw, Box::new([1, 4]), GgmlType::F32).unwrap();

            let combined = QMatrix::concat_rows(&[&first, &second]).unwrap();
            let dequantized = combined.dequantize::<f32>();
            let values = dequantized.as_slice::<2, f32>().await.unwrap();

            assert_eq!(combined.shape(), &[3, 4]);
            assert_eq!(values.shape(), &[3, 4]);
            for row in 0..3 {
                for col in 0..4 {
                    assert_eq!(values[[row, col]], (row * 4 + col + 1) as f32);
                }
            }
        });
    }
}
