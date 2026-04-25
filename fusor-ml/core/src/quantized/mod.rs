use crate::Device;
use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock,
    GgufMetadata, GgufReadError, GgufTensorMetadata,
};
use std::sync::Arc;

pub(crate) mod dequantize;
pub(crate) mod matmul;

#[derive(Clone, Debug)]
pub struct QMatrix {
    device: Device,
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
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
            Some(rope_freq_weight) => {
                let rope_freq_weight = QMatrix::read(
                    device,
                    rope_freq_weight,
                    reader,
                    metadata.tensor_data_offset,
                )?;
                Some(rope_freq_weight)
            }
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
        let ty = metadata.ty;
        QMatrix::from_parts(device, &bytes, shape, ty)
    }

    pub fn from_parts(
        device: &Device,
        bytes: &[u8],
        shape: Box<[usize]>,
        ty: GgmlType,
    ) -> Result<Self, GgufReadError> {
        let use_f16 = device.f16_supported();
        let bytes: Box<[u8]> = match ty {
            GgmlType::Q4_0 => {
                let map = if use_f16 {
                    BlockQ4_0::into_wgsl_bytes
                } else {
                    BlockQ4_0::into_wgsl_bytes_f32
                };
                bytemuck::cast_slice::<_, BlockQ4_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(map)
                    .collect()
            }
            GgmlType::Q5_0 => {
                let map = if use_f16 {
                    BlockQ5_0::into_wgsl_bytes
                } else {
                    BlockQ5_0::into_wgsl_bytes_f32
                };
                bytemuck::cast_slice::<_, BlockQ5_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(map)
                    .collect()
            }
            GgmlType::Q8_0 => {
                let map = if use_f16 {
                    BlockQ8_0::into_wgsl_bytes
                } else {
                    BlockQ8_0::into_wgsl_bytes_f32
                };
                bytemuck::cast_slice::<_, BlockQ8_0>(bytes)
                    .iter()
                    .copied()
                    .flat_map(map)
                    .collect()
            }
            GgmlType::Q4K => {
                let slice = bytemuck::cast_slice::<_, BlockQ4K>(bytes);
                if use_f16 {
                    slice
                        .iter()
                        .copied()
                        .flat_map(BlockQ4K::into_wgsl_bytes)
                        .collect()
                } else {
                    slice
                        .iter()
                        .copied()
                        .flat_map(BlockQ4K::into_wgsl_bytes_f32)
                        .collect()
                }
            }
            GgmlType::Q5K => {
                let slice = bytemuck::cast_slice::<_, BlockQ5K>(bytes);
                if use_f16 {
                    slice
                        .iter()
                        .copied()
                        .flat_map(BlockQ5K::into_wgsl_bytes)
                        .collect()
                } else {
                    slice
                        .iter()
                        .copied()
                        .flat_map(BlockQ5K::into_wgsl_bytes_f32)
                        .collect()
                }
            }
            GgmlType::Q6K => {
                let map = if use_f16 {
                    BlockQ6K::into_wgsl_bytes
                } else {
                    BlockQ6K::into_wgsl_bytes_f32
                };
                bytemuck::cast_slice::<_, BlockQ6K>(bytes)
                    .iter()
                    .copied()
                    .flat_map(map)
                    .collect()
            }
            GgmlType::F16 => {
                if use_f16 {
                    bytes.into()
                } else {
                    bytemuck::cast_slice::<_, half::f16>(bytes)
                        .iter()
                        .flat_map(|f| f.to_f32().to_le_bytes())
                        .collect()
                }
            }
            GgmlType::F32 => bytes.into(),
            _ => todo!(),
        };
        let datatype = if ty == GgmlType::F16 && !use_f16 {
            GgmlType::F32
        } else {
            ty
        };
        let buffer = device.create_buffer_init(
            &bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Ok(QMatrix {
            device: device.clone(),
            shape,
            buffer,
            datatype,
        })
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
