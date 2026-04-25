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
    dequantized_f32: Arc<wgpu::Buffer>,
    datatype: GgmlType,
}

impl PartialEq for QMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.datatype == other.datatype
            && self.buffer == other.buffer
            && self.dequantized_f32 == other.dequantized_f32
    }
}

fn dequantize_blocks<B>(bytes: &[u8]) -> Vec<f32>
where
    B: GgufBlock,
    B::Dequantized: AsRef<[f32]>,
{
    bytemuck::cast_slice::<_, B>(bytes)
        .iter()
        .flat_map(|block| {
            block
                .dequantize()
                .as_ref()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn dequantize_bytes_to_f32(bytes: &[u8], ty: GgmlType) -> Vec<f32> {
    match ty {
        GgmlType::Q4_0 => dequantize_blocks::<BlockQ4_0>(bytes),
        GgmlType::Q5_0 => dequantize_blocks::<BlockQ5_0>(bytes),
        GgmlType::Q8_0 => dequantize_blocks::<BlockQ8_0>(bytes),
        GgmlType::Q4K => dequantize_blocks::<BlockQ4K>(bytes),
        GgmlType::Q5K => dequantize_blocks::<BlockQ5K>(bytes),
        GgmlType::Q6K => dequantize_blocks::<BlockQ6K>(bytes),
        GgmlType::F16 => bytemuck::cast_slice::<_, half::f16>(bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        GgmlType::F32 => bytemuck::cast_slice::<_, f32>(bytes).to_vec(),
        _ => todo!(),
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
        let dequantized = dequantize_bytes_to_f32(bytes, ty);
        assert_eq!(dequantized.len(), shape.iter().product::<usize>());
        let dequantized_f32 = device.create_buffer_init(
            bytemuck::cast_slice(&dequantized),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buffer = device.create_buffer_init(
            bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Ok(QMatrix {
            device: device.clone(),
            shape,
            buffer,
            dequantized_f32,
            datatype: ty,
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

    pub(crate) fn dequantized_f32_buffer(&self) -> Arc<wgpu::Buffer> {
        self.dequantized_f32.clone()
    }
}
