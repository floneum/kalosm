//! Shared checked-WebGPU replay and numerical checks for layout probes.
use fusor_ir::dtype::{QFmt, QLayout};
use std::{borrow::Cow, error::Error, sync::mpsc};
pub(super) type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(super) fn buffer(d: &wgpu::Device, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage,
        mapped_at_creation: false,
    })
}
pub(super) fn read(d: &wgpu::Device, b: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = b.slice(..).get_mapped_range().to_vec();
    b.unmap();
    Ok(bytes)
}
pub(super) fn pipeline(d: &wgpu::Device, source: &str) -> Result<wgpu::ComputePipeline> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Q4 column layout"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    Ok(d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    }))
}
pub(super) fn bind(
    d: &wgpu::Device,
    p: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &entries,
    })
}
pub(super) fn expected(
    fmt: QFmt,
    weights: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
    all: bool,
) -> Vec<(usize, f64)> {
    let columns: Vec<_> = if all {
        (0..n).collect()
    } else {
        (0..n.min(33))
            .map(|i| i * (n - 1) / (n.min(33) - 1).max(1))
            .collect()
    };
    let size = fmt.block_bytes(QLayout::Native) as usize;
    let mut decoded = [0f32; 256];
    columns
        .into_iter()
        .map(|col| {
            let mut sum = 0.;
            for (bk, values) in x.chunks_exact(256).enumerate() {
                let at = (col * k / 256 + bk) * size;
                fusor_gguf::blocks::cpu_dequantize_block(
                    fmt,
                    QLayout::Native,
                    &weights[at..at + size],
                    &mut decoded,
                );
                sum += decoded
                    .iter()
                    .zip(values)
                    .map(|(w, v)| f64::from(*w) * f64::from(*v))
                    .sum::<f64>();
            }
            (col, sum)
        })
        .collect()
}
pub(super) fn check(bytes: &[u8], reference: &[(usize, f64)], n: usize) -> Result<Vec<f32>> {
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if let Some(i) = values[..n].iter().position(|v| !v.is_finite()) {
        return Err(format!("nonfinite or unwritten output at column {i}").into());
    }
    if let Some(i) = values[n..].iter().position(|v| !v.is_nan()) {
        return Err(format!("overwritten tail canary at column {}", n + i).into());
    }
    for &(i, want) in reference {
        let error = (f64::from(values[i]) - want).abs();
        if error > 0.002 + want.abs() * 0.0002 {
            return Err(format!("column {i}: {} != {want} (error {error})", values[i]).into());
        }
    }
    Ok(values)
}
