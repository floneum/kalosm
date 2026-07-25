//! Cooperative-matrix conformance over every tile-table geometry: through
//! the selector on shapes derived from each entry, and with each entry's
//! tile forced through the kernel builder.

use fusor_core::{Device, Layout, Tensor};
use fusor_tile_ir::{CoopMatrixToken, ScalarElement, Shape, SubgroupToken, tile};
use fusor_tile_ir_kernels::{
    DEFAULT_SWIZZLE_GROUP_M, DenseCoopMatmulConfig, DenseCoopMatmulTile, DenseMatmulEpilogues,
    DenseMatmulShape, DenseMatmulTensors, SubgroupConfig, coop_tile_entries,
    try_batched_coop_matmul,
};

fn values(len: usize, freq: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * freq).sin()).collect()
}

/// f64-accumulated reference for `a[batch, m, k] @ b[batch, k, n]`.
fn cpu_matmul(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f64;
                for ki in 0..k {
                    let a_val = a[(bi * m + mi) * k + ki] as f64;
                    let b_val = b[(bi * k + ki) * n + ni] as f64;
                    acc += a_val * b_val;
                }
                out[(bi * m + mi) * n + ni] = acc as f32;
            }
        }
    }
    out
}

async fn check_automatic(
    device: &Device,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) {
    let a_data = values(batch * m * k, 0.13);
    let b_data = values(batch * k * n, 0.07);

    let a = Tensor::from_slice(device, [batch, m, k], &a_data);
    let b = if transpose_b {
        let b_t_data: Vec<f32> = (0..batch * n * k)
            .map(|i| {
                let (bi, rest) = (i / (n * k), i % (n * k));
                let (ni, ki) = (rest / k, rest % k);
                b_data[(bi * k + ki) * n + ni]
            })
            .collect();
        let b_t = Tensor::from_slice(device, [batch, n, k], &b_t_data);
        b_t.restride_layout(Layout::from_parts(
            0,
            vec![batch, k, n].into(),
            vec![n * k, 1, k].into(),
        ))
    } else {
        Tensor::from_slice(device, [batch, k, n], &b_data)
    };

    let out = a.mat_mul(&b);
    let actual = out.as_slice::<3, f32>().await.unwrap();
    let expected = cpu_matmul(&a_data, &b_data, batch, m, k, n);
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let want = expected[(bi * m + mi) * n + ni];
                let got = actual[[bi, mi, ni]];
                assert!(
                    (got - want).abs() < 1e-3 + want.abs() * 1e-3,
                    "batch={batch} m={m} k={k} n={n} transpose_b={transpose_b} \
                     [{bi}, {mi}, {ni}]: got {got}, expected {want}",
                );
            }
        }
    }
}

/// Build one tile geometry's kernel directly and check its whole logical
/// output against the host reference. The output buffer covers whole tiles,
/// so the rows and columns past `m` and `n` hold pad values the comparison
/// skips.
fn check_forced(device: &Device, tile: DenseCoopMatmulTile, m: u32, k: u32, n: u32) {
    let a_data = values((m * k) as usize, 0.13);
    let b_data = values((k * n) as usize, 0.07);
    let m_pad = m.div_ceil(tile.bm) * tile.bm;
    let n_pad = n.div_ceil(tile.bn) * tile.bn;
    let a_buf = device.create_buffer_init(
        bytemuck::cast_slice(&a_data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let b_buf = device.create_buffer_init(
        bytemuck::cast_slice(&b_data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let y_buf = device.create_buffer(
        u64::from(m_pad) * u64::from(n_pad) * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let element = ScalarElement::F32.element();
    let mut built = false;
    let ir = tile::build(|program| {
        let a = program.storage_read(element, Shape::new([m, k]));
        let b = program.storage_read(element, Shape::new([k, n]));
        let y = program.storage_write(element, Shape::new([m_pad, n_pad]));
        built = try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            DenseMatmulShape { batch: 1, m, k, n },
            &DenseMatmulEpilogues::empty(),
            65535,
            DenseCoopMatmulConfig {
                coop: CoopMatrixToken::new_unchecked(),
                subgroups: SubgroupConfig::fixed(SubgroupToken::new_unchecked(), 32),
                tile,
                staging: None,
                swizzle_group_m: DEFAULT_SWIZZLE_GROUP_M,
            },
        );
    });
    assert!(built, "{tile:?} declined {m}x{k}x{n}");
    let grid = ir.grid;
    let kernel = ir
        .lower_to_naga()
        .unwrap_or_else(|error| panic!("{tile:?} lowering failed: {error}"));

    let wgpu_device = device.wgpu_device();
    let module = unsafe {
        wgpu_device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(kernel.module().clone())),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    });
    let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buf.as_entire_binding(),
            },
        ],
    });
    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: y_buf.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = wgpu_device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
    }
    encoder.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, staging.size());
    device.wgpu_queue().submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device.poll_wait();
    receiver.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();
    let actual: &[f32] = bytemuck::cast_slice(&view);

    let expected = cpu_matmul(&a_data, &b_data, 1, m as usize, k as usize, n as usize);
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let want = expected[mi * n as usize + ni];
            let got = actual[mi * n_pad as usize + ni];
            assert!(
                (got - want).abs() < 1e-3 + want.abs() * 1e-3,
                "{tile:?} m={m} k={k} n={n} [{mi}, {ni}]: got {got}, expected {want}",
            );
        }
    }
}

/// The scorer orders least-padded first and fewest passes next, and padded
/// MACs are monotone in tile extent, so every multi-pass entry — and the
/// single-buffered one selection skips outright — is unreachable through the
/// automatic sweep below. The 128x128 profile shipped an all-zero miscompile
/// behind exactly that gap, so each entry's kernel is driven directly here,
/// tile-aligned and with every extent past its tile.
#[test]
fn forced_coop_tiles_compute_their_own_geometry_correctly() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        if device.fixed_width_subgroup_size() != Some(32)
            || !device
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        {
            return;
        }
        for entry in coop_tile_entries() {
            let tile = entry.tile;
            check_forced(&device, tile, tile.bm, 48, tile.bn);
            check_forced(&device, tile, tile.bm + 7, 30, tile.bn + 5);
        }
    });
}

#[test]
fn automatic_coop_selection_computes_table_derived_shapes_correctly() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let geometries: Vec<(u32, u32)> = coop_tile_entries()
            .iter()
            .map(|entry| (entry.tile.bm, entry.tile.bn))
            .collect();
        for (bm, bn) in geometries {
            let (m, n) = (2 * bm as usize, 2 * bn as usize);
            // Aligned, all edges masked at once, batched, and transposed-B.
            check_automatic(&device, 1, m, 64, n, false).await;
            check_automatic(&device, 1, m - 13, 50, n.saturating_sub(9).max(1), false).await;
            check_automatic(&device, 3, m, 64, n, false).await;
            check_automatic(&device, 2, m, 64, n, true).await;
        }
    });
}
