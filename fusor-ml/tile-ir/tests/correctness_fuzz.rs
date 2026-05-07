use std::{borrow::Cow, error::Error, sync::mpsc};

use fusor_tile_ir::{tile, GgmlQuantFormat, KernelIr, Layout, MemoryLevel, Shape, Strides, F32};
use wgpu::util::DeviceExt;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Copy, Clone, Debug)]
enum StorageVariant {
    Contiguous,
    Padded,
    Transposed,
}

impl StorageVariant {
    const ALL: [Self; 3] = [Self::Contiguous, Self::Padded, Self::Transposed];

    fn name(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Padded => "padded",
            Self::Transposed => "transposed",
        }
    }
}

#[test]
#[ignore = "requires a WGPU adapter"]
fn fuzz_gemv_gemm_qgemv_qgemm_correctness() -> TestResult {
    pollster::block_on(run_fuzz())
}

async fn run_fuzz() -> TestResult {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    let adapter_features = adapter.features();
    if !adapter_features.contains(wgpu::Features::SUBGROUP) {
        eprintln!(
            "skipping GPU correctness fuzz: adapter {:?} lacks subgroup support",
            adapter.get_info(),
        );
        return Ok(());
    }
    let required_features = wgpu::Features::SUBGROUP;

    if !adapter_features.contains(required_features) {
        eprintln!(
            "skipping GPU correctness fuzz: adapter {:?} lacks {:?}",
            adapter.get_info(),
            required_features - adapter_features
        );
        return Ok(());
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("fusor-tile-ir-correctness-fuzz-device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await?;

    let mut rng = FuzzRng::new(0x5eed_f00d_cafe_babe);

    for case in 0..8 {
        for variant in StorageVariant::ALL {
            fuzz_gemm_case(&device, &queue, &mut rng, case, variant)?;
            fuzz_gemv_case(&device, &queue, &mut rng, case, variant)?;
        }
    }

    for format in ggml_formats() {
        for case in 0..4 {
            for variant in StorageVariant::ALL {
                fuzz_qgemm_case(&device, &queue, &mut rng, format, case, variant)?;
                fuzz_qgemv_case(&device, &queue, &mut rng, format, case, variant)?;
            }
        }
        fuzz_qgemv_split_workgroups_case(&device, &queue, &mut rng, format)?;
        fuzz_qgemm_skewed_activation_case(&device, &queue, &mut rng, format)?;
        fuzz_qgemm_im2col_nhwc_case(&device, &queue, &mut rng, format)?;
    }

    Ok(())
}

fn fuzz_gemm_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    case: usize,
    variant: StorageVariant,
) -> TestResult {
    let (m, n, k) = match case % 4 {
        0 => (16, 16, 8),
        1 => (16, 16, 16),
        2 => (32, 16, 8),
        _ => (16, 32, 16),
    };
    let a = random_f32s(rng, m * k, 0.75);
    let b = random_f32s(rng, k * n, 0.75);
    let a_layout = matrix_layout(m, k, variant);
    let b_layout = matrix_layout(k, n, variant);
    let y_layout = matrix_layout(m, n, variant);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let b_physical = pack_f32_matrix(&b, k, n, &b_layout);
    let expected = cpu_matmul(&a, &b, m, k, n);
    let ir = gemm_ir(&a_layout, &b_layout, &y_layout);
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&b_physical),
        allocation_len(&y_layout),
        (n as u32, m as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!("gemm {} case {case} m={m} n={n} k={k}", variant.name()),
        &actual,
        &expected,
        2.0e-3,
    );
    Ok(())
}

fn fuzz_gemv_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    case: usize,
    variant: StorageVariant,
) -> TestResult {
    let (m, k, rows_per_workgroup, vector_width) = match case % 4 {
        0 => (4, 64, 4, 1),
        1 => (8, 96, 4, 2),
        2 => (8, 128, 2, 4),
        _ => (12, 80, 4, 2),
    };
    let a = random_f32s(rng, m * k, 0.75);
    let x = random_f32s(rng, k, 0.75);
    let a_layout = matrix_layout(m, k, variant);
    let x_layout = matrix_layout(k, 1, variant);
    let y_layout = matrix_layout(m, 1, variant);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let x_physical = pack_f32_matrix(&x, k, 1, &x_layout);
    let expected = cpu_gemv(&a, &x, m, k);
    let ir = gemv_ir(
        m,
        k,
        rows_per_workgroup,
        vector_width,
        &a_layout,
        &x_layout,
        &y_layout,
    );
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&x_physical),
        allocation_len(&y_layout),
        (1, m as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, 1, &y_layout);

    assert_close(
        &format!(
            "gemv {} case {case} m={m} k={k} rows_per_workgroup={rows_per_workgroup} vector_width={vector_width}",
            variant.name()
        ),
        &actual,
        &expected,
        2.0e-3,
    );
    Ok(())
}

fn fuzz_qgemm_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
    case: usize,
    variant: StorageVariant,
) -> TestResult {
    let m = 32;
    let n = 32;
    let k = format.block_elements() as usize * (1 + case % 2);
    let a = random_f32s(rng, m * k, 0.25);
    let a_layout = matrix_layout(m, k, variant);
    let y_layout = matrix_layout(m, n, variant);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let (packed_b, dequantized_b) = pack_random_quantized_matrix(rng, format, k, n);
    let expected = cpu_matmul(&a, &dequantized_b, m, k, n);
    let ir = qmatmul_ir(m, n, k, format, true, &a_layout, &y_layout);
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&packed_b),
        allocation_len(&y_layout),
        (n as u32, m as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!(
            "qgemm {format:?} {} case {case} m={m} n={n} k={k}",
            variant.name()
        ),
        &actual,
        &expected,
        3.0e-2,
    );
    Ok(())
}

fn fuzz_qgemm_skewed_activation_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
) -> TestResult {
    let m = 32;
    let n = 32;
    let k = format.block_elements() as usize * 2;
    let a = random_f32s(rng, m * k, 0.25);
    let a_layout = skewed_activation_layout(m, k);
    let y_layout = matrix_layout(m, n, StorageVariant::Contiguous);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let (packed_b, dequantized_b) = pack_random_quantized_matrix(rng, format, k, n);
    let expected = cpu_matmul(&a, &dequantized_b, m, k, n);
    let ir = qmatmul_ir(m, n, k, format, true, &a_layout, &y_layout);
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&packed_b),
        allocation_len(&y_layout),
        (n as u32, m as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!("qgemm {format:?} skewed activation m={m} n={n} k={k}"),
        &actual,
        &expected,
        3.0e-2,
    );
    Ok(())
}

fn fuzz_qgemm_im2col_nhwc_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
) -> TestResult {
    let batch = 1;
    let input_h = 5;
    let input_w = 9;
    let kernel_h = 2;
    let kernel_w = 2;
    let channels = format.block_elements() as usize / (kernel_h * kernel_w);
    let out_h = 4;
    let out_w = 8;
    let m = batch * out_h * out_w;
    let n = 32;
    let k = kernel_h * kernel_w * channels;
    let input = random_f32s(rng, batch * input_h * input_w * channels, 0.25);
    let input_layout = Layout::contiguous(
        MemoryLevel::Storage,
        shape([batch, input_h, input_w, channels]),
    );
    let y_layout = matrix_layout(m, n, StorageVariant::Contiguous);
    let a = im2col_nhwc_matrix(
        &input, batch, input_h, input_w, channels, out_h, out_w, kernel_h, kernel_w,
    );
    let (packed_b, dequantized_b) = pack_random_quantized_matrix(rng, format, k, n);
    let expected = cpu_matmul(&a, &dequantized_b, m, k, n);
    let ir = qmatmul_im2col_nhwc_ir(
        m,
        n,
        k,
        format,
        &input_layout,
        [out_h, out_w],
        [kernel_h, kernel_w],
        &y_layout,
    );
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&input),
        bytemuck::cast_slice(&packed_b),
        allocation_len(&y_layout),
        (n as u32, m as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!("qgemm {format:?} im2col_nhwc m={m} n={n} k={k}"),
        &actual,
        &expected,
        3.0e-2,
    );
    Ok(())
}

fn fuzz_qgemv_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
    case: usize,
    variant: StorageVariant,
) -> TestResult {
    let m = 1;
    let n = 17 + case * 3;
    let k = format.block_elements() as usize * (1 + case % 2);
    let a = random_f32s(rng, m * k, 0.25);
    let a_layout = matrix_layout(m, k, variant);
    let y_layout = matrix_layout(m, n, variant);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let (packed_b, dequantized_b) = pack_random_quantized_matrix(rng, format, k, n);
    let expected = cpu_matmul(&a, &dequantized_b, m, k, n);
    let ir = qmatmul_ir(m, n, k, format, false, &a_layout, &y_layout);
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&packed_b),
        allocation_len(&y_layout),
        (1, n as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!(
            "qgemv {format:?} {} case {case} n={n} k={k}",
            variant.name()
        ),
        &actual,
        &expected,
        qgemv_tolerance(format),
    );
    Ok(())
}

fn fuzz_qgemv_split_workgroups_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
) -> TestResult {
    let m = 1;
    let n = qgemv_cols_per_workgroup(format) * 3 + 1;
    let k = format.block_elements() as usize;
    let a = random_f32s(rng, m * k, 0.25);
    let a_layout = matrix_layout(m, k, StorageVariant::Contiguous);
    let y_layout = matrix_layout(m, n, StorageVariant::Contiguous);
    let a_physical = pack_f32_matrix(&a, m, k, &a_layout);
    let (packed_b, dequantized_b) = pack_random_quantized_matrix(rng, format, k, n);
    let expected = cpu_matmul(&a, &dequantized_b, m, k, n);
    let ir = qmatmul_split_workgroups_ir(m, n, k, format, 2, &a_layout, &y_layout);
    let actual_physical = run_three_buffer_kernel(
        device,
        queue,
        &ir,
        bytemuck::cast_slice(&a_physical),
        bytemuck::cast_slice(&packed_b),
        allocation_len(&y_layout),
        (2, n.div_ceil(2) as u32, 1),
    )?;
    let actual = gather_f32_matrix(&actual_physical, m, n, &y_layout);

    assert_close(
        &format!("split-grid qgemv {format:?} n={n} k={k}"),
        &actual,
        &expected,
        qgemv_tolerance(format),
    );
    Ok(())
}

fn gemm_ir(a_layout: &Layout, b_layout: &Layout, y_layout: &Layout) -> KernelIr {
    let a_layout = a_layout.clone();
    let b_layout = b_layout.clone();
    let y_layout = y_layout.clone();
    tile::build(move |phase| {
        let a = phase.storage_read_with_layout::<F32, 2>(a_layout);
        let b = phase.storage_read_with_layout::<F32, 2>(b_layout);
        let y = phase.storage_write_with_layout::<F32, 2>(y_layout);
        phase.matmul::<256>(&a, &b, &y);
    })
}

fn gemv_ir(
    _m: usize,
    _k: usize,
    _rows_per_workgroup: usize,
    _vector_width: usize,
    a_layout: &Layout,
    x_layout: &Layout,
    y_layout: &Layout,
) -> KernelIr {
    let a_layout = a_layout.clone();
    let x_layout = x_layout.clone();
    let y_layout = y_layout.clone();
    tile::build(move |phase| {
        let a = phase.storage_read_with_layout::<F32, 2>(a_layout);
        let x = phase.storage_read_with_layout::<F32, 2>(x_layout);
        let y = phase.storage_write_with_layout::<F32, 2>(y_layout);
        phase.matmul::<256>(&a, &x, &y);
    })
}

fn qmatmul_ir(
    m: usize,
    n: usize,
    k: usize,
    format: GgmlQuantFormat,
    force_gemm: bool,
    a_layout: &Layout,
    y_layout: &Layout,
) -> KernelIr {
    let a_layout = a_layout.clone();
    let y_layout = y_layout.clone();
    tile::build(move |phase| {
        let a = phase.storage_read_with_layout::<F32, 2>(a_layout);
        let b = phase.quantized_matrix(format, k as u32, n as u32);
        let y = phase.storage_write_with_layout::<F32, 2>(y_layout);
        if force_gemm {
            phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4);
        } else if m == 1 {
            phase.qgemv::<4, 64>(&a, &b, &y, 4, 1);
        } else {
            phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4);
        }
    })
}

fn qmatmul_split_workgroups_ir(
    _m: usize,
    n: usize,
    k: usize,
    format: GgmlQuantFormat,
    workgroups_x: u32,
    a_layout: &Layout,
    y_layout: &Layout,
) -> KernelIr {
    let a_layout = a_layout.clone();
    let y_layout = y_layout.clone();
    tile::build(move |phase| {
        let a = phase.storage_read_with_layout::<F32, 2>(a_layout);
        let b = phase.quantized_matrix(format, k as u32, n as u32);
        let y = phase.storage_write_with_layout::<F32, 2>(y_layout);
        phase.qgemv::<4, 64>(&a, &b, &y, 4, workgroups_x);
    })
}

fn qmatmul_im2col_nhwc_ir(
    _m: usize,
    n: usize,
    k: usize,
    format: GgmlQuantFormat,
    input_layout: &Layout,
    output_hw: [usize; 2],
    kernel_hw: [usize; 2],
    y_layout: &Layout,
) -> KernelIr {
    let input_layout = input_layout.clone();
    let y_layout = y_layout.clone();
    tile::build(move |phase| {
        let input = phase.storage_read_with_layout::<F32, 4>(input_layout);
        let a = input.im2col_nhwc(
            [output_hw[0] as u32, output_hw[1] as u32],
            [kernel_hw[0] as u32, kernel_hw[1] as u32],
            [1, 1],
            [1, 1],
        );
        let b = phase.quantized_matrix(format, k as u32, n as u32);
        let y = phase.storage_write_with_layout::<F32, 2>(y_layout);
        phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4);
    })
}

fn run_three_buffer_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ir: &KernelIr,
    first_input: &[u8],
    second_input: &[u8],
    output_len: usize,
    _dispatch: (u32, u32, u32),
) -> TestResult<Vec<f32>> {
    let dispatch = ir
        .single_tile_program_grid()
        .ok_or("correctness fuzz expects one tile program")?;
    let lowered = ir.lower_to_naga()?;
    let shader = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some("correctness-fuzz-shader"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("correctness-fuzz-bind-group-layout"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, true),
            storage_layout_entry(2, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("correctness-fuzz-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("correctness-fuzz-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let first_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("correctness-fuzz-input-0"),
        contents: first_input,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let second_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("correctness-fuzz-input-1"),
        contents: second_input,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (output_len * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("correctness-fuzz-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("correctness-fuzz-readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("correctness-fuzz-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            buffer_entry(0, &first_buffer),
            buffer_entry(1, &second_buffer),
            buffer_entry(2, &output_buffer),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("correctness-fuzz-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("correctness-fuzz-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let data = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buffer.unmap();
    Ok(values)
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn matrix_layout(rows: usize, cols: usize, variant: StorageVariant) -> Layout {
    let shape = shape([rows, cols]);
    match variant {
        StorageVariant::Contiguous => Layout::contiguous(MemoryLevel::Storage, shape),
        StorageVariant::Padded if rows == 1 && cols > 1 => Layout::strided(
            MemoryLevel::Storage,
            shape,
            Strides::new([(cols * 2 + 3) as u32, 2]),
        ),
        StorageVariant::Padded => Layout::strided(
            MemoryLevel::Storage,
            shape,
            Strides::new([(cols + 5) as u32, 1]),
        ),
        StorageVariant::Transposed => {
            let strides = Strides::col_major_for(&shape);
            Layout::strided(MemoryLevel::Storage, shape, strides)
        }
    }
}

fn skewed_activation_layout(rows: usize, cols: usize) -> Layout {
    let shape = shape([rows, cols]);
    Layout::strided(
        MemoryLevel::Storage,
        shape,
        Strides::new([2, (rows * 2 + 7) as u32]),
    )
}

fn allocation_len(layout: &Layout) -> usize {
    layout.allocation_element_count().get() as usize
}

fn matrix_storage_index(layout: &Layout, row: usize, col: usize) -> usize {
    let strides = layout.strides().values();
    row * strides[0] as usize + col * strides[1] as usize
}

fn pack_f32_matrix(logical: &[f32], rows: usize, cols: usize, layout: &Layout) -> Vec<f32> {
    assert_eq!(logical.len(), rows * cols);
    let mut physical = vec![17.0; allocation_len(layout)];
    for row in 0..rows {
        for col in 0..cols {
            physical[matrix_storage_index(layout, row, col)] = logical[row * cols + col];
        }
    }
    physical
}

fn gather_f32_matrix(physical: &[f32], rows: usize, cols: usize, layout: &Layout) -> Vec<f32> {
    let mut logical = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            logical[row * cols + col] = physical[matrix_storage_index(layout, row, col)];
        }
    }
    logical
}

fn pack_random_quantized_matrix(
    rng: &mut FuzzRng,
    format: GgmlQuantFormat,
    rows: usize,
    cols: usize,
) -> (Vec<u32>, Vec<f32>) {
    let block_elems = format.block_elements() as usize;
    assert_eq!(rows % block_elems, 0);

    let blocks_per_col = rows / block_elems;
    let words_per_block = format.block_words() as usize;
    let mut packed = Vec::with_capacity(cols * blocks_per_col * words_per_block);
    let mut dequantized = vec![0.0; rows * cols];

    for col in 0..cols {
        for block in 0..blocks_per_col {
            let (block_words, values) = pack_random_quantized_block(rng, format);
            packed.extend(block_words);
            for (i, value) in values.into_iter().enumerate() {
                let row = block * block_elems + i;
                dequantized[row * cols + col] = value;
            }
        }
    }

    (packed, dequantized)
}

fn pack_random_quantized_block(rng: &mut FuzzRng, format: GgmlQuantFormat) -> (Vec<u32>, Vec<f32>) {
    match format {
        GgmlQuantFormat::Q4_0 => pack_q4_0_block(rng),
        GgmlQuantFormat::Q4_1 => pack_q4_1_block(rng),
        GgmlQuantFormat::Q5_0 => pack_q5_0_block(rng),
        GgmlQuantFormat::Q5_1 => pack_q5_1_block(rng),
        GgmlQuantFormat::Q8_0 => pack_q8_0_block(rng),
        GgmlQuantFormat::Q8_1 => pack_q8_1_block(rng),
        GgmlQuantFormat::Q2K => pack_q2k_block(rng),
        GgmlQuantFormat::Q3K => pack_q3k_block(rng),
        GgmlQuantFormat::Q4K => pack_q4k_block(rng),
        GgmlQuantFormat::Q5K => pack_q5k_block(rng),
        GgmlQuantFormat::Q6K => pack_q6k_block(rng),
        GgmlQuantFormat::Q8K => pack_q8k_block(rng),
    }
}

fn pack_q4_0_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.125;
    let mut words = vec![0; GgmlQuantFormat::Q4_0.block_words() as usize];
    words[0] = scale.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0xf) as u8;
        set_nibble(&mut words, 4 + (i & 15), i >= 16, q);
        *value = (q as i32 - 8) as f32 * scale;
    }

    (words, values)
}

fn pack_q4_1_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.125;
    let min: f32 = -1.0;
    let mut words = vec![0; GgmlQuantFormat::Q4_1.block_words() as usize];
    words[0] = scale.to_bits();
    words[1] = min.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0xf) as u8;
        set_nibble(&mut words, 8 + (i & 15), i >= 16, q);
        *value = q as f32 * scale + min;
    }

    (words, values)
}

fn pack_q5_0_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.125;
    let mut words = vec![0; GgmlQuantFormat::Q5_0.block_words() as usize];
    words[0] = scale.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0x1f) as u8;
        if q & 0x10 != 0 {
            let high_bit = if i < 16 { i } else { (i & 15) + 16 };
            words[1] |= 1u32 << high_bit;
        }
        set_nibble(&mut words, 8 + (i & 15), i >= 16, q & 0xf);
        *value = (q as i32 - 16) as f32 * scale;
    }

    (words, values)
}

fn pack_q5_1_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.125;
    let min: f32 = -2.0;
    let mut words = vec![0; GgmlQuantFormat::Q5_1.block_words() as usize];
    words[0] = scale.to_bits();
    words[1] = min.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0x1f) as u8;
        if q & 0x10 != 0 {
            let high_bit = if i < 16 { i } else { (i & 15) + 16 };
            words[2] |= 1u32 << high_bit;
        }
        set_nibble(&mut words, 12 + (i & 15), i >= 16, q & 0xf);
        *value = q as f32 * scale + min;
    }

    (words, values)
}

fn pack_q8_0_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.03125;
    let mut words = vec![0; GgmlQuantFormat::Q8_0.block_words() as usize];
    words[0] = scale.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() % 63) as i32 - 31;
        set_byte(&mut words, 4 + i, q as i8 as u8);
        *value = q as f32 * scale;
    }

    (words, values)
}

fn pack_q8_1_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.03125;
    let mut words = vec![0; GgmlQuantFormat::Q8_1.block_words() as usize];
    words[0] = scale.to_bits();
    let mut values = vec![0.0; 32];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() % 63) as i32 - 31;
        set_byte(&mut words, 8 + i, q as i8 as u8);
        *value = q as f32 * scale;
    }

    (words, values)
}

fn pack_q2k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.0625;
    let min_scale: f32 = 0.015625;
    let mut words = vec![0; GgmlQuantFormat::Q2K.block_words() as usize];
    for byte in 0..16 {
        set_byte(&mut words, byte, 0x11);
    }
    words[20] = scale.to_bits();
    words[21] = min_scale.to_bits();
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0x3) as u8;
        let group = i >> 4;
        let local = i & 15;
        let chunk = group >> 3;
        let group_in_chunk = group & 7;
        let byte = 4 * 4 + chunk * 32 + (group_in_chunk & 1) * 16 + local;
        let bit = (group_in_chunk >> 1) * 2;
        set_two_bits(&mut words, byte, bit, q);
        *value = q as f32 * scale - min_scale;
    }

    (words, values)
}

fn pack_q3k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.0625;
    let mut words = vec![0; GgmlQuantFormat::Q3K.block_words() as usize];
    for group in 0..16 {
        set_q3k_scale(&mut words, group, 33);
    }
    words[27] = scale.to_bits();
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q_signed = (rng.next_u32() & 0x7) as i32 - 4;
        let low = if q_signed >= 0 {
            q_signed as u8
        } else {
            (q_signed + 4) as u8
        };
        let group = i >> 4;
        let local = i & 15;
        let chunk = group >> 3;
        let group_in_chunk = group & 7;
        let pair = group_in_chunk & 1;
        let byte = 8 * 4 + chunk * 32 + pair * 16 + local;
        let bit = (group_in_chunk >> 1) * 2;
        set_two_bits(&mut words, byte, bit, low);
        if q_signed >= 0 {
            let hmask_byte = pair * 16 + local;
            let hmask_bit = chunk * 4 + (group_in_chunk >> 1);
            set_bit_in_byte(&mut words, hmask_byte, hmask_bit);
        }
        *value = q_signed as f32 * scale;
    }

    (words, values)
}

fn pack_q4k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.0625;
    let mut words = vec![0; GgmlQuantFormat::Q4K.block_words() as usize];
    words[0] = scale.to_bits();
    words[1] = 0.0_f32.to_bits();
    words[2] = 0x0101_0101;
    words[3] = 0;
    words[4] = 0x0101_0101;
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0xf) as u8;
        set_k_nibble(&mut words, 5, i, q);
        *value = q as f32 * scale;
    }

    (words, values)
}

fn pack_q5k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.0625;
    let mut words = vec![0; GgmlQuantFormat::Q5K.block_words() as usize];
    words[0] = scale.to_bits();
    words[1] = 0.0_f32.to_bits();
    words[2] = 0x0101_0101;
    words[3] = 0;
    words[4] = 0x0101_0101;
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0x1f) as u8;
        if q & 0x10 != 0 {
            let qh_byte = 5 * 4 + (i & 31);
            let qh_bit = i >> 5;
            set_bit_in_byte(&mut words, qh_byte, qh_bit);
        }
        set_k_nibble(&mut words, 13, i, q & 0xf);
        *value = q as f32 * scale;
    }

    (words, values)
}

fn pack_q6k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.0625;
    let mut words = vec![0; GgmlQuantFormat::Q6K.block_words() as usize];
    for byte in 48 * 4..52 * 4 {
        set_byte(&mut words, byte, 1);
    }
    words[52] = scale.to_bits();
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() & 0x3f) as u8;
        let chunk = i >> 7;
        let local = i & 127;
        let high_byte_index = local & 31;
        let low_group = local >> 5;

        let lower_byte = chunk * 64 + high_byte_index + (low_group & 1) * 32;
        set_nibble(&mut words, lower_byte, low_group >= 2, q & 0xf);

        let higher_byte = 32 * 4 + chunk * 32 + high_byte_index;
        set_two_bits(&mut words, higher_byte, low_group * 2, (q >> 4) & 0x3);

        *value = (q as i32 - 32) as f32 * scale;
    }

    (words, values)
}

fn pack_q8k_block(rng: &mut FuzzRng) -> (Vec<u32>, Vec<f32>) {
    let scale: f32 = 0.03125;
    let mut words = vec![0; GgmlQuantFormat::Q8K.block_words() as usize];
    words[0] = scale.to_bits();
    let mut values = vec![0.0; 256];

    for (i, value) in values.iter_mut().enumerate() {
        let q = (rng.next_u32() % 127) as i32 - 63;
        set_byte(&mut words, 4 + i, q as i8 as u8);
        *value = q as f32 * scale;
    }

    (words, values)
}

fn set_k_nibble(words: &mut [u32], word_offset: usize, element: usize, value: u8) {
    let group = element >> 5;
    let in_group = element & 31;
    let byte = word_offset * 4 + (group >> 1) * 32 + in_group;
    set_nibble(words, byte, group & 1 != 0, value);
}

fn set_q3k_scale(words: &mut [u32], group: usize, value: u8) {
    let lane = group & 3;
    let low = value & 0x0f;
    let high = (value >> 4) & 0x03;
    let s0 = 24 * 4 + lane;
    let s1 = 25 * 4 + lane;
    let s2 = 26 * 4 + lane;
    match group {
        0..=3 => {
            set_nibble(words, s0, false, low);
            set_two_bits(words, s2, 0, high);
        }
        4..=7 => {
            set_nibble(words, s1, false, low);
            set_two_bits(words, s2, 2, high);
        }
        8..=11 => {
            set_nibble(words, s0, true, low);
            set_two_bits(words, s2, 4, high);
        }
        12..=15 => {
            set_nibble(words, s1, true, low);
            set_two_bits(words, s2, 6, high);
        }
        _ => unreachable!("q3k has 16 scale groups"),
    }
}

fn set_bit_in_byte(words: &mut [u32], byte: usize, bit: usize) {
    let word = byte / 4;
    let shift = ((byte % 4) * 8 + bit) as u32;
    words[word] |= 1u32 << shift;
}

fn set_two_bits(words: &mut [u32], byte: usize, bit: usize, value: u8) {
    let word = byte / 4;
    let shift = ((byte % 4) * 8 + bit) as u32;
    let mask = !(0x3u32 << shift);
    words[word] = (words[word] & mask) | (((value & 0x3) as u32) << shift);
}

fn set_nibble(words: &mut [u32], byte: usize, high: bool, value: u8) {
    let word = byte / 4;
    let shift = ((byte % 4) * 8 + usize::from(high) * 4) as u32;
    let mask = !(0xfu32 << shift);
    words[word] = (words[word] & mask) | (((value & 0xf) as u32) << shift);
}

fn set_byte(words: &mut [u32], byte: usize, value: u8) {
    let word = byte / 4;
    let shift = ((byte % 4) * 8) as u32;
    let mask = !(0xffu32 << shift);
    words[word] = (words[word] & mask) | ((value as u32) << shift);
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            y[row * n + col] = acc;
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn im2col_nhwc_matrix(
    input: &[f32],
    batch: usize,
    input_h: usize,
    input_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Vec<f32> {
    let m = batch * out_h * out_w;
    let k = kernel_h * kernel_w * channels;
    let mut matrix = vec![0.0; m * k];
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let row = (b * out_h + oh) * out_w + ow;
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        for c in 0..channels {
                            let col = (kh * kernel_w + kw) * channels + c;
                            let input_index =
                                ((b * input_h + oh + kh) * input_w + ow + kw) * channels + c;
                            matrix[row * k + col] = input[input_index];
                        }
                    }
                }
            }
        }
    }
    matrix
}

fn cpu_gemv(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows];
    for row in 0..rows {
        let mut acc = 0.0;
        for col in 0..cols {
            acc += a[row * cols + col] * x[col];
        }
        y[row] = acc;
    }
    y
}

fn random_f32s(rng: &mut FuzzRng, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| rng.f32_range(-scale, scale)).collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");

    let mut worst_index = 0;
    let mut worst_error = 0.0;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}: non-finite value at {index}, actual={actual}, expected={expected}"
        );
        let error = (actual - expected).abs();
        if error > worst_error {
            worst_error = error;
            worst_index = index;
        }
    }

    assert!(
        worst_error <= tolerance,
        "{label}: worst error {worst_error} at {worst_index}, actual={}, expected={}, tolerance={tolerance}",
        actual[worst_index],
        expected[worst_index],
    );
}

fn ggml_formats() -> [GgmlQuantFormat; 12] {
    [
        GgmlQuantFormat::Q4_0,
        GgmlQuantFormat::Q4_1,
        GgmlQuantFormat::Q5_0,
        GgmlQuantFormat::Q5_1,
        GgmlQuantFormat::Q8_0,
        GgmlQuantFormat::Q8_1,
        GgmlQuantFormat::Q2K,
        GgmlQuantFormat::Q3K,
        GgmlQuantFormat::Q4K,
        GgmlQuantFormat::Q5K,
        GgmlQuantFormat::Q6K,
        GgmlQuantFormat::Q8K,
    ]
}

fn qgemv_cols_per_workgroup(format: GgmlQuantFormat) -> usize {
    format.qgemv_cols_per_workgroup() as usize
}

fn qgemv_tolerance(format: GgmlQuantFormat) -> f32 {
    match format {
        // Q4K/Q6K qgemv intentionally quantizes activations to q8 before
        // dot4; the CPU reference still multiplies dequantized f32 values.
        GgmlQuantFormat::Q4K | GgmlQuantFormat::Q6K => 5.0e-2,
        _ => 3.0e-2,
    }
}

fn shape<const R: usize>(dims: [usize; R]) -> Shape {
    Shape::new(dims.map(|dim| dim as u32))
}

#[derive(Clone)]
struct FuzzRng {
    state: u64,
}

impl FuzzRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn f32_range(&mut self, min: f32, max: f32) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        min + unit * (max - min)
    }
}
