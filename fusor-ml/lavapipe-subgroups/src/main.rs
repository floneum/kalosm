use std::{
    ffi::{CStr, CString},
    mem::size_of,
    process::ExitCode,
    ptr,
};

use ash::{Entry, vk};
use fusor_tile_ir::{F32, F32Bits, KernelBuilder, KernelTensorRef, Shape, TileLiteral, U32, tile};
use fusor_tile_ir_kernels::{
    FlashAttentionDims, FlashAttentionMeta, TensorMeta, flash_attention,
    flash_outputs_per_workgroup, linear_storage_layout,
};
use naga::back::spv;
use thiserror::Error;

const SKIP: u8 = 77;
const DEFAULT_WIDTHS: &[u32] = &[4, 8, 16, 32, 64];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelChoice {
    All,
    SubgroupProbe,
    SubgroupReduce,
    FlashAttention,
}

#[derive(Debug)]
struct Args {
    subgroup_size: u32,
    kernel: KernelChoice,
    validation: bool,
}

#[derive(Debug, Error)]
enum RunError {
    #[error("{0}")]
    Skip(String),
    #[error("{0}")]
    Fatal(String),
    #[error("vulkan error: {0:?}")]
    Vk(#[from] vk::Result),
    #[error("naga SPIR-V write failed: {0}")]
    Spv(#[from] spv::Error),
    #[error("tile IR lowering failed: {0}")]
    Lower(#[from] fusor_tile_ir::LowerError),
}

type Result<T> = std::result::Result<T, RunError>;

fn main() -> ExitCode {
    match Args::parse().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(RunError::Skip(message)) => {
            eprintln!("SKIP: {message}");
            ExitCode::from(SKIP)
        }
        Err(error) => {
            eprintln!("ERROR: {error}");
            ExitCode::FAILURE
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut subgroup_size = None;
        let mut kernel = KernelChoice::All;
        let mut validation = std::env::var_os("FUSOR_VK_VALIDATION").is_some();
        let mut iter = std::env::args().skip(1);

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--subgroup-size" => {
                    let Some(value) = iter.next() else {
                        return Err(RunError::Fatal("--subgroup-size needs a value".into()));
                    };
                    subgroup_size = Some(value.parse::<u32>().map_err(|_| {
                        RunError::Fatal(format!("invalid subgroup size {value:?}"))
                    })?);
                }
                "--kernel" => {
                    let Some(value) = iter.next() else {
                        return Err(RunError::Fatal("--kernel needs a value".into()));
                    };
                    kernel = match value.as_str() {
                        "all" => KernelChoice::All,
                        "subgroup-probe" => KernelChoice::SubgroupProbe,
                        "subgroup-reduce" => KernelChoice::SubgroupReduce,
                        "flash-attention" => KernelChoice::FlashAttention,
                        _ => return Err(RunError::Fatal(format!("unknown kernel {value:?}"))),
                    };
                }
                "--validation" => validation = true,
                "--help" | "-h" => {
                    println!(
                        "usage: fusor-lavapipe-subgroups --subgroup-size <{}> [--kernel all|subgroup-probe|subgroup-reduce|flash-attention] [--validation]",
                        DEFAULT_WIDTHS
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join("|")
                    );
                    return Err(RunError::Skip("help requested".into()));
                }
                _ => return Err(RunError::Fatal(format!("unknown argument {arg:?}"))),
            }
        }

        let subgroup_size =
            subgroup_size.ok_or_else(|| RunError::Fatal("--subgroup-size is required".into()))?;
        if !DEFAULT_WIDTHS.contains(&subgroup_size) {
            return Err(RunError::Skip(format!(
                "subgroup size {subgroup_size} is outside the local smoke matrix"
            )));
        }

        Ok(Self {
            subgroup_size,
            kernel,
            validation,
        })
    }
}

fn run(args: Args) -> Result<()> {
    log_verbose("creating Vulkan context");
    let vk = VkContext::new(args.validation)?;
    vk.require_subgroup_size(args.subgroup_size)?;

    log_verbose("building test kernels");
    let mut tests = Vec::new();
    match args.kernel {
        KernelChoice::All => {
            tests.push(TestCase::subgroup_probe(args.subgroup_size)?);
            tests.push(TestCase::subgroup_reduce(args.subgroup_size)?);
            tests.push(TestCase::flash_attention(args.subgroup_size)?);
        }
        KernelChoice::SubgroupProbe => tests.push(TestCase::subgroup_probe(args.subgroup_size)?),
        KernelChoice::SubgroupReduce => tests.push(TestCase::subgroup_reduce(args.subgroup_size)?),
        KernelChoice::FlashAttention => tests.push(TestCase::flash_attention(args.subgroup_size)?),
    }

    for mut test in tests {
        log_verbose(&format!("running {}", test.name));
        vk.run_test(args.subgroup_size, &mut test)?;
        (test.verify)(&test.buffers, args.subgroup_size)?;
        eprintln!("ok: {} @ subgroup {}", test.name, args.subgroup_size);
    }

    Ok(())
}

fn log_verbose(message: &str) {
    if std::env::var_os("FUSOR_LAVAPIPE_VERBOSE").is_some() {
        eprintln!("lavapipe-subgroups: {message}");
    }
}

struct TestCase {
    name: &'static str,
    spirv: Vec<u32>,
    dispatch: [u32; 3],
    buffers: Vec<TestBuffer>,
    verify: fn(&[TestBuffer], u32) -> Result<()>,
}

struct TestBuffer {
    data: Vec<u8>,
    read_only: bool,
}

impl TestBuffer {
    fn read_only<T: bytemuck::Pod>(data: &[T]) -> Self {
        Self {
            data: bytemuck::cast_slice(data).to_vec(),
            read_only: true,
        }
    }

    fn read_write_zeroed<T: bytemuck::Pod>(len: usize) -> Self {
        Self {
            data: vec![0; len * size_of::<T>()],
            read_only: false,
        }
    }

    fn read_write<T: bytemuck::Pod>(data: &[T]) -> Self {
        Self {
            data: bytemuck::cast_slice(data).to_vec(),
            read_only: false,
        }
    }

    fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }
}

impl TestCase {
    fn subgroup_probe(_subgroup_size: u32) -> Result<Self> {
        let ir = tile::build(|program| {
            let output = program.storage_write::<U32, 1>(Shape::new([512]));
            program.program_grid::<256>([1, 1, 1], |block| {
                let lane = block.lane();
                let subgroup_id = block.subgroup_id();
                let subgroup_size_value = block.subgroup_size();
                let subgroup_lane = block.subgroup_lane();
                block.store(output.at(0), subgroup_size_value, lane.eq(0u32));
                block.store(output.at(1), subgroup_id, lane.eq(0u32));
                block.store(
                    output.at(lane.clone() + 4u32),
                    subgroup_lane,
                    lane.lt(256u32),
                );
            });
        });

        Ok(Self {
            name: "subgroup-probe",
            spirv: lower_to_spirv(&ir)?,
            dispatch: [1, 1, 1],
            buffers: vec![TestBuffer::read_write_zeroed::<u32>(512)],
            verify: move_probe_result,
        })
    }

    fn subgroup_reduce(_subgroup_size: u32) -> Result<Self> {
        let ir = tile::build(|program| {
            let output = program.storage_write::<F32, 1>(Shape::new([1]));
            program.program_grid::<256>([1, 1, 1], |block| {
                let lane = block.lane();
                let one = fusor_tile_ir::tile::Tile::literal(TileLiteral::f32(1.0));
                let sum = block.subgroup_reduce_sum(one);
                block.store(output.at(0), sum, lane.eq(0u32));
            });
        });

        Ok(Self {
            name: "subgroup-reduce",
            spirv: lower_to_spirv(&ir)?,
            dispatch: [1, 1, 1],
            buffers: vec![TestBuffer::read_write_zeroed::<f32>(1)],
            verify: move_reduce_result,
        })
    }

    fn flash_attention(subgroup_size: u32) -> Result<Self> {
        let dims = FlashAttentionDims {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 1,
            kv_seq_len: subgroup_size,
            head_dim: 64,
        };
        let q_len = (dims.batch * dims.num_heads * dims.q_seq_len * dims.head_dim) as usize;
        let kv_len = (dims.batch * dims.num_kv_heads * dims.kv_seq_len * dims.head_dim) as usize;
        let out_len = q_len;
        let q = (0..q_len)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.125)
            .collect::<Vec<_>>();
        let k = (0..kv_len)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.0625)
            .collect::<Vec<_>>();
        let v = vec![0.0f32; kv_len];
        let scale = 1.0 / (dims.head_dim as f32).sqrt();
        let output_meta = TensorMeta::new(
            vec![
                dims.num_heads * dims.q_seq_len * dims.head_dim,
                dims.q_seq_len * dims.head_dim,
                dims.head_dim,
                1,
            ],
            0,
        );
        let meta = FlashAttentionMeta {
            dims,
            scale: F32Bits::new(scale),
            q_meta: TensorMeta::new(
                vec![
                    dims.num_heads * dims.q_seq_len * dims.head_dim,
                    dims.q_seq_len * dims.head_dim,
                    dims.head_dim,
                    1,
                ],
                0,
            ),
            k_meta: TensorMeta::new(
                vec![
                    dims.num_kv_heads * dims.kv_seq_len * dims.head_dim,
                    dims.kv_seq_len * dims.head_dim,
                    dims.head_dim,
                    1,
                ],
                0,
            ),
            v_meta: TensorMeta::new(
                vec![
                    dims.num_kv_heads * dims.kv_seq_len * dims.head_dim,
                    dims.kv_seq_len * dims.head_dim,
                    dims.head_dim,
                    1,
                ],
                0,
            ),
            mask_meta: None,
            output_meta,
            dispatch_size: [
                dims.head_dim
                    .div_ceil(flash_outputs_per_workgroup(subgroup_size)),
                dims.batch * dims.num_heads * dims.q_seq_len,
                1,
            ],
        };

        let mut kb = KernelBuilder::<u32>::new();
        dispatch_flash(&mut kb, subgroup_size, meta.clone())?;
        let (ir, _) = kb.finish();
        let expected = flash_reference(&q, &k, &v, dims, scale);
        let output = vec![123.0f32; out_len];

        Ok(Self {
            name: "flash-attention",
            spirv: lower_to_spirv(&ir)?,
            dispatch: meta.dispatch_size,
            buffers: vec![
                TestBuffer::read_only(&q),
                TestBuffer::read_only(&k),
                TestBuffer::read_only(&v),
                TestBuffer::read_write(&output),
                TestBuffer::read_only(&expected),
            ],
            verify: move_flash_result,
        })
    }
}

fn dispatch_flash(
    kb: &mut KernelBuilder<u32>,
    subgroup_size: u32,
    meta: FlashAttentionMeta,
) -> Result<()> {
    let q = KernelTensorRef::new(0, linear_storage_layout());
    let k = KernelTensorRef::new(1, linear_storage_layout());
    let v = KernelTensorRef::new(2, linear_storage_layout());
    let out = KernelTensorRef::new(3, linear_storage_layout());
    let ok = match subgroup_size {
        4 => flash_attention::<F32, 4, u32>(kb, q, k, v, None, out, meta),
        8 => flash_attention::<F32, 8, u32>(kb, q, k, v, None, out, meta),
        16 => flash_attention::<F32, 16, u32>(kb, q, k, v, None, out, meta),
        32 => flash_attention::<F32, 32, u32>(kb, q, k, v, None, out, meta),
        64 => flash_attention::<F32, 64, u32>(kb, q, k, v, None, out, meta),
        _ => None,
    };
    ok.ok_or_else(|| {
        RunError::Fatal(format!(
            "failed to build flash kernel for subgroup {subgroup_size}"
        ))
    })
}

fn lower_to_spirv(ir: &fusor_tile_ir::KernelIr) -> Result<Vec<u32>> {
    let lowered = ir.lower_to_naga()?;
    if std::env::var_os("FUSOR_LAVAPIPE_VERBOSE").is_some() {
        for (_, global) in lowered.module().global_variables.iter() {
            if let Some(binding) = &global.binding {
                eprintln!(
                    "lavapipe-subgroups: binding group={} binding={} space={:?}",
                    binding.group, binding.binding, global.space
                );
            }
        }
    }
    let options = spv::Options {
        lang_version: (1, 3),
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
        ..Default::default()
    };
    let pipeline = spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
    };
    Ok(spv::write_vec(
        lowered.module(),
        lowered.info(),
        &options,
        Some(&pipeline),
    )?)
}

fn move_probe_result(buffers: &[TestBuffer], subgroup_size: u32) -> Result<()> {
    let values = buffers[0].as_slice::<u32>();
    let observed = values[0];
    if observed != subgroup_size {
        return Err(RunError::Fatal(format!(
            "expected subgroup size {subgroup_size}, shader observed {observed}"
        )));
    }
    for lane in 0..256usize {
        let expected = lane as u32 % subgroup_size;
        let actual = values[lane + 4];
        if actual != expected {
            return Err(RunError::Fatal(format!(
                "probe lane {lane}: expected subgroup lane {expected}, got {actual}"
            )));
        }
    }
    Ok(())
}

fn move_reduce_result(buffers: &[TestBuffer], subgroup_size: u32) -> Result<()> {
    let values = buffers[0].as_slice::<f32>();
    let expected = subgroup_size as f32;
    for (index, value) in values.iter().copied().enumerate() {
        if (value - expected).abs() > 1e-6 {
            return Err(RunError::Fatal(format!(
                "reduce subgroup {index}: expected {expected}, got {value}"
            )));
        }
    }
    Ok(())
}

fn move_flash_result(buffers: &[TestBuffer], _subgroup_size: u32) -> Result<()> {
    let actual = buffers[3].as_slice::<f32>();
    let expected = buffers[4].as_slice::<f32>();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if (actual - expected).abs() > 1e-4 {
            return Err(RunError::Fatal(format!(
                "flash output {index}: expected {expected}, got {actual}"
            )));
        }
    }
    Ok(())
}

fn flash_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: FlashAttentionDims,
    scale: f32,
) -> Vec<f32> {
    let mut output =
        vec![0.0; (dims.batch * dims.num_heads * dims.q_seq_len * dims.head_dim) as usize];
    for dim in 0..dims.head_dim as usize {
        let mut scores = Vec::with_capacity(dims.kv_seq_len as usize);
        for kv in 0..dims.kv_seq_len as usize {
            let mut dot = 0.0;
            for d in 0..dims.head_dim as usize {
                dot += q[d] * k[kv * dims.head_dim as usize + d];
            }
            scores.push(dot * scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores.iter().map(|s| (*s - max).exp()).sum::<f32>();
        let mut value = 0.0;
        for kv in 0..dims.kv_seq_len as usize {
            let prob = (scores[kv] - max).exp() / denom;
            value += prob * v[kv * dims.head_dim as usize + dim];
        }
        output[dim] = value;
    }
    output
}

struct VkContext {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    subgroup_range: (u32, u32),
}

struct VkBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl VkContext {
    fn new(validation: bool) -> Result<Self> {
        log_verbose("loading Vulkan loader");
        let entry = unsafe {
            Entry::load()
                .map_err(|err| RunError::Skip(format!("failed to load Vulkan loader: {err}")))?
        };
        let app_name = CString::new("fusor-lavapipe-subgroups").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let enable_validation = validation
            && available_layers.iter().any(|layer| unsafe {
                CStr::from_ptr(layer.layer_name.as_ptr()) == validation_layer.as_c_str()
            });
        if validation && !enable_validation {
            eprintln!("warning: VK_LAYER_KHRONOS_validation not available");
        }
        let layer_names = if enable_validation {
            vec![validation_layer.as_ptr()]
        } else {
            Vec::new()
        };
        let available_instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None)? };
        let enable_validation_features = enable_validation
            && available_instance_extensions
                .iter()
                .any(|extension| unsafe {
                    CStr::from_ptr(extension.extension_name.as_ptr())
                        == vk::EXT_VALIDATION_FEATURES_NAME
                });
        let instance_extensions = if enable_validation_features {
            vec![vk::EXT_VALIDATION_FEATURES_NAME.as_ptr()]
        } else {
            Vec::new()
        };

        let mut validation_features = vk::ValidationFeaturesEXT::default()
            .enabled_validation_features(&[
                vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            ]);
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);
        if enable_validation_features {
            instance_info = instance_info.push_next(&mut validation_features);
        } else if validation {
            eprintln!(
                "warning: VK_EXT_validation_features not available; sync validation not enabled"
            );
        }
        log_verbose("creating Vulkan instance");
        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        log_verbose("choosing CPU physical device");
        let physical_device = choose_cpu_device(&instance)?;
        let (min_subgroup, max_subgroup) = subgroup_range(&instance, physical_device);
        let queue_family_index = compute_queue_family(&instance, physical_device)?;

        let priority = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priority)];
        let extensions = device_extensions(&instance, physical_device)?;
        let extension_ptrs = extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let mut subgroup_features =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extension_ptrs)
            .push_next(&mut subgroup_features);
        log_verbose("creating Vulkan device");
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        log_verbose("getting device queue");
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        log_verbose("creating command pool");
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            command_pool,
            subgroup_range: (min_subgroup, max_subgroup),
        })
    }

    fn require_subgroup_size(&self, subgroup_size: u32) -> Result<()> {
        let (min, max) = self.subgroup_range;
        if subgroup_size < min || subgroup_size > max {
            return Err(RunError::Skip(format!(
                "requested subgroup size {subgroup_size} outside device range {min}..={max}"
            )));
        }
        Ok(())
    }

    fn run_test(&self, subgroup_size: u32, test: &mut TestCase) -> Result<()> {
        log_verbose("creating buffers");
        let buffers = test
            .buffers
            .iter()
            .map(|buffer| self.create_buffer(buffer))
            .collect::<Result<Vec<_>>>()?;
        log_verbose("creating descriptor set layout");
        let descriptor_set_layout = self.create_descriptor_set_layout(&test.buffers)?;
        log_verbose("creating pipeline layout");
        let pipeline_layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
                None,
            )?
        };
        log_verbose("creating shader module");
        let shader_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&test.spirv),
                None,
            )?
        };
        let main = CString::new("main").unwrap();
        let mut required_subgroup =
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup_size);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&main)
            .push_next(&mut required_subgroup);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        log_verbose("creating compute pipeline");
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?[0]
        };
        log_verbose("creating descriptor pool");
        let descriptor_pool = self.create_descriptor_pool(test.buffers.len() as u32)?;
        let descriptor_set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )?[0]
        };
        log_verbose("writing descriptor set");
        self.write_descriptor_set(descriptor_set, &buffers);
        log_verbose("dispatching");
        self.dispatch(pipeline, pipeline_layout, descriptor_set, test.dispatch)?;
        log_verbose("reading buffers");
        for (gpu_buffer, test_buffer) in buffers.iter().zip(&mut test.buffers) {
            if !test_buffer.read_only {
                self.read_buffer(gpu_buffer, test_buffer)?;
            }
        }

        unsafe {
            self.device.destroy_descriptor_pool(descriptor_pool, None);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(shader_module, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            for buffer in buffers {
                self.device.destroy_buffer(buffer.buffer, None);
                self.device.free_memory(buffer.memory, None);
            }
        }
        Ok(())
    }

    fn create_buffer(&self, input: &TestBuffer) -> Result<VkBuffer> {
        let size = input.data.len() as vk::DeviceSize;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = self.host_memory_type(requirements.memory_type_bits)?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )?
        };
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let mapped = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
            ptr::copy_nonoverlapping(input.data.as_ptr(), mapped.cast::<u8>(), input.data.len());
            self.device.unmap_memory(memory);
        }
        Ok(VkBuffer {
            buffer,
            memory,
            size,
        })
    }

    fn read_buffer(&self, gpu: &VkBuffer, output: &mut TestBuffer) -> Result<()> {
        unsafe {
            let mapped =
                self.device
                    .map_memory(gpu.memory, 0, gpu.size, vk::MemoryMapFlags::empty())?;
            ptr::copy_nonoverlapping(
                mapped.cast::<u8>(),
                output.data.as_mut_ptr(),
                output.data.len(),
            );
            self.device.unmap_memory(gpu.memory);
        }
        Ok(())
    }

    fn create_descriptor_set_layout(
        &self,
        buffers: &[TestBuffer],
    ) -> Result<vk::DescriptorSetLayout> {
        let bindings = buffers
            .iter()
            .enumerate()
            .map(|(binding, _)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        Ok(unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        })
    }

    fn create_descriptor_pool(&self, count: u32) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: count,
        }];
        Ok(unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        })
    }

    fn write_descriptor_set(&self, descriptor_set: vk::DescriptorSet, buffers: &[VkBuffer]) {
        let infos = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size)
            })
            .collect::<Vec<_>>();
        let writes = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn dispatch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        dispatch: [u32; 3],
    ) -> Result<()> {
        let command_buffer = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        unsafe {
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device
                .cmd_dispatch(command_buffer, dispatch[0], dispatch[1], dispatch[2]);
            self.device.end_command_buffer(command_buffer)?;
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
        Ok(())
    }

    fn host_memory_type(&self, mask: u32) -> Result<u32> {
        let memory = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        for index in 0..memory.memory_type_count {
            let flags = memory.memory_types[index as usize].property_flags;
            if (mask & (1 << index)) != 0
                && flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            {
                return Ok(index);
            }
        }
        Err(RunError::Fatal(
            "no host-visible coherent memory type".into(),
        ))
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn choose_cpu_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    devices
        .into_iter()
        .find(|device| {
            let props = unsafe { instance.get_physical_device_properties(*device) };
            props.device_type == vk::PhysicalDeviceType::CPU
        })
        .ok_or_else(|| {
            RunError::Skip("no CPU Vulkan physical device found; check VK_DRIVER_FILES".into())
        })
}

fn subgroup_range(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> (u32, u32) {
    let mut subgroup = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props2);
    }
    (subgroup.min_subgroup_size, subgroup.max_subgroup_size)
}

fn compute_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|index| index as u32)
        .ok_or_else(|| RunError::Skip("no compute queue family on CPU Vulkan device".into()))
}

fn device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<&'static CStr>> {
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
    let has_ext = extensions.iter().any(|extension| unsafe {
        CStr::from_ptr(extension.extension_name.as_ptr()) == vk::EXT_SUBGROUP_SIZE_CONTROL_NAME
    });
    if props.api_version < vk::API_VERSION_1_3 && !has_ext {
        return Err(RunError::Skip(
            "device lacks Vulkan 1.3/VK_EXT_subgroup_size_control".into(),
        ));
    }
    Ok(if props.api_version < vk::API_VERSION_1_3 {
        vec![vk::EXT_SUBGROUP_SIZE_CONTROL_NAME]
    } else {
        Vec::new()
    })
}
