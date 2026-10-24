use super::*;
use fusor_ir::device::{DeviceKind, Limits};
use fusor_ir::ir::kernel::{BufferAccess, BufferDecl, MemoryLevel, StorageView, TileLayout};
use std::sync::Arc;

fn caps() -> Caps {
    Caps {
        kind: DeviceKind::Gpu,
        name: "portable vector load test".into(),
        limits: Limits::default(),
        subgroups: None,
        f16: false,
        bf16: false,
        coop: Default::default(),
        atomic_f32: false,
        workgroup_alias: false,
        mixed_precision_coop_store: false,
        pipeline_cache: false,
        timestamp_query: false,
        simd_widths: Default::default(),
        threads: 1,
    }
}

fn kernel(elements: u32, components: u32, offset: u32, writable: bool, masked: bool) -> KernelIr {
    let b = fusor_tile::build::Kernel::new();
    let storage = |binding, access| {
        let buffer = Arc::new(BufferDecl {
            binding,
            element: ScalarElement::F32.element(),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[elements]),
            access,
        });
        StorageView {
            layout: buffer.layout.clone(),
            buffer,
            offset: 0,
        }
    };
    let mut input = storage(
        1,
        if writable {
            BufferAccess::ReadWrite
        } else {
            BufferAccess::Read
        },
    );
    input.offset = offset;
    let output = storage(2, BufferAccess::ReadWrite);
    let base = b.mul(b.builtin(Builtin::ProgramId(0)), b.u32(4));
    let body = (0..components)
        .map(|i| {
            let index = b.add(base.clone(), b.u32(i));
            Stmt::Store {
                dst: output.clone(),
                addr: Addr::Linear(index.clone()),
                value: b.load(
                    Source::Storage(input.clone()),
                    Addr::Linear(index),
                    b.bool(!masked),
                    b.f32(-7.),
                ),
                mask: b.bool(true),
            }
        })
        .collect();
    KernelIr {
        buffers: vec![
            storage(0, BufferAccess::Read).buffer,
            input.buffer,
            output.buffer,
        ],
        grid: [4, 1, 1],
        block: 1,
        body,
        byte_arena: None,
        name: "grouped_storage_loads",
    }
}

fn input_type(ir: &KernelIr) -> naga::TypeInner {
    let emitted = emit(ir, &caps()).unwrap();
    let global = emitted
        .module
        .global_variables
        .iter()
        .find(|(_, g)| g.binding.as_ref().is_some_and(|b| b.binding == 1))
        .unwrap()
        .1;
    let naga::TypeInner::Array { base, .. } = emitted.module.types[global.ty].inner else {
        panic!("storage array")
    };
    emitted.module.types[base].inner.clone()
}

#[test]
fn complete_aligned_scalar_reads_use_one_vector_storage_element() {
    assert!(
        matches!(
            input_type(&kernel(32, 4, 4, false, false)),
            naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                ..
            }
        ),
        "four adjacent reads still use scalar storage"
    );
}

#[test]
fn incomplete_or_unproven_storage_groups_remain_scalar() {
    for args in [
        (31, 4, 0, false, false),
        (32, 3, 0, false, false),
        (32, 4, 1, false, false),
        (32, 4, 0, true, false),
        (32, 4, 0, false, true),
    ] {
        assert!(
            matches!(
                input_type(&kernel(args.0, args.1, args.2, args.3, args.4)),
                naga::TypeInner::Scalar(_)
            ),
            "unproven group {args:?} became vector storage"
        );
    }
}

#[test]
fn grouped_storage_reads_preserve_all_bits_on_webgpu() {
    use fusor_ir::dtype::Persistence;
    use fusor_ir::target::{Target, Uniforms};
    let target = match crate::target::GpuTarget::new_blocking() {
        Ok(t) => t,
        Err(e) if std::env::var_os("FUSOR_CONFORMANCE_REQUIRE_GPU").is_none() => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
        Err(e) => panic!("GPU required: {e}"),
    };
    let ir = kernel(32, 4, 0, false, false);
    let artifact = target.emit(&ir).unwrap();
    let uniform = target.alloc(128, Persistence::Step).unwrap();
    let output = target.alloc(128, Persistence::Step).unwrap();
    for round in 0..3u32 {
        let mut words: Vec<u32> = (0..32u32)
            .map(|i| (i.wrapping_mul(0x182d7615) ^ 0x3f800000).wrapping_add(round))
            .collect();
        words[..8].copy_from_slice(&[
            0, 0x80000000, 1, 0x007fffff, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678,
        ]);
        let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = target
            .pool()
            .create_buffer_init(
                &bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )
            .unwrap();
        target
            .launch(
                &artifact,
                ir.grid,
                &[uniform.clone(), input, output.clone()],
                &Uniforms::default(),
            )
            .unwrap();
        let actual = pollster::block_on(target.readback(&output, 64)).unwrap();
        assert_eq!(
            actual,
            bytes[..64],
            "changed storage bits on replay {round}"
        );
    }
}
