//! Bounded diagnostic of ordinary cache growth, without model weights.
use fusor::{Tensor, cache::KvCache, device::Device};
use fusor_gpu::launch::{BufferTransfer, CommandRecord};
use std::{sync::Arc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    assert!(
        args.len() == 4,
        "cache_growth_lab PAIRS HEADS CAPACITY WIDTH"
    );
    let [pairs, heads, capacity, width] = args[..] else {
        unreachable!()
    };
    assert!((1..=64).contains(&pairs));
    assert!((1..=64).contains(&heads));
    assert!((1..=4096).contains(&capacity));
    assert!((1..=256).contains(&width));
    let elements = heads * capacity * width;
    let old_bytes = pairs * 2 * elements * 4;
    assert!(
        old_bytes * 3 <= 512 * 1024 * 1024,
        "old plus new stores exceed 512 MiB"
    );
    let device = Device::gpu_blocking()?;
    let data: Vec<f32> = (0..elements).map(|i| (i % 251) as f32 / 128.).collect();
    let input = Tensor::from_slice(&device, [1, heads, capacity, width], &data);
    let mut caches: Vec<_> = (0..pairs)
        .map(|_| KvCache::with_capacity(2, capacity as u64))
        .collect();
    for cache in &mut caches {
        cache.append(&input, &input);
    }
    KvCache::resolve_and_commit_all(&mut caches, &[])?;
    device.session().wait()?;
    let target = device.backend().gpu_target().unwrap();
    let compiles_before = target.launcher().pipeline_compiles();
    let start = Instant::now();
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        let mut dispatches = 0;
        let mut copies = 0;
        let mut copied_bytes = 0;
        let mut cleared_bytes = 0;
        for command in commands {
            match command {
                CommandRecord::Dispatch { name, .. } => {
                    dispatches += 1;
                    eprintln!("CACHE_GROWTH_DISPATCH {name}");
                }
                CommandRecord::CopyBuffer { bytes, .. } => {
                    copies += 1;
                    copied_bytes += bytes;
                }
                CommandRecord::Transfers { copies: transfers } => {
                    for transfer in transfers {
                        match transfer {
                            BufferTransfer::Copy { bytes, .. } => {
                                copies += 1;
                                copied_bytes += bytes;
                            }
                            BufferTransfer::Clear { bytes, .. } => cleared_bytes += bytes,
                        }
                    }
                }
            }
        }
        eprintln!(
            "CACHE_GROWTH_COMMANDS {}",
            serde_json::json!({
                "host_ms": start.elapsed().as_secs_f64() * 1000.,
                "dispatches": dispatches, "copies": copies,
                "copied_bytes": copied_bytes, "cleared_bytes": cleared_bytes,
            })
        );
    })));
    KvCache::reserve_all(&mut caches, 1)?;
    let submitted = start.elapsed();
    device.session().wait()?;
    let completed = start.elapsed();
    target.set_plan_observer(None);
    eprintln!(
        "CACHE_GROWTH_TIMING {}",
        serde_json::json!({
            "pairs": pairs, "heads": heads, "capacity": capacity, "width": width,
            "old_bytes": old_bytes,
            "resolve_ms": submitted.as_secs_f64() * 1000.,
            "wait_ms": (completed - submitted).as_secs_f64() * 1000.,
            "total_ms": completed.as_secs_f64() * 1000.,
            "new_pipeline_compiles": target.launcher().pipeline_compiles() - compiles_before,
        })
    );
    for cache in &caches {
        assert_eq!(cache.storage_capacity(), Some(2 * capacity as u64));
        assert_eq!(cache.current_len(), Some(capacity as u64));
        for value in [cache.k().unwrap(), cache.v().unwrap()] {
            assert_eq!(value.as_dyn().narrow(2, 0, capacity)?.to_vec_f32()?, data);
        }
    }
    println!("CACHE_GROWTH_VERIFIED {} values", elements * pairs * 2);
    Ok(())
}
