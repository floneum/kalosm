//! THROWAWAY: can logical-access stages fuse without owning physical layouts?
mod bench;
mod emit;
mod graph;
mod plan;
mod run;
mod storage;
use graph::{CASES, Graph};
use plan::{Config, Plan};
use std::io::{self, IsTerminal, Write};
use std::time::Instant;
use storage::{Allocation, Packing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn allocations(g: &Graph, a: &Allocation) {
    println!(
        "  {:<18} {:>9} {:>10} {:>8}",
        "value", "live", "offset B", "size B"
    );
    for (s, off) in a.slots.iter().zip(&a.offsets) {
        let name = g
            .values
            .get(s.id)
            .map(|v| v.name.clone())
            .unwrap_or_else(|| format!("fold_scratch_{}", usize::MAX - s.id));
        println!(
            "  {name:<18} {:>9} {:>10} {:>8}",
            format!("{}..{}", s.first, s.last),
            off * 4,
            s.len * 4
        );
    }
    println!(
        "  allocated={} B; peak-live lower bound={} B; interference edges={}",
        a.len * 4,
        a.lower_bound * 4,
        a.conflicts
    );
}
fn show(g: &Graph, p: &Plan, cfg: &Config, ms: f64, detail: bool) {
    println!(
        "\n{}  input={:?}  packing={:?}  shared cap={} B  fusion={}",
        g.name, g.values[0].shape, cfg.packing, cfg.shared_bytes, cfg.fuse
    );
    println!(
        "  {} compute stages + {} views → {} dispatches; planning={ms:.3} ms",
        g.stages().len(),
        g.values
            .iter()
            .filter(|v| matches!(v.op, graph::Op::View(..)))
            .count(),
        p.regions.len()
    );
    println!(
        "  {} legal partitions; ownership rejects={}; capacity rejects={}",
        p.partitions, p.ownership_rejects, p.capacity_rejects
    );
    println!(
        "  forwarded={} values; serial reduction limit={}; legacy={}",
        p.forwarded.len(),
        p.serial_limit,
        p.legacy
    );
    println!(
        "  modeled score={:.2} us (uncalibrated); global arena={} B",
        p.score_ns / 1000.0,
        p.global.len * 4
    );
    for (i, r) in p.regions.iter().enumerate() {
        println!(
            "  R{i}: [{}] groups={} shared={} B global traffic={} B",
            r.members
                .iter()
                .map(|id| g.values[*id].name.as_str())
                .collect::<Vec<_>>()
                .join(" → "),
            r.groups,
            r.shared.len * 4,
            r.traffic_bytes
        );
        if detail {
            allocations(g, &r.shared);
        }
    }
    if detail {
        println!("  Global allocations (lifetimes are dispatch positions):");
        allocations(g, &p.global);
    }
}
fn execute(
    g: &Graph,
    cfg: &Config,
    gpu: Option<&fusor_gpu::GpuDevice>,
    baseline: bool,
    dump: bool,
    detail: bool,
    iterations: usize,
) -> Result<()> {
    let start = Instant::now();
    let p = plan::compile(g, cfg)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    show(g, &p, cfg, ms, detail);
    let sources: Vec<String> = p.regions.iter().map(|r| emit::shader(g, &p, r)).collect();
    for (i, source) in sources.iter().enumerate() {
        run::validate_shader(source)?;
        if dump {
            println!("\n--- R{i} WGSL ---\n{source}");
        }
    }
    println!("  WGSL validation: PASS");
    if let Some(gpu) = gpu {
        let executable = run::Executable::build(gpu, g, &p, &sources)?;
        let mut worst = 0.0f32;
        for seed in 0..3 {
            worst = worst.max(executable.check(gpu, g, &p, seed)?);
        }
        let us = executable.time(gpu, iterations)?;
        println!(
            "  GPU: PASS (3 inputs, poisoned/reused arena), max error={worst:.3e}; build={:.2} ms; batched median={us:.2} us/step",
            executable.build_ms
        );
    }
    if baseline {
        match run::baseline(g) {
            Ok(b) => println!(
                "  Existing Fusor: PASS, {} dispatches; first resolve={:.2} ms; max error={:.3e}",
                b.launches, b.first_ms, b.error
            ),
            Err(e) => println!("  Existing Fusor: ERROR — {e}"),
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |s: &str| args.iter().any(|a| a == s);
    let value = |s: &str| {
        args.iter()
            .position(|a| a == s)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    if has("--help") {
        println!(
            "--bench                  Matched warm GPU benchmark, including original prototype\n--legacy                 Original emission and scheduling policies\n--no-forwarding          Materialize scalar producers"
        );
        println!(
            "THROWAWAY Fusor storage/fusion prototype\n\ncargo run -p fusor-storage-prototype --release -- [options]\n\n--all                    Run every built-in case\n--case NAME              {}\n--plan-only              Plan and validate WGSL without requesting a GPU\n--baseline               Also execute the current Fusor compiler\n--rows N --cols N        Shape (default 16 × 128; cols positive/even)\n--shared-bytes N         Workgroup budget (default 2048)\n--global-bytes N         Output/intermediate arena budget\n--packing MODE           bestfit | colored | dedicated\n--no-fusion              One dispatch per compute stage\n--details                Print interference lifetimes and allocation offsets\n--dump-wgsl              Print generated shaders\n--iterations N           Steps per timing batch (default 50)\n--interactive            Explore cases and policies in a terminal",
            CASES.join(" | ")
        );
        return Ok(());
    }
    let rows = value("--rows").unwrap_or("16").parse()?;
    let cols = value("--cols").unwrap_or("128").parse()?;
    let iterations = value("--iterations")
        .unwrap_or("50")
        .parse::<usize>()?
        .max(1);
    let mut cfg = Config::default();
    cfg.forwarding = !has("--no-forwarding");
    if has("--legacy") {
        cfg.legacy = true;
        cfg.forwarding = false;
        cfg.serial_limit = 8;
    }
    cfg.fuse = !has("--no-fusion");
    if let Some(x) = value("--shared-bytes") {
        cfg.shared_bytes = x.parse()?;
    }
    if let Some(x) = value("--global-bytes") {
        cfg.global_bytes = x.parse()?;
    }
    cfg.packing = match value("--packing").unwrap_or("bestfit") {
        "colored" => Packing::Colored,
        "dedicated" => Packing::Dedicated,
        "bestfit" => Packing::BestFit,
        x => panic!("unknown packing {x}"),
    };
    let interactive = has("--interactive") || (args.is_empty() && io::stdin().is_terminal());
    if has("--bench") {
        let cases = if has("--all") {
            CASES.to_vec()
        } else {
            vec![value("--case").unwrap_or(CASES[0])]
        };
        for name in cases {
            bench::compare(&graph::case(name, rows, cols), &cfg, iterations)?;
        }
        return Ok(());
    }
    let gpu = if has("--plan-only") || interactive {
        None
    } else {
        Some(pollster::block_on(fusor_gpu::GpuDevice::request(None))?)
    };
    if let Some(gpu) = &gpu {
        println!("GPU: {}", gpu.adapter_info().name);
    }
    if interactive {
        let mut current = 0;
        loop {
            print!("\x1b[2J\x1b[H");
            println!("THROWAWAY — logical indexing → region ownership → storage assignment");
            let g = graph::case(CASES[current], rows, cols);
            let start = Instant::now();
            match plan::compile(&g, &cfg) {
                Ok(p) => show(&g, &p, &cfg, start.elapsed().as_secs_f64() * 1000.0, false),
                Err(e) => println!("{e}"),
            };
            println!(
                "\n[n] next case  [p] packing  [m] memory cap  [f] fusion  [s] scalar forwarding\n[d] allocation details  [g] execute GPU  [b] existing Fusor  [q] quit\nType a key and Enter:"
            );
            io::stdout().flush()?;
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            match line.trim() {
                "q" => break,
                "n" => current = (current + 1) % CASES.len(),
                "p" => {
                    cfg.packing = match cfg.packing {
                        Packing::BestFit => Packing::Colored,
                        Packing::Colored => Packing::Dedicated,
                        Packing::Dedicated => Packing::BestFit,
                    }
                }
                "m" => {
                    cfg.shared_bytes = match cfg.shared_bytes {
                        0 => 256,
                        256 => 512,
                        512 => 1024,
                        1024 => 2048,
                        2048 => 4096,
                        4096 => 16384,
                        _ => 0,
                    }
                }
                "f" => cfg.fuse = !cfg.fuse,
                "s" => cfg.forwarding = !cfg.forwarding,
                "d" | "g" | "b" => {
                    let gpu = if line.trim() == "g" {
                        Some(pollster::block_on(fusor_gpu::GpuDevice::request(None))?)
                    } else {
                        None
                    };
                    if let Err(e) = execute(
                        &g,
                        &cfg,
                        gpu.as_ref(),
                        line.trim() == "b",
                        false,
                        true,
                        iterations,
                    ) {
                        println!("{e}");
                    }
                    println!("Enter to return");
                    let mut wait = String::new();
                    io::stdin().read_line(&mut wait)?;
                }
                _ => {}
            }
        }
    } else {
        let cases = if has("--all") {
            CASES.to_vec()
        } else {
            vec![value("--case").unwrap_or(CASES[0])]
        };
        for case in cases {
            execute(
                &graph::case(case, rows, cols),
                &cfg,
                gpu.as_ref(),
                has("--baseline"),
                has("--dump-wgsl"),
                has("--details"),
                iterations,
            )?;
        }
    }
    Ok(())
}
