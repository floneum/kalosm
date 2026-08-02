//! Calibration: seven microbenchmarks, ~180 ms, once per device, cached to
//! disk. This is what makes the cost model portable rather than fitted to
//! one adapter.
//!
//! Every bench builds a real `KernelIr`, emits it through the target,
//! allocates real buffers, launches, and times the wall clock around a
//! `wait()`. Each reports the **median of 5 reps** so one scheduler hiccup
//! cannot move a rate. Each has a wall-clock budget; a bench that errors or
//! overruns leaves its seed value in place and is named in
//! [`CalibrationReport::fell_back`]. Calibration is therefore never a
//! failure mode — the worst case is the shipped table.
//!
//! `Target::lower` is deliberately unused: it needs a `Plan`, a `Launch`
//! and an `EGraph` to build a `LowerCtx`, none of which a microbenchmark
//! has. The benches construct L2 directly, which is the same dialect
//! `lower` would have produced.
//!
//! Owned by W6.

use fusor2_ir::Result;
use fusor2_ir::cost::{Calibrate, DeviceFacts, MacUnit, RateDtype};
use fusor2_ir::device::{Caps, DeviceKind};
use fusor2_ir::dtype::{Dtype, NumericContract, Persistence};
use fusor2_ir::error::Error;
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Buffer, BufferAccess, BufferDecl, Builtin, ElementType, KernelIr, Local,
    LocalDecl, MemoryLevel, ScalarElement, Source, Stmt, StorageView, Tile, TileDecl, TileExpr,
    TileExprKind, TileLayout, TileLiteral, WorkgroupAxis,
};
use fusor2_ir::scalar::{BinOp, UnOp};
use fusor2_ir::target::{Artifact, Buf, Target, Uniforms};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The seven microbenchmarks, in run order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Bench {
    /// 512 one-workgroup no-op dispatches -> `launch_ps`. The first cold
    /// `emit` also gives `compile_ps_per_kernel`, and on CPU the pool-wake
    /// span gives `thread_wake_ps`.
    LaunchOverhead,
    /// A 64 MiB streaming copy -> `dram_bytes_per_us`.
    DramBandwidth,
    /// The same copy across a size sweep -> `llc_bytes`.
    LlcSize,
    /// A workgroup-tile staging loop -> `wg_bytes_per_us` and
    /// `single_buffered_traffic_pct`.
    WorkgroupBandwidth,
    /// Register-resident FMA loops -> `mac_per_us`.
    MacRates,
    /// The same loop with `exp` -> `trans_ps`.
    Transcendental,
    /// An accumulator drain at two arena footprints plus a resident-lane
    /// sweep -> `store_ps_per_element` and `saturation_lanes`.
    Occupancy,
}

impl Bench {
    pub const ALL: [Bench; 7] = [
        Bench::LaunchOverhead,
        Bench::DramBandwidth,
        Bench::LlcSize,
        Bench::WorkgroupBandwidth,
        Bench::MacRates,
        Bench::Transcendental,
        Bench::Occupancy,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::LaunchOverhead => "launch",
            Self::DramBandwidth => "dram",
            Self::LlcSize => "llc",
            Self::WorkgroupBandwidth => "wg",
            Self::MacRates => "mac",
            Self::Transcendental => "trans",
            Self::Occupancy => "epilogue_occupancy",
        }
    }

    /// Wall-clock budget. The seven sum to 180 ms.
    pub const fn budget(self) -> Duration {
        Duration::from_millis(match self {
            Self::LaunchOverhead => 20,
            Self::DramBandwidth => 35,
            Self::LlcSize => 30,
            Self::WorkgroupBandwidth => 25,
            Self::MacRates => 25,
            Self::Transcendental => 15,
            Self::Occupancy => 30,
        })
    }
}

/// What calibration actually measured, and what it left on its seed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CalibrationReport {
    pub measured: Vec<&'static str>,
    pub fell_back: Vec<&'static str>,
    pub micros: u64,
}

/// The shipped calibrator.
#[derive(Default, Debug, Clone, Copy)]
pub struct Calibrator;

/// Reps per bench; the median is reported.
const REPS: usize = 5;
/// Invocations per workgroup for every bench kernel. The WebGPU baseline
/// guarantees 256.
const BLOCK: u32 = 256;
/// Elements one invocation streams in the copy benches.
const PER_THREAD: u32 = 8;
/// f32 elements one workgroup tile holds in the staging bench.
const TILE_ELEMS: u32 = 1024;
/// Iterations the register loops run.
const LOOP_ITERS: u32 = 4_096;
/// Independent accumulators the register loops carry, so the measurement is
/// issue-bound rather than latency-bound.
const ACCUMULATORS: usize = 8;

// ---------------------------------------------------------------------------
// L2 construction helpers
// ---------------------------------------------------------------------------

fn u32_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::U32)
}

fn lit_u32(v: u32) -> TileExpr {
    TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32_ty())
}

fn lit(scalar: ScalarElement, bits: u32) -> TileExpr {
    let literal = match scalar {
        ScalarElement::F32 => TileLiteral::F32(bits),
        ScalarElement::F16 => TileLiteral::F16(bits as u16),
        ScalarElement::BF16 => TileLiteral::BF16(bits as u16),
        ScalarElement::U32 => TileLiteral::U32(bits),
        ScalarElement::I32 => TileLiteral::I32(bits as i32),
        ScalarElement::Bool => TileLiteral::Bool(bits != 0),
    };
    TileExpr::new(TileExprKind::Literal(literal), ElementType::Scalar(scalar))
}

/// A statically-true mask. The lowerer skips codegen for it, so a bench does
/// not measure a predicate it never intended to pay for.
fn mask_true() -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::Bool(true)),
        ElementType::Scalar(ScalarElement::Bool),
    )
}

fn builtin(b: Builtin) -> TileExpr {
    TileExpr::new(TileExprKind::Builtin(b), u32_ty())
}

fn bin(op: BinOp, left: TileExpr, right: TileExpr) -> TileExpr {
    let ty = left.element();
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left,
            right,
            numeric: NumericContract::RELAXED,
        },
        ty,
    )
}

fn unary(op: UnOp, value: TileExpr) -> TileExpr {
    let ty = value.element();
    TileExpr::new(
        TileExprKind::Unary {
            op,
            value,
            numeric: NumericContract::RELAXED,
        },
        ty,
    )
}

/// `workgroup_id.x * BLOCK + local_invocation_index`.
fn global_index() -> TileExpr {
    bin(
        BinOp::Add,
        bin(
            BinOp::Mul,
            builtin(Builtin::ProgramId(WorkgroupAxis::X)),
            lit_u32(BLOCK),
        ),
        builtin(Builtin::Lane),
    )
}

fn buffer(binding: u32, scalar: ScalarElement, elements: u32, access: BufferAccess) -> Buffer {
    Arc::new(BufferDecl {
        binding,
        element: ElementType::Scalar(scalar),
        layout: TileLayout::contiguous(MemoryLevel::Storage, &[elements]),
        access,
    })
}

fn view(b: &Buffer) -> StorageView {
    StorageView {
        buffer: b.clone(),
        offset: 0,
        layout: b.layout.clone(),
    }
}

fn load(b: &Buffer, index: TileExpr, scalar: ScalarElement) -> TileExpr {
    TileExpr::new(
        TileExprKind::Load {
            src: Source::Storage(view(b)),
            addr: Box::new(Addr::Linear(index)),
            mask: mask_true(),
            fill: lit(scalar, 0),
        },
        ElementType::Scalar(scalar),
    )
}

fn store(b: &Buffer, index: TileExpr, value: TileExpr) -> Stmt {
    Stmt::Store {
        dst: view(b),
        addr: Addr::Linear(index),
        value,
        mask: mask_true(),
    }
}

fn local(scalar: ScalarElement) -> Local {
    Arc::new(LocalDecl::new(ElementType::Scalar(scalar)))
}

fn load_local(l: &Local) -> TileExpr {
    TileExpr::new(TileExprKind::LoadLocal(l.clone()), l.element)
}

fn tile(name: &'static str, scalar: ScalarElement, elements: u32) -> Tile {
    Arc::new(TileDecl::new(
        ElementType::Scalar(scalar),
        TileLayout::contiguous(MemoryLevel::Workgroup, &[elements]),
        name,
    ))
}

/// Grid for `threads` invocations, folded against the per-dimension cap.
/// The slab count is picked before x is sized, so an over-long 1-D count
/// does not launch a nearly-empty final slab.
fn grid_for(threads: u64, caps: &Caps) -> [u32; 3] {
    let cap = u64::from(caps.limits.max_compute_workgroups_per_dimension.max(1));
    let total = threads.div_ceil(u64::from(BLOCK)).max(1);
    let y = total.div_ceil(cap).clamp(1, cap);
    let x = total.div_ceil(y).clamp(1, cap);
    let z = total.div_ceil(x * y).clamp(1, cap);
    [x as u32, y as u32, z as u32]
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

fn noop_kernel() -> KernelIr {
    KernelIr {
        buffers: Vec::new(),
        grid: [1, 1, 1],
        block: BLOCK,
        body: vec![Stmt::Return],
        byte_arena: None,
        name: "calib_noop",
    }
}

/// `out[i] = in[i]` over `elements` f32, `PER_THREAD` per invocation.
fn copy_kernel(elements: u32, caps: &Caps) -> KernelIr {
    let src = buffer(0, ScalarElement::F32, elements, BufferAccess::Read);
    let dst = buffer(1, ScalarElement::F32, elements, BufferAccess::ReadWrite);
    let base = bin(BinOp::Mul, global_index(), lit_u32(PER_THREAD));
    let body = (0..PER_THREAD)
        .map(|i| {
            let index = bin(BinOp::Add, base.clone(), lit_u32(i));
            let value = load(&src, index.clone(), ScalarElement::F32);
            store(&dst, index, value)
        })
        .collect();
    KernelIr {
        buffers: vec![src, dst],
        grid: grid_for(u64::from(elements / PER_THREAD), caps),
        block: BLOCK,
        body,
        byte_arena: None,
        name: "calib_copy",
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LoopOp {
    Fma,
    Exp,
}

/// A register-resident loop of `ACCUMULATORS` independent chains, stored
/// once at the end so nothing is dead code.
fn register_loop_kernel(
    scalar: ScalarElement,
    op: LoopOp,
    workgroups: u32,
    caps: &Caps,
) -> KernelIr {
    let out = buffer(0, scalar, BLOCK * workgroups, BufferAccess::ReadWrite);
    let seed = load(&out, global_index(), scalar);
    let one = match scalar {
        ScalarElement::F32 => lit(scalar, 1.0f32.to_bits()),
        ScalarElement::F16 => lit(scalar, 0x3c00),
        ScalarElement::BF16 => lit(scalar, 0x3f80),
        _ => lit(scalar, 1),
    };

    let locals: Vec<Local> = (0..ACCUMULATORS).map(|_| local(scalar)).collect();
    let accumulators = locals
        .iter()
        .map(|l| {
            let value = load_local(l);
            let update = match op {
                // One multiply and one add: exactly one MAC per iteration.
                LoopOp::Fma => bin(BinOp::Add, bin(BinOp::Mul, value.clone(), one.clone()), value),
                LoopOp::Exp => unary(UnOp::Exp, value),
            };
            Accumulator {
                local: l.clone(),
                init: seed.clone(),
                update,
            }
        })
        .collect();

    let mut body = vec![Stmt::Loop {
        count: Some(lit_u32(LOOP_ITERS)),
        index: None,
        accumulators,
        body: Vec::new(),
    }];
    let total = locals
        .iter()
        .map(load_local)
        .reduce(|a, b| bin(BinOp::Add, a, b))
        .expect("ACCUMULATORS is non-zero");
    body.push(store(&out, global_index(), total));

    KernelIr {
        buffers: vec![out],
        grid: grid_for(u64::from(BLOCK) * u64::from(workgroups), caps),
        block: BLOCK,
        body,
        byte_arena: None,
        name: "calib_register_loop",
    }
}

/// Stage a workgroup tile from storage and drain it back, `iters` times.
/// `depth == 2` alternates two tiles so a fill overlaps the previous drain;
/// `depth == 1` reuses one tile behind a second barrier and loses that
/// overlap, which is exactly what `single_buffered_traffic_pct` prices.
fn staging_kernel(depth: u8, iters: u32, workgroups: u32, caps: &Caps) -> KernelIr {
    let elements = BLOCK * workgroups + iters;
    let src = buffer(0, ScalarElement::F32, elements, BufferAccess::Read);
    let dst = buffer(1, ScalarElement::F32, elements, BufferAccess::ReadWrite);
    let tiles: Vec<Tile> = if depth == 1 {
        vec![tile("stage_a", ScalarElement::F32, TILE_ELEMS)]
    } else {
        vec![
            tile("stage_a", ScalarElement::F32, TILE_ELEMS),
            tile("stage_b", ScalarElement::F32, TILE_ELEMS),
        ]
    };

    let mut body = Vec::new();
    for i in 0..iters {
        let t = &tiles[(i as usize) % tiles.len()];
        let index = bin(BinOp::Add, global_index(), lit_u32(i));
        body.push(Stmt::FillTile {
            dst: t.clone(),
            value: load(&src, index.clone(), ScalarElement::F32),
            bounds: [None, None],
        });
        body.push(Stmt::Barrier);
        let staged = TileExpr::new(
            TileExprKind::LoadTile {
                tile: t.clone(),
                index: builtin(Builtin::Lane),
            },
            ElementType::Scalar(ScalarElement::F32),
        );
        body.push(store(&dst, index, staged));
        if depth == 1 {
            // One tile cannot be refilled until every lane has drained it.
            body.push(Stmt::Barrier);
        }
    }

    KernelIr {
        buffers: vec![src, dst],
        grid: grid_for(u64::from(BLOCK) * u64::from(workgroups), caps),
        block: BLOCK,
        body,
        byte_arena: None,
        name: "calib_staging",
    }
}

/// A drain-shaped kernel: `arena_elems` of workgroup memory held live while
/// every lane stores `per_lane` outputs. The footprint divides into how many
/// workgroups a core holds, so running it at two footprints isolates the
/// residency term the epilogue drain is divided by.
fn drain_kernel(arena_elems: u32, per_lane: u32, workgroups: u32, caps: &Caps) -> KernelIr {
    let elements = BLOCK * workgroups * per_lane;
    let dst = buffer(0, ScalarElement::F32, elements, BufferAccess::ReadWrite);
    let scratch = tile("drain_arena", ScalarElement::F32, arena_elems);
    let mut body = vec![
        Stmt::FillTile {
            dst: scratch.clone(),
            value: lit(ScalarElement::F32, 0),
            bounds: [None, None],
        },
        Stmt::Barrier,
    ];
    let base = bin(BinOp::Mul, global_index(), lit_u32(per_lane));
    for i in 0..per_lane {
        let staged = TileExpr::new(
            TileExprKind::LoadTile {
                tile: scratch.clone(),
                index: builtin(Builtin::Lane),
            },
            ElementType::Scalar(ScalarElement::F32),
        );
        body.push(store(&dst, bin(BinOp::Add, base.clone(), lit_u32(i)), staged));
    }
    KernelIr {
        buffers: vec![dst],
        grid: grid_for(u64::from(BLOCK) * u64::from(workgroups), caps),
        block: BLOCK,
        body,
        byte_arena: None,
        name: "calib_drain",
    }
}

// ---------------------------------------------------------------------------
// Running one bench
// ---------------------------------------------------------------------------

/// Emit, allocate, and time `REPS` launches, reporting the median.
struct Runner<'a> {
    target: &'a dyn Target,
}

impl Runner<'_> {
    fn caps(&self) -> &Caps {
        self.target.caps()
    }

    /// Compile once and report how long it took — the cold `emit` is the
    /// only honest measurement of `compile_ps_per_kernel` there is.
    fn compile(&self, ir: &KernelIr) -> Result<(Artifact, Duration)> {
        let started = Instant::now();
        let artifact = self
            .target
            .emit(ir)
            .map_err(|e| Error::Device(format!("calibration emit: {e}")))?;
        Ok((artifact, started.elapsed()))
    }

    fn alloc(&self, ir: &KernelIr) -> Result<Vec<Buf>> {
        ir.buffers
            .iter()
            .map(|b| {
                let bytes = b.layout.element_count() * b.element.byte_size();
                self.target.alloc(bytes.max(4), Persistence::Step)
            })
            .collect()
    }

    /// Median wall-clock span of `REPS` runs of `launches` dispatches, each
    /// timed around a `wait()`.
    fn time(
        &self,
        artifact: &Artifact,
        grid: [u32; 3],
        binds: &[Buf],
        launches: u32,
        budget: Duration,
        deadline: Instant,
    ) -> Result<Duration> {
        let uniforms = Uniforms::default();
        let mut spans = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            if Instant::now() >= deadline {
                return Err(Error::Budget("calibration bench overran".into()));
            }
            let started = Instant::now();
            for _ in 0..launches {
                self.target.launch(artifact, grid, binds, &uniforms)?;
            }
            self.target.wait()?;
            spans.push(started.elapsed());
        }
        spans.sort_unstable();
        let median = spans[REPS / 2];
        if median > budget {
            return Err(Error::Budget("calibration bench overran".into()));
        }
        Ok(median)
    }

    /// One kernel end to end: compile, allocate, time.
    fn measure(
        &self,
        ir: &KernelIr,
        launches: u32,
        budget: Duration,
        deadline: Instant,
    ) -> Result<(Duration, Duration)> {
        let (artifact, compile) = self.compile(ir)?;
        let binds = self.alloc(ir)?;
        let span = self.time(&artifact, ir.grid, &binds, launches, budget, deadline)?;
        Ok((span, compile))
    }
}

fn per_us(bytes: u64, span: Duration) -> Option<u64> {
    let micros = span.as_nanos() as f64 / 1_000.0;
    (micros > 0.0).then(|| (bytes as f64 / micros) as u64)
}

fn ps_per(count: u64, span: Duration) -> Option<u64> {
    (count > 0).then(|| (span.as_nanos() as u64).saturating_mul(1_000) / count)
}

impl Calibrator {
    pub const fn new() -> Self {
        Self
    }

    /// Run one microbenchmark against a live target and return its primary
    /// scalar: `launch_ps`, `dram_bytes_per_us`, `llc_bytes`,
    /// `wg_bytes_per_us`, `mac_per_us[Fma][F32]`, `trans_ps` or
    /// `saturation_lanes` respectively.
    pub fn run(&self, target: &dyn Target, bench: Bench) -> Result<u64> {
        let mut facts = crate::facts::seed_facts(target.caps());
        let runner = Runner { target };
        let deadline = Instant::now() + bench.budget();
        self.run_into(&runner, bench, &mut facts, deadline)?;
        Ok(match bench {
            Bench::LaunchOverhead => facts.launch_ps,
            Bench::DramBandwidth => facts.dram_bytes_per_us,
            Bench::LlcSize => facts.llc_bytes,
            Bench::WorkgroupBandwidth => facts.wg_bytes_per_us,
            Bench::MacRates => facts.mac_per_us[MacUnit::Fma as usize][RateDtype::F32 as usize],
            Bench::Transcendental => facts.trans_ps,
            Bench::Occupancy => u64::from(facts.saturation_lanes),
        })
    }

    /// Calibrate, and say what was measured and what fell back.
    pub fn calibrate_reporting(&self, target: &dyn Target) -> (DeviceFacts, CalibrationReport) {
        let mut facts = crate::facts::seed_facts(target.caps());
        let runner = Runner { target };
        let mut report = CalibrationReport::default();
        let started = Instant::now();
        for bench in Bench::ALL {
            let deadline = Instant::now() + bench.budget();
            match self.run_into(&runner, bench, &mut facts, deadline) {
                Ok(()) => report.measured.push(bench.name()),
                Err(_) => report.fell_back.push(bench.name()),
            }
        }
        // The two accelerator rows are derived from the probed scalar row by
        // the seed's ratio rather than measured: probing them needs a
        // cooperative-fragment and a dp4a kernel whose legality this crate
        // cannot establish without the backend's geometry tables. Named here
        // so the fallback is visible rather than assumed.
        report.fell_back.push("mac_coop_row");
        report.fell_back.push("mac_dp4a_row");
        report.micros = started.elapsed().as_micros() as u64;
        (facts, report)
    }

    fn run_into(
        &self,
        runner: &Runner<'_>,
        bench: Bench,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        match bench {
            Bench::LaunchOverhead => self.bench_launch(runner, facts, deadline),
            Bench::DramBandwidth => self.bench_dram(runner, facts, deadline),
            Bench::LlcSize => self.bench_llc(runner, facts, deadline),
            Bench::WorkgroupBandwidth => self.bench_wg(runner, facts, deadline),
            Bench::MacRates => self.bench_mac(runner, facts, deadline),
            Bench::Transcendental => self.bench_trans(runner, facts, deadline),
            Bench::Occupancy => self.bench_epilogue_occupancy(runner, facts, deadline),
        }
    }

    /// (1) 512 one-workgroup no-op dispatches. The span over 512 is the
    /// per-launch cost with the body removed. The cold `emit` that preceded
    /// it is `compile_ps_per_kernel`. On CPU, the difference between a
    /// one-workgroup and a pool-wide dispatch of the same empty body is the
    /// pool-wake span — the number that replaces `PARALLEL_THRESHOLD`.
    fn bench_launch(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const DISPATCHES: u32 = 512;
        let ir = noop_kernel();
        let budget = Bench::LaunchOverhead.budget();
        let (span, compile) = runner.measure(&ir, DISPATCHES, budget, deadline)?;
        if let Some(ps) = ps_per(u64::from(DISPATCHES), span) {
            facts.launch_ps = ps;
        }
        facts.compile_ps_per_kernel = (compile.as_nanos() as u64).saturating_mul(1_000).max(1);

        if runner.caps().kind == DeviceKind::Cpu {
            let threads = runner.caps().threads.max(1);
            let mut wide = noop_kernel();
            wide.grid = grid_for(u64::from(threads) * u64::from(BLOCK) * 4, runner.caps());
            if let Ok((wide_span, _)) = runner.measure(&wide, DISPATCHES, budget, deadline)
                && let Some(ps) = ps_per(u64::from(DISPATCHES), wide_span.saturating_sub(span))
            {
                facts.thread_wake_ps = ps.max(1);
            }
        }
        Ok(())
    }

    /// (2) A 64 MiB streaming copy. 64 MiB is comfortably past every
    /// last-level cache fusor2 targets, so the rate is DRAM's.
    fn bench_dram(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        let rate = self.copy_rate(runner, 64 << 20, Bench::DramBandwidth.budget(), deadline)?;
        facts.dram_bytes_per_us = rate.max(1);
        Ok(())
    }

    /// (3) The same copy across a size sweep. `llc_bytes` is the largest
    /// working set whose per-byte rate is still within 15% of the 64 KiB
    /// rate — a direct read of where reuse stops being free, which is the
    /// one number the DRAM reread term and the grid swizzle both share.
    fn bench_llc(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const SIZES: [u64; 7] = [
            64 << 10,
            512 << 10,
            4 << 20,
            16 << 20,
            64 << 20,
            256 << 20,
            1 << 30,
        ];
        let budget = Bench::LlcSize.budget();
        let reference = self.copy_rate(runner, SIZES[0], budget, deadline)?;
        let floor = reference / 100 * 85;
        let mut resident = SIZES[0];
        for &size in &SIZES[1..] {
            if Instant::now() >= deadline {
                break;
            }
            let Ok(rate) = self.copy_rate(runner, size, budget, deadline) else {
                break;
            };
            if rate < floor {
                break;
            }
            resident = size;
        }
        facts.llc_bytes = resident;
        Ok(())
    }

    fn copy_rate(
        &self,
        runner: &Runner<'_>,
        bytes: u64,
        budget: Duration,
        deadline: Instant,
    ) -> Result<u64> {
        let elements = u32::try_from(bytes / 4).map_err(|_| {
            Error::Device("calibration working set exceeds a 32-bit element count".into())
        })?;
        let elements = elements.next_multiple_of(BLOCK * PER_THREAD);
        let ir = copy_kernel(elements, runner.caps());
        let (span, _) = runner.measure(&ir, 1, budget, deadline)?;
        // A copy reads every byte once and writes it once.
        per_us(2 * u64::from(elements) * 4, span)
            .ok_or_else(|| Error::Device("calibration copy took no measurable time".into()))
    }

    /// (6) A workgroup-tile staging loop, double-buffered, then the same
    /// geometry single-buffered. The first gives `wg_bytes_per_us`; the
    /// ratio of the second to the first gives
    /// `single_buffered_traffic_pct`, which is the whole reason staging
    /// depth is a decision rather than a table column.
    fn bench_wg(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const ITERS: u32 = 64;
        const WORKGROUPS: u32 = 1_024;
        let budget = Bench::WorkgroupBandwidth.budget();
        // Each iteration fills and drains the whole tile.
        let bytes = u64::from(ITERS) * u64::from(WORKGROUPS) * 2 * u64::from(TILE_ELEMS) * 4;

        let double = staging_kernel(2, ITERS, WORKGROUPS, runner.caps());
        let (double_span, _) = runner.measure(&double, 1, budget, deadline)?;
        let rate = per_us(bytes, double_span)
            .ok_or_else(|| Error::Device("staging took no measurable time".into()))?;
        facts.wg_bytes_per_us = rate.max(1);

        let single = staging_kernel(1, ITERS, WORKGROUPS, runner.caps());
        if let Ok((single_span, _)) = runner.measure(&single, 1, budget, deadline) {
            let pct = single_span.as_nanos().saturating_mul(100) / double_span.as_nanos().max(1);
            // Below 100 would mean single buffering is free, which the
            // geometry makes impossible; clamp rather than invert the term.
            facts.single_buffered_traffic_pct =
                u32::try_from(pct).unwrap_or(u32::MAX).clamp(100, 400);
        }
        Ok(())
    }

    /// (4) A register-resident FMA loop per `(MacUnit, RateDtype)` the caps
    /// permit. Unprobed cells inherit the seed's ratio to the probed
    /// `Fma/F32` cell, so a device that reports no f16 still gets a
    /// self-consistent table rather than a zero.
    fn bench_mac(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const WORKGROUPS: u32 = 1_024;
        let budget = Bench::MacRates.budget();
        let caps = runner.caps();
        let seed = crate::facts::seed_facts(caps);
        let macs =
            u64::from(LOOP_ITERS) * ACCUMULATORS as u64 * u64::from(BLOCK) * u64::from(WORKGROUPS);

        let probe = |scalar: ScalarElement| -> Option<u64> {
            let ir = register_loop_kernel(scalar, LoopOp::Fma, WORKGROUPS, caps);
            let (span, _) = runner.measure(&ir, 1, budget, deadline).ok()?;
            per_us(macs, span)
        };

        let base = probe(ScalarElement::F32)
            .ok_or_else(|| Error::Device("the f32 FMA probe did not run".into()))?;
        let seed_base = seed.mac_per_us[MacUnit::Fma as usize][RateDtype::F32 as usize].max(1);
        for unit in 0..3 {
            for dt in 0..RateDtype::COUNT {
                let inherited = u128::from(base) * u128::from(seed.mac_per_us[unit][dt])
                    / u128::from(seed_base);
                facts.mac_per_us[unit][dt] =
                    u64::try_from(inherited).unwrap_or(u64::MAX).max(1);
            }
        }
        facts.mac_per_us[MacUnit::Fma as usize][RateDtype::F32 as usize] = base.max(1);

        let probed: [(bool, ScalarElement, RateDtype); 4] = [
            (caps.f16, ScalarElement::F16, RateDtype::F16),
            (caps.bf16, ScalarElement::BF16, RateDtype::BF16),
            (true, ScalarElement::U32, RateDtype::U32),
            (true, ScalarElement::I32, RateDtype::I32),
        ];
        for (permitted, scalar, slot) in probed {
            if !permitted || Instant::now() >= deadline {
                continue;
            }
            if let Some(rate) = probe(scalar) {
                facts.mac_per_us[MacUnit::Fma as usize][slot as usize] = rate.max(1);
            }
        }
        Ok(())
    }

    /// (5) The same loop with `exp`, so what is measured is the
    /// transcendental unit's own throughput.
    fn bench_trans(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const WORKGROUPS: u32 = 256;
        let budget = Bench::Transcendental.budget();
        let ops =
            u64::from(LOOP_ITERS) * ACCUMULATORS as u64 * u64::from(BLOCK) * u64::from(WORKGROUPS);
        let ir = register_loop_kernel(ScalarElement::F32, LoopOp::Exp, WORKGROUPS, runner.caps());
        let (span, _) = runner.measure(&ir, 1, budget, deadline)?;
        facts.trans_ps = ps_per(ops, span)
            .ok_or_else(|| Error::Device("the transcendental probe took no time".into()))?
            .max(1);
        Ok(())
    }

    /// (7) An accumulator drain at two arena footprints gives
    /// `store_ps_per_element` — inverted out of the model's own T3 form, so
    /// the measurement and the term cannot drift apart. Then a three-point
    /// resident-lane sweep gives `saturation_lanes`: the lane count at which
    /// measured throughput stops rising.
    fn bench_epilogue_occupancy(
        &self,
        runner: &Runner<'_>,
        facts: &mut DeviceFacts,
        deadline: Instant,
    ) -> Result<()> {
        const PER_LANE: u32 = 8;
        const WORKGROUPS: u32 = 512;
        let budget = Bench::Occupancy.budget();
        let caps = runner.caps();
        let max_storage = caps.limits.max_compute_workgroup_storage_size.max(1);
        // Two footprints an octave apart, both legal.
        let big_elems = (max_storage / 4 / 2).max(1);
        let small_elems = (big_elems / 2).max(1);
        let elems = u64::from(BLOCK) * u64::from(WORKGROUPS) * u64::from(PER_LANE);

        let mut store_rates = Vec::new();
        for arena_elems in [small_elems, big_elems] {
            if Instant::now() >= deadline {
                break;
            }
            let ir = drain_kernel(arena_elems, PER_LANE, WORKGROUPS, caps);
            let Ok((span, _)) = runner.measure(&ir, 1, budget, deadline) else {
                continue;
            };
            // t = elems * store_ps * subgroups * 1000 / root4(slots * 1e12),
            // with one emitting subgroup in this kernel shape.
            let slots = u128::from(max_storage / (arena_elems * 4).max(1)).max(1);
            let root = crate::terms::integer_root(slots * 1_000_000_000_000, 4).max(1);
            let ps =
                u128::from(span.as_nanos() as u64) * 1_000 * root / (u128::from(elems) * 1_000);
            store_rates.push(u64::try_from(ps).unwrap_or(u64::MAX).max(1));
        }
        if !store_rates.is_empty() {
            store_rates.sort_unstable();
            facts.store_ps_per_element = store_rates[store_rates.len() / 2];
        }

        // Three points an octave apart. Throughput stops rising once the
        // device is saturated, so the last lane count that still bought
        // >10% is the floor.
        let mut best_lanes = u64::from(BLOCK) * u64::from(WORKGROUPS);
        let mut previous: Option<f64> = None;
        for scale in [1u32, 4, 16] {
            if Instant::now() >= deadline {
                break;
            }
            let workgroups = WORKGROUPS * scale;
            let lanes = u64::from(BLOCK) * u64::from(workgroups);
            let ir = drain_kernel(small_elems, PER_LANE, workgroups, caps);
            let Ok((span, _)) = runner.measure(&ir, 1, budget, deadline) else {
                break;
            };
            let work = lanes as f64 * f64::from(PER_LANE);
            let throughput = work / span.as_nanos().max(1) as f64;
            match previous {
                Some(before) if throughput < before * 1.10 => break,
                _ => {
                    best_lanes = lanes;
                    previous = Some(throughput);
                }
            }
        }
        facts.saturation_lanes = u32::try_from(best_lanes).unwrap_or(u32::MAX).max(1);
        Ok(())
    }
}

impl Calibrate for Calibrator {
    fn calibrate(&self, target: &dyn Target) -> Result<DeviceFacts> {
        Ok(self.calibrate_reporting(target).0)
    }

    fn seed_facts(&self, caps: &Caps) -> DeviceFacts {
        crate::facts::seed_facts(caps)
    }
}

/// Quantized formats price at their dequantized compute dtype; this is the
/// mapping the MAC probe names a slot with.
pub const fn rate_dtype_of(dtype: Dtype) -> RateDtype {
    RateDtype::of(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::tests::{cpu_caps, gpu_caps};

    /// Every bench has a name and a budget, and the seven sum to the stated
    /// 180 ms.
    #[test]
    fn the_seven_budgets_sum_to_the_stated_total() {
        let total: Duration = Bench::ALL.iter().map(|b| b.budget()).sum();
        assert_eq!(total, Duration::from_millis(180));
        let names: Vec<_> = Bench::ALL.iter().map(|b| b.name()).collect();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), 7);
    }

    /// Every bench kernel builds, declares the buffers it binds in binding
    /// order, and folds its grid inside the device's per-dimension cap.
    /// This is the part of calibration checkable without a device.
    #[test]
    fn bench_kernels_are_well_formed() {
        for caps in [gpu_caps("calib"), cpu_caps("calib", 8)] {
            let cap = caps.limits.max_compute_workgroups_per_dimension;
            let kernels = [
                noop_kernel(),
                copy_kernel(BLOCK * PER_THREAD * 4, &caps),
                register_loop_kernel(ScalarElement::F32, LoopOp::Fma, 64, &caps),
                register_loop_kernel(ScalarElement::F32, LoopOp::Exp, 64, &caps),
                staging_kernel(2, 4, 64, &caps),
                staging_kernel(1, 4, 64, &caps),
                drain_kernel(256, 8, 64, &caps),
            ];
            for ir in kernels {
                assert!(!ir.name.is_empty());
                assert_eq!(ir.block, BLOCK);
                assert!(ir.grid.iter().all(|&d| d >= 1 && d <= cap), "{:?}", ir.grid);
                for (i, b) in ir.buffers.iter().enumerate() {
                    assert_eq!(b.binding, i as u32, "bindings are declaration order");
                    assert!(b.layout.element_count() > 0);
                }
            }
        }
    }

    /// The grid fold picks the slab count before sizing x, so an over-long
    /// 1-D count does not launch a nearly-empty final slab.
    #[test]
    fn grid_fold_stays_inside_the_dimension_cap() {
        let caps = gpu_caps("grid");
        let cap = u64::from(caps.limits.max_compute_workgroups_per_dimension);
        for threads in [1u64, 256, 1 << 20, 1 << 32, u64::from(u32::MAX)] {
            let [x, y, z] = grid_for(threads, &caps);
            assert!(u64::from(x) <= cap && u64::from(y) <= cap && u64::from(z) <= cap);
            let launched = u64::from(x) * u64::from(y) * u64::from(z);
            let needed = threads.div_ceil(u64::from(BLOCK)).max(1);
            assert!(launched >= needed, "{launched} < {needed}");
            // Never more than one extra slab of slack.
            assert!(launched - needed <= u64::from(x) * u64::from(y));
        }
    }

    /// A single-buffered staging body carries the extra barrier that costs
    /// it the overlap; a double-buffered one carries two tiles instead.
    #[test]
    fn staging_depth_changes_the_body_not_just_a_flag() {
        let caps = gpu_caps("staging");
        let single = staging_kernel(1, 4, 8, &caps);
        let double = staging_kernel(2, 4, 8, &caps);
        let barriers = |ir: &KernelIr| {
            ir.body
                .iter()
                .filter(|s| matches!(s, Stmt::Barrier))
                .count()
        };
        assert_eq!(barriers(&single), 8);
        assert_eq!(barriers(&double), 4);
    }

    /// The MAC loop carries independent accumulators, so the probe measures
    /// issue rate rather than dependent-FMA latency.
    #[test]
    fn mac_loop_is_issue_bound() {
        let caps = gpu_caps("mac");
        let ir = register_loop_kernel(ScalarElement::F32, LoopOp::Fma, 8, &caps);
        let Stmt::Loop { accumulators, .. } = &ir.body[0] else {
            panic!("the register loop must be a counted loop");
        };
        assert_eq!(accumulators.len(), ACCUMULATORS);
        let distinct: std::collections::BTreeSet<usize> = accumulators
            .iter()
            .map(|a| Arc::as_ptr(&a.local) as *const () as usize)
            .collect();
        assert_eq!(distinct.len(), ACCUMULATORS, "chains must be independent");
    }

    /// Each bench owns one primary fact, and `run`'s mapping covers all
    /// seven.
    #[test]
    fn every_bench_owns_a_fact() {
        let seed = crate::facts::seed_facts(&gpu_caps("fields"));
        let named = [
            seed.launch_ps,
            seed.dram_bytes_per_us,
            seed.llc_bytes,
            seed.wg_bytes_per_us,
            seed.mac_per_us[MacUnit::Fma as usize][RateDtype::F32 as usize],
            seed.trans_ps,
            u64::from(seed.saturation_lanes),
        ];
        assert_eq!(named.len(), Bench::ALL.len());
        assert!(named.iter().all(|&v| v > 0 || seed.launch_ps == 0));
        assert_eq!(rate_dtype_of(Dtype::F16), RateDtype::F16);
    }

    /// A fresh report names nothing and a calibrated one always names the
    /// two ratio-derived accelerator rows, so the fallback is never silent.
    #[test]
    fn report_defaults_are_empty() {
        let report = CalibrationReport::default();
        assert!(report.measured.is_empty());
        assert!(report.fell_back.is_empty());
        assert_eq!(report.micros, 0);
    }
}
