//! `KernelIr` -> a runnable CPU loop nest.
//!
//! `KernelIr` is **compiled**, not interpreted per element. [`compile`] lowers
//! one kernel into a [`Program`]: a flat SSA `tape` plus a list of segments.
//! Flattening the hash-consed `TileExpr` DAG into the tape is the CSE.
//!
//! One grid point is one workgroup. `block` lanes are walked in chunks of `W`,
//! and `Stmt::Barrier` lowers to a **segment split** (see
//! [`stmt::block`]) so a lane chunk can never read a tile slot a later chunk
//! has not written.

pub mod access;
pub mod expr;
pub mod quantized;
pub mod reduce;
pub mod stmt;

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::ir::kernel::{
    Addr, ArenaPlanner, Builtin, BufferDecl, ElementType, KernelIr, LocalDecl, ScalarElement,
    Source, Stmt, Tile, TileExpr, TileExprKind, TileLiteral, TileReduceOp,
};
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::MultiFlattenMap;
use fusor2_ir::target::{Buf, EmitError, Uniforms};
use fusor2_ir::Result;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use access::AccessForm;
use expr::{Instr, NumTy, RKind, Reg, Slot, UniformSrc};
use stmt::{CAcc, CStmt, LaneLoop};

/// One workgroup tile's placement in the thread-local scratch arena.
#[derive(Clone, Debug)]
pub struct TileInfo {
    pub elem: ScalarElement,
    pub elements: u32,
    pub byte_offset: u32,
    pub extents: [u32; 2],
}

/// A compiled kernel body.
#[derive(Debug)]
pub struct Program {
    /// The SSA tape: one instruction per distinct expression node, referenced
    /// by half-open ranges from the statements.
    pub tape: Vec<Instr>,
    /// Top-level segments, in order. A `Barrier` between two statements is a
    /// boundary here, not a runtime no-op.
    pub segments: Vec<LaneLoop>,
    pub regs: usize,
    pub locals: usize,
    pub tiles: Vec<TileInfo>,
    pub maps: Vec<MultiFlattenMap>,
    pub buffer_elements: Vec<ScalarElement>,
    pub arena_bytes: u32,
    pub block: u32,
    pub width: u32,
    /// Set when the body contains `Stmt::AtomicAdd`; the launcher then runs on
    /// one worker so the accumulation stays deterministic.
    pub has_atomic: bool,
}

impl Program {
    /// Total `Store` statements at every depth. Used by
    /// `matmul_epilogue_fuses_in_k_loop` to assert nothing is materialized in
    /// between.
    pub fn store_count(&self) -> usize {
        fn go(s: &[CStmt]) -> usize {
            s.iter()
                .map(|s| match s {
                    CStmt::Store { .. } => 1,
                    CStmt::If { accept, reject, .. } => go(accept) + go(reject),
                    CStmt::Loop { body, .. } => go(body),
                    CStmt::Lanes(b) => go(b),
                    _ => 0,
                })
                .sum()
        }
        self.segments.iter().map(|l| go(&l.stmts)).sum()
    }

    /// Fused multiply-adds on the tape. Zero under `NumericContract::STRICT`.
    pub fn fma_count(&self) -> usize {
        self.tape.iter().filter(|i| i.is_fma()).count()
    }
}

/// A compiled CPU kernel: the program plus the geometry it was built for.
#[derive(Clone, Debug)]
pub struct CpuArtifact {
    pub prog: Arc<Program>,
    pub grid: [u32; 3],
    pub block: u32,
    pub name: &'static str,
    pub arena_bytes: u32,
}

/// The public handle the `Target` hands back.
#[derive(Clone, Debug)]
pub struct CpuKernel {
    pub name: &'static str,
    pub block: u32,
    pub vector_width: u32,
    pub artifact: CpuArtifact,
}

impl CpuKernel {
    /// Run one dispatch. `binds` is positional, exactly as on GPU.
    pub fn run(&self, grid: [u32; 3], binds: &[Buf], uniforms: &Uniforms) -> Result<()> {
        crate::launch::run(self, grid, binds, uniforms)
    }
}

/// Emit a kernel for this device.
pub fn emit(ir: &KernelIr, caps: &Caps) -> Result<CpuKernel, EmitError> {
    let artifact = compile(ir, caps, None)?;
    Ok(CpuKernel {
        name: artifact.name,
        block: artifact.block,
        vector_width: artifact.prog.width,
        artifact,
    })
}

/// Compile one `KernelIr`.
///
/// When a planner is supplied its `verify_uniformity` and `verify_arena` run
/// **before** compilation and a failure is `EmitError::Validation`, never a
/// silent fallback. Without one the arena falls back to a sequential packing,
/// which is always legal on CPU because thread-local scratch aliases freely.
pub fn compile(
    ir: &KernelIr,
    caps: &Caps,
    planner: Option<&dyn ArenaPlanner>,
) -> std::result::Result<CpuArtifact, EmitError> {
    let width = pick_width(caps, ir.block);

    let arena = match planner {
        Some(p) => {
            p.verify_uniformity(ir)
                .map_err(|e| EmitError::Validation(e.to_string()))?;
            let plan = p
                .arena_plan(ir, caps)
                .map_err(|e| EmitError::Validation(e.to_string()))?;
            p.verify_arena(ir, &plan)
                .map_err(|e| EmitError::Validation(e.to_string()))?;
            Some(plan)
        }
        None => None,
    };

    let mut c = Compiler::new(ir);
    if let Some(plan) = &arena {
        c.seed_arena(plan);
    }
    let body = c.compile_stmts(&ir.body)?;
    let segments = stmt::block(&body, ir.block.max(1), width)?;

    let prog = Program {
        tape: c.tape,
        segments,
        regs: c.regs as usize,
        locals: c.locals.len(),
        tiles: c.tiles,
        maps: c.maps,
        buffer_elements: ir
            .buffers
            .iter()
            .map(|b| scalar_of(b.element))
            .collect::<std::result::Result<_, _>>()?,
        arena_bytes: c.arena_bytes,
        block: ir.block.max(1),
        width,
        has_atomic: c.has_atomic,
    };

    Ok(CpuArtifact {
        prog: Arc::new(prog),
        grid: ir.grid,
        block: ir.block.max(1),
        name: ir.name,
        arena_bytes: prog_arena(&arena, c.arena_bytes),
    })
}

fn prog_arena(plan: &Option<fusor2_ir::ir::kernel::ArenaPlan>, fallback: u32) -> u32 {
    plan.as_ref().map_or(fallback, |p| p.total_bytes.max(fallback))
}

/// The widest legal instantiation that still divides the work sensibly.
fn pick_width(caps: &Caps, block: u32) -> u32 {
    let mut best = 4;
    for w in caps.simd_widths.iter().copied() {
        if crate::caps::WIDTHS.contains(&w) && w <= block.max(1) {
            best = best.max(w);
        }
    }
    best
}

fn scalar_of(e: ElementType) -> std::result::Result<ScalarElement, EmitError> {
    match e {
        ElementType::Scalar(s) => Ok(s),
        other => Err(EmitError::Unsupported(format!(
            "buffer element {other:?} is not a scalar"
        ))),
    }
}

struct Compiler<'a> {
    ir: &'a KernelIr,
    tape: Vec<Instr>,
    regs: u32,
    memo: FxHashMap<TileExpr, Slot>,
    /// Workgroup reduces already staged, mapped to the tile read that replaces
    /// them.
    redirect: FxHashMap<TileExpr, TileExpr>,
    tiles: Vec<TileInfo>,
    tile_index: FxHashMap<usize, u16>,
    locals: Vec<Arc<LocalDecl>>,
    local_index: FxHashMap<usize, u16>,
    maps: Vec<MultiFlattenMap>,
    arena_bytes: u32,
    has_atomic: bool,
    /// Collective statements hoisted in front of the statement being compiled.
    pre: Vec<CStmt>,
}

impl<'a> Compiler<'a> {
    fn new(ir: &'a KernelIr) -> Self {
        Self {
            ir,
            tape: Vec::new(),
            regs: 0,
            memo: FxHashMap::default(),
            redirect: FxHashMap::default(),
            tiles: Vec::new(),
            tile_index: FxHashMap::default(),
            locals: Vec::new(),
            local_index: FxHashMap::default(),
            maps: Vec::new(),
            arena_bytes: 0,
            has_atomic: false,
            pre: Vec::new(),
        }
    }

    /// Adopt the planner's placements, so the CPU arena and the Launch occupancy
    /// term read the same `arena_plan` value.
    fn seed_arena(&mut self, plan: &fusor2_ir::ir::kernel::ArenaPlan) {
        for p in &plan.placements {
            let key = Arc::as_ptr(&p.tile) as usize;
            if self.tile_index.contains_key(&key) {
                continue;
            }
            let idx = self.tiles.len() as u16;
            self.tiles.push(TileInfo {
                elem: match p.tile.element {
                    ElementType::Scalar(s) => s,
                    _ => ScalarElement::F32,
                },
                elements: p.tile.layout.element_count() as u32,
                byte_offset: p.byte_offset,
                extents: extents2(&p.tile),
            });
            self.tile_index.insert(key, idx);
            self.arena_bytes = self.arena_bytes.max(p.byte_offset + p.byte_len);
        }
    }

    fn slot(&mut self) -> Slot {
        let s = self.regs;
        self.regs += 1;
        s
    }

    fn slots(&mut self, n: u32) -> Slot {
        let s = self.regs;
        self.regs += n;
        s
    }

    fn begin(&mut self) -> u32 {
        self.memo.clear();
        self.tape.len() as u32
    }

    fn end(&self) -> u32 {
        self.tape.len() as u32
    }

    fn buffer_of(&self, b: &Arc<BufferDecl>) -> std::result::Result<u16, EmitError> {
        self.ir
            .buffers
            .iter()
            .position(|x| Arc::ptr_eq(x, b))
            .map(|i| i as u16)
            .ok_or_else(|| {
                EmitError::Validation(format!(
                    "storage view references buffer @{} which is not declared",
                    b.binding
                ))
            })
    }

    fn tile_of(&mut self, t: &Tile) -> u16 {
        let key = Arc::as_ptr(t) as usize;
        if let Some(&i) = self.tile_index.get(&key) {
            return i;
        }
        let elements = t.layout.element_count() as u32;
        let elem = match t.element {
            ElementType::Scalar(s) => s,
            _ => ScalarElement::F32,
        };
        // Sequential packing: legal on CPU because thread-local scratch
        // aliases freely.
        let offset = align_up(self.arena_bytes, 64);
        let len = elements * elem.byte_size() as u32;
        self.arena_bytes = offset + len;
        let idx = self.tiles.len() as u16;
        self.tiles.push(TileInfo {
            elem,
            elements,
            byte_offset: offset,
            extents: extents2(t),
        });
        self.tile_index.insert(key, idx);
        idx
    }

    fn local_of(&mut self, l: &Arc<LocalDecl>) -> u16 {
        let key = Arc::as_ptr(l) as usize;
        if let Some(&i) = self.local_index.get(&key) {
            return i;
        }
        let idx = self.locals.len() as u16;
        self.locals.push(Arc::clone(l));
        self.local_index.insert(key, idx);
        idx
    }

    fn map_of(&mut self, m: &MultiFlattenMap) -> u16 {
        if let Some(i) = self.maps.iter().position(|x| x == m) {
            return i as u16;
        }
        self.maps.push(m.clone());
        (self.maps.len() - 1) as u16
    }

    fn push(&mut self, i: Instr) -> Slot {
        let out = i.out();
        self.tape.push(i);
        out
    }

    fn konst(&mut self, bits: u32) -> Slot {
        let out = self.slot();
        self.push(Instr::Const { out, bits })
    }

    fn compile_stmts(&mut self, body: &[Stmt]) -> std::result::Result<Vec<CStmt>, EmitError> {
        let mut out = Vec::with_capacity(body.len());
        for stmt in body {
            let desugared = desugar_fast_reduce(stmt);
            let s = desugared.as_ref().unwrap_or(stmt);
            self.pre.clear();
            self.stage_reduces_in(s)?;
            let pre = std::mem::take(&mut self.pre);
            out.extend(pre);
            out.push(self.compile_stmt(s)?);
        }
        Ok(out)
    }

    fn compile_stmt(&mut self, s: &Stmt) -> std::result::Result<CStmt, EmitError> {
        Ok(match s {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            }
            | Stmt::AtomicAdd {
                dst,
                addr,
                value,
                mask,
            } => {
                let buf = self.buffer_of(&dst.buffer)?;
                let elem = scalar_of(dst.buffer.element)?;
                let prep = self.begin();
                let index = self.compile_addr(&dst.layout, dst.offset, addr)?;
                let v = self.compile_expr(value)?;
                let v = self.coerce_store(v, value, elem)?;
                let m = self.compile_mask(mask)?;
                let prep = prep..self.end();
                if matches!(s, Stmt::AtomicAdd { .. }) {
                    self.has_atomic = true;
                    CStmt::AtomicAdd {
                        prep,
                        buf,
                        elem,
                        index,
                        value: v,
                        mask: m,
                    }
                } else {
                    CStmt::Store {
                        prep,
                        buf,
                        elem,
                        index,
                        value: v,
                        mask: m,
                    }
                }
            }
            Stmt::StoreLocal { dst, value } => {
                let local = self.local_of(dst);
                let prep = self.begin();
                let v = self.compile_expr(value)?;
                CStmt::StoreLocal {
                    prep: prep..self.end(),
                    local,
                    value: v,
                }
            }
            Stmt::StoreTile { dst, index, value } => {
                let tile = self.tile_of(dst);
                let elem = self.tiles[tile as usize].elem;
                let prep = self.begin();
                let i = self.compile_expr(index)?;
                let v = self.compile_expr(value)?;
                CStmt::StoreTile {
                    prep: prep..self.end(),
                    tile,
                    elem,
                    index: i,
                    value: v,
                }
            }
            Stmt::FillTile { dst, value, bounds } => {
                let tile = self.tile_of(dst);
                let info = self.tiles[tile as usize].clone();
                let prep = self.begin();
                let v = self.compile_expr(value)?;
                let lo = match &bounds[0] {
                    Some(e) => Some(self.compile_expr(e)?),
                    None => None,
                };
                let hi = match &bounds[1] {
                    Some(e) => Some(self.compile_expr(e)?),
                    None => None,
                };
                CStmt::FillTile {
                    prep: prep..self.end(),
                    tile,
                    elem: info.elem,
                    value: v,
                    extents: info.extents,
                    lo,
                    hi,
                }
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                let uniform = access::is_lane_uniform(condition);
                let prep = self.begin();
                let c = self.compile_mask(condition)?;
                let prep = prep..self.end();
                let a = self.compile_stmts(accept)?;
                let r = self.compile_stmts(reject)?;
                CStmt::If {
                    prep,
                    cond: c,
                    uniform,
                    accept: a,
                    reject: r,
                }
            }
            Stmt::Loop {
                count,
                index,
                accumulators,
                body,
            } => {
                let idx = index.as_ref().map(|l| self.local_of(l));
                let prep = self.begin();
                let cnt = match count {
                    Some(e) => Some(self.compile_expr(e)?),
                    None => None,
                };
                let mut accs = Vec::with_capacity(accumulators.len());
                for a in accumulators {
                    let local = self.local_of(&a.local);
                    let s = self.tape.len() as u32;
                    let init = self.compile_expr(&a.init)?;
                    accs.push((local, s..self.tape.len() as u32, init, a.clone()));
                }
                let prep = prep..self.end();
                let body = self.compile_stmts(body)?;
                // Updates run once per iteration, so each gets its own range.
                let mut cacc = Vec::with_capacity(accs.len());
                for (local, init_prep, init, a) in accs {
                    let s = self.begin();
                    let update = self.compile_expr(&a.update)?;
                    cacc.push(CAcc {
                        local,
                        init_prep,
                        init,
                        update_prep: s..self.end(),
                        update,
                    });
                }
                CStmt::Loop {
                    prep,
                    count: cnt,
                    index: idx,
                    accs: cacc,
                    body,
                }
            }
            // The N-ary reduction. A one-lane node with a hardware operator has
            // already been rewritten into the expression form by
            // [`desugar_fast_reduce`], so this is the general merge.
            Stmt::Reduce {
                kind,
                values,
                merge,
                fast,
                outs,
                scratch,
            } => {
                let group = match &**kind {
                    fusor2_ir::ir::kernel::ReduceKind::Workgroup { group_size, .. } => *group_size,
                    fusor2_ir::ir::kernel::ReduceKind::Subgroup => {
                        return Err(EmitError::Unsupported(
                            "a multi-lane merge has no horizontal-reduce form: the SIMD \
                             butterfly folds one register with one operator"
                                .into(),
                        ));
                    }
                    fusor2_ir::ir::kernel::ReduceKind::Loop { .. } => {
                        return Err(EmitError::Unsupported(
                            "a multi-lane loop reduction seeds from the carrier's identities, \
                             which the lowering carries: build the per-lane loop with Stmt::Loop \
                             and close with ReduceKind::Workgroup"
                                .into(),
                        ));
                    }
                };
                let block = self.ir.block.max(1);
                if group == 0 || !group.is_power_of_two() || group > block || !block.is_multiple_of(group) {
                    return Err(EmitError::Unsupported(format!(
                        "tree reduce needs a power-of-two group size dividing the block, got \
                         {group} with block {block}"
                    )));
                }
                let tiles: Vec<u16> = scratch.iter().map(|t| self.tile_of(t)).collect();
                let lhs: Vec<u16> = merge.lhs.iter().map(|l| self.local_of(l)).collect();
                let rhs: Vec<u16> = merge.rhs.iter().map(|l| self.local_of(l)).collect();
                let outs: Vec<u16> = outs.iter().map(|l| self.local_of(l)).collect();
                let prep = self.begin();
                let values: Vec<Slot> = values
                    .iter()
                    .map(|v| self.compile_expr(v))
                    .collect::<std::result::Result<_, _>>()?;
                let prep = prep..self.end();
                let merge_prep = self.begin();
                let merged: Vec<Slot> = merge
                    .body
                    .iter()
                    .map(|v| self.compile_expr(v))
                    .collect::<std::result::Result<_, _>>()?;
                let merge_prep = merge_prep..self.end();
                CStmt::CarrierTree {
                    prep,
                    tiles,
                    values,
                    lhs,
                    rhs,
                    merge_prep,
                    merged,
                    outs,
                    group,
                    fast: *fast,
                }
            }
            Stmt::Break => CStmt::Break,
            Stmt::Return => CStmt::Return,
            Stmt::Barrier | Stmt::StorageBarrier => CStmt::Barrier,
            Stmt::CoopStore { .. } | Stmt::CoopStoreTile { .. } => {
                return Err(EmitError::MissingCapability(
                    "cooperative matrix: the CPU target reports no coop config",
                ));
            }
        })
    }

    /// Hoist every `Reduce{Workgroup|Loop}` reachable from a statement into a
    /// collective staging pass in front of it, and record the tile read that
    /// replaces it.
    fn stage_reduces_in(&mut self, s: &Stmt) -> std::result::Result<(), EmitError> {
        let mut found = Vec::new();
        let mut seen: rustc_hash::FxHashSet<TileExpr> = rustc_hash::FxHashSet::default();
        visit_stmt_exprs(s, &mut |e| collect_group_reduces(e, &mut found, &mut seen));
        for e in found {
            if self.redirect.contains_key(&e) {
                continue;
            }
            self.stage_one_reduce(&e)?;
        }
        Ok(())
    }

    fn stage_one_reduce(&mut self, e: &TileExpr) -> std::result::Result<(), EmitError> {
        let TileExprKind::Reduce { op, kind, value } = e.kind() else {
            return Ok(());
        };
        let (tile, group, iterations, index) = match &**kind {
            fusor2_ir::ir::kernel::ReduceKind::Workgroup {
                scratch,
                group_size,
            } => (self.tile_of(scratch), *group_size, None, None),
            fusor2_ir::ir::kernel::ReduceKind::Loop {
                iterations,
                index,
                scratch,
                group_size,
            } => {
                let li = self.local_of(index);
                (
                    self.tile_of(scratch),
                    *group_size,
                    Some(*iterations),
                    Some(li),
                )
            }
            fusor2_ir::ir::kernel::ReduceKind::Subgroup => return Ok(()),
        };
        let start = self.begin();
        let v = self.compile_expr(value)?;
        let prep = start..self.end();
        let group = group.max(1);
        self.pre.push(match (iterations, index) {
            (Some(iterations), Some(index)) => CStmt::LoopTree {
                prep,
                tile,
                value: v,
                op: *op,
                group,
                iterations,
                index,
            },
            _ => CStmt::StageTree {
                prep,
                tile,
                value: v,
                op: *op,
                group,
            },
        });
        // The replacement read: `tile[(lane / group) * group]`.
        let u32_ty = ElementType::Scalar(ScalarElement::U32);
        let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty);
        let g = TileExpr::new(TileExprKind::Literal(TileLiteral::U32(group)), u32_ty);
        let base = tile_bin(BinOp::Mul, tile_bin(BinOp::Div, lane, g.clone(), u32_ty), g, u32_ty);
        let tile_arc = Arc::clone(match &**kind {
            fusor2_ir::ir::kernel::ReduceKind::Workgroup { scratch, .. }
            | fusor2_ir::ir::kernel::ReduceKind::Loop { scratch, .. } => scratch,
            fusor2_ir::ir::kernel::ReduceKind::Subgroup => unreachable!(),
        });
        let read = TileExpr::new(
            TileExprKind::LoadTile {
                tile: tile_arc,
                index: base,
            },
            e.element(),
        );
        self.redirect.insert(e.clone(), read);
        Ok(())
    }

    fn compile_addr(
        &mut self,
        layout: &fusor2_ir::ir::kernel::TileLayout,
        offset: u32,
        addr: &Addr,
    ) -> std::result::Result<Slot, EmitError> {
        let base = match addr {
            Addr::Linear(e) => self.compile_expr(e)?,
            Addr::Rc2 { row, col } => {
                // A rank-2 address runs both coordinates through the declared
                // divmod chain; nothing else does.
                let r = self.compile_expr(row)?;
                let c = self.compile_expr(col)?;
                let map = self.map_of(&layout.indexing);
                let out = self.slot();
                self.push(Instr::Rc2Index {
                    out,
                    row: r,
                    col: c,
                    map,
                })
            }
        };
        if offset == 0 {
            return Ok(base);
        }
        let off = self.konst(offset);
        let out = self.slot();
        Ok(self.push(Instr::Bin {
            out,
            op: BinOp::Add,
            a: base,
            b: off,
            ty: NumTy::U32,
        }))
    }

    /// Compile an expression consumed as a lane mask.
    fn compile_mask(&mut self, e: &TileExpr) -> std::result::Result<Slot, EmitError> {
        if e.is_constant_true() {
            let out = self.slot();
            return Ok(self.push(Instr::Const {
                out,
                bits: u32::MAX,
            }));
        }
        let s = self.compile_expr(e)?;
        if produces_mask(e) {
            return Ok(s);
        }
        let ty = num_ty(e.element());
        let out = self.slot();
        Ok(self.push(Instr::ValueToMask { out, x: s, ty }))
    }

    /// Compile a value, materializing a mask to 1/0 when the consumer wants a
    /// number.
    fn compile_value(&mut self, e: &TileExpr) -> std::result::Result<Slot, EmitError> {
        let s = self.compile_expr(e)?;
        if !produces_mask(e) {
            return Ok(s);
        }
        let ty = num_ty(e.element());
        let out = self.slot();
        Ok(self.push(Instr::MaskToValue { out, x: s, ty }))
    }

    fn coerce_store(
        &mut self,
        s: Slot,
        e: &TileExpr,
        _elem: ScalarElement,
    ) -> std::result::Result<Slot, EmitError> {
        if !produces_mask(e) {
            return Ok(s);
        }
        let ty = num_ty(e.element());
        let out = self.slot();
        Ok(self.push(Instr::MaskToValue { out, x: s, ty }))
    }

    fn compile_expr(&mut self, e: &TileExpr) -> std::result::Result<Slot, EmitError> {
        if let Some(rep) = self.redirect.get(e).cloned() {
            return self.compile_expr(&rep);
        }
        if let Some(&s) = self.memo.get(e) {
            return Ok(s);
        }
        let s = self.compile_expr_uncached(e)?;
        self.memo.insert(e.clone(), s);
        Ok(s)
    }

    fn compile_expr_uncached(&mut self, e: &TileExpr) -> std::result::Result<Slot, EmitError> {
        use TileExprKind as K;
        let ty = num_ty(e.element());
        Ok(match e.kind() {
            K::Literal(l) => {
                let bits = match l {
                    TileLiteral::F32(b) => *b,
                    TileLiteral::F16(b) => half::f16::from_bits(*b).to_f32().to_bits(),
                    TileLiteral::BF16(b) => half::bf16::from_bits(*b).to_f32().to_bits(),
                    TileLiteral::U32(v) => *v,
                    TileLiteral::I32(v) => *v as u32,
                    TileLiteral::Bool(b) => {
                        if *b {
                            u32::MAX
                        } else {
                            0
                        }
                    }
                };
                self.konst(bits)
            }
            K::Builtin(b) => {
                let out = self.slot();
                match b {
                    Builtin::Lane => self.push(Instr::LaneId { out }),
                    other => {
                        let which = match other {
                            Builtin::ProgramId(fusor2_ir::ir::kernel::WorkgroupAxis::X) => {
                                UniformSrc::ProgramX
                            }
                            Builtin::ProgramId(fusor2_ir::ir::kernel::WorkgroupAxis::Y) => {
                                UniformSrc::ProgramY
                            }
                            Builtin::ProgramId(fusor2_ir::ir::kernel::WorkgroupAxis::Z) => {
                                UniformSrc::ProgramZ
                            }
                            Builtin::NumWorkgroups(fusor2_ir::ir::kernel::WorkgroupAxis::X) => {
                                UniformSrc::GridX
                            }
                            Builtin::NumWorkgroups(fusor2_ir::ir::kernel::WorkgroupAxis::Y) => {
                                UniformSrc::GridY
                            }
                            Builtin::NumWorkgroups(fusor2_ir::ir::kernel::WorkgroupAxis::Z) => {
                                UniformSrc::GridZ
                            }
                            Builtin::SubgroupId => UniformSrc::SubgroupId,
                            Builtin::SubgroupLane => UniformSrc::SubgroupLane,
                            Builtin::SubgroupSize => UniformSrc::SubgroupSize,
                            Builtin::NumSubgroups => UniformSrc::NumSubgroups,
                            Builtin::Lane => unreachable!(),
                        };
                        self.push(Instr::Uniform { out, which })
                    }
                }
            }
            K::LoadLocal(l) => {
                let local = self.local_of(l);
                let out = self.slot();
                self.push(Instr::LoadLocal { out, local })
            }
            K::Load {
                src,
                addr,
                mask,
                fill,
            } => match src {
                Source::Storage(view) => {
                    let buf = self.buffer_of(&view.buffer)?;
                    let elem = scalar_of(view.buffer.element)?;
                    let form = access::form_of(&view.layout, addr);
                    access::note_form(form);
                    let index = self.compile_addr(&view.layout, view.offset, addr)?;
                    let m = self.compile_mask(mask)?;
                    let f = self.compile_value(fill)?;
                    let out = self.slot();
                    self.push(Instr::Load {
                        out,
                        buf,
                        elem,
                        index,
                        mask: m,
                        fill: f,
                        form,
                    })
                }
                Source::Quantized(q) => {
                    // A quantized load is one lane of a decoded block; the
                    // decode program supplies the expression.
                    let (k_base, col) = match &**addr {
                        Addr::Rc2 { row, col } => (row.clone(), col.clone()),
                        Addr::Linear(e) => (e.clone(), zero_u32()),
                    };
                    let element = quantized::expand_dequantize(q, &k_base, &col, mask, fill)?;
                    return self.compile_expr(&element);
                }
            },
            K::LoadTile { tile, index } => {
                let t = self.tile_of(tile);
                let elem = self.tiles[t as usize].elem;
                let i = self.compile_expr(index)?;
                let out = self.slot();
                self.push(Instr::LoadTile {
                    out,
                    tile: t,
                    elem,
                    index: i,
                })
            }
            K::Unary { op, value, numeric } => {
                let _ = numeric;
                if *op == fusor2_ir::scalar::UnOp::Unpack2x16Float {
                    let x = self.compile_value(value)?;
                    let out = self.slots(2);
                    self.push(Instr::Unpack2x16 { out, x })
                } else {
                    let x = self.compile_value(value)?;
                    let out = self.slot();
                    self.push(Instr::Un {
                        out,
                        op: *op,
                        x,
                        ty: num_ty(value.element()),
                    })
                }
            }
            K::Binary {
                op,
                left,
                right,
                numeric,
            } => {
                // `contract: false` forbids fusing a mul+add into a mul_add.
                if *op == BinOp::Add && numeric.contract && ty == NumTy::F32 {
                    if let Some((a, b)) = mul_operands(left, numeric) {
                        let sa = self.compile_value(&a)?;
                        let sb = self.compile_value(&b)?;
                        let sc = self.compile_value(right)?;
                        let out = self.slot();
                        return Ok(self.push(Instr::Fma {
                            out,
                            a: sa,
                            b: sb,
                            c: sc,
                        }));
                    }
                    if let Some((a, b)) = mul_operands(right, numeric) {
                        let sa = self.compile_value(&a)?;
                        let sb = self.compile_value(&b)?;
                        let sc = self.compile_value(left)?;
                        let out = self.slot();
                        return Ok(self.push(Instr::Fma {
                            out,
                            a: sa,
                            b: sb,
                            c: sc,
                        }));
                    }
                }
                let a = self.compile_value(left)?;
                let b = self.compile_value(right)?;
                let out = self.slot();
                self.push(Instr::Bin {
                    out,
                    op: *op,
                    a,
                    b,
                    ty: num_ty(left.element()),
                })
            }
            K::Compare { op, left, right } => {
                let a = self.compile_value(left)?;
                let b = self.compile_value(right)?;
                let out = self.slot();
                self.push(Instr::Cmp {
                    out,
                    op: *op,
                    a,
                    b,
                    ty: num_ty(left.element()),
                })
            }
            K::Round { mode, value } => {
                let x = self.compile_value(value)?;
                let out = self.slot();
                self.push(Instr::Round {
                    out,
                    mode: *mode,
                    x,
                })
            }
            K::Cast { value, to } => {
                let x = self.compile_value(value)?;
                let from = num_ty(value.element());
                let to_s = scalar_of(*to)?;
                let out = self.slot();
                let widened = self.push(Instr::Cast {
                    out,
                    x,
                    from,
                    to: NumTy::of(to_s),
                });
                if matches!(to_s, ScalarElement::F16 | ScalarElement::BF16) {
                    let out = self.slot();
                    self.push(Instr::Narrow {
                        out,
                        x: widened,
                        to: to_s,
                    })
                } else {
                    widened
                }
            }
            K::Bitcast { value, .. } => {
                let x = self.compile_value(value)?;
                let out = self.slot();
                self.push(Instr::Bitcast { out, x })
            }
            K::Select {
                condition,
                accept,
                reject,
            } => {
                let c = self.compile_mask(condition)?;
                let t = self.compile_value(accept)?;
                let f = self.compile_value(reject)?;
                // A vector occupies `lanes` consecutive registers, so a
                // vector-typed select is `lanes` selects.
                match e.element() {
                    ElementType::Vector { lanes, .. } if lanes > 1 => {
                        let out = self.slots(lanes);
                        for i in 0..lanes {
                            self.push(Instr::Select {
                                out: out + i,
                                c,
                                t: t + i,
                                f: f + i,
                            });
                        }
                        out
                    }
                    _ => {
                        let out = self.slot();
                        self.push(Instr::Select { out, c, t, f })
                    }
                }
            }
            K::Vec { lanes, parts, .. } => {
                let mut ps = Vec::with_capacity(parts.len());
                for p in parts {
                    ps.push(self.compile_value(p)?);
                }
                let out = self.slots(*lanes);
                self.push(Instr::VecCompose { out, parts: ps })
            }
            K::VecComponent { vector, component } => {
                let base = self.compile_expr(vector)?;
                let out = self.slot();
                self.push(Instr::VecComponent {
                    out,
                    base,
                    component: *component,
                })
            }
            K::Dot { left, right } => {
                let a = self.compile_expr(left)?;
                let b = self.compile_expr(right)?;
                let lanes = match left.element() {
                    ElementType::Vector { lanes, .. } => lanes,
                    _ => 1,
                };
                let out = self.slot();
                self.push(Instr::Dot { out, a, b, lanes })
            }
            K::Reduce { op, kind, value } => match &**kind {
                fusor2_ir::ir::kernel::ReduceKind::Subgroup => {
                    let x = self.compile_value(value)?;
                    let out = self.slot();
                    self.push(Instr::Reduce {
                        out,
                        op: *op,
                        x,
                        kind: RKind::Subgroup,
                        group_base: 0,
                    })
                }
                _ => {
                    return Err(EmitError::Validation(
                        "a workgroup reduce was not staged before its consumer".into(),
                    ));
                }
            },
            K::CoopLoad { .. } | K::CoopMma { .. } | K::CoopZero { .. } => {
                return Err(EmitError::MissingCapability(
                    "cooperative matrix: the CPU target reports no coop config",
                ));
            }
        })
    }
}

fn align_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

fn extents2(t: &Tile) -> [u32; 2] {
    let e = &t.layout.extents;
    match e.len() {
        0 => [1, 1],
        1 => [e[0], 1],
        _ => [e[0], e[1]],
    }
}

fn num_ty(e: ElementType) -> NumTy {
    match e {
        ElementType::Scalar(s) | ElementType::Vector { scalar: s, .. } => NumTy::of(s),
        ElementType::CoopMatrix { scalar, .. } => NumTy::of(scalar),
    }
}

/// A comparison node yields a lane mask rather than a value.
fn produces_mask(e: &TileExpr) -> bool {
    matches!(e.kind(), TileExprKind::Compare { .. })
        || matches!(
            e.kind(),
            TileExprKind::Literal(TileLiteral::Bool(_))
        )
}

fn zero_u32() -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::U32(0)),
        ElementType::Scalar(ScalarElement::U32),
    )
}

fn tile_bin(op: BinOp, a: TileExpr, b: TileExpr, ty: ElementType) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left: a,
            right: b,
            numeric: NumericContract::RELAXED,
        },
        ty,
    )
}

/// The operands of a `mul` node, when contraction into an fma is permitted by
/// *both* the enclosing add and the multiply itself.
fn mul_operands(e: &TileExpr, outer: &NumericContract) -> Option<(TileExpr, TileExpr)> {
    match e.kind() {
        TileExprKind::Binary {
            op: BinOp::Mul,
            left,
            right,
            numeric,
        } if numeric.contract && outer.contract => Some((left.clone(), right.clone())),
        _ => None,
    }
}

/// `seen` is required for termination: a Kernel term is a hash-consed **DAG**,
/// so walking it as a tree is exponential in the sharing depth.
fn collect_group_reduces(
    e: &TileExpr,
    out: &mut Vec<TileExpr>,
    seen: &mut rustc_hash::FxHashSet<TileExpr>,
) {
    if !seen.insert(e.clone()) {
        return;
    }
    // Children first: a reduce nested inside another reduce's value must be
    // staged before it.
    for c in children_of(e) {
        collect_group_reduces(&c, out, seen);
    }
    if let TileExprKind::Reduce { kind, .. } = e.kind() {
        if !matches!(&**kind, fusor2_ir::ir::kernel::ReduceKind::Subgroup)
            && !out.contains(e)
        {
            out.push(e.clone());
        }
    }
}

fn children_of(e: &TileExpr) -> Vec<TileExpr> {
    use TileExprKind as K;
    match e.kind() {
        K::Literal(_) | K::Builtin(_) | K::LoadLocal(_) => vec![],
        K::Load {
            addr, mask, fill, ..
        } => {
            let mut v = match &**addr {
                Addr::Linear(e) => vec![e.clone()],
                Addr::Rc2 { row, col } => vec![row.clone(), col.clone()],
            };
            v.push(mask.clone());
            v.push(fill.clone());
            v
        }
        K::LoadTile { index, .. } => vec![index.clone()],
        K::Unary { value, .. }
        | K::Round { value, .. }
        | K::Cast { value, .. }
        | K::Bitcast { value, .. } => vec![value.clone()],
        K::Binary { left, right, .. } | K::Compare { left, right, .. } | K::Dot { left, right } => {
            vec![left.clone(), right.clone()]
        }
        K::Select {
            condition,
            accept,
            reject,
        } => vec![condition.clone(), accept.clone(), reject.clone()],
        K::Vec { parts, .. } => parts.clone(),
        K::VecComponent { vector, .. } => vec![vector.clone()],
        K::Reduce { value, .. } => vec![value.clone()],
        K::CoopLoad { .. } | K::CoopMma { .. } | K::CoopZero { .. } => vec![],
    }
}

/// Rewrite a one-lane `Stmt::Reduce` carrying a hardware operator into the
/// expression form, so it goes down the same staging path. `None` when nothing
/// is to rewrite.
fn desugar_fast_reduce(s: &Stmt) -> Option<Stmt> {
    let Stmt::Reduce {
        kind,
        values,
        fast: Some(op),
        outs,
        ..
    } = s
    else {
        return None;
    };
    if values.len() != 1 {
        return None;
    }
    let value = values[0].clone();
    let element = value.element();
    Some(Stmt::StoreLocal {
        dst: outs[0].clone(),
        value: TileExpr::new(
            TileExprKind::Reduce {
                op: *op,
                kind: kind.clone(),
                value,
            },
            element,
        ),
    })
}

fn visit_stmt_exprs(s: &Stmt, f: &mut impl FnMut(&TileExpr)) {
    match s {
        Stmt::Store {
            addr, value, mask, ..
        }
        | Stmt::AtomicAdd {
            addr, value, mask, ..
        } => {
            match addr {
                Addr::Linear(e) => f(e),
                Addr::Rc2 { row, col } => {
                    f(row);
                    f(col);
                }
            }
            f(value);
            f(mask);
        }
        Stmt::StoreLocal { value, .. } => f(value),
        Stmt::StoreTile { index, value, .. } => {
            f(index);
            f(value);
        }
        Stmt::FillTile { value, bounds, .. } => {
            f(value);
            for b in bounds.iter().flatten() {
                f(b);
            }
        }
        Stmt::CoopStore { acc, addr, .. } => {
            f(acc);
            match addr {
                Addr::Linear(e) => f(e),
                Addr::Rc2 { row, col } => {
                    f(row);
                    f(col);
                }
            }
        }
        Stmt::CoopStoreTile { acc, row, col, .. } => {
            f(acc);
            f(row);
            f(col);
        }
        // Nested bodies get their own staging pass when they are compiled.
        Stmt::If { condition, .. } => f(condition),
        Stmt::Loop {
            count,
            accumulators,
            ..
        } => {
            if let Some(c) = count {
                f(c);
            }
            for a in accumulators {
                f(&a.init);
            }
        }
        // The merge reads only its formals, so nothing in it can need staging;
        // the per-lane partials can.
        Stmt::Reduce { values, .. } => {
            for v in values {
                f(v);
            }
        }
        Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
    }
}

/// A raw view of one bound buffer.
#[derive(Copy, Clone)]
pub struct RawBuf {
    pub ptr: *mut u8,
    pub bytes: usize,
}

// SAFETY: the launcher only hands a `RawBuf` to workers whose lane ranges write
// disjoint elements — `verify_launch` invariant 3.
unsafe impl Send for RawBuf {}
// SAFETY: as above.
unsafe impl Sync for RawBuf {}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Return,
}

struct Exec<'a, const W: usize> {
    prog: &'a Program,
    regs: Vec<Reg<W>>,
    locals: Vec<u32>,
    bufs: &'a [RawBuf],
    scratch: *mut u8,
    gid: [u32; 3],
    grid: [u32; 3],
    lane_base: u32,
    active: Reg<W>,
}

impl<'a, const W: usize> Exec<'a, W> {
    fn tile_f32(&self, tile: u16) -> &mut [f32] {
        let info = &self.prog.tiles[tile as usize];
        // SAFETY: the arena is sized to `arena_bytes`, which covers every
        // placement, and each worker owns its own scratch.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.scratch.add(info.byte_offset as usize) as *mut f32,
                info.elements as usize,
            )
        }
    }

    fn tile_ptr(&self, tile: u16) -> *mut u8 {
        let info = &self.prog.tiles[tile as usize];
        // SAFETY: as above.
        unsafe { self.scratch.add(info.byte_offset as usize) }
    }

    fn lane_mask(&self) -> Reg<W> {
        let mut m = [0u32; W];
        for (k, slot) in m.iter_mut().enumerate() {
            *slot = if self.lane_base + (k as u32) < self.prog.block {
                u32::MAX
            } else {
                0
            };
        }
        Reg(m)
    }

    fn eval(&mut self, range: &std::ops::Range<u32>) {
        for pc in range.clone() {
            self.step(pc as usize);
        }
    }

    fn step(&mut self, pc: usize) {
        // Copy the shared reference out of `self` first: `prog` is `&'a
        // Program`, so the instruction borrow does not alias `&mut self`.
        let prog: &'a Program = self.prog;
        let instr = &prog.tape[pc];
        let v = match instr {
            Instr::Const { bits, .. } => Reg::splat_bits(*bits),
            Instr::LaneId { .. } => {
                Reg(core::array::from_fn(|k| self.lane_base + k as u32))
            }
            Instr::Uniform { which, .. } => {
                let w = self.prog.width;
                let v = match which {
                    UniformSrc::ProgramX => self.gid[0],
                    UniformSrc::ProgramY => self.gid[1],
                    UniformSrc::ProgramZ => self.gid[2],
                    UniformSrc::GridX => self.grid[0],
                    UniformSrc::GridY => self.grid[1],
                    UniformSrc::GridZ => self.grid[2],
                    UniformSrc::SubgroupSize => w,
                    UniformSrc::NumSubgroups => self.prog.block.div_ceil(w),
                    UniformSrc::SubgroupId => self.lane_base / w,
                    UniformSrc::SubgroupLane => 0,
                };
                match which {
                    UniformSrc::SubgroupLane => {
                        Reg(core::array::from_fn(|k| (self.lane_base + k as u32) % w))
                    }
                    _ => Reg::splat_u32(v),
                }
            }
            Instr::LoadLocal { local, .. } => {
                let base = *local as usize * self.prog.block as usize;
                Reg(core::array::from_fn(|k| {
                    let lane = self.lane_base as usize + k;
                    if lane < self.prog.block as usize {
                        self.locals[base + lane]
                    } else {
                        0
                    }
                }))
            }
            Instr::Load {
                buf,
                elem,
                index,
                mask,
                fill,
                form,
                ..
            } => {
                let b = self.bufs[*buf as usize];
                let idx = self.regs[*index as usize];
                let m = self.regs[*mask as usize];
                let fl = self.regs[*fill as usize];
                let act = self.active;
                let elems = b.bytes / elem.byte_size() as usize;
                let mut o = [0u32; W];
                match form {
                    AccessForm::Broadcast => {
                        let i = idx.u(0) as usize;
                        let val = if i < elems && m.u(0) != 0 && act.u(0) != 0 {
                            // SAFETY: bounds-checked above.
                            unsafe { expr::read_elem(*elem, b.ptr, i) }
                        } else {
                            fl.u(0)
                        };
                        o = [val; W];
                    }
                    AccessForm::Contiguous | AccessForm::UnitInnerStride => {
                        let base = idx.u(0) as usize;
                        for (k, slot) in o.iter_mut().enumerate() {
                            let i = base + k;
                            *slot = if i < elems && m.u(k) != 0 && act.u(k) != 0 {
                                // SAFETY: bounds-checked above.
                                unsafe { expr::read_elem(*elem, b.ptr, i) }
                            } else {
                                fl.u(k)
                            };
                        }
                    }
                    AccessForm::Gather => {
                        for (k, slot) in o.iter_mut().enumerate() {
                            let i = idx.u(k) as usize;
                            *slot = if i < elems && m.u(k) != 0 && act.u(k) != 0 {
                                // SAFETY: bounds-checked above.
                                unsafe { expr::read_elem(*elem, b.ptr, i) }
                            } else {
                                fl.u(k)
                            };
                        }
                    }
                }
                Reg(o)
            }
            Instr::LoadTile {
                tile, elem, index, ..
            } => {
                let info = &self.prog.tiles[*tile as usize];
                let ptr = self.tile_ptr(*tile);
                let idx = self.regs[*index as usize];
                Reg(core::array::from_fn(|k| {
                    let i = idx.u(k) as usize;
                    if i < info.elements as usize {
                        // SAFETY: bounds-checked above.
                        unsafe { expr::read_elem(*elem, ptr, i) }
                    } else {
                        0
                    }
                }))
            }
            Instr::Un { op, x, ty, .. } => expr::apply_un(*op, *ty, self.regs[*x as usize]),
            Instr::Bin { op, a, b, ty, .. } => expr::apply_bin(
                *op,
                *ty,
                self.regs[*a as usize],
                self.regs[*b as usize],
            ),
            Instr::Fma { a, b, c, .. } => {
                let (ra, rb, rc) = (
                    self.regs[*a as usize],
                    self.regs[*b as usize],
                    self.regs[*c as usize],
                );
                let mut o = [0u32; W];
                for (k, slot) in o.iter_mut().enumerate() {
                    *slot = ra.f(k).mul_add(rb.f(k), rc.f(k)).to_bits();
                }
                Reg(o)
            }
            Instr::Cmp { op, a, b, ty, .. } => expr::apply_cmp(
                *op,
                *ty,
                self.regs[*a as usize],
                self.regs[*b as usize],
            ),
            Instr::MaskToValue { x, ty, .. } => {
                let m = self.regs[*x as usize];
                let one = match ty {
                    NumTy::F32 => 1.0f32.to_bits(),
                    _ => 1,
                };
                Reg(core::array::from_fn(|k| if m.u(k) != 0 { one } else { 0 }))
            }
            Instr::ValueToMask { x, ty, .. } => {
                let v = self.regs[*x as usize];
                Reg(core::array::from_fn(|k| {
                    let nz = match ty {
                        NumTy::F32 => v.f(k) != 0.0,
                        _ => v.u(k) != 0,
                    };
                    if nz { u32::MAX } else { 0 }
                }))
            }
            Instr::Round { mode, x, .. } => self.regs[*x as usize]
                .mapf(|v| expr::round_mode(*mode, v)),
            Instr::Cast { x, from, to, .. } => {
                expr::apply_cast(*from, *to, self.regs[*x as usize])
            }
            Instr::Narrow { x, to, .. } => expr::apply_narrow(*to, self.regs[*x as usize]),
            Instr::Bitcast { x, .. } => self.regs[*x as usize],
            Instr::Select { c, t, f, .. } => Reg::select(
                self.regs[*c as usize],
                self.regs[*t as usize],
                self.regs[*f as usize],
            ),
            Instr::VecCompose { out, parts } => {
                for (i, p) in parts.iter().enumerate() {
                    let v = self.regs[*p as usize];
                    self.regs[*out as usize + i] = v;
                }
                return;
            }
            Instr::VecComponent {
                base, component, ..
            } => self.regs[*base as usize + *component as usize],
            Instr::Dot { a, b, lanes, .. } => {
                let mut acc = [0f32; W];
                for i in 0..*lanes as usize {
                    let ra = self.regs[*a as usize + i];
                    let rb = self.regs[*b as usize + i];
                    for (k, s) in acc.iter_mut().enumerate() {
                        *s += ra.f(k) * rb.f(k);
                    }
                }
                Reg::from_f(acc)
            }
            Instr::Reduce { op, x, kind, .. } => match kind {
                RKind::Subgroup => {
                    reduce::horizontal_masked(*op, self.regs[*x as usize], self.active)
                }
                RKind::TileGroup { tile, group } => {
                    let t = self.tile_f32(*tile);
                    let g = (*group).max(1);
                    Reg(core::array::from_fn(|k| {
                        let lane = self.lane_base + k as u32;
                        let base = ((lane / g) * g) as usize;
                        t.get(base).copied().unwrap_or(0.0).to_bits()
                    }))
                }
            },
            Instr::Unpack2x16 { out, x } => {
                let v = self.regs[*x as usize];
                let lo = v.mapf(|f| f);
                let _ = lo;
                let a = Reg::<W>(core::array::from_fn(|k| {
                    half::f16::from_bits((v.u(k) & 0xFFFF) as u16).to_f32().to_bits()
                }));
                let b = Reg::<W>(core::array::from_fn(|k| {
                    half::f16::from_bits((v.u(k) >> 16) as u16).to_f32().to_bits()
                }));
                self.regs[*out as usize] = a;
                self.regs[*out as usize + 1] = b;
                return;
            }
            Instr::Copy { x, .. } => self.regs[*x as usize],
            Instr::Rc2Index { row, col, map, .. } => {
                let m = &self.prog.maps[*map as usize];
                let r = self.regs[*row as usize];
                let c = self.regs[*col as usize];
                Reg(core::array::from_fn(|k| {
                    access::rc2_offset(m, r.u(k), c.u(k))
                }))
            }
        };
        self.regs[instr.out() as usize] = v;
    }

    fn run(&mut self, stmts: &[CStmt]) -> Flow {
        for s in stmts {
            match self.run_one(s) {
                Flow::Normal => {}
                other => return other,
            }
        }
        Flow::Normal
    }

    fn run_one(&mut self, s: &CStmt) -> Flow {
        match s {
            CStmt::Lanes(body) => {
                let mut base = 0;
                while base < self.prog.block {
                    self.lane_base = base;
                    self.active = self.lane_mask();
                    if self.run(body) == Flow::Return {
                        return Flow::Return;
                    }
                    base += W as u32;
                }
                self.lane_base = 0;
                self.active = self.lane_mask();
            }
            CStmt::Store {
                prep,
                buf,
                elem,
                index,
                value,
                mask,
            } => {
                self.eval(prep);
                let b = self.bufs[*buf as usize];
                let elems = b.bytes / elem.byte_size() as usize;
                let idx = self.regs[*index as usize];
                let v = self.regs[*value as usize];
                let m = self.regs[*mask as usize];
                let act = self.active;
                for k in 0..W {
                    let i = idx.u(k) as usize;
                    if m.u(k) != 0 && act.u(k) != 0 && i < elems {
                        // SAFETY: bounds-checked; the write map is injective.
                        unsafe { expr::write_elem(*elem, b.ptr, i, v.u(k)) };
                    }
                }
            }
            CStmt::AtomicAdd {
                prep,
                buf,
                elem,
                index,
                value,
                mask,
            } => {
                self.eval(prep);
                let b = self.bufs[*buf as usize];
                let elems = b.bytes / elem.byte_size() as usize;
                let idx = self.regs[*index as usize];
                let v = self.regs[*value as usize];
                let m = self.regs[*mask as usize];
                let act = self.active;
                for k in 0..W {
                    let i = idx.u(k) as usize;
                    if m.u(k) != 0 && act.u(k) != 0 && i < elems {
                        // SAFETY: bounds-checked. The launcher runs an
                        // atomic-carrying program on a single worker, so this
                        // read-modify-write is unshared and deterministic.
                        unsafe {
                            let old = expr::read_elem(*elem, b.ptr, i);
                            let sum = f32::from_bits(old) + v.f(k);
                            expr::write_elem(*elem, b.ptr, i, sum.to_bits());
                        }
                    }
                }
            }
            CStmt::StoreLocal { prep, local, value } => {
                self.eval(prep);
                let v = self.regs[*value as usize];
                let act = self.active;
                let base = *local as usize * self.prog.block as usize;
                for k in 0..W {
                    let lane = self.lane_base as usize + k;
                    if act.u(k) != 0 && lane < self.prog.block as usize {
                        self.locals[base + lane] = v.u(k);
                    }
                }
            }
            CStmt::StoreTile {
                prep,
                tile,
                elem,
                index,
                value,
            } => {
                self.eval(prep);
                let info = self.prog.tiles[*tile as usize].clone();
                let ptr = self.tile_ptr(*tile);
                let idx = self.regs[*index as usize];
                let v = self.regs[*value as usize];
                let act = self.active;
                for k in 0..W {
                    let i = idx.u(k) as usize;
                    if act.u(k) != 0 && i < info.elements as usize {
                        // SAFETY: bounds-checked, thread-local scratch.
                        unsafe { expr::write_elem(*elem, ptr, i, v.u(k)) };
                    }
                }
            }
            CStmt::FillTile {
                prep,
                tile,
                elem,
                value,
                extents,
                lo,
                hi,
            } => {
                self.lane_base = 0;
                self.active = self.lane_mask();
                self.eval(prep);
                let info = self.prog.tiles[*tile as usize].clone();
                let ptr = self.tile_ptr(*tile);
                let bits = self.regs[*value as usize].u(0);
                let b0 = lo.map_or(extents[0], |s| self.regs[s as usize].u(0));
                let b1 = hi.map_or(extents[1], |s| self.regs[s as usize].u(0));
                let cols = extents[1].max(1);
                for i in 0..info.elements {
                    let (r, c) = (i / cols, i % cols);
                    if r < b0 && c < b1 {
                        // SAFETY: `i < elements`, thread-local scratch.
                        unsafe { expr::write_elem(*elem, ptr, i as usize, bits) };
                    }
                }
            }
            CStmt::If {
                prep,
                cond,
                uniform,
                accept,
                reject,
            } => {
                self.eval(prep);
                let c = self.regs[*cond as usize];
                if *uniform {
                    let taken = c.u(0) != 0;
                    return if taken {
                        self.run(accept)
                    } else {
                        self.run(reject)
                    };
                }
                // Divergent: both arms run under complementary masks, so every
                // store merges rather than branching.
                let saved = self.active;
                self.active = Reg(core::array::from_fn(|k| saved.u(k) & c.u(k)));
                let f1 = self.run(accept);
                self.active = Reg(core::array::from_fn(|k| saved.u(k) & !c.u(k)));
                let f2 = self.run(reject);
                self.active = saved;
                if f1 == Flow::Return || f2 == Flow::Return {
                    return Flow::Return;
                }
            }
            CStmt::Loop {
                prep,
                count,
                index,
                accs,
                body,
            } => {
                self.eval(prep);
                // Every accumulator is read at the value it had entering the
                // step, then all are written.
                let mut next: Vec<Reg<W>> = Vec::with_capacity(accs.len());
                for a in accs {
                    self.eval(&a.init_prep);
                    next.push(self.regs[a.init as usize]);
                }
                for (a, v) in accs.iter().zip(&next) {
                    self.write_local(a.local, *v);
                }
                let n = count.map_or(u32::MAX, |s| self.regs[s as usize].u(0));
                let mut it = 0u32;
                while it < n {
                    if let Some(l) = index {
                        self.write_local(*l, Reg::splat_u32(it));
                    }
                    let flow = self.run(body);
                    next.clear();
                    for a in accs {
                        self.eval(&a.update_prep);
                        next.push(self.regs[a.update as usize]);
                    }
                    for (a, v) in accs.iter().zip(&next) {
                        self.write_local(a.local, *v);
                    }
                    match flow {
                        Flow::Break => break,
                        Flow::Return => return Flow::Return,
                        Flow::Normal => {}
                    }
                    it += 1;
                }
            }
            CStmt::Break => return Flow::Break,
            CStmt::Return => return Flow::Return,
            CStmt::StageTree {
                prep,
                tile,
                value,
                op,
                group,
            } => {
                self.stage(prep, *tile, *value, None, 1);
                self.tree(*tile, *op, *group);
            }
            CStmt::LoopTree {
                prep,
                tile,
                value,
                op,
                group,
                iterations,
                index,
            } => {
                self.stage_loop(prep, *tile, *value, *index, *iterations, *op);
                self.tree(*tile, *op, *group);
            }
            CStmt::CarrierTree {
                prep,
                tiles,
                values,
                lhs,
                rhs,
                merge_prep,
                merged,
                outs,
                group,
                fast,
            } => {
                self.carrier_stage(prep, tiles, values);
                self.carrier_tree(tiles, lhs, rhs, merge_prep, merged, *group, *fast);
                self.carrier_broadcast(tiles, outs, *group);
            }
            CStmt::Barrier => {}
        }
        Flow::Normal
    }

    /// Write one partial per lane into each lane's scratch tile.
    fn carrier_stage(&mut self, prep: &std::ops::Range<u32>, tiles: &[u16], values: &[Slot]) {
        let block = self.prog.block;
        let mut base = 0;
        while base < block {
            self.lane_base = base;
            self.active = self.lane_mask();
            self.eval(prep);
            for (tile, value) in tiles.iter().zip(values) {
                let v = self.regs[*value as usize];
                let t = self.tile_f32(*tile);
                for k in 0..W {
                    let lane = (base as usize) + k;
                    if lane < block as usize && lane < t.len() {
                        t[lane] = v.f(k);
                    }
                }
            }
            base += W as u32;
        }
        self.lane_base = 0;
        self.active = self.lane_mask();
    }

    /// The log-tree, one level at a time, applying `merge` to `W` independent
    /// pairs at once. A merge reads only its formals, so the whole level is
    /// gathered before any of it is written back and no pair can observe a
    /// half-merged sibling.
    #[allow(clippy::too_many_arguments)]
    fn carrier_tree(
        &mut self,
        tiles: &[u16],
        lhs: &[u16],
        rhs: &[u16],
        merge_prep: &std::ops::Range<u32>,
        merged: &[Slot],
        group: u32,
        fast: Option<TileReduceOp>,
    ) {
        let block = self.prog.block as usize;
        let group = (group.max(1) as usize).min(block.max(1));
        let chunk = W.min(block.max(1));
        self.lane_base = 0;
        self.active = Reg::splat_bits(u32::MAX);
        let mut base = 0;
        while base < block {
            let mut stride = group / 2;
            while stride >= 1 {
                let mut i = 0;
                while i < stride {
                    let take = chunk.min(stride - i);
                    let mut left = vec![[0f32; W]; tiles.len()];
                    let mut right = vec![[0f32; W]; tiles.len()];
                    for (s, tile) in tiles.iter().enumerate() {
                        let t = self.tile_f32(*tile);
                        for k in 0..take {
                            let li = base + i + k;
                            left[s][k] = t[li];
                            right[s][k] = t[li + stride];
                        }
                    }
                    let out_lanes: Vec<[f32; W]> = match fast {
                        // One lane with a hardware operator folds without the
                        // tape, exactly as the single-slot path does.
                        Some(op) => (0..tiles.len())
                            .map(|s| {
                                core::array::from_fn(|k| {
                                    reduce::combine_f32(op, left[s][k], right[s][k])
                                })
                            })
                            .collect(),
                        None => {
                            for s in 0..tiles.len() {
                                self.write_local(lhs[s], Reg::from_f(left[s]));
                                self.write_local(rhs[s], Reg::from_f(right[s]));
                            }
                            self.eval(merge_prep);
                            merged
                                .iter()
                                .map(|slot| {
                                    let v = self.regs[*slot as usize];
                                    core::array::from_fn(|k| v.f(k))
                                })
                                .collect()
                        }
                    };
                    for (s, tile) in tiles.iter().enumerate() {
                        let t = self.tile_f32(*tile);
                        for k in 0..take {
                            t[base + i + k] = out_lanes[s][k];
                        }
                    }
                    i += take;
                }
                stride /= 2;
            }
            base += group;
        }
        self.lane_base = 0;
        self.active = self.lane_mask();
    }

    /// Broadcast each group's result over the group, then load it per lane.
    fn carrier_broadcast(&mut self, tiles: &[u16], outs: &[u16], group: u32) {
        let block = self.prog.block as usize;
        let group = (group.max(1) as usize).min(block.max(1));
        for tile in tiles {
            let t = self.tile_f32(*tile);
            let len = block.min(t.len());
            let mut base = 0;
            while base < len {
                let hi = (base + group).min(len);
                let v = t[base];
                for x in &mut t[base..hi] {
                    *x = v;
                }
                base = hi;
            }
        }
        let mut base = 0;
        while base < self.prog.block {
            self.lane_base = base;
            self.active = self.lane_mask();
            for (tile, out) in tiles.iter().zip(outs) {
                let t = self.tile_f32(*tile);
                let mut v = [0f32; W];
                for (k, slot) in v.iter_mut().enumerate() {
                    let lane = (base as usize) + k;
                    if lane < t.len() {
                        *slot = t[lane];
                    }
                }
                self.write_local(*out, Reg::from_f(v));
            }
            base += W as u32;
        }
        self.lane_base = 0;
        self.active = self.lane_mask();
    }

    fn write_local(&mut self, local: u16, v: Reg<W>) {
        let base = local as usize * self.prog.block as usize;
        for k in 0..W {
            let lane = self.lane_base as usize + k;
            if lane < self.prog.block as usize {
                self.locals[base + lane] = v.u(k);
            }
        }
    }

    fn stage(
        &mut self,
        prep: &std::ops::Range<u32>,
        tile: u16,
        value: Slot,
        _idx: Option<u16>,
        _iters: u32,
    ) {
        let block = self.prog.block;
        let mut base = 0;
        while base < block {
            self.lane_base = base;
            self.active = self.lane_mask();
            self.eval(prep);
            let v = self.regs[value as usize];
            let t = self.tile_f32(tile);
            for k in 0..W {
                let lane = (base as usize) + k;
                if lane < block as usize && lane < t.len() {
                    t[lane] = v.f(k);
                }
            }
            base += W as u32;
        }
        self.lane_base = 0;
        self.active = self.lane_mask();
    }

    fn stage_loop(
        &mut self,
        prep: &std::ops::Range<u32>,
        tile: u16,
        value: Slot,
        index: u16,
        iterations: u32,
        op: TileReduceOp,
    ) {
        let block = self.prog.block;
        let ident = reduce::identity_f32(op);
        let mut base = 0;
        while base < block {
            self.lane_base = base;
            self.active = self.lane_mask();
            let mut acc = Reg::<W>::splat_f32(ident);
            for it in 0..iterations {
                self.write_local(index, Reg::splat_u32(it));
                self.eval(prep);
                let v = self.regs[value as usize];
                acc = reduce::combine_reg(op, acc, v);
            }
            let t = self.tile_f32(tile);
            for k in 0..W {
                let lane = (base as usize) + k;
                if lane < block as usize && lane < t.len() {
                    t[lane] = acc.f(k);
                }
            }
            base += W as u32;
        }
        self.lane_base = 0;
        self.active = self.lane_mask();
    }

    fn tree(&mut self, tile: u16, op: TileReduceOp, group: u32) {
        let block = self.prog.block as usize;
        let t = self.tile_f32(tile);
        let len = block.min(t.len());
        reduce::tree_in_place(op, t, len, group as usize);
    }
}

/// Execute one workgroup at a statically-known lane width.
#[inline(always)]
pub fn run_workgroup<const W: usize>(
    prog: &Program,
    gid: [u32; 3],
    grid: [u32; 3],
    bufs: &[RawBuf],
    scratch: *mut u8,
) {
    let mut ex = Exec::<W> {
        prog,
        regs: vec![Reg::default(); prog.regs.max(1)],
        locals: vec![0u32; prog.locals * prog.block as usize],
        bufs,
        scratch,
        gid,
        grid,
        lane_base: 0,
        active: Reg::splat_bits(u32::MAX),
    };
    for seg in &prog.segments {
        // A barrier between two segments is this boundary: the previous
        // segment has completed for *every* lane before the next one starts.
        let collective = seg.stmts.iter().any(CStmt::is_collective);
        let flow = if collective {
            ex.run(&seg.stmts)
        } else {
            ex.run_one(&CStmt::Lanes(seg.stmts.clone()))
        };
        if flow == Flow::Return {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::AlignedBuf;
    use fusor2_ir::dtype::RoundMode;
    use fusor2_ir::ir::kernel::{
        Accumulator, BufferAccess, LocalDecl, MemoryLevel, ReduceKind, StorageView, TileDecl,
        TileLayout, WorkgroupAxis,
    };
    use fusor2_ir::scalar::{CmpOp, UnOp};

    fn f32_ty() -> ElementType {
        ElementType::Scalar(ScalarElement::F32)
    }
    fn u32_ty() -> ElementType {
        ElementType::Scalar(ScalarElement::U32)
    }
    fn bool_ty() -> ElementType {
        ElementType::Scalar(ScalarElement::Bool)
    }

    fn decl(binding: u32, elem: ScalarElement, n: u32, rw: bool) -> Arc<BufferDecl> {
        Arc::new(BufferDecl {
            binding,
            element: ElementType::Scalar(elem),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[n]),
            access: if rw {
                BufferAccess::ReadWrite
            } else {
                BufferAccess::Read
            },
        })
    }

    fn view(b: &Arc<BufferDecl>) -> StorageView {
        StorageView {
            buffer: Arc::clone(b),
            offset: 0,
            layout: b.layout.clone(),
        }
    }

    fn lane() -> TileExpr {
        TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty())
    }
    fn pid() -> TileExpr {
        TileExpr::new(
            TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
            u32_ty(),
        )
    }
    fn ulit(v: u32) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32_ty())
    }
    fn flit(v: f32) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::F32(v.to_bits())), f32_ty())
    }
    fn tlit() -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::Bool(true)), bool_ty())
    }
    fn ubin(op: BinOp, a: TileExpr, b: TileExpr) -> TileExpr {
        TileExpr::new(
            TileExprKind::Binary {
                op,
                left: a,
                right: b,
                numeric: NumericContract::RELAXED,
            },
            u32_ty(),
        )
    }
    fn fbin(op: BinOp, a: TileExpr, b: TileExpr, nc: NumericContract) -> TileExpr {
        TileExpr::new(
            TileExprKind::Binary {
                op,
                left: a,
                right: b,
                numeric: nc,
            },
            f32_ty(),
        )
    }
    fn fload(b: &Arc<BufferDecl>, index: TileExpr) -> TileExpr {
        TileExpr::new(
            TileExprKind::Load {
                src: Source::Storage(view(b)),
                addr: Box::new(Addr::Linear(index)),
                mask: tlit(),
                fill: flit(0.0),
            },
            ElementType::Scalar(match b.element {
                ElementType::Scalar(s) => s,
                _ => ScalarElement::F32,
            }),
        )
    }

    /// Compile and run one kernel over f32 buffers, returning the outputs.
    fn run_f32(ir: &KernelIr, inputs: &[Vec<f32>], out_len: usize) -> Vec<f32> {
        let caps = crate::caps::cpu_caps();
        let art = compile(ir, caps, None).expect("compile");
        let kernel = CpuKernel {
            name: art.name,
            block: art.block,
            vector_width: art.prog.width,
            artifact: art,
        };
        let mut binds = Vec::new();
        for v in inputs {
            let mut b = AlignedBuf::zeroed(v.len() * 4).unwrap();
            b.as_mut_slice()
                .copy_from_slice(bytemuck::cast_slice(v.as_slice()));
            binds.push(Buf::new(b));
        }
        let out = Buf::new(AlignedBuf::zeroed(out_len * 4).unwrap());
        binds.push(out.clone());
        kernel
            .run(ir.grid, &binds, &Uniforms::default())
            .expect("launch");
        let ab = out.downcast_ref::<AlignedBuf>().unwrap();
        bytemuck::cast_slice::<u8, f32>(ab.as_slice()).to_vec()
    }

    #[test]
    fn barrier_splits_lane_loop() {
        const BLOCK: u32 = 256;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let out = decl(1, ScalarElement::F32, BLOCK, true);
        let scratch: Arc<TileDecl> = Arc::new(TileDecl::new(
            f32_ty(),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[BLOCK]),
            "rev",
        ));

        let lane_f = TileExpr::new(
            TileExprKind::Cast {
                value: lane(),
                to: f32_ty(),
            },
            f32_ty(),
        );
        let mirrored = ubin(BinOp::Sub, ulit(BLOCK - 1), lane());
        let body = vec![
            Stmt::StoreTile {
                dst: Arc::clone(&scratch),
                index: lane(),
                value: lane_f,
            },
            Stmt::Barrier,
            Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(lane()),
                value: TileExpr::new(
                    TileExprKind::LoadTile {
                        tile: Arc::clone(&scratch),
                        index: mirrored,
                    },
                    f32_ty(),
                ),
                mask: tlit(),
            },
        ];
        let ir = KernelIr {
            buffers: vec![uni, out],
            grid: [1, 1, 1],
            block: BLOCK,
            body,
            byte_arena: None,
            name: "reverse",
        };

        let art = compile(&ir, crate::caps::cpu_caps(), None).unwrap();
        assert_eq!(
            art.prog.segments.len(),
            2,
            "a barrier must cut the lane loop in two"
        );
        assert!(
            art.prog.width >= 8 && BLOCK % art.prog.width == 0,
            "the block must split into whole lane chunks (W = {})",
            art.prog.width
        );

        let got = run_f32(&ir, &[], BLOCK as usize);
        for i in 0..BLOCK as usize {
            assert_eq!(
                got[i],
                (BLOCK as usize - 1 - i) as f32,
                "lane {i} read a stale tile slot: a no-op barrier would leave \
                 every lane past the first chunk reading zeros"
            );
        }
        // The specific failure a no-op barrier produces.
        assert_ne!(got[8], 0.0);
    }

    fn fma_program(nc: NumericContract) -> Arc<Program> {
        let uni = decl(0, ScalarElement::U32, 1, false);
        let a = decl(1, ScalarElement::F32, 8, false);
        let b = decl(2, ScalarElement::F32, 8, false);
        let c = decl(3, ScalarElement::F32, 8, false);
        let out = decl(4, ScalarElement::F32, 8, true);
        let prod = fbin(BinOp::Mul, fload(&a, lane()), fload(&b, lane()), nc);
        let sum = fbin(BinOp::Add, prod, fload(&c, lane()), nc);
        let ir = KernelIr {
            buffers: vec![uni, a, b, c, out.clone()],
            grid: [1, 1, 1],
            block: 8,
            body: vec![Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(lane()),
                value: sum,
                mask: tlit(),
            }],
            byte_arena: None,
            name: "fma",
        };
        compile(&ir, crate::caps::cpu_caps(), None).unwrap().prog
    }

    #[test]
    fn numeric_contract_blocks_contraction() {
        let relaxed = fma_program(NumericContract::RELAXED);
        assert_eq!(
            relaxed.fma_count(),
            1,
            "a relaxed a*b+c must contract into one mul_add"
        );

        let strict = fma_program(NumericContract::STRICT);
        assert_eq!(
            strict.fma_count(),
            0,
            "contract: false forbids fusing a mul and an add"
        );
        let muls = strict
            .tape
            .iter()
            .filter(|i| matches!(i, Instr::Bin { op: BinOp::Mul, .. }))
            .count();
        let adds = strict
            .tape
            .iter()
            .filter(|i| matches!(i, Instr::Bin { op: BinOp::Add, .. }))
            .count();
        assert_eq!((muls, adds), (1, 1), "strict must emit separate mul and add");

        // `round(x, HalfAwayFromZero)` at an exact .5 rounds away from zero:
        // MSQ1 export idempotence depends on it.
        for i in -16i32..=16 {
            if i % 2 == 0 {
                continue;
            }
            let x = i as f32 * 0.5;
            assert_eq!(expr::round_mode(RoundMode::HalfAwayFromZero, x).abs(), x.abs() + 0.5);
        }
    }

    #[test]
    fn four_access_lowerings() {
        access::reset_form_counts();
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, 64, false);
        let out = decl(2, ScalarElement::F32, 4 * 8, true);

        // The same logical tensor read through four addressings.
        let addrs = [
            lane(),                                    // contiguous
            ulit(3),                                   // broadcast
            ubin(BinOp::Add, ulit(16), lane()),        // unit inner stride
            ubin(BinOp::Mul, lane(), ulit(3)),         // general gather
        ];
        let mut body = Vec::new();
        for (i, a) in addrs.iter().enumerate() {
            body.push(Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(ubin(BinOp::Add, ulit(i as u32 * 8), lane())),
                value: fload(&src, a.clone()),
                mask: tlit(),
            });
        }
        let ir = KernelIr {
            buffers: vec![uni, src, out],
            grid: [1, 1, 1],
            block: 8,
            body,
            byte_arena: None,
            name: "access",
        };

        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let got = run_f32(&ir, &[data.clone()], 32);
        for k in 0..8usize {
            assert_eq!(got[k], data[k], "contiguous lane {k}");
            assert_eq!(got[8 + k], data[3], "broadcast lane {k}");
            assert_eq!(got[16 + k], data[16 + k], "unit-inner lane {k}");
            assert_eq!(got[24 + k], data[3 * k], "gather lane {k}");
        }
        let counts = access::form_counts();
        for (i, f) in access::AccessForm::ALL.iter().enumerate() {
            assert!(counts[i] > 0, "{f:?} was never selected");
        }
    }

    /// A vector-typed `Select` selects **every** lane.
    ///
    /// Pins the regression where one `Instr::Select` wrote register `out`
    /// only, leaving lanes 1.. uninitialized.
    #[test]
    fn a_masked_vector_select_writes_every_lane() {
        const LANES: u32 = 8;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, LANES, false);
        let out = decl(2, ScalarElement::F32, LANES, true);

        let vec_ty = ElementType::Vector {
            scalar: ScalarElement::F32,
            lanes: LANES,
        };
        let parts: Vec<TileExpr> = (0..LANES).map(|i| fload(&src, ulit(i))).collect();
        let taken = TileExpr::new(
            TileExprKind::Vec {
                scalar: ScalarElement::F32,
                lanes: LANES,
                parts,
            },
            vec_ty,
        );
        let zeros = TileExpr::new(
            TileExprKind::Vec {
                scalar: ScalarElement::F32,
                lanes: LANES,
                parts: (0..LANES).map(|_| flit(-1.0)).collect(),
            },
            vec_ty,
        );
        let selected = TileExpr::new(
            TileExprKind::Select {
                condition: tlit(),
                accept: taken,
                reject: zeros,
            },
            vec_ty,
        );
        let body = (0..LANES)
            .map(|i| Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(ulit(i)),
                value: TileExpr::new(
                    TileExprKind::VecComponent {
                        vector: selected.clone(),
                        component: i,
                    },
                    f32_ty(),
                ),
                mask: tlit(),
            })
            .collect();
        let ir = KernelIr {
            buffers: vec![uni, src, out],
            grid: [1, 1, 1],
            block: 1,
            body,
            byte_arena: None,
            name: "vector_select",
        };

        let data: Vec<f32> = (0..LANES).map(|i| i as f32 + 0.5).collect();
        let got = run_f32(&ir, &[data.clone()], LANES as usize);
        assert_eq!(got, data, "every lane of the selected vector must survive");
    }

    #[test]
    fn f16_bf16_widen_compute() {
        for (elem, to_bits) in [
            (
                ScalarElement::F16,
                (|x: f32| half::f16::from_f32(x).to_bits() as u32) as fn(f32) -> u32,
            ),
            (
                ScalarElement::BF16,
                (|x: f32| half::bf16::from_f32(x).to_bits() as u32) as fn(f32) -> u32,
            ),
        ] {
            let n = 256u32;
            let uni = decl(0, ScalarElement::U32, 1, false);
            let src = decl(1, elem, n, false);
            let out = decl(2, elem, n, true);
            let idx = ubin(BinOp::Add, ubin(BinOp::Mul, pid(), ulit(64)), lane());
            let value = TileExpr::new(
                TileExprKind::Unary {
                    op: UnOp::Exp,
                    value: fload(&src, idx.clone()),
                    numeric: NumericContract::RELAXED,
                },
                f32_ty(),
            );
            let ir = KernelIr {
                buffers: vec![uni, src, out.clone()],
                grid: [4, 1, 1],
                block: 64,
                body: vec![Stmt::Store {
                    dst: view(&out),
                    addr: Addr::Linear(idx),
                    value,
                    mask: tlit(),
                }],
                byte_arena: None,
                name: "widen",
            };
            let art = compile(&ir, crate::caps::cpu_caps(), None).unwrap();
            // The register file holds f32: every ALU instruction is typed F32,
            // never a one-lane narrow float.
            assert!(art.prog.tape.iter().all(|i| !matches!(
                i,
                Instr::Un { ty: NumTy::U32, .. } | Instr::Un { ty: NumTy::I32, .. }
            )));
            assert!(art
                .prog
                .tape
                .iter()
                .any(|i| matches!(i, Instr::Un { ty: NumTy::F32, op: UnOp::Exp, .. })));

            // Bitwise equality against `f16::from_f32(exp(f32::from(x)))`.
            let kernel = CpuKernel {
                name: art.name,
                block: art.block,
                vector_width: art.prog.width,
                artifact: art,
            };
            let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 3.0).collect();
            let mut inb = AlignedBuf::zeroed(n as usize * 2).unwrap();
            for (i, v) in vals.iter().enumerate() {
                let raw = to_bits(*v) as u16;
                inb.as_mut_slice()[i * 2..i * 2 + 2].copy_from_slice(&raw.to_le_bytes());
            }
            let outb = Buf::new(AlignedBuf::zeroed(n as usize * 2).unwrap());
            kernel
                .run([4, 1, 1], &[Buf::new(inb), outb.clone()], &Uniforms::default())
                .unwrap();
            let ab = outb.downcast_ref::<AlignedBuf>().unwrap();
            for i in 0..n as usize {
                let raw = u16::from_le_bytes([ab.as_slice()[i * 2], ab.as_slice()[i * 2 + 1]]);
                let x = match elem {
                    ScalarElement::F16 => half::f16::from_bits(to_bits(vals[i]) as u16).to_f32(),
                    _ => half::bf16::from_bits(to_bits(vals[i]) as u16).to_f32(),
                };
                let want = to_bits(expr::expf(x)) as u16;
                assert_eq!(raw, want, "{elem:?} element {i}");
            }
        }
    }

    #[test]
    fn workgroup_reduce_sums_every_lane() {
        const BLOCK: u32 = 64;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, BLOCK, false);
        let out = decl(2, ScalarElement::F32, 1, true);
        let scratch: Arc<TileDecl> = Arc::new(TileDecl::new(
            f32_ty(),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[BLOCK]),
            "tree",
        ));
        let reduced = TileExpr::new(
            TileExprKind::Reduce {
                op: TileReduceOp::Sum,
                kind: Box::new(ReduceKind::Workgroup {
                    scratch,
                    group_size: BLOCK,
                }),
                value: fload(&src, lane()),
            },
            f32_ty(),
        );
        let ir = KernelIr {
            buffers: vec![uni, src, out.clone()],
            grid: [1, 1, 1],
            block: BLOCK,
            body: vec![Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(ulit(0)),
                value: reduced,
                mask: TileExpr::new(
                    TileExprKind::Compare {
                        op: CmpOp::Eq,
                        left: lane(),
                        right: ulit(0),
                    },
                    bool_ty(),
                ),
            }],
            byte_arena: None,
            name: "wg_sum",
        };
        let data: Vec<f32> = (0..BLOCK).map(|i| i as f32).collect();
        let got = run_f32(&ir, &[data], 1);
        assert_eq!(got[0], (0..BLOCK).map(|i| i as f32).sum::<f32>());
    }

    #[test]
    fn subgroup_reduce_is_a_horizontal_reduce() {
        const BLOCK: u32 = 8;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, BLOCK, false);
        let out = decl(2, ScalarElement::F32, BLOCK, true);
        let reduced = TileExpr::new(
            TileExprKind::Reduce {
                op: TileReduceOp::Max,
                kind: Box::new(ReduceKind::Subgroup),
                value: fload(&src, lane()),
            },
            f32_ty(),
        );
        let ir = KernelIr {
            buffers: vec![uni, src, out.clone()],
            grid: [1, 1, 1],
            block: BLOCK,
            body: vec![Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(lane()),
                value: reduced,
                mask: tlit(),
            }],
            byte_arena: None,
            name: "sg_max",
        };
        let data = vec![1.0, -2.0, 7.5, 3.0, 0.0, -9.0, 2.0, 4.0];
        let got = run_f32(&ir, &[data], BLOCK as usize);
        assert!(got.iter().all(|v| *v == 7.5), "{got:?}");
    }

    #[test]
    fn matmul_epilogue_fuses_in_the_k_loop() {
        // out[j] = gelu-ish(sum_k a[k] * b[k, j] + bias[j]) for one row.
        const N: u32 = 16;
        const K: u32 = 12;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let a = decl(1, ScalarElement::F32, K, false);
        let b = decl(2, ScalarElement::F32, K * N, false);
        let bias = decl(3, ScalarElement::F32, N, false);
        let out = decl(4, ScalarElement::F32, N, true);

        let acc_local = Arc::new(LocalDecl::new(f32_ty()));
        let kk = Arc::new(LocalDecl::new(u32_ty()));
        let k_idx = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&kk)), u32_ty());
        let a_v = fload(&a, k_idx.clone());
        let b_v = fload(
            &b,
            ubin(BinOp::Add, ubin(BinOp::Mul, k_idx, ulit(N)), lane()),
        );
        let prev = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc_local)), f32_ty());
        let update = fbin(
            BinOp::Add,
            prev,
            fbin(BinOp::Mul, a_v, b_v, NumericContract::RELAXED),
            NumericContract::RELAXED,
        );
        let acc = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc_local)), f32_ty());
        let epilogue = TileExpr::new(
            TileExprKind::Unary {
                op: UnOp::Tanh,
                value: fbin(
                    BinOp::Add,
                    acc,
                    fload(&bias, lane()),
                    NumericContract::RELAXED,
                ),
                numeric: NumericContract::RELAXED,
            },
            f32_ty(),
        );
        let ir = KernelIr {
            buffers: vec![uni, a, b, bias, out.clone()],
            grid: [1, 1, 1],
            block: N,
            body: vec![
                Stmt::Loop {
                    count: Some(ulit(K)),
                    index: Some(kk),
                    accumulators: vec![Accumulator {
                        local: acc_local,
                        init: flit(0.0),
                        update,
                    }],
                    body: vec![],
                },
                Stmt::Store {
                    dst: view(&out),
                    addr: Addr::Linear(lane()),
                    value: epilogue,
                    mask: tlit(),
                },
            ],
            byte_arena: None,
            name: "gemv_epilogue",
        };

        let art = compile(&ir, crate::caps::cpu_caps(), None).unwrap();
        assert_eq!(art.prog.segments.len(), 1, "no barrier, so one segment");
        assert_eq!(
            art.prog.store_count(),
            1,
            "the epilogue must fuse: no intermediate materialization"
        );

        let av: Vec<f32> = (0..K).map(|i| 0.1 * (i as f32) - 0.5).collect();
        let bv: Vec<f32> = (0..K * N).map(|i| 0.01 * (i as f32) - 0.3).collect();
        let biasv: Vec<f32> = (0..N).map(|i| 0.05 * (i as f32)).collect();
        let got = run_f32(&ir, &[av.clone(), bv.clone(), biasv.clone()], N as usize);
        for j in 0..N as usize {
            let mut acc = 0f64;
            for k in 0..K as usize {
                acc += av[k] as f64 * bv[k * N as usize + j] as f64;
            }
            let want = ((acc + biasv[j] as f64).tanh()) as f32;
            assert!((got[j] - want).abs() < 1e-5, "col {j}: {} vs {want}", got[j]);
        }
    }

    #[test]
    fn scatter_add_accumulates_duplicates() {
        // 4096 indices into 64 bins x 8 lanes, 7% of them in one bin, plus a
        // pure-padding tail. One workgroup per bin, so the accumulation order
        // is fixed and the result is bit-reproducible at any thread count.
        const BINS: u32 = 64;
        const WIDTH: u32 = 8;
        const N: u32 = 4096;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let base = decl(1, ScalarElement::F32, BINS * WIDTH, false);
        let idx = decl(2, ScalarElement::U32, N, false);
        let upd = decl(3, ScalarElement::F32, N * WIDTH, false);
        let out = decl(4, ScalarElement::F32, BINS * WIDTH, true);

        let dst = ubin(BinOp::Add, ubin(BinOp::Mul, pid(), ulit(WIDTH)), lane());
        let kk = Arc::new(LocalDecl::new(u32_ty()));
        let k_idx = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&kk)), u32_ty());
        let this = TileExpr::new(
            TileExprKind::Load {
                src: Source::Storage(view(&idx)),
                addr: Box::new(Addr::Linear(k_idx.clone())),
                mask: tlit(),
                fill: ulit(0),
            },
            u32_ty(),
        );
        let hit = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Eq,
                left: this,
                right: pid(),
            },
            bool_ty(),
        );
        let contribution = fload(
            &upd,
            ubin(BinOp::Add, ubin(BinOp::Mul, k_idx, ulit(WIDTH)), lane()),
        );
        let acc_local = Arc::new(LocalDecl::new(f32_ty()));
        let prev = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc_local)), f32_ty());
        let update = TileExpr::new(
            TileExprKind::Select {
                condition: hit,
                accept: fbin(
                    BinOp::Add,
                    prev.clone(),
                    contribution,
                    NumericContract::STRICT,
                ),
                reject: prev,
            },
            f32_ty(),
        );
        let ir = KernelIr {
            buffers: vec![uni, base.clone(), idx, upd, out.clone()],
            grid: [BINS, 1, 1],
            block: WIDTH,
            body: vec![
                Stmt::Loop {
                    count: Some(ulit(N)),
                    index: Some(kk),
                    accumulators: vec![Accumulator {
                        local: Arc::clone(&acc_local),
                        init: fload(&base, dst.clone()),
                        update,
                    }],
                    body: vec![],
                },
                Stmt::Store {
                    dst: view(&out),
                    addr: Addr::Linear(dst),
                    value: TileExpr::new(TileExprKind::LoadLocal(acc_local), f32_ty()),
                    mask: tlit(),
                },
            ],
            byte_arena: None,
            name: "scatter_add",
        };

        // 7% of the indices land in bin 3; the last 256 are pure padding into
        // a bin nothing reads back.
        let indices: Vec<u32> = (0..N)
            .map(|i| {
                if i % 14 == 0 {
                    3
                } else if i >= N - 256 {
                    BINS - 1
                } else {
                    (i * 7 + 11) % BINS
                }
            })
            .collect();
        let basev = vec![0.0f32; (BINS * WIDTH) as usize];
        let updv: Vec<f32> = (0..N * WIDTH).map(|i| ((i % 97) as f32) * 0.25 - 6.0).collect();

        let mut want = basev.clone();
        for (i, b) in indices.iter().enumerate() {
            for l in 0..WIDTH as usize {
                want[*b as usize * WIDTH as usize + l] += updv[i * WIDTH as usize + l];
            }
        }

        let caps = crate::caps::cpu_caps();
        let art = compile(&ir, caps, None).unwrap();
        let kernel = CpuKernel {
            name: art.name,
            block: art.block,
            vector_width: art.prog.width,
            artifact: art,
        };
        let mk_f32 = |v: &[f32]| {
            let mut b = AlignedBuf::zeroed(v.len() * 4).unwrap();
            b.as_mut_slice().copy_from_slice(bytemuck::cast_slice(v));
            Buf::new(b)
        };
        let mut ib = AlignedBuf::zeroed(indices.len() * 4).unwrap();
        ib.as_mut_slice()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        let outb = Buf::new(AlignedBuf::zeroed(basev.len() * 4).unwrap());
        kernel
            .run(
                [BINS, 1, 1],
                &[mk_f32(&basev), Buf::new(ib), mk_f32(&updv), outb.clone()],
                &Uniforms::default(),
            )
            .unwrap();
        let ab = outb.downcast_ref::<AlignedBuf>().unwrap();
        let got = bytemuck::cast_slice::<u8, f32>(ab.as_slice());
        for i in 0..got.len() {
            assert_eq!(got[i], want[i], "bin element {i}");
        }
    }

    #[test]
    fn parallel_for_is_deterministic() {
        const N: u32 = 4096;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, N, false);
        let out = decl(2, ScalarElement::F32, N, true);
        let idx = ubin(BinOp::Add, ubin(BinOp::Mul, pid(), ulit(64)), lane());
        let ir = KernelIr {
            buffers: vec![uni, Arc::clone(&src), out.clone()],
            grid: [N / 64, 1, 1],
            block: 64,
            body: vec![Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(idx.clone()),
                value: TileExpr::new(
                    TileExprKind::Unary {
                        op: UnOp::Tanh,
                        value: fload(&src, idx),
                        numeric: NumericContract::RELAXED,
                    },
                    f32_ty(),
                ),
                mask: tlit(),
            }],
            byte_arena: None,
            name: "det",
        };
        let data: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let a = run_f32(&ir, &[data.clone()], N as usize);
        let b = run_f32(&ir, &[data.clone()], N as usize);
        assert_eq!(a, b, "two runs over the same pool must be bit-identical");
        for i in 0..N as usize {
            assert_eq!(a[i], expr::tanhf(data[i]), "element {i}");
        }
    }

    #[test]
    fn level_dispatched_once_per_worker_not_per_row() {
        const N: u32 = 65536;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let out = decl(1, ScalarElement::F32, N, true);
        let idx = ubin(BinOp::Add, ubin(BinOp::Mul, pid(), ulit(64)), lane());
        let ir = KernelIr {
            buffers: vec![uni, out.clone()],
            grid: [N / 64, 1, 1],
            block: 64,
            body: vec![Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(idx),
                value: flit(1.0),
                mask: tlit(),
            }],
            byte_arena: None,
            name: "count",
        };
        let threads = crate::caps::CpuCaps::threads() as u64;
        let bound = threads * 4 + 1;

        crate::launch::reset_dispatch_count();
        let _ = run_f32(&ir, &[], N as usize);
        let small = crate::launch::dispatch_count();
        assert!(small >= 1);
        assert!(
            small <= bound,
            "{small} dispatches for {threads} threads at grid {} looks per-row",
            N / 64
        );

        // Sixteen times the workgroups, same bound: the dispatch count tracks
        // the worker pool, not the grid. A per-row dispatch would be 1024x
        // this.
        let wide = KernelIr {
            block: 4,
            grid: [N / 4, 1, 1],
            ..ir.clone()
        };
        crate::launch::reset_dispatch_count();
        let _ = run_f32(&wide, &[], N as usize);
        let big = crate::launch::dispatch_count();
        assert!(
            big <= bound,
            "{big} dispatches at grid {} exceeds the pool bound {bound}",
            N / 4
        );
    }

    #[test]
    fn a_divergent_if_merges_both_arms_under_a_mask() {
        const BLOCK: u32 = 32;
        let uni = decl(0, ScalarElement::U32, 1, false);
        let out = decl(1, ScalarElement::F32, BLOCK, true);
        let even = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Eq,
                left: ubin(BinOp::Rem, lane(), ulit(2)),
                right: ulit(0),
            },
            bool_ty(),
        );
        let ir = KernelIr {
            buffers: vec![uni, out.clone()],
            grid: [1, 1, 1],
            block: BLOCK,
            body: vec![Stmt::If {
                condition: even,
                accept: vec![Stmt::Store {
                    dst: view(&out),
                    addr: Addr::Linear(lane()),
                    value: flit(1.0),
                    mask: tlit(),
                }],
                reject: vec![Stmt::Store {
                    dst: view(&out),
                    addr: Addr::Linear(lane()),
                    value: flit(-1.0),
                    mask: tlit(),
                }],
            }],
            byte_arena: None,
            name: "divergent",
        };
        let got = run_f32(&ir, &[], BLOCK as usize);
        for i in 0..BLOCK as usize {
            assert_eq!(got[i], if i % 2 == 0 { 1.0 } else { -1.0 }, "lane {i}");
        }
    }
    /// **Loop accumulators step simultaneously.**
    ///
    /// Two accumulators that swap — `a' = b`, `b' = a` — must still swap after N
    /// iterations. Writing them back one at a time makes `b'` read the already
    /// updated `a`, so both collapse to the original `b`; a `(n, mean, m2)`
    /// carrier hits the same edge, where `mean`'s update reads `n` and the
    /// variance comes back about half right rather than obviously broken.
    #[test]
    fn loop_accumulators_read_the_values_they_entered_the_step_with() {
        let uni = decl(0, ScalarElement::U32, 1, false);
        let src = decl(1, ScalarElement::F32, 2, false);
        let out = decl(2, ScalarElement::F32, 2, true);

        let a = Arc::new(LocalDecl::new(f32_ty()));
        let b = Arc::new(LocalDecl::new(f32_ty()));
        let read = |l: &Arc<LocalDecl>| TileExpr::new(TileExprKind::LoadLocal(Arc::clone(l)), f32_ty());
        let accumulators = vec![
            Accumulator {
                local: Arc::clone(&a),
                init: fload(&src, ulit(0)),
                update: read(&b),
            },
            Accumulator {
                local: Arc::clone(&b),
                init: fload(&src, ulit(1)),
                update: read(&a),
            },
        ];
        let body = vec![
            Stmt::Loop {
                count: Some(ulit(3)),
                index: None,
                accumulators,
                body: Vec::new(),
            },
            Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(ulit(0)),
                value: read(&a),
                mask: cmp_eq_lane_zero(),
            },
            Stmt::Store {
                dst: view(&out),
                addr: Addr::Linear(ulit(1)),
                value: read(&b),
                mask: cmp_eq_lane_zero(),
            },
        ];
        let ir = KernelIr {
            buffers: vec![uni, src, out],
            grid: [1, 1, 1],
            block: 8,
            body,
            byte_arena: None,
            name: "swap",
        };
        let got = run_f32(&ir, &[vec![0.0], vec![1.0, 2.0]], 2);
        // Three swaps of (1, 2) give (2, 1).
        assert_eq!(
            got,
            vec![2.0, 1.0],
            "sequential write-back would give (2, 2)"
        );
    }

    fn cmp_eq_lane_zero() -> TileExpr {
        TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Eq,
                left: lane(),
                right: ulit(0),
            },
            bool_ty(),
        )
    }
}
