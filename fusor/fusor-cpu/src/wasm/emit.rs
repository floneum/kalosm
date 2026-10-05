//! `Program` -> a wasm module with one exported function:
//! `run(bufs, gid.x, gid.y, gid.z, grid.x, grid.y, grid.z, frame)`.
//!
//! The lowering is the Cranelift emitter's, instruction for instruction: one
//! `i32` local per tape slot (f32 values travel as their bit patterns and are
//! reinterpreted around arithmetic), lane loops as `block`/`loop`, per-lane
//! locals and workgroup tiles in a caller-provided `frame` of host memory, and
//! the same eight host helpers for everything not inlined. `bufs` points at an
//! array of wasm32 `RawBuf`s (`ptr: i32, bytes: i32`).

use fusor_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor_ir::scalar::{BinOp, CmpOp, UnOp};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction as I, MemArg, MemoryType, Module, TypeSection, ValType,
};

use crate::emit::Program;
use crate::emit::expr::{Instr, NumTy, UniformSrc};
use crate::emit::stmt::CStmt;

/// The imported helpers in import order: `(name, parameters, returns a value)`.
pub(crate) const HELPERS: [(&str, usize, bool); 8] = [
    ("fusor2_jit_read", 6, true),
    ("fusor2_jit_write", 6, false),
    ("fusor2_jit_un", 3, true),
    ("fusor2_jit_bin", 4, true),
    ("fusor2_jit_cast", 3, true),
    ("fusor2_jit_round", 2, true),
    ("fusor2_jit_narrow", 2, true),
    ("fusor2_jit_unpack", 2, true),
];
const READ: u32 = 0;
const WRITE: u32 = 1;
const UN: u32 = 2;
const BIN: u32 = 3;
const CAST: u32 = 4;
const ROUND: u32 = 5;
const NARROW: u32 = 6;
const UNPACK: u32 = 7;
/// The entry's parameters: bufs, gid x/y/z, grid x/y/z, frame.
pub(crate) const ENTRY_PARAMS: u32 = 8;
const BUFS: u32 = 0;
const GID: [u32; 3] = [1, 2, 3];
const GRID: [u32; 3] = [4, 5, 6];
const FRAME: u32 = 7;
/// Size of a wasm32 `RawBuf`.
const RAW_BUF: u64 = 8;

fn locals_bytes(prog: &Program) -> usize {
    prog.locals * prog.block as usize * 4
}

/// Scratch bytes the entry's `frame` parameter must point at: per-lane locals
/// followed by the workgroup tiles.
pub(crate) fn frame_bytes(prog: &Program) -> usize {
    (locals_bytes(prog) + prog.arena_bytes as usize).max(4)
}

/// Encode `prog` as a wasm module.
pub(crate) fn module(prog: &Program) -> Result<Vec<u8>, String> {
    let mut types = TypeSection::new();
    for (_, params, returns) in HELPERS {
        types.ty().function(
            std::iter::repeat_n(ValType::I32, params),
            if returns { vec![ValType::I32] } else { vec![] },
        );
    }
    let entry_type = HELPERS.len() as u32;
    types
        .ty()
        .function(std::iter::repeat_n(ValType::I32, ENTRY_PARAMS as usize), []);
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    for (i, (name, ..)) in HELPERS.iter().enumerate() {
        imports.import("env", name, EntityType::Function(i as u32));
    }
    let mut functions = FunctionSection::new();
    functions.function(entry_type);
    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, HELPERS.len() as u32);
    let body = Emitter::new(prog).function()?;
    let mut code = CodeSection::new();
    code.function(&body);
    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&functions)
        .section(&exports)
        .section(&code);
    Ok(module.finish())
}

struct Emitter<'a> {
    prog: &'a Program,
    code: Vec<I<'static>>,
    /// Temporaries allocated past the tape registers.
    temps: u32,
}

/// The local holding tape slot `slot`.
fn reg(slot: u32) -> u32 {
    ENTRY_PARAMS + slot
}

fn mem(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 2,
        memory_index: 0,
    }
}

fn is_word(elem: ScalarElement) -> bool {
    matches!(
        elem,
        ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool
    )
}

impl<'a> Emitter<'a> {
    fn new(prog: &'a Program) -> Self {
        Self {
            prog,
            code: Vec::new(),
            temps: 0,
        }
    }

    fn function(mut self) -> Result<Function, String> {
        let prog = self.prog;
        // Collective statements run once for the workgroup, at lane 0.
        let lane0 = self.temp();
        for segment in &prog.segments {
            if segment.iter().any(CStmt::is_collective) {
                for stmt in segment {
                    self.stmt(stmt, lane0)?;
                }
            } else {
                self.lanes(segment)?;
            }
        }
        self.code.push(I::End);
        let mut function = Function::new_with_locals_types(std::iter::repeat_n(
            ValType::I32,
            prog.regs + self.temps as usize,
        ));
        for instruction in &self.code {
            function.instruction(instruction);
        }
        Ok(function)
    }

    fn temp(&mut self) -> u32 {
        let local = ENTRY_PARAMS + self.prog.regs as u32 + self.temps;
        self.temps += 1;
        local
    }
    fn get(&mut self, local: u32) {
        self.code.push(I::LocalGet(local));
    }
    fn set(&mut self, local: u32) {
        self.code.push(I::LocalSet(local));
    }
    fn c(&mut self, value: u32) {
        self.code.push(I::I32Const(value as i32));
    }
    /// Push a register as f32.
    fn f(&mut self, local: u32) {
        self.get(local);
        self.code.push(I::F32ReinterpretI32);
    }
    /// Push an f32 constant (as reinterpreted bits).
    fn fc(&mut self, value: f32) {
        self.c(value.to_bits());
        self.code.push(I::F32ReinterpretI32);
    }
    /// Store the f32 on the stack into an i32 register.
    fn fset(&mut self, local: u32) {
        self.code.push(I::I32ReinterpretF32);
        self.set(local);
    }
    /// `0 - cond`: a 0/1 comparison becomes the 0/-1 lane mask.
    fn mask(&mut self, cmp: impl FnOnce(&mut Self)) {
        self.c(0);
        cmp(self);
        self.code.push(I::I32Sub);
    }
    /// `select` on the condition just pushed, taken through a local: wasmi
    /// 2.0 mis-executes a `select` fused with the comparison feeding it.
    fn select_pushed(&mut self) {
        let cond = self.temp();
        self.set(cond);
        self.get(cond);
        self.code.push(I::Select);
    }
    /// `select(cond ? t : f)` over locals.
    fn select(&mut self, t: u32, f: u32, cond: u32) {
        self.get(t);
        self.get(f);
        self.get(cond);
        self.code.push(I::Select);
    }

    /// `for (i = 0; i < limit; i += step) body(i)`.
    fn const_loop(
        &mut self,
        limit: u32,
        step: u32,
        body: impl FnOnce(&mut Self, u32) -> Result<(), String>,
    ) -> Result<(), String> {
        let i = self.temp();
        self.c(0);
        self.set(i);
        self.code.push(I::Block(BlockType::Empty));
        self.code.push(I::Loop(BlockType::Empty));
        self.get(i);
        self.c(limit);
        self.code.push(I::I32GeU);
        self.code.push(I::BrIf(1));
        body(self, i)?;
        self.get(i);
        self.c(step);
        self.code.push(I::I32Add);
        self.set(i);
        self.code.push(I::Br(0));
        self.code.push(I::End);
        self.code.push(I::End);
        Ok(())
    }

    fn lanes(&mut self, stmts: &[CStmt]) -> Result<(), String> {
        let block = self.prog.block;
        self.const_loop(block, 1, |e, lane| {
            for stmt in stmts {
                e.stmt(stmt, lane)?;
            }
            Ok(())
        })
    }

    fn stmt(&mut self, stmt: &CStmt, lane: u32) -> Result<(), String> {
        match stmt {
            CStmt::Store {
                prep,
                buf,
                elem,
                index,
                value,
                mask,
            } => {
                self.range(prep, lane)?;
                self.bound_store(*buf, *elem, reg(*index), reg(*value), reg(*mask));
            }
            CStmt::StoreLocal { prep, local, value } => {
                self.range(prep, lane)?;
                self.store_local(*local, lane, reg(*value));
            }
            CStmt::StoreTile {
                prep,
                tile,
                elem: ScalarElement::F32,
                index,
                value,
            } => {
                self.range(prep, lane)?;
                self.store_tile(*tile, reg(*index), reg(*value));
            }
            CStmt::If {
                prep,
                cond,
                accept,
                reject,
                ..
            } => {
                self.range(prep, lane)?;
                self.get(reg(*cond));
                self.code.push(I::If(BlockType::Empty));
                for stmt in accept {
                    self.stmt(stmt, lane)?;
                }
                self.code.push(I::Else);
                for stmt in reject {
                    self.stmt(stmt, lane)?;
                }
                self.code.push(I::End);
            }
            CStmt::Loop {
                prep,
                count: Some(count),
                index,
                accs,
                body,
            } => {
                self.range(prep, lane)?;
                // Every tape slot is its own local, so all initial values can
                // be evaluated before any accumulator is written.
                for acc in accs {
                    self.range(&acc.init_prep, lane)?;
                }
                for acc in accs {
                    self.store_local(acc.local, lane, reg(acc.init));
                }
                let iteration = self.temp();
                self.c(0);
                self.set(iteration);
                self.code.push(I::Block(BlockType::Empty));
                self.code.push(I::Loop(BlockType::Empty));
                self.get(iteration);
                self.get(reg(*count));
                self.code.push(I::I32GeU);
                self.code.push(I::BrIf(1));
                if let Some(index) = index {
                    self.store_local(*index, lane, iteration);
                }
                for stmt in body {
                    self.stmt(stmt, lane)?;
                }
                // Every update observes the old accumulator tuple.
                for acc in accs {
                    self.range(&acc.update_prep, lane)?;
                }
                for acc in accs {
                    self.store_local(acc.local, lane, reg(acc.update));
                }
                self.get(iteration);
                self.c(1);
                self.code.push(I::I32Add);
                self.set(iteration);
                self.code.push(I::Br(0));
                self.code.push(I::End);
                self.code.push(I::End);
            }
            CStmt::StageTree {
                prep,
                tile,
                value,
                op,
                group,
            } => {
                self.carrier_tree(
                    prep,
                    &[*tile],
                    &[*value],
                    &[],
                    &[],
                    &(0..0),
                    &[],
                    &[],
                    *group,
                    Some(*op),
                )?;
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
            } if tiles.len() == values.len()
                && tiles.len() == lhs.len()
                && tiles.len() == rhs.len()
                && tiles.len() == merged.len()
                && tiles.len() == outs.len() =>
            {
                self.carrier_tree(
                    prep, tiles, values, lhs, rhs, merge_prep, merged, outs, *group, *fast,
                )?;
            }
            CStmt::Lanes(body) => self.lanes(body)?,
            _ => return Err(format!("unsupported wasm statement {stmt:?}")),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn carrier_tree(
        &mut self,
        prep: &std::ops::Range<u32>,
        tiles: &[u16],
        values: &[u32],
        lhs: &[u16],
        rhs: &[u16],
        merge_prep: &std::ops::Range<u32>,
        merged: &[u32],
        outs: &[u16],
        group: u32,
        fast: Option<TileReduceOp>,
    ) -> Result<(), String> {
        if !group.is_power_of_two()
            || tiles
                .iter()
                .any(|tile| self.prog.tiles[*tile as usize].elem != ScalarElement::F32)
        {
            return Err("wasm reduction requires F32 scratch and a power-of-two group".into());
        }
        let block = self.prog.block;
        // Stage one partial per logical lane.
        self.const_loop(block, 1, |e, lane| {
            e.range(prep, lane)?;
            for (tile, value) in tiles.iter().zip(values) {
                e.store_tile(*tile, lane, reg(*value));
            }
            Ok(())
        })?;
        // Reduce every group in place: for each group base, halve the stride
        // and fold every pair.
        let left = self.temp();
        let right = self.temp();
        let lefts: Vec<u32> = tiles.iter().map(|_| self.temp()).collect();
        let rights: Vec<u32> = tiles.iter().map(|_| self.temp()).collect();
        let result = self.temp();
        let lane0 = self.temp();
        self.const_loop(block, group, |e, base| {
            let stride = e.temp();
            e.c(group / 2);
            e.set(stride);
            e.code.push(I::Block(BlockType::Empty));
            e.code.push(I::Loop(BlockType::Empty));
            e.get(stride);
            e.code.push(I::I32Eqz);
            e.code.push(I::BrIf(1));
            let pair = e.temp();
            e.c(0);
            e.set(pair);
            e.code.push(I::Block(BlockType::Empty));
            e.code.push(I::Loop(BlockType::Empty));
            e.get(pair);
            e.get(stride);
            e.code.push(I::I32GeU);
            e.code.push(I::BrIf(1));
            e.get(base);
            e.get(pair);
            e.code.push(I::I32Add);
            e.set(left);
            e.get(left);
            e.get(stride);
            e.code.push(I::I32Add);
            e.set(right);
            for (i, tile) in tiles.iter().enumerate() {
                e.load_tile(*tile, left);
                e.set(lefts[i]);
                e.load_tile(*tile, right);
                e.set(rights[i]);
            }
            if let Some(op) = fast {
                for (i, tile) in tiles.iter().enumerate() {
                    e.bin(op.binary(), NumTy::F32, lefts[i], rights[i]);
                    e.set(result);
                    e.store_tile(*tile, left, result);
                }
            } else {
                for (i, (local_lhs, local_rhs)) in lhs.iter().zip(rhs).enumerate() {
                    e.store_local(*local_lhs, lane0, lefts[i]);
                    e.store_local(*local_rhs, lane0, rights[i]);
                }
                e.range(merge_prep, lane0)?;
                for (tile, slot) in tiles.iter().zip(merged) {
                    e.store_tile(*tile, left, reg(*slot));
                }
            }
            e.get(pair);
            e.c(1);
            e.code.push(I::I32Add);
            e.set(pair);
            e.code.push(I::Br(0));
            e.code.push(I::End);
            e.code.push(I::End);
            e.get(stride);
            e.c(1);
            e.code.push(I::I32ShrU);
            e.set(stride);
            e.code.push(I::Br(0));
            e.code.push(I::End);
            e.code.push(I::End);
            Ok(())
        })?;
        // Materialize each group's result into the output locals.
        let group_base = self.temp();
        self.const_loop(block, 1, |e, lane| {
            if group == 1 {
                e.get(lane);
            } else {
                e.get(lane);
                e.c(group);
                e.code.push(I::I32DivU);
                e.c(group);
                e.code.push(I::I32Mul);
            }
            e.set(group_base);
            for (tile, out) in tiles.iter().zip(outs) {
                e.load_tile(*tile, group_base);
                e.set(result);
                e.store_local(*out, lane, result);
            }
            Ok(())
        })
    }

    /// Evaluate a tape range at `lane`.
    fn range(&mut self, range: &std::ops::Range<u32>, lane: u32) -> Result<(), String> {
        let prog = self.prog;
        for pc in range.clone() {
            self.instr(&prog.tape[pc as usize], lane)?;
        }
        Ok(())
    }

    fn instr(&mut self, instr: &Instr, lane: u32) -> Result<(), String> {
        let out = reg(instr.out());
        match instr {
            Instr::Const { bits, .. } => {
                self.c(*bits);
                self.set(out);
            }
            Instr::LaneId { .. } => {
                self.get(lane);
                self.set(out);
            }
            Instr::Uniform { which, .. } => {
                let (block, width) = (self.prog.block, self.prog.width);
                match which {
                    UniformSrc::ProgramX => self.get(GID[0]),
                    UniformSrc::ProgramY => self.get(GID[1]),
                    UniformSrc::ProgramZ => self.get(GID[2]),
                    UniformSrc::GridX => self.get(GRID[0]),
                    UniformSrc::GridY => self.get(GRID[1]),
                    UniformSrc::GridZ => self.get(GRID[2]),
                    UniformSrc::SubgroupSize => self.c(width),
                    UniformSrc::NumSubgroups => self.c(block.div_ceil(width)),
                    UniformSrc::SubgroupId => {
                        self.get(lane);
                        self.c(width);
                        self.code.push(I::I32DivU);
                    }
                    UniformSrc::SubgroupLane => {
                        self.get(lane);
                        self.c(width);
                        self.code.push(I::I32RemU);
                    }
                }
                self.set(out);
            }
            Instr::LoadLocal { local, .. } => {
                self.local_address(*local, lane);
                self.code.push(I::I32Load(mem(0)));
                self.set(out);
            }
            Instr::LoadTile {
                tile,
                elem: ScalarElement::F32,
                index,
                ..
            } => {
                self.load_tile(*tile, reg(*index));
                self.set(out);
            }
            Instr::Load {
                buf,
                elem,
                index,
                mask,
                fill,
                ..
            } => {
                let (ptr, bytes) = self.raw_buf(*buf);
                if is_word(*elem) {
                    self.direct_load(ptr, bytes, reg(*index), reg(*mask), reg(*fill));
                } else {
                    self.get(ptr);
                    self.get(bytes);
                    self.get(reg(*index));
                    self.get(reg(*mask));
                    self.get(reg(*fill));
                    self.c(*elem as u32);
                    self.code.push(I::Call(READ));
                }
                self.set(out);
            }
            Instr::Un { op, x, ty, .. } => {
                self.un(*op, *ty, reg(*x));
                self.set(out);
            }
            Instr::Bin { op, a, b, ty, .. } => {
                self.bin(*op, *ty, reg(*a), reg(*b));
                self.set(out);
            }
            Instr::Fma { a, b, c, .. } => {
                self.f(reg(*a));
                self.f(reg(*b));
                self.code.push(I::F32Mul);
                self.f(reg(*c));
                self.code.push(I::F32Add);
                self.fset(out);
            }
            Instr::Cmp { op, a, b, ty, .. } => {
                self.cmp(*op, *ty, reg(*a), reg(*b));
                self.set(out);
            }
            Instr::MaskToValue { x, ty, .. } => {
                self.c(match ty {
                    NumTy::F32 => 1.0f32.to_bits(),
                    _ => 1,
                });
                self.c(0);
                self.get(reg(*x));
                self.code.push(I::Select);
                self.set(out);
            }
            Instr::ValueToMask { x, ty, .. } => {
                let x = reg(*x);
                self.mask(|e| match ty {
                    NumTy::F32 => {
                        e.f(x);
                        e.fc(0.0);
                        e.code.push(I::F32Ne);
                    }
                    _ => {
                        e.get(x);
                        e.c(0);
                        e.code.push(I::I32Ne);
                    }
                });
                self.set(out);
            }
            Instr::Round { mode, x, .. } => {
                self.c(*mode as u32);
                self.get(reg(*x));
                self.code.push(I::Call(ROUND));
                self.set(out);
            }
            Instr::Cast { x, from, to, .. } => {
                self.c(*from as u32);
                self.c(*to as u32);
                self.get(reg(*x));
                self.code.push(I::Call(CAST));
                self.set(out);
            }
            Instr::Narrow { x, to, .. } => {
                self.c(*to as u32);
                self.get(reg(*x));
                self.code.push(I::Call(NARROW));
                self.set(out);
            }
            Instr::Bitcast { x, .. } => {
                self.get(reg(*x));
                self.set(out);
            }
            Instr::Select { c, t, f, .. } => {
                self.get(reg(*t));
                self.get(reg(*f));
                // `select` already takes any nonzero condition.
                self.get(reg(*c));
                self.code.push(I::Select);
                self.set(out);
            }
            Instr::VecCompose { parts, .. } => {
                for (i, part) in parts.iter().enumerate() {
                    self.get(reg(*part));
                    self.set(out + i as u32);
                }
            }
            Instr::VecComponent {
                base, component, ..
            } => {
                self.get(reg(*base + *component));
                self.set(out);
            }
            Instr::Unpack2x16 { x, .. } => {
                for high in 0..2u32 {
                    self.get(reg(*x));
                    self.c(high);
                    self.code.push(I::Call(UNPACK));
                    self.set(out + high);
                }
            }
            _ => return Err(format!("unsupported wasm instruction {instr:?}")),
        }
        Ok(())
    }

    /// Push the `(ptr, bytes)` of bound buffer `buf` into two temporaries.
    fn raw_buf(&mut self, buf: u16) -> (u32, u32) {
        let ptr = self.temp();
        let bytes = self.temp();
        self.get(BUFS);
        self.code.push(I::I32Load(mem(buf as u64 * RAW_BUF)));
        self.set(ptr);
        self.get(BUFS);
        self.code.push(I::I32Load(mem(buf as u64 * RAW_BUF + 4)));
        self.set(bytes);
        (ptr, bytes)
    }

    /// `valid = index < bytes / 4 && mask != 0` and `address = ptr + 4 * index`,
    /// into two temporaries.
    fn word_address(&mut self, ptr: u32, bytes: u32, index: u32, mask: u32) -> (u32, u32) {
        let valid = self.temp();
        let address = self.temp();
        self.get(index);
        self.get(bytes);
        self.c(2);
        self.code.push(I::I32ShrU);
        self.code.push(I::I32LtU);
        self.get(mask);
        self.c(0);
        self.code.push(I::I32Ne);
        self.code.push(I::I32And);
        self.set(valid);
        self.get(ptr);
        self.get(index);
        self.c(2);
        self.code.push(I::I32Shl);
        self.code.push(I::I32Add);
        self.set(address);
        (valid, address)
    }

    /// The common masked 32-bit load: an invalid lane reads element zero as a
    /// safe speculative address and the select returns its fill.
    fn direct_load(&mut self, ptr: u32, bytes: u32, index: u32, mask: u32, fill: u32) {
        let (valid, address) = self.word_address(ptr, bytes, index, mask);
        self.select(address, ptr, valid);
        self.code.push(I::I32Load(mem(0)));
        self.get(fill);
        self.get(valid);
        self.code.push(I::Select);
    }

    fn bound_store(&mut self, buf: u16, elem: ScalarElement, index: u32, value: u32, mask: u32) {
        let (ptr, bytes) = self.raw_buf(buf);
        if is_word(elem) {
            let (valid, address) = self.word_address(ptr, bytes, index, mask);
            self.get(valid);
            self.code.push(I::If(BlockType::Empty));
            self.get(address);
            self.get(value);
            self.code.push(I::I32Store(mem(0)));
            self.code.push(I::End);
        } else {
            self.get(ptr);
            self.get(bytes);
            self.get(index);
            self.get(value);
            self.get(mask);
            self.c(elem as u32);
            self.code.push(I::Call(WRITE));
        }
    }

    /// Push `frame + 4 * (local * block + lane)`.
    fn local_address(&mut self, local: u16, lane: u32) {
        self.get(FRAME);
        self.get(lane);
        self.c(local as u32 * self.prog.block);
        self.code.push(I::I32Add);
        self.c(2);
        self.code.push(I::I32Shl);
        self.code.push(I::I32Add);
    }
    fn store_local(&mut self, local: u16, lane: u32, value: u32) {
        self.local_address(local, lane);
        self.get(value);
        self.code.push(I::I32Store(mem(0)));
    }
    /// Push `frame + 4 * index`; the tile's placement is the static offset.
    fn tile_offset(&mut self, tile: u16, index: u32) -> u64 {
        self.get(FRAME);
        self.get(index);
        self.c(2);
        self.code.push(I::I32Shl);
        self.code.push(I::I32Add);
        (locals_bytes(self.prog) + self.prog.tiles[tile as usize].byte_offset as usize) as u64
    }
    fn load_tile(&mut self, tile: u16, index: u32) {
        let offset = self.tile_offset(tile, index);
        self.code.push(I::I32Load(mem(offset)));
    }
    fn store_tile(&mut self, tile: u16, index: u32, value: u32) {
        let offset = self.tile_offset(tile, index);
        self.get(value);
        self.code.push(I::I32Store(mem(offset)));
    }

    fn bin(&mut self, op: BinOp, ty: NumTy, a: u32, b: u32) {
        let helper = |e: &mut Self| {
            e.c(op as u32);
            e.c(ty as u32);
            e.get(a);
            e.get(b);
            e.code.push(I::Call(BIN));
        };
        if matches!(op, BinOp::Pow) {
            return helper(self);
        }
        match ty {
            NumTy::F32 => {
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        self.f(a);
                        self.f(b);
                        self.code.push(match op {
                            BinOp::Add => I::F32Add,
                            BinOp::Sub => I::F32Sub,
                            BinOp::Mul => I::F32Mul,
                            _ => I::F32Div,
                        });
                    }
                    BinOp::Rem => return helper(self),
                    // `y < x ? y : x`, the comparison-select form, not the
                    // NaN-propagating hardware minimum.
                    BinOp::Min | BinOp::Max => {
                        self.f(b);
                        self.f(a);
                        self.f(b);
                        self.f(a);
                        self.code
                            .push(if op == BinOp::Min { I::F32Lt } else { I::F32Gt });
                        self.select_pushed();
                    }
                    BinOp::LogicalAnd | BinOp::LogicalOr => {
                        self.c(1.0f32.to_bits());
                        self.c(0);
                        self.f(a);
                        self.fc(0.0);
                        self.code.push(I::F32Ne);
                        self.f(b);
                        self.fc(0.0);
                        self.code.push(I::F32Ne);
                        self.code.push(if op == BinOp::LogicalAnd {
                            I::I32And
                        } else {
                            I::I32Or
                        });
                        self.select_pushed();
                        return;
                    }
                    BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::Shr
                    | BinOp::Shl
                    | BinOp::Pow => unreachable!(),
                }
                self.code.push(I::I32ReinterpretF32);
            }
            NumTy::U32 | NumTy::I32 => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    self.get(a);
                    self.get(b);
                    self.code.push(match op {
                        BinOp::Add => I::I32Add,
                        BinOp::Sub => I::I32Sub,
                        _ => I::I32Mul,
                    });
                }
                BinOp::Div | BinOp::Rem => self.int_divrem(op, ty, a, b),
                BinOp::Min | BinOp::Max => {
                    self.get(a);
                    self.get(b);
                    self.get(a);
                    self.get(b);
                    self.code.push(match (ty, op) {
                        (NumTy::U32, BinOp::Min) => I::I32LtU,
                        (NumTy::U32, _) => I::I32GtU,
                        (_, BinOp::Min) => I::I32LtS,
                        _ => I::I32GtS,
                    });
                    self.select_pushed();
                }
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    self.get(a);
                    self.get(b);
                    self.code.push(match op {
                        BinOp::BitAnd => I::I32And,
                        BinOp::BitOr => I::I32Or,
                        BinOp::BitXor => I::I32Xor,
                        BinOp::Shl => I::I32Shl,
                        _ if ty == NumTy::U32 => I::I32ShrU,
                        _ => I::I32ShrS,
                    });
                }
                BinOp::LogicalAnd | BinOp::LogicalOr => {
                    self.get(a);
                    self.c(0);
                    self.code.push(I::I32Ne);
                    self.get(b);
                    self.c(0);
                    self.code.push(I::I32Ne);
                    self.code.push(if op == BinOp::LogicalAnd {
                        I::I32And
                    } else {
                        I::I32Or
                    });
                }
                BinOp::Pow => unreachable!(),
            },
        }
    }

    /// Total integer division, as the Cranelift emitter defines it: `x / 0 ==
    /// MAX`, `x % 0 == 0`, signed overflow wraps. wasm traps on both.
    fn int_divrem(&mut self, op: BinOp, ty: NumTy, a: u32, b: u32) {
        let by_zero = self.temp();
        let safe = self.temp();
        let value = self.temp();
        self.get(b);
        self.code.push(I::I32Eqz);
        self.set(by_zero);
        if ty == NumTy::U32 {
            self.c(1);
            self.get(b);
            self.get(by_zero);
            self.code.push(I::Select);
            self.set(safe);
            self.get(a);
            self.get(safe);
            self.code.push(if op == BinOp::Div {
                I::I32DivU
            } else {
                I::I32RemU
            });
            self.set(value);
            self.c(if op == BinOp::Div { u32::MAX } else { 0 });
            self.get(value);
            self.get(by_zero);
            self.code.push(I::Select);
            return;
        }
        let overflow = self.temp();
        let invalid = self.temp();
        self.get(a);
        self.c(i32::MIN as u32);
        self.code.push(I::I32Eq);
        self.get(b);
        self.c(u32::MAX);
        self.code.push(I::I32Eq);
        self.code.push(I::I32And);
        self.set(overflow);
        self.get(by_zero);
        self.get(overflow);
        self.code.push(I::I32Or);
        self.set(invalid);
        self.c(1);
        self.get(b);
        self.get(invalid);
        self.code.push(I::Select);
        self.set(safe);
        self.get(a);
        self.get(safe);
        self.code.push(if op == BinOp::Div {
            I::I32DivS
        } else {
            I::I32RemS
        });
        self.set(value);
        if op == BinOp::Rem {
            self.c(0);
            self.get(value);
            self.get(invalid);
            self.code.push(I::Select);
            return;
        }
        self.c(i32::MIN as u32);
        self.get(value);
        self.get(overflow);
        self.code.push(I::Select);
        self.set(value);
        self.c(u32::MAX);
        self.get(value);
        self.get(by_zero);
        self.code.push(I::Select);
    }

    fn un(&mut self, op: UnOp, ty: NumTy, x: u32) {
        match (op, ty) {
            (UnOp::Abs, NumTy::F32) => {
                self.f(x);
                self.code.push(I::F32Abs);
                self.code.push(I::I32ReinterpretF32);
            }
            (UnOp::Neg, NumTy::F32) => {
                self.f(x);
                self.code.push(I::F32Neg);
                self.code.push(I::I32ReinterpretF32);
            }
            (UnOp::Sqrt, NumTy::F32) => {
                self.f(x);
                self.code.push(I::F32Sqrt);
                self.code.push(I::I32ReinterpretF32);
            }
            (UnOp::InverseSqrt, NumTy::F32) => {
                self.fc(1.0);
                self.f(x);
                self.code.push(I::F32Sqrt);
                self.code.push(I::F32Div);
                self.code.push(I::I32ReinterpretF32);
            }
            (UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp, NumTy::F32) => {
                self.expf(x);
            }
            (UnOp::Abs, NumTy::U32) => self.get(x),
            (UnOp::Neg, NumTy::U32 | NumTy::I32) => {
                self.c(0);
                self.get(x);
                self.code.push(I::I32Sub);
            }
            (UnOp::Abs, NumTy::I32) => {
                self.c(0);
                self.get(x);
                self.code.push(I::I32Sub);
                self.get(x);
                self.get(x);
                self.c(0);
                self.code.push(I::I32LtS);
                self.select_pushed();
            }
            _ => {
                self.c(op as u32);
                self.c(ty as u32);
                self.get(x);
                self.code.push(I::Call(UN));
            }
        }
    }

    /// The Cody-Waite `exp` the Cranelift emitter inlines, including its
    /// subnormal tail and saturation.
    fn expf(&mut self, bits: u32) {
        let n = self.temp();
        let r = self.temp();
        let p = self.temp();
        let exponent = self.temp();
        let normal = self.temp();
        let rest = self.temp();
        let too_low = self.temp();
        let subnormal = self.temp();
        let value = self.temp();
        // n = nearest(x * log2 e); r = x - n * ln2 (hi + lo).
        self.f(bits);
        self.fc(std::f32::consts::LOG2_E);
        self.code.push(I::F32Mul);
        self.code.push(I::F32Nearest);
        self.fset(n);
        self.f(bits);
        self.f(n);
        self.fc(0.693_145_75);
        self.code.push(I::F32Mul);
        self.code.push(I::F32Sub);
        self.f(n);
        self.fc(1.428_606_8e-6);
        self.code.push(I::F32Mul);
        self.code.push(I::F32Sub);
        self.fset(r);
        self.fc(2.480_158_7e-5);
        self.fset(p);
        for coefficient in [
            1.984_127e-4,
            1.388_888_9e-3,
            8.333_333e-3,
            4.166_666_8e-2,
            0.166_666_67,
            0.5,
            1.0,
            1.0,
        ] {
            self.f(p);
            self.f(r);
            self.code.push(I::F32Mul);
            self.fc(coefficient);
            self.code.push(I::F32Add);
            self.fset(p);
        }
        self.f(n);
        self.code.push(I::I32TruncSatF32S);
        self.set(exponent);
        // normal = p * 2^exponent by exponent-bit assembly.
        self.f(p);
        self.get(exponent);
        self.c(127);
        self.code.push(I::I32Add);
        self.c(23);
        self.code.push(I::I32Shl);
        self.code.push(I::F32ReinterpretI32);
        self.code.push(I::F32Mul);
        self.fset(normal);
        // The subnormal construction: p * 2^-100 * 2^(exponent + 100).
        self.get(exponent);
        self.c(100);
        self.code.push(I::I32Add);
        self.set(rest);
        self.get(rest);
        self.c((-126i32) as u32);
        self.code.push(I::I32LtS);
        self.set(too_low);
        self.f(p);
        self.fc(f32::from_bits(27 << 23));
        self.code.push(I::F32Mul);
        self.c((-126i32) as u32);
        self.get(rest);
        self.get(too_low);
        self.code.push(I::Select);
        self.c(127);
        self.code.push(I::I32Add);
        self.c(23);
        self.code.push(I::I32Shl);
        self.code.push(I::F32ReinterpretI32);
        self.code.push(I::F32Mul);
        self.fset(subnormal);
        self.c(0);
        self.get(subnormal);
        self.get(too_low);
        self.code.push(I::Select);
        self.get(normal);
        self.get(exponent);
        self.c((-126i32) as u32);
        self.code.push(I::I32LtS);
        self.select_pushed();
        self.set(value);
        // Saturate: +inf above 88.72, 0 below -103.
        self.c(f32::INFINITY.to_bits());
        self.get(value);
        self.f(bits);
        self.fc(88.72);
        self.code.push(I::F32Gt);
        self.select_pushed();
        self.set(value);
        self.c(0);
        self.get(value);
        self.f(bits);
        self.fc(-103.0);
        self.code.push(I::F32Lt);
        self.select_pushed();
    }

    fn cmp(&mut self, op: CmpOp, ty: NumTy, a: u32, b: u32) {
        self.mask(|e| match ty {
            NumTy::F32 => {
                e.f(a);
                e.f(b);
                e.code.push(match op {
                    CmpOp::Lt => I::F32Lt,
                    CmpOp::Le => I::F32Le,
                    CmpOp::Gt => I::F32Gt,
                    CmpOp::Ge => I::F32Ge,
                    CmpOp::Eq => I::F32Eq,
                    CmpOp::Ne => I::F32Ne,
                });
            }
            NumTy::U32 | NumTy::I32 => {
                e.get(a);
                e.get(b);
                e.code.push(match (ty, op) {
                    (_, CmpOp::Eq) => I::I32Eq,
                    (_, CmpOp::Ne) => I::I32Ne,
                    (NumTy::U32, CmpOp::Lt) => I::I32LtU,
                    (NumTy::U32, CmpOp::Le) => I::I32LeU,
                    (NumTy::U32, CmpOp::Gt) => I::I32GtU,
                    (NumTy::U32, CmpOp::Ge) => I::I32GeU,
                    (_, CmpOp::Lt) => I::I32LtS,
                    (_, CmpOp::Le) => I::I32LeS,
                    (_, CmpOp::Gt) => I::I32GtS,
                    (_, CmpOp::Ge) => I::I32GeS,
                });
            }
        });
    }
}
