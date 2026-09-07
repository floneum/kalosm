//! Cranelift emitter for CPU map, gather, and structured-fold programs.
//!
//! Contracts use the optimized GEMM route. Every other program must produce a
//! native artifact; unsupported IR is rejected during emission.

use std::collections::HashMap;
use std::mem;
use std::sync::{Mutex, OnceLock};

use cranelift_codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlags, StackSlotData, StackSlotKind, UserFuncName, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use fusor_ir::dtype::RoundMode;
use fusor_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor_ir::scalar::{BinOp, CmpOp, UnOp};

use crate::emit::expr::{Instr, NumTy, UniformSrc};
use crate::emit::stmt::CStmt;
use crate::emit::{Program, RawBuf};

type Entry = unsafe extern "C" fn(*const RawBuf, u32, u32, u32, u32, u32, u32);
#[cfg(target_os = "macos")]
type GeluEntry = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize);

#[derive(Copy, Clone, Debug)]
pub struct JitKernel(Entry);

#[cfg(target_os = "macos")]
#[derive(Copy, Clone)]
struct GeluKernel(GeluEntry);

impl JitKernel {
    pub fn run(&self, bufs: &[RawBuf], gid: [u32; 3], grid: [u32; 3]) {
        unsafe {
            (self.0)(
                bufs.as_ptr(),
                gid[0],
                gid[1],
                gid[2],
                grid[0],
                grid[1],
                grid[2],
            );
        }
    }
}

#[cfg(target_os = "macos")]
/// Run the fused-contraction input transform as native Cranelift code. GEMM
/// itself remains in the platform BLAS; this keeps the non-GEMM arithmetic on
/// the same JIT path as standalone maps.
pub(crate) fn gelu_dense(a: &[f32], bias: &[f32], out: &mut [f32]) -> Result<(), String> {
    if a.len() != bias.len() || a.len() != out.len() {
        return Err("dense GELU slices have different lengths".into());
    }
    static KERNEL: OnceLock<Result<GeluKernel, String>> = OnceLock::new();
    let kernel = match KERNEL.get_or_init(compile_gelu) {
        Ok(kernel) => kernel,
        Err(error) => return Err(error.clone()),
    };
    unsafe { (kernel.0)(a.as_ptr(), bias.as_ptr(), out.as_mut_ptr(), out.len()) };
    Ok(())
}

#[cfg(target_os = "macos")]
fn compile_gelu() -> Result<GeluKernel, String> {
    let mut module = JITModule::new(jit_builder()?);
    let ptr_ty = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.extend([
        AbiParam::new(ptr_ty),
        AbiParam::new(ptr_ty),
        AbiParam::new(ptr_ty),
        AbiParam::new(ptr_ty),
    ]);
    let id = module
        .declare_function("fusor2_gelu", Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, id.as_u32());
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = b.create_block();
        let head4 = b.create_block();
        let body4 = b.create_block();
        let tail = b.create_block();
        let tail_body = b.create_block();
        let done = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(head4, ptr_ty);
        b.append_block_param(tail, ptr_ty);
        b.switch_to_block(entry);
        let params = b.block_params(entry).to_vec();
        let zero = b.ins().iconst(ptr_ty, 0);
        b.ins().jump(head4, &[zero.into()]);

        b.switch_to_block(head4);
        let index = b.block_params(head4)[0];
        let remaining = b.ins().isub(params[3], index);
        let four = b.ins().iconst(ptr_ty, 4);
        let enough = b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, remaining, four);
        b.ins().brif(enough, body4, &[], tail, &[index.into()]);

        b.switch_to_block(body4);
        for offset in 0..4 {
            let at = if offset == 0 {
                index
            } else {
                b.ins().iadd_imm(index, offset)
            };
            emit_dense_gelu_at(&mut b, params[0], params[1], params[2], at);
        }
        let next = b.ins().iadd_imm(index, 4);
        b.ins().jump(head4, &[next.into()]);

        b.switch_to_block(tail);
        let index = b.block_params(tail)[0];
        let more = b.ins().icmp(IntCC::UnsignedLessThan, index, params[3]);
        b.ins().brif(more, tail_body, &[], done, &[]);

        b.switch_to_block(tail_body);
        emit_dense_gelu_at(&mut b, params[0], params[1], params[2], index);
        let next = b.ins().iadd_imm(index, 1);
        b.ins().jump(tail, &[next.into()]);

        b.switch_to_block(done);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize();
    }
    module
        .define_function(id, &mut ctx)
        .map_err(|e| format!("{e:?}"))?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let entry =
        unsafe { mem::transmute::<*const u8, GeluEntry>(module.get_finalized_function(id)) };
    Box::leak(Box::new(module));
    Ok(GeluKernel(entry))
}

#[cfg(target_os = "macos")]
fn emit_dense_gelu_at(
    b: &mut FunctionBuilder<'_>,
    a: Value,
    bias: Value,
    out: Value,
    index: Value,
) {
    let offset = b.ins().imul_imm(index, 4);
    let a = b.ins().iadd(a, offset);
    let bias = b.ins().iadd(bias, offset);
    let out = b.ins().iadd(out, offset);
    let a = b.ins().load(types::F32, memory(), a, 0);
    let bias = b.ins().load(types::F32, memory(), bias, 0);
    let x = b.ins().fadd(a, bias);
    let value = emit_gelu_value(b, x);
    b.ins().store(memory(), value, out, 0);
}

#[cfg(target_os = "macos")]
fn emit_gelu_value(b: &mut FunctionBuilder<'_>, x: Value) -> Value {
    let x2 = b.ins().fmul(x, x);
    let x3 = b.ins().fmul(x, x2);
    let cubic_scale = b.ins().f32const(0.044_715);
    let cubic = b.ins().fmul(x3, cubic_scale);
    let sum = b.ins().fadd(x, cubic);
    let sqrt_scale = b.ins().f32const(0.797_884_6);
    let inner = b.ins().fmul(sum, sqrt_scale);

    // The [9/8] Pade form is branchless in the JIT. Four independent lanes
    // are emitted per loop body so their divisions overlap on the CPU.
    let z = b.ins().fmul(inner, inner);
    let numerator = horner(
        b,
        z,
        &[
            2.901_951_3e-8,
            2.872_939_4e-5,
            0.003_921_566_5,
            0.137_254_88,
            1.0,
        ],
    );
    let numerator = b.ins().fmul(inner, numerator);
    let denominator = horner(
        b,
        z,
        &[
            1.305_880_2e-6,
            0.000_402_211_77,
            0.027_450_973,
            0.470_588_2,
            1.0,
        ],
    );
    let rational = b.ins().fdiv(numerator, denominator);
    let one = b.ins().f32const(1.0);
    let sign = b.ins().fcopysign(one, inner);
    let abs = b.ins().fabs(inner);
    let limit = b.ins().f32const(6.0);
    let saturated = b.ins().fcmp(FloatCC::GreaterThanOrEqual, abs, limit);
    let t = b.ins().select(saturated, sign, rational);
    let one_plus_t = b.ins().fadd(one, t);
    let half = b.ins().f32const(0.5);
    let half_x = b.ins().fmul(half, x);
    b.ins().fmul(half_x, one_plus_t)
}

#[cfg(target_os = "macos")]
fn horner(b: &mut FunctionBuilder<'_>, x: Value, coefficients: &[f32]) -> Value {
    let mut value = b.ins().f32const(coefficients[0]);
    for &coefficient in &coefficients[1..] {
        let product = b.ins().fmul(value, x);
        let coefficient = b.ins().f32const(coefficient);
        value = b.ins().fadd(product, coefficient);
    }
    value
}

#[inline(never)]
extern "C" fn jit_read(
    ptr: *const u8,
    bytes: usize,
    index: u32,
    mask: u32,
    fill: u32,
    elem: u32,
) -> u32 {
    let elem = decode_elem(elem);
    if mask == 0 || index as usize >= bytes / elem.byte_size() as usize {
        fill
    } else {
        unsafe { crate::emit::expr::read_elem(elem, ptr, index as usize) }
    }
}

#[inline(never)]
extern "C" fn jit_write(ptr: *mut u8, bytes: usize, index: u32, value: u32, mask: u32, elem: u32) {
    let elem = decode_elem(elem);
    if mask != 0 && (index as usize) < bytes / elem.byte_size() as usize {
        unsafe { crate::emit::expr::write_elem(elem, ptr, index as usize, value) };
    }
}

#[inline(never)]
extern "C" fn jit_un(code: u32, ty: u32, bits: u32) -> u32 {
    let op = decode_un(code);
    let ty = decode_ty(ty);
    crate::emit::expr::apply_un(op, ty, bits)
}

#[inline(never)]
extern "C" fn jit_bin(code: u32, ty: u32, a: u32, b: u32) -> u32 {
    let op = decode_bin(code);
    let ty = decode_ty(ty);
    crate::emit::expr::apply_bin(op, ty, a, b)
}

#[inline(never)]
extern "C" fn jit_cast(from: u32, to: u32, bits: u32) -> u32 {
    crate::emit::expr::apply_cast(decode_ty(from), decode_ty(to), bits)
}

#[inline(never)]
extern "C" fn jit_round(mode: u32, bits: u32) -> u32 {
    let mode = match mode {
        0 => RoundMode::HalfToEven,
        1 => RoundMode::HalfAwayFromZero,
        2 => RoundMode::Floor,
        3 => RoundMode::Ceil,
        _ => RoundMode::Trunc,
    };
    crate::emit::expr::round_mode(mode, f32::from_bits(bits)).to_bits()
}

#[inline(never)]
extern "C" fn jit_narrow(elem: u32, bits: u32) -> u32 {
    crate::emit::expr::apply_narrow(decode_elem(elem), bits)
}

#[inline(never)]
extern "C" fn jit_unpack(bits: u32, high: u32) -> u32 {
    let raw = if high == 0 {
        bits as u16
    } else {
        (bits >> 16) as u16
    };
    half::f16::from_bits(raw).to_f32().to_bits()
}

fn jit_builder() -> Result<JITBuilder, String> {
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .map_err(|e| format!("unsupported host: {e}"))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    Ok(JITBuilder::with_isa(isa, default_libcall_names()))
}

struct NativeModule {
    module: JITModule,
    ptr_ty: cranelift_codegen::ir::Type,
    imports: [cranelift_module::FuncId; 8],
}

impl NativeModule {
    fn new() -> Result<Self, String> {
        let mut jit = jit_builder()?;
        for (name, address) in [
            ("fusor2_jit_read", jit_read as *const u8),
            ("fusor2_jit_write", jit_write as *const u8),
            ("fusor2_jit_un", jit_un as *const u8),
            ("fusor2_jit_bin", jit_bin as *const u8),
            ("fusor2_jit_cast", jit_cast as *const u8),
            ("fusor2_jit_round", jit_round as *const u8),
            ("fusor2_jit_narrow", jit_narrow as *const u8),
            ("fusor2_jit_unpack", jit_unpack as *const u8),
        ] {
            jit.symbol(name, address);
        }
        let mut module = JITModule::new(jit);
        let ptr_ty = module.target_config().pointer_type();
        let rw = [
            ptr_ty,
            ptr_ty,
            types::I32,
            types::I32,
            types::I32,
            types::I32,
        ];
        let imports = [
            import(&mut module, "fusor2_jit_read", &rw, Some(types::I32))?,
            import(&mut module, "fusor2_jit_write", &rw, None)?,
            import(
                &mut module,
                "fusor2_jit_un",
                &[types::I32, types::I32, types::I32],
                Some(types::I32),
            )?,
            import(
                &mut module,
                "fusor2_jit_bin",
                &[types::I32; 4],
                Some(types::I32),
            )?,
            import(
                &mut module,
                "fusor2_jit_cast",
                &[types::I32; 3],
                Some(types::I32),
            )?,
            import(
                &mut module,
                "fusor2_jit_round",
                &[types::I32; 2],
                Some(types::I32),
            )?,
            import(
                &mut module,
                "fusor2_jit_narrow",
                &[types::I32; 2],
                Some(types::I32),
            )?,
            import(
                &mut module,
                "fusor2_jit_unpack",
                &[types::I32; 2],
                Some(types::I32),
            )?,
        ];
        Ok(Self {
            module,
            ptr_ty,
            imports,
        })
    }

    fn function(
        &mut self,
        name: &str,
        namespace: u32,
    ) -> Result<
        (
            cranelift_module::FuncId,
            cranelift_codegen::Context,
            Helpers,
        ),
        String,
    > {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(self.ptr_ty));
        sig.params.extend((0..6).map(|_| AbiParam::new(types::I32)));
        let id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .map_err(|e| e.to_string())?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(namespace, id.as_u32());
        let refs = self
            .imports
            .map(|id| self.module.declare_func_in_func(id, &mut ctx.func));
        let [read, write, un, bin, cast, round, narrow, unpack] = refs;
        Ok((
            id,
            ctx,
            Helpers {
                read,
                write,
                un,
                bin,
                cast,
                round,
                narrow,
                unpack,
            },
        ))
    }

    fn finish(
        mut self,
        id: cranelift_module::FuncId,
        mut ctx: cranelift_codegen::Context,
    ) -> Result<JitKernel, String> {
        self.module
            .define_function(id, &mut ctx)
            .map_err(|e| format!("{e:?}"))?;
        self.module
            .finalize_definitions()
            .map_err(|e| e.to_string())?;
        let entry =
            unsafe { mem::transmute::<*const u8, Entry>(self.module.get_finalized_function(id)) };
        Box::leak(Box::new(self.module));
        Ok(JitKernel(entry))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MapKey {
    tape: Vec<Instr>,
    block: u32,
    width: u32,
    regs: usize,
    stores: Vec<StoreKey>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    prep: std::ops::Range<u32>,
    buf: u16,
    elem: ScalarElement,
    index: u32,
    value: u32,
    mask: u32,
}

impl MapKey {
    fn of(prog: &Program) -> Option<Self> {
        let [segment] = prog.segments.as_slice() else {
            return None;
        };
        if segment.stmts.is_empty() {
            return None;
        }
        let stores = segment
            .stmts
            .iter()
            .map(|stmt| {
                let CStmt::Store {
                    prep,
                    buf,
                    elem,
                    index,
                    value,
                    mask,
                } = stmt
                else {
                    return None;
                };
                Some(StoreKey {
                    prep: prep.clone(),
                    buf: *buf,
                    elem: *elem,
                    index: *index,
                    value: *value,
                    mask: *mask,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let tape_end = stores.iter().map(|store| store.prep.end).max()? as usize;
        if prog.locals != 0 || !prog.tiles.is_empty() || tape_end > prog.tape.len() {
            return None;
        }
        Some(Self {
            tape: prog.tape[..tape_end].to_vec(),
            block: prog.block,
            width: prog.width,
            regs: prog.regs,
            stores,
        })
    }
}

pub(crate) fn compile(prog: &Program) -> Result<Option<JitKernel>, String> {
    static CACHE: OnceLock<Mutex<HashMap<MapKey, JitKernel>>> = OnceLock::new();
    if let Some(key) = MapKey::of(prog) {
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(hit) = cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .copied()
        {
            return Ok(Some(hit));
        }
        if let Some(kernel) = compile_uncached(&key)? {
            cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, kernel);
            return Ok(Some(kernel));
        }
    }
    compile_fold_cached(prog)
}

pub(crate) fn unsupported_reason(prog: &Program) -> String {
    fn stmt_reason(stmt: &CStmt) -> Option<String> {
        match stmt {
            CStmt::Store { .. } | CStmt::StoreLocal { .. } | CStmt::StoreTile { .. } => None,
            CStmt::If { accept, reject, .. } => accept.iter().chain(reject).find_map(stmt_reason),
            CStmt::Loop {
                count: Some(_),
                body,
                ..
            } => body.iter().find_map(stmt_reason),
            CStmt::Loop { count: None, .. } => Some("unbounded loop".into()),
            CStmt::StageTree { group, .. } if *group > 0 && group.is_power_of_two() => None,
            CStmt::CarrierTree {
                tiles,
                values,
                lhs,
                rhs,
                merged,
                outs,
                group,
                ..
            } if *group > 0
                && group.is_power_of_two()
                && tiles.len() == values.len()
                && tiles.len() == lhs.len()
                && tiles.len() == rhs.len()
                && tiles.len() == merged.len()
                && tiles.len() == outs.len() =>
            {
                None
            }
            CStmt::CarrierTree {
                tiles,
                values,
                lhs,
                rhs,
                merged,
                outs,
                group,
                ..
            } => Some(format!(
                "invalid carrier tree group={group}, arities={}/{}/{}/{}/{}/{}",
                tiles.len(),
                values.len(),
                lhs.len(),
                rhs.len(),
                merged.len(),
                outs.len()
            )),
            other => Some(format!("unsupported statement {other:?}")),
        }
    }

    if let Some(tile) = prog
        .tiles
        .iter()
        .find(|tile| tile.elem != ScalarElement::F32)
    {
        return format!("fold scratch is {:?}, not F32", tile.elem);
    }
    if let Some(reason) = prog
        .segments
        .iter()
        .flat_map(|segment| &segment.stmts)
        .find_map(stmt_reason)
    {
        return reason;
    }
    if let Some(instr) = prog.tape.iter().find(|instr| {
        matches!(
            instr,
            Instr::Dot { .. } | Instr::Reduce { .. } | Instr::Rc2Index { .. }
        )
    }) {
        return format!("unsupported fold instruction {instr:?}");
    }
    "program did not match a native map or fold form".into()
}

fn compile_uncached(key: &MapKey) -> Result<Option<JitKernel>, String> {
    if key.tape.iter().any(unsupported) {
        return Ok(None);
    }

    let mut module = NativeModule::new()?;
    let ptr_ty = module.ptr_ty;
    let (id, mut ctx, helpers) = module.function("fusor2_map", 0)?;
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = b.create_block();
        let head = b.create_block();
        let body = b.create_block();
        let done = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(head, types::I32);
        b.switch_to_block(entry);
        let params = b.block_params(entry).to_vec();
        let zero = b.ins().iconst(types::I32, 0);
        b.ins().jump(head, &[zero.into()]);

        b.switch_to_block(head);
        let lane_base = b.block_params(head)[0];
        let limit = b.ins().iconst(types::I32, key.block as i64);
        let more = b.ins().icmp(IntCC::UnsignedLessThan, lane_base, limit);
        b.ins().brif(more, body, &[], done, &[]);

        b.switch_to_block(body);
        // Cranelift intentionally has no loop vectorizer. Unrolling four
        // independent lanes exposes their address arithmetic and scalar math
        // to its scheduler while deleting three quarters of the loop control.
        // CPU blocks are normally powers of two; retain the scalar loop for a
        // non-divisible custom block so no tail mask has to be invented.
        let unroll = if key.block >= 4 && key.block.is_multiple_of(4) {
            4
        } else {
            1
        };
        for offset in 0..unroll {
            let lane = if offset == 0 {
                lane_base
            } else {
                b.ins().iadd_imm(lane_base, offset as i64)
            };
            let mut regs = vec![None; key.regs.max(1)];
            let env = Env {
                bufs: params[0],
                gid: [params[1], params[2], params[3]],
                grid: [params[4], params[5], params[6]],
                lane,
                block: key.block,
                width: key.width,
                ptr_ty,
            };
            for store in &key.stores {
                for instr in &key.tape[store.prep.start as usize..store.prep.end as usize] {
                    emit_instr(&mut b, instr, &mut regs, &env, &helpers)?;
                }
                let (dst_ptr, dst_bytes) = raw_buf(&mut b, params[0], store.buf, ptr_ty);
                if matches!(
                    store.elem,
                    ScalarElement::F32
                        | ScalarElement::U32
                        | ScalarElement::I32
                        | ScalarElement::Bool
                ) {
                    let write = b.create_block();
                    let next_store = b.create_block();
                    let (valid, address) = direct_u32_address(
                        &mut b,
                        dst_ptr,
                        dst_bytes,
                        reg(&regs, store.index)?,
                        reg(&regs, store.mask)?,
                        ptr_ty,
                    );
                    b.ins().brif(valid, write, &[], next_store, &[]);
                    b.switch_to_block(write);
                    b.ins()
                        .store(memory(), reg(&regs, store.value)?, address, 0);
                    b.ins().jump(next_store, &[]);
                    b.switch_to_block(next_store);
                } else {
                    let elem_code = b.ins().iconst(types::I32, elem_code(store.elem) as i64);
                    b.ins().call(
                        helpers.write,
                        &[
                            dst_ptr,
                            dst_bytes,
                            reg(&regs, store.index)?,
                            reg(&regs, store.value)?,
                            reg(&regs, store.mask)?,
                            elem_code,
                        ],
                    );
                }
            }
        }
        let next = b.ins().iadd_imm(lane_base, unroll as i64);
        b.ins().jump(head, &[next.into()]);

        b.switch_to_block(done);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize();
    }
    Ok(Some(module.finish(id, ctx)?))
}

fn compile_fold_cached(prog: &Program) -> Result<Option<JitKernel>, String> {
    if !fold_supported(prog) {
        return Ok(None);
    }
    static CACHE: OnceLock<Mutex<HashMap<String, JitKernel>>> = OnceLock::new();
    let key = format!("{prog:?}");
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .copied()
    {
        return Ok(Some(hit));
    }
    let kernel = compile_fold_uncached(prog)?;
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, kernel);
    Ok(Some(kernel))
}

fn fold_supported(prog: &Program) -> bool {
    fn stmt(stmt: &CStmt) -> bool {
        match stmt {
            CStmt::Store { .. } | CStmt::StoreLocal { .. } | CStmt::StoreTile { .. } => true,
            CStmt::If { accept, reject, .. } => {
                accept.iter().chain(reject).all(fold_stmt_supported)
            }
            CStmt::StageTree { group, .. } => *group > 0 && group.is_power_of_two(),
            CStmt::Loop {
                count: Some(_),
                body,
                ..
            } => body.iter().all(fold_stmt_supported),
            CStmt::CarrierTree {
                tiles,
                values,
                lhs,
                rhs,
                merged,
                outs,
                group,
                ..
            } => {
                *group > 0
                    && group.is_power_of_two()
                    && tiles.len() == values.len()
                    && tiles.len() == lhs.len()
                    && tiles.len() == rhs.len()
                    && tiles.len() == merged.len()
                    && tiles.len() == outs.len()
            }
            _ => false,
        }
    }

    fn fold_stmt_supported(value: &CStmt) -> bool {
        stmt(value)
    }

    prog.tiles
        .iter()
        .all(|tile| tile.elem == ScalarElement::F32)
        && prog
            .segments
            .iter()
            .all(|segment| segment.stmts.iter().all(stmt))
        && prog.tape.iter().all(|instr| {
            !matches!(
                instr,
                Instr::Dot { .. } | Instr::Reduce { .. } | Instr::Rc2Index { .. }
            )
        })
}

fn compile_fold_uncached(prog: &Program) -> Result<JitKernel, String> {
    let mut module = NativeModule::new()?;
    let ptr_ty = module.ptr_ty;
    let (id, mut ctx, helpers) = module.function("fusor2_fold", 1)?;
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry_block = b.create_block();
        b.append_block_params_for_function_params(entry_block);
        b.switch_to_block(entry_block);
        let params = b.block_params(entry_block).to_vec();
        let locals_bytes = prog
            .locals
            .checked_mul(prog.block as usize)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| "native fold local frame overflows usize".to_string())?;
        let frame_bytes = locals_bytes
            .checked_add(prog.arena_bytes as usize)
            .ok_or_else(|| "native fold frame overflows usize".to_string())?
            .max(4);
        let frame_size =
            u32::try_from(frame_bytes).map_err(|_| "native fold frame exceeds u32".to_string())?;
        let frame_slot = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            frame_size,
            2,
        ));
        let frame = b.ins().stack_addr(ptr_ty, frame_slot, 0);
        let fold = FoldEnv {
            prog,
            env: Env {
                bufs: params[0],
                gid: [params[1], params[2], params[3]],
                grid: [params[4], params[5], params[6]],
                lane: b.ins().iconst(types::I32, 0),
                block: prog.block,
                width: prog.width,
                ptr_ty,
            },
            frame,
            locals_bytes,
            helpers: &helpers,
        };
        for segment in &prog.segments {
            if segment.stmts.iter().any(CStmt::is_collective) {
                for stmt in &segment.stmts {
                    emit_fold_stmt(&mut b, stmt, fold.env.lane, &fold)?;
                }
            } else {
                emit_fold_lanes(&mut b, &segment.stmts, &fold)?;
            }
        }
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize();
    }
    module.finish(id, ctx)
}

struct FoldEnv<'a> {
    prog: &'a Program,
    env: Env,
    frame: Value,
    locals_bytes: usize,
    helpers: &'a Helpers,
}

fn emit_fold_lanes(
    b: &mut FunctionBuilder<'_>,
    stmts: &[CStmt],
    fold: &FoldEnv<'_>,
) -> Result<(), String> {
    emit_const_loop(b, fold.prog.block, |b, lane| {
        for stmt in stmts {
            emit_fold_stmt(b, stmt, lane, fold)?;
        }
        Ok(())
    })
}

fn emit_const_loop(
    b: &mut FunctionBuilder<'_>,
    limit: u32,
    body_fn: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> Result<(), String>,
) -> Result<(), String> {
    let head = b.create_block();
    let body = b.create_block();
    let done = b.create_block();
    b.append_block_param(head, types::I32);
    let zero = b.ins().iconst(types::I32, 0);
    b.ins().jump(head, &[zero.into()]);
    b.switch_to_block(head);
    let index = b.block_params(head)[0];
    let more = b
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, index, limit as i64);
    b.ins().brif(more, body, &[], done, &[]);
    b.switch_to_block(body);
    body_fn(b, index)?;
    let next = b.ins().iadd_imm(index, 1);
    b.ins().jump(head, &[next.into()]);
    b.switch_to_block(done);
    Ok(())
}

fn emit_fold_stmt(
    b: &mut FunctionBuilder<'_>,
    stmt: &CStmt,
    lane: Value,
    fold: &FoldEnv<'_>,
) -> Result<(), String> {
    match stmt {
        CStmt::Store {
            prep,
            buf,
            elem,
            index,
            value,
            mask,
        } => {
            let regs = emit_fold_range(b, prep, lane, fold)?;
            emit_bound_store(
                b,
                *buf,
                *elem,
                fold_reg(&regs, *index)?,
                fold_reg(&regs, *value)?,
                fold_reg(&regs, *mask)?,
                fold,
            )?;
        }
        CStmt::StoreLocal { prep, local, value } => {
            let regs = emit_fold_range(b, prep, lane, fold)?;
            store_local(b, *local, lane, fold_reg(&regs, *value)?, fold);
        }
        CStmt::StoreTile {
            prep,
            tile,
            elem: ScalarElement::F32,
            index,
            value,
        } => {
            let regs = emit_fold_range(b, prep, lane, fold)?;
            store_tile_f32(
                b,
                *tile,
                fold_reg(&regs, *index)?,
                fold_reg(&regs, *value)?,
                fold,
            );
        }
        CStmt::If {
            prep,
            cond,
            accept,
            reject,
            ..
        } => {
            let regs = emit_fold_range(b, prep, lane, fold)?;
            let condition = b
                .ins()
                .icmp_imm(IntCC::NotEqual, fold_reg(&regs, *cond)?, 0);
            let accept_block = b.create_block();
            let reject_block = b.create_block();
            let done = b.create_block();
            b.ins()
                .brif(condition, accept_block, &[], reject_block, &[]);
            b.switch_to_block(accept_block);
            for stmt in accept {
                emit_fold_stmt(b, stmt, lane, fold)?;
            }
            b.ins().jump(done, &[]);
            b.switch_to_block(reject_block);
            for stmt in reject {
                emit_fold_stmt(b, stmt, lane, fold)?;
            }
            b.ins().jump(done, &[]);
            b.switch_to_block(done);
        }
        CStmt::Loop {
            prep,
            count: Some(count),
            index,
            accs,
            body,
        } => {
            let mut prep_regs = vec![None; fold.prog.regs.max(1)];
            emit_fold_range_into(b, prep, lane, fold, &mut prep_regs)?;
            let count = fold_reg(&prep_regs, *count)?;
            let mut initial = Vec::with_capacity(accs.len());
            for acc in accs {
                emit_fold_range_into(b, &acc.init_prep, lane, fold, &mut prep_regs)?;
                initial.push(fold_reg(&prep_regs, acc.init)?);
            }
            for (acc, value) in accs.iter().zip(initial) {
                store_local(b, acc.local, lane, value, fold);
            }

            let head = b.create_block();
            let loop_body = b.create_block();
            let done = b.create_block();
            b.append_block_param(head, types::I32);
            let zero = b.ins().iconst(types::I32, 0);
            b.ins().jump(head, &[zero.into()]);
            b.switch_to_block(head);
            let iteration = b.block_params(head)[0];
            let more = b.ins().icmp(IntCC::UnsignedLessThan, iteration, count);
            b.ins().brif(more, loop_body, &[], done, &[]);
            b.switch_to_block(loop_body);
            if let Some(index) = index {
                store_local(b, *index, lane, iteration, fold);
            }
            for stmt in body {
                emit_fold_stmt(b, stmt, lane, fold)?;
            }
            // Every update observes the old accumulator tuple; commit only
            // after all update expressions have been evaluated.
            let mut updated = Vec::with_capacity(accs.len());
            let mut update_regs = prep_regs.clone();
            for acc in accs {
                emit_fold_range_into(b, &acc.update_prep, lane, fold, &mut update_regs)?;
                updated.push(fold_reg(&update_regs, acc.update)?);
            }
            for (acc, value) in accs.iter().zip(updated) {
                store_local(b, acc.local, lane, value, fold);
            }
            let next = b.ins().iadd_imm(iteration, 1);
            b.ins().jump(head, &[next.into()]);
            b.switch_to_block(done);
        }
        CStmt::StageTree {
            prep,
            tile,
            value,
            op,
            group,
        } => {
            let tiles = [*tile];
            let values = [*value];
            emit_carrier_tree(
                b,
                prep,
                &tiles,
                &values,
                &[],
                &[],
                &(0..0),
                &[],
                &[],
                *group,
                Some(*op),
                fold,
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
        } => emit_carrier_tree(
            b, prep, tiles, values, lhs, rhs, merge_prep, merged, outs, *group, *fast, fold,
        )?,
        _ => return Err("unsupported native fold statement".into()),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_carrier_tree(
    b: &mut FunctionBuilder<'_>,
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
    fold: &FoldEnv<'_>,
) -> Result<(), String> {
    // Stage one partial per logical lane.
    emit_const_loop(b, fold.prog.block, |b, lane| {
        let regs = emit_fold_range(b, prep, lane, fold)?;
        for (tile, value) in tiles.iter().zip(values) {
            store_tile_f32(b, *tile, lane, fold_reg(&regs, *value)?, fold);
        }
        Ok(())
    })?;

    // Reduce every group in place. These are runtime loops, so native code
    // size is independent of the 256-lane workgroup and carrier width.
    let outer_head = b.create_block();
    let outer_body = b.create_block();
    let outer_done = b.create_block();
    let stride_head = b.create_block();
    let stride_body = b.create_block();
    let stride_done = b.create_block();
    let pair_head = b.create_block();
    let pair_body = b.create_block();
    let pair_done = b.create_block();
    b.append_block_param(outer_head, types::I32);
    b.append_block_param(stride_head, types::I32);
    b.append_block_param(pair_head, types::I32);

    let zero = b.ins().iconst(types::I32, 0);
    b.ins().jump(outer_head, &[zero.into()]);
    b.switch_to_block(outer_head);
    let base = b.block_params(outer_head)[0];
    let more_groups = b
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, base, fold.prog.block as i64);
    b.ins().brif(more_groups, outer_body, &[], outer_done, &[]);

    b.switch_to_block(outer_body);
    let first_stride = b.ins().iconst(types::I32, (group / 2) as i64);
    b.ins().jump(stride_head, &[first_stride.into()]);
    b.switch_to_block(stride_head);
    let stride = b.block_params(stride_head)[0];
    let has_stride = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, stride, 0);
    b.ins().brif(has_stride, stride_body, &[], stride_done, &[]);

    b.switch_to_block(stride_body);
    let pair_zero = b.ins().iconst(types::I32, 0);
    b.ins().jump(pair_head, &[pair_zero.into()]);
    b.switch_to_block(pair_head);
    let pair = b.block_params(pair_head)[0];
    let more_pairs = b.ins().icmp(IntCC::UnsignedLessThan, pair, stride);
    b.ins().brif(more_pairs, pair_body, &[], pair_done, &[]);

    b.switch_to_block(pair_body);
    let left_index = b.ins().iadd(base, pair);
    let right_index = b.ins().iadd(left_index, stride);
    let mut left = Vec::with_capacity(tiles.len());
    let mut right = Vec::with_capacity(tiles.len());
    for tile in tiles {
        left.push(load_tile_f32(b, *tile, left_index, fold));
        right.push(load_tile_f32(b, *tile, right_index, fold));
    }
    let results = if let Some(op) = fast {
        left.iter()
            .zip(&right)
            .map(|(&left, &right)| emit_fold_fast(b, op, left, right, fold.helpers))
            .collect::<Vec<_>>()
    } else {
        let merge_lane = b.ins().iconst(types::I32, 0);
        for (((local_lhs, local_rhs), &left), &right) in lhs.iter().zip(rhs).zip(&left).zip(&right)
        {
            store_local(b, *local_lhs, merge_lane, left, fold);
            store_local(b, *local_rhs, merge_lane, right, fold);
        }
        let regs = emit_fold_range(b, merge_prep, merge_lane, fold)?;
        merged
            .iter()
            .map(|slot| fold_reg(&regs, *slot))
            .collect::<Result<Vec<_>, _>>()?
    };
    for (tile, value) in tiles.iter().zip(results) {
        store_tile_f32(b, *tile, left_index, value, fold);
    }
    let next_pair = b.ins().iadd_imm(pair, 1);
    b.ins().jump(pair_head, &[next_pair.into()]);

    b.switch_to_block(pair_done);
    let next_stride = b.ins().ushr_imm(stride, 1);
    b.ins().jump(stride_head, &[next_stride.into()]);
    b.switch_to_block(stride_done);
    let next_base = b.ins().iadd_imm(base, group as i64);
    b.ins().jump(outer_head, &[next_base.into()]);
    b.switch_to_block(outer_done);

    // Materialize the group result into the output locals expected by the
    // post-reduction store expressions.
    emit_const_loop(b, fold.prog.block, |b, lane| {
        let group_base = if group == 1 {
            lane
        } else {
            let group_id = b.ins().udiv_imm(lane, group as i64);
            b.ins().imul_imm(group_id, group as i64)
        };
        for (tile, out) in tiles.iter().zip(outs) {
            let value = load_tile_f32(b, *tile, group_base, fold);
            store_local(b, *out, lane, value, fold);
        }
        Ok(())
    })?;
    Ok(())
}

fn emit_fold_range(
    b: &mut FunctionBuilder<'_>,
    range: &std::ops::Range<u32>,
    lane: Value,
    fold: &FoldEnv<'_>,
) -> Result<Vec<Option<Value>>, String> {
    let mut regs = vec![None; fold.prog.regs.max(1)];
    emit_fold_range_into(b, range, lane, fold, &mut regs)?;
    Ok(regs)
}

fn emit_fold_range_into(
    b: &mut FunctionBuilder<'_>,
    range: &std::ops::Range<u32>,
    lane: Value,
    fold: &FoldEnv<'_>,
    regs: &mut [Option<Value>],
) -> Result<(), String> {
    let env = Env {
        bufs: fold.env.bufs,
        gid: fold.env.gid,
        grid: fold.env.grid,
        lane,
        block: fold.env.block,
        width: fold.env.width,
        ptr_ty: fold.env.ptr_ty,
    };
    for pc in range.clone() {
        let instr = &fold.prog.tape[pc as usize];
        match instr {
            Instr::LoadLocal { out, local } => {
                let address = local_address(b, *local, lane, fold);
                regs[*out as usize] = Some(b.ins().load(types::I32, memory(), address, 0));
            }
            Instr::LoadTile {
                out,
                tile,
                elem,
                index,
            } if *elem == ScalarElement::F32 => {
                let index = fold_reg(regs, *index)?;
                regs[*out as usize] = Some(load_tile_f32(b, *tile, index, fold));
            }
            _ => emit_instr(b, instr, regs, &env, fold.helpers)?,
        }
    }
    Ok(())
}

fn emit_bound_store(
    b: &mut FunctionBuilder<'_>,
    buf: u16,
    elem: ScalarElement,
    index: Value,
    value: Value,
    mask: Value,
    fold: &FoldEnv<'_>,
) -> Result<(), String> {
    let (ptr, bytes) = raw_buf(b, fold.env.bufs, buf, fold.env.ptr_ty);
    if matches!(
        elem,
        ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool
    ) {
        let write = b.create_block();
        let done = b.create_block();
        let (valid, address) = direct_u32_address(b, ptr, bytes, index, mask, fold.env.ptr_ty);
        b.ins().brif(valid, write, &[], done, &[]);
        b.switch_to_block(write);
        b.ins().store(memory(), value, address, 0);
        b.ins().jump(done, &[]);
        b.switch_to_block(done);
    } else {
        let elem = b.ins().iconst(types::I32, elem_code(elem) as i64);
        b.ins()
            .call(fold.helpers.write, &[ptr, bytes, index, value, mask, elem]);
    }
    Ok(())
}

fn local_address(
    b: &mut FunctionBuilder<'_>,
    local: u16,
    lane: Value,
    fold: &FoldEnv<'_>,
) -> Value {
    let base = local as i64 * fold.prog.block as i64;
    let index = b.ins().iadd_imm(lane, base);
    let index = if fold.env.ptr_ty == types::I32 {
        index
    } else {
        b.ins().uextend(fold.env.ptr_ty, index)
    };
    let offset = b.ins().imul_imm(index, 4);
    b.ins().iadd(fold.frame, offset)
}

fn store_local(
    b: &mut FunctionBuilder<'_>,
    local: u16,
    lane: Value,
    value: Value,
    fold: &FoldEnv<'_>,
) {
    let address = local_address(b, local, lane, fold);
    b.ins().store(memory(), value, address, 0);
}

fn tile_address(b: &mut FunctionBuilder<'_>, tile: u16, index: Value, fold: &FoldEnv<'_>) -> Value {
    let info = &fold.prog.tiles[tile as usize];
    let base = (fold.locals_bytes + info.byte_offset as usize) as i64;
    let index = if fold.env.ptr_ty == types::I32 {
        index
    } else {
        b.ins().uextend(fold.env.ptr_ty, index)
    };
    let offset = b.ins().imul_imm(index, 4);
    let address = b.ins().iadd(fold.frame, offset);
    b.ins().iadd_imm(address, base)
}

fn load_tile_f32(
    b: &mut FunctionBuilder<'_>,
    tile: u16,
    index: Value,
    fold: &FoldEnv<'_>,
) -> Value {
    let address = tile_address(b, tile, index, fold);
    b.ins().load(types::I32, memory(), address, 0)
}

fn store_tile_f32(
    b: &mut FunctionBuilder<'_>,
    tile: u16,
    index: Value,
    value: Value,
    fold: &FoldEnv<'_>,
) {
    let address = tile_address(b, tile, index, fold);
    b.ins().store(memory(), value, address, 0);
}

fn emit_fold_fast(
    b: &mut FunctionBuilder<'_>,
    op: TileReduceOp,
    left: Value,
    right: Value,
    helpers: &Helpers,
) -> Value {
    emit_bin(b, op.binary(), NumTy::F32, left, right, helpers)
}

fn fold_reg(regs: &[Option<Value>], slot: u32) -> Result<Value, String> {
    reg(regs, slot)
}

struct Env {
    bufs: Value,
    gid: [Value; 3],
    grid: [Value; 3],
    lane: Value,
    block: u32,
    width: u32,
    ptr_ty: cranelift_codegen::ir::Type,
}

struct Helpers {
    read: FuncRef,
    write: FuncRef,
    un: FuncRef,
    bin: FuncRef,
    cast: FuncRef,
    round: FuncRef,
    narrow: FuncRef,
    unpack: FuncRef,
}

fn emit_instr(
    b: &mut FunctionBuilder<'_>,
    instr: &Instr,
    regs: &mut [Option<Value>],
    env: &Env,
    h: &Helpers,
) -> Result<(), String> {
    let out = instr.out() as usize;
    let value = match instr {
        Instr::Const { bits, .. } => b.ins().iconst(types::I32, *bits as i64),
        Instr::LaneId { .. } => env.lane,
        Instr::Uniform { which, .. } => match which {
            UniformSrc::ProgramX => env.gid[0],
            UniformSrc::ProgramY => env.gid[1],
            UniformSrc::ProgramZ => env.gid[2],
            UniformSrc::GridX => env.grid[0],
            UniformSrc::GridY => env.grid[1],
            UniformSrc::GridZ => env.grid[2],
            UniformSrc::SubgroupSize => b.ins().iconst(types::I32, env.width as i64),
            UniformSrc::NumSubgroups => b
                .ins()
                .iconst(types::I32, env.block.div_ceil(env.width) as i64),
            UniformSrc::SubgroupId => b.ins().udiv_imm(env.lane, env.width as i64),
            UniformSrc::SubgroupLane => b.ins().urem_imm(env.lane, env.width as i64),
        },
        Instr::Load {
            buf,
            elem,
            index,
            mask,
            fill,
            ..
        } => {
            let (ptr, bytes) = raw_buf(b, env.bufs, *buf, env.ptr_ty);
            if matches!(
                elem,
                ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool
            ) {
                direct_load_u32(
                    b,
                    ptr,
                    bytes,
                    reg(regs, *index)?,
                    reg(regs, *mask)?,
                    reg(regs, *fill)?,
                    env.ptr_ty,
                )
            } else {
                let elem_code = b.ins().iconst(types::I32, elem_code(*elem) as i64);
                let call = b.ins().call(
                    h.read,
                    &[
                        ptr,
                        bytes,
                        reg(regs, *index)?,
                        reg(regs, *mask)?,
                        reg(regs, *fill)?,
                        elem_code,
                    ],
                );
                b.inst_results(call)[0]
            }
        }
        Instr::Un { op, x, ty, .. } => emit_un(b, *op, *ty, reg(regs, *x)?, h),
        Instr::Bin {
            op, a, b: rhs, ty, ..
        } => emit_bin(b, *op, *ty, reg(regs, *a)?, reg(regs, *rhs)?, h),
        Instr::Fma { a, b: rhs, c, .. } => {
            let a = as_f32(b, reg(regs, *a)?);
            let rhs = as_f32(b, reg(regs, *rhs)?);
            let c = as_f32(b, reg(regs, *c)?);
            let value = b.ins().fma(a, rhs, c);
            as_i32(b, value)
        }
        Instr::Cmp {
            op, a, b: rhs, ty, ..
        } => emit_cmp(b, *op, *ty, reg(regs, *a)?, reg(regs, *rhs)?),
        Instr::MaskToValue { x, ty, .. } => {
            let nz = b.ins().icmp_imm(IntCC::NotEqual, reg(regs, *x)?, 0);
            let one = match ty {
                NumTy::F32 => 1.0f32.to_bits(),
                _ => 1,
            };
            let t = b.ins().iconst(types::I32, one as i64);
            let f = b.ins().iconst(types::I32, 0);
            b.ins().select(nz, t, f)
        }
        Instr::ValueToMask { x, ty, .. } => {
            let nz = match ty {
                NumTy::F32 => {
                    let value = as_f32(b, reg(regs, *x)?);
                    let zero = b.ins().f32const(0.0);
                    b.ins().fcmp(FloatCC::NotEqual, value, zero)
                }
                _ => b.ins().icmp_imm(IntCC::NotEqual, reg(regs, *x)?, 0),
            };
            mask_value(b, nz)
        }
        Instr::Round { mode, x, .. } => {
            let mode = b.ins().iconst(types::I32, round_code(*mode) as i64);
            let call = b.ins().call(h.round, &[mode, reg(regs, *x)?]);
            b.inst_results(call)[0]
        }
        Instr::Cast { x, from, to, .. } => {
            let from = b.ins().iconst(types::I32, ty_code(*from) as i64);
            let to = b.ins().iconst(types::I32, ty_code(*to) as i64);
            let call = b.ins().call(h.cast, &[from, to, reg(regs, *x)?]);
            b.inst_results(call)[0]
        }
        Instr::Narrow { x, to, .. } => {
            let to = b.ins().iconst(types::I32, elem_code(*to) as i64);
            let call = b.ins().call(h.narrow, &[to, reg(regs, *x)?]);
            b.inst_results(call)[0]
        }
        Instr::Bitcast { x, .. } | Instr::Copy { x, .. } => reg(regs, *x)?,
        Instr::Select { c, t, f, .. } => {
            let cond = b.ins().icmp_imm(IntCC::NotEqual, reg(regs, *c)?, 0);
            b.ins().select(cond, reg(regs, *t)?, reg(regs, *f)?)
        }
        Instr::VecCompose { parts, .. } => {
            for (i, part) in parts.iter().enumerate() {
                regs[out + i] = Some(reg(regs, *part)?);
            }
            return Ok(());
        }
        Instr::VecComponent {
            base, component, ..
        } => reg(regs, *base + *component)?,
        Instr::Unpack2x16 { x, .. } => {
            for high in 0..2 {
                let high_value = b.ins().iconst(types::I32, high);
                let call = b.ins().call(h.unpack, &[reg(regs, *x)?, high_value]);
                regs[out + high as usize] = Some(b.inst_results(call)[0]);
            }
            return Ok(());
        }
        _ => return Err("unsupported Cranelift map instruction".into()),
    };
    regs[out] = Some(value);
    Ok(())
}

fn emit_bin(
    b: &mut FunctionBuilder<'_>,
    op: BinOp,
    ty: NumTy,
    a: Value,
    rhs: Value,
    h: &Helpers,
) -> Value {
    if matches!(op, BinOp::Pow) {
        let op_value = b.ins().iconst(types::I32, bin_code(op) as i64);
        let ty_value = b.ins().iconst(types::I32, ty_code(ty) as i64);
        let call = b.ins().call(h.bin, &[op_value, ty_value, a, rhs]);
        return b.inst_results(call)[0];
    }
    match ty {
        NumTy::F32 => {
            let x = as_f32(b, a);
            let y = as_f32(b, rhs);
            let f = match op {
                BinOp::Add => b.ins().fadd(x, y),
                BinOp::Sub => b.ins().fsub(x, y),
                BinOp::Mul => b.ins().fmul(x, y),
                BinOp::Div => b.ins().fdiv(x, y),
                BinOp::Rem => {
                    let op_value = b.ins().iconst(types::I32, bin_code(op) as i64);
                    let ty_value = b.ins().iconst(types::I32, ty_code(ty) as i64);
                    let call = b.ins().call(h.bin, &[op_value, ty_value, a, rhs]);
                    return b.inst_results(call)[0];
                }
                BinOp::Min => {
                    let c = b.ins().fcmp(FloatCC::LessThan, y, x);
                    b.ins().select(c, y, x)
                }
                BinOp::Max => {
                    let c = b.ins().fcmp(FloatCC::GreaterThan, y, x);
                    b.ins().select(c, y, x)
                }
                BinOp::LogicalAnd | BinOp::LogicalOr => {
                    let zero = b.ins().f32const(0.0);
                    let zx = b.ins().fcmp(FloatCC::NotEqual, x, zero);
                    let zy = b.ins().fcmp(FloatCC::NotEqual, y, zero);
                    let c = if op == BinOp::LogicalAnd {
                        b.ins().band(zx, zy)
                    } else {
                        b.ins().bor(zx, zy)
                    };
                    let one = b.ins().f32const(1.0);
                    let zero = b.ins().f32const(0.0);
                    b.ins().select(c, one, zero)
                }
                BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shr
                | BinOp::Shl
                | BinOp::Pow => unreachable!(),
            };
            as_i32(b, f)
        }
        NumTy::U32 | NumTy::I32 => match op {
            BinOp::Add => b.ins().iadd(a, rhs),
            BinOp::Sub => b.ins().isub(a, rhs),
            BinOp::Mul => b.ins().imul(a, rhs),
            BinOp::Div => emit_int_divrem(b, BinOp::Div, ty, a, rhs),
            BinOp::Rem => emit_int_divrem(b, BinOp::Rem, ty, a, rhs),
            BinOp::Min | BinOp::Max => {
                let cc = match (ty, op) {
                    (NumTy::U32, BinOp::Min) => IntCC::UnsignedLessThan,
                    (NumTy::U32, _) => IntCC::UnsignedGreaterThan,
                    (_, BinOp::Min) => IntCC::SignedLessThan,
                    _ => IntCC::SignedGreaterThan,
                };
                let c = b.ins().icmp(cc, a, rhs);
                b.ins().select(c, a, rhs)
            }
            BinOp::BitAnd => b.ins().band(a, rhs),
            BinOp::BitOr => b.ins().bor(a, rhs),
            BinOp::BitXor => b.ins().bxor(a, rhs),
            BinOp::Shr => {
                if ty == NumTy::U32 {
                    b.ins().ushr(a, rhs)
                } else {
                    b.ins().sshr(a, rhs)
                }
            }
            BinOp::Shl => b.ins().ishl(a, rhs),
            BinOp::LogicalAnd | BinOp::LogicalOr => {
                let x = b.ins().icmp_imm(IntCC::NotEqual, a, 0);
                let y = b.ins().icmp_imm(IntCC::NotEqual, rhs, 0);
                let c = if op == BinOp::LogicalAnd {
                    b.ins().band(x, y)
                } else {
                    b.ins().bor(x, y)
                };
                let one = b.ins().iconst(types::I32, 1);
                let zero = b.ins().iconst(types::I32, 0);
                b.ins().select(c, one, zero)
            }
            BinOp::Pow => unreachable!(),
        },
    }
}

/// Cranelift integer division traps, while Kernel integer arithmetic is
/// explicitly total (`x / 0 == MAX`, `x % 0 == 0`, signed overflow wraps).
/// Totality is also essential for masked tail lanes: their loads deliberately
/// produce zero and their stores are suppressed only after the value tree has
/// been evaluated.
fn emit_int_divrem(
    b: &mut FunctionBuilder<'_>,
    op: BinOp,
    ty: NumTy,
    a: Value,
    rhs: Value,
) -> Value {
    let zero = b.ins().iconst(types::I32, 0);
    let one = b.ins().iconst(types::I32, 1);
    let by_zero = b.ins().icmp_imm(IntCC::Equal, rhs, 0);
    if ty == NumTy::U32 {
        let safe_rhs = b.ins().select(by_zero, one, rhs);
        let value = if op == BinOp::Div {
            b.ins().udiv(a, safe_rhs)
        } else {
            b.ins().urem(a, safe_rhs)
        };
        let on_zero = if op == BinOp::Div {
            b.ins().iconst(types::I32, -1)
        } else {
            zero
        };
        return b.ins().select(by_zero, on_zero, value);
    }

    let min = b.ins().icmp_imm(IntCC::Equal, a, i32::MIN as i64);
    let negative_one = b.ins().icmp_imm(IntCC::Equal, rhs, -1);
    let overflow = b.ins().band(min, negative_one);
    let invalid = b.ins().bor(by_zero, overflow);
    let safe_rhs = b.ins().select(invalid, one, rhs);
    let value = if op == BinOp::Div {
        b.ins().sdiv(a, safe_rhs)
    } else {
        b.ins().srem(a, safe_rhs)
    };
    if op == BinOp::Rem {
        return b.ins().select(invalid, zero, value);
    }
    let wrapped = b.ins().iconst(types::I32, i32::MIN as i64);
    let value = b.ins().select(overflow, wrapped, value);
    let minus_one = b.ins().iconst(types::I32, -1);
    b.ins().select(by_zero, minus_one, value)
}

fn emit_un(b: &mut FunctionBuilder<'_>, op: UnOp, ty: NumTy, x: Value, h: &Helpers) -> Value {
    match (op, ty) {
        (UnOp::Abs, NumTy::F32) => {
            let x = as_f32(b, x);
            let value = b.ins().fabs(x);
            as_i32(b, value)
        }
        (UnOp::Neg, NumTy::F32) => {
            let x = as_f32(b, x);
            let value = b.ins().fneg(x);
            as_i32(b, value)
        }
        (UnOp::Sqrt, NumTy::F32) => {
            let x = as_f32(b, x);
            let value = b.ins().sqrt(x);
            as_i32(b, value)
        }
        (UnOp::InverseSqrt, NumTy::F32) => {
            let x = as_f32(b, x);
            let root = b.ins().sqrt(x);
            let one = b.ins().f32const(1.0);
            let value = b.ins().fdiv(one, root);
            as_i32(b, value)
        }
        (UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp, NumTy::F32) => {
            emit_expf(b, x)
        }
        (UnOp::Abs, NumTy::U32) => x,
        (UnOp::Neg, NumTy::U32) => b.ins().ineg(x),
        (UnOp::Abs, NumTy::I32) => {
            let negative = b.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
            let negated = b.ins().ineg(x);
            b.ins().select(negative, negated, x)
        }
        (UnOp::Neg, NumTy::I32) => b.ins().ineg(x),
        _ => {
            let op = b.ins().iconst(types::I32, un_code(op) as i64);
            let ty = b.ins().iconst(types::I32, ty_code(ty) as i64);
            let call = b.ins().call(h.un, &[op, ty, x]);
            b.inst_results(call)[0]
        }
    }
}

/// Inline the same Cody-Waite `exp` used by the reference executor. Keeping
/// this arithmetic in the JIT removes one host call per attention score and
/// lets Cranelift schedule the polynomial with the surrounding map.
fn emit_expf(b: &mut FunctionBuilder<'_>, bits: Value) -> Value {
    let x = as_f32(b, bits);
    let log2_e = b.ins().f32const(std::f32::consts::LOG2_E);
    let scaled = b.ins().fmul(x, log2_e);
    let n = b.ins().nearest(scaled);
    let ln2_hi = b.ins().f32const(0.693_145_75);
    let ln2_lo = b.ins().f32const(1.428_606_8e-6);
    let hi = b.ins().fmul(n, ln2_hi);
    let r = b.ins().fsub(x, hi);
    let lo = b.ins().fmul(n, ln2_lo);
    let r = b.ins().fsub(r, lo);

    let mut p = b.ins().f32const(2.480_158_7e-5);
    for coefficient in [
        1.984_127e-4,
        1.388_888_9e-3,
        8.333_333e-3,
        4.166_666_8e-2,
        0.166_666_67,
        0.5,
        1.0,
    ] {
        let product = b.ins().fmul(p, r);
        let coefficient = b.ins().f32const(coefficient);
        p = b.ins().fadd(product, coefficient);
    }
    let product = b.ins().fmul(p, r);
    let one = b.ins().f32const(1.0);
    p = b.ins().fadd(product, one);

    let exponent = b.ins().fcvt_to_sint_sat(types::I32, n);
    let biased = b.ins().iadd_imm(exponent, 127);
    let scale_bits = b.ins().ishl_imm(biased, 23);
    let scale = b.ins().bitcast(types::F32, MemFlags::new(), scale_bits);
    let normal = b.ins().fmul(p, scale);

    // Match `ldexp`'s subnormal construction instead of flushing the masked
    // tail of a softmax to zero prematurely.
    let rest = b.ins().iadd_imm(exponent, 100);
    let rest_too_low = b.ins().icmp_imm(IntCC::SignedLessThan, rest, -126);
    let floor = b.ins().iconst(types::I32, -126);
    let safe_rest = b.ins().select(rest_too_low, floor, rest);
    let rest_biased = b.ins().iadd_imm(safe_rest, 127);
    let rest_bits = b.ins().ishl_imm(rest_biased, 23);
    let rest_scale = b.ins().bitcast(types::F32, MemFlags::new(), rest_bits);
    let step = b.ins().f32const(f32::from_bits(27 << 23));
    let subnormal = b.ins().fmul(p, step);
    let subnormal = b.ins().fmul(subnormal, rest_scale);
    let zero = b.ins().f32const(0.0);
    let subnormal = b.ins().select(rest_too_low, zero, subnormal);
    let is_subnormal = b.ins().icmp_imm(IntCC::SignedLessThan, exponent, -126);
    let value = b.ins().select(is_subnormal, subnormal, normal);

    let high_limit = b.ins().f32const(88.72);
    let over = b.ins().fcmp(FloatCC::GreaterThan, x, high_limit);
    let infinity = b.ins().f32const(f32::INFINITY);
    let value = b.ins().select(over, infinity, value);
    let low_limit = b.ins().f32const(-103.0);
    let under = b.ins().fcmp(FloatCC::LessThan, x, low_limit);
    let value = b.ins().select(under, zero, value);
    as_i32(b, value)
}

fn emit_cmp(b: &mut FunctionBuilder<'_>, op: CmpOp, ty: NumTy, a: Value, rhs: Value) -> Value {
    let c = match ty {
        NumTy::F32 => {
            let a = as_f32(b, a);
            let rhs = as_f32(b, rhs);
            b.ins().fcmp(
                match op {
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                },
                a,
                rhs,
            )
        }
        NumTy::U32 | NumTy::I32 => b.ins().icmp(
            match (ty, op) {
                (_, CmpOp::Eq) => IntCC::Equal,
                (_, CmpOp::Ne) => IntCC::NotEqual,
                (NumTy::U32, CmpOp::Lt) => IntCC::UnsignedLessThan,
                (NumTy::U32, CmpOp::Le) => IntCC::UnsignedLessThanOrEqual,
                (NumTy::U32, CmpOp::Gt) => IntCC::UnsignedGreaterThan,
                (NumTy::U32, CmpOp::Ge) => IntCC::UnsignedGreaterThanOrEqual,
                (_, CmpOp::Lt) => IntCC::SignedLessThan,
                (_, CmpOp::Le) => IntCC::SignedLessThanOrEqual,
                (_, CmpOp::Gt) => IntCC::SignedGreaterThan,
                (_, CmpOp::Ge) => IntCC::SignedGreaterThanOrEqual,
            },
            a,
            rhs,
        ),
    };
    mask_value(b, c)
}

fn raw_buf(
    b: &mut FunctionBuilder<'_>,
    bufs: Value,
    index: u16,
    ptr_ty: cranelift_codegen::ir::Type,
) -> (Value, Value) {
    let addr = b.ins().iadd_imm(bufs, index as i64 * 16);
    (
        b.ins().load(ptr_ty, memory(), addr, 0),
        b.ins().load(ptr_ty, memory(), addr, 8),
    )
}

/// Inline the overwhelmingly common 32-bit masked load. Buffers are always
/// at least four bytes, so redirecting an invalid lane to element zero is a
/// safe speculative load; the final select returns the requested fill value.
fn direct_load_u32(
    b: &mut FunctionBuilder<'_>,
    ptr: Value,
    bytes: Value,
    index: Value,
    mask: Value,
    fill: Value,
    ptr_ty: cranelift_codegen::ir::Type,
) -> Value {
    let (valid, address) = direct_u32_address(b, ptr, bytes, index, mask, ptr_ty);
    // Invalid lanes use element zero only as a safe speculative address.
    let safe_address = b.ins().select(valid, address, ptr);
    let loaded = b.ins().load(types::I32, memory(), safe_address, 0);
    b.ins().select(valid, loaded, fill)
}

fn direct_u32_address(
    b: &mut FunctionBuilder<'_>,
    ptr: Value,
    bytes: Value,
    index: Value,
    mask: Value,
    ptr_ty: cranelift_codegen::ir::Type,
) -> (Value, Value) {
    let index_ptr = if ptr_ty == types::I32 {
        index
    } else {
        b.ins().uextend(ptr_ty, index)
    };
    let elements = b.ins().udiv_imm(bytes, 4);
    let in_bounds = b.ins().icmp(IntCC::UnsignedLessThan, index_ptr, elements);
    let enabled = b.ins().icmp_imm(IntCC::NotEqual, mask, 0);
    let valid = b.ins().band(in_bounds, enabled);
    let offset = b.ins().imul_imm(index_ptr, 4);
    let address = b.ins().iadd(ptr, offset);
    (valid, address)
}

fn import(
    module: &mut JITModule,
    name: &str,
    params: &[cranelift_codegen::ir::Type],
    ret: Option<cranelift_codegen::ir::Type>,
) -> Result<cranelift_module::FuncId, String> {
    let mut sig = module.make_signature();
    sig.params.extend(params.iter().copied().map(AbiParam::new));
    if let Some(ret) = ret {
        sig.returns.push(AbiParam::new(ret));
    }
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| e.to_string())
}

fn memory() -> MemFlags {
    MemFlags::new().with_notrap()
}
fn reg(regs: &[Option<Value>], slot: u32) -> Result<Value, String> {
    regs.get(slot as usize)
        .and_then(|v| *v)
        .ok_or_else(|| format!("missing register {slot}"))
}
fn as_f32(b: &mut FunctionBuilder<'_>, v: Value) -> Value {
    b.ins().bitcast(types::F32, MemFlags::new(), v)
}
fn as_i32(b: &mut FunctionBuilder<'_>, v: Value) -> Value {
    b.ins().bitcast(types::I32, MemFlags::new(), v)
}
fn mask_value(b: &mut FunctionBuilder<'_>, c: Value) -> Value {
    let t = b.ins().iconst(types::I32, -1);
    let f = b.ins().iconst(types::I32, 0);
    b.ins().select(c, t, f)
}

fn unsupported(i: &Instr) -> bool {
    matches!(
        i,
        Instr::LoadLocal { .. }
            | Instr::LoadTile { .. }
            | Instr::Dot { .. }
            | Instr::Reduce { .. }
            | Instr::Rc2Index { .. }
    )
}
fn ty_code(v: NumTy) -> u32 {
    match v {
        NumTy::F32 => 0,
        NumTy::U32 => 1,
        NumTy::I32 => 2,
    }
}
fn decode_ty(v: u32) -> NumTy {
    match v {
        0 => NumTy::F32,
        1 => NumTy::U32,
        _ => NumTy::I32,
    }
}
fn elem_code(v: ScalarElement) -> u32 {
    match v {
        ScalarElement::F32 => 0,
        ScalarElement::F16 => 1,
        ScalarElement::BF16 => 2,
        ScalarElement::U32 => 3,
        ScalarElement::I32 => 4,
        ScalarElement::Bool => 5,
    }
}
fn decode_elem(v: u32) -> ScalarElement {
    match v {
        0 => ScalarElement::F32,
        1 => ScalarElement::F16,
        2 => ScalarElement::BF16,
        3 => ScalarElement::U32,
        4 => ScalarElement::I32,
        _ => ScalarElement::Bool,
    }
}
fn un_code(v: UnOp) -> u32 {
    v as u32
}
fn decode_un(v: u32) -> UnOp {
    const OPS: [UnOp; 23] = [
        UnOp::Exp,
        UnOp::ApproximateExp,
        UnOp::LessApproximateExp,
        UnOp::Exp2,
        UnOp::Log,
        UnOp::Log2,
        UnOp::Sqrt,
        UnOp::InverseSqrt,
        UnOp::Sin,
        UnOp::Cos,
        UnOp::Tan,
        UnOp::Tanh,
        UnOp::Asin,
        UnOp::Acos,
        UnOp::Atan,
        UnOp::Sinh,
        UnOp::Cosh,
        UnOp::Asinh,
        UnOp::Acosh,
        UnOp::Atanh,
        UnOp::Abs,
        UnOp::Neg,
        UnOp::Unpack2x16Float,
    ];
    OPS[v as usize]
}
fn bin_code(v: BinOp) -> u32 {
    v as u32
}
fn decode_bin(v: u32) -> BinOp {
    const OPS: [BinOp; 15] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::Pow,
        BinOp::Min,
        BinOp::Max,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shr,
        BinOp::Shl,
        BinOp::LogicalAnd,
        BinOp::LogicalOr,
    ];
    OPS[v as usize]
}
fn round_code(v: RoundMode) -> u32 {
    match v {
        RoundMode::HalfToEven => 0,
        RoundMode::HalfAwayFromZero => 1,
        RoundMode::Floor => 2,
        RoundMode::Ceil => 3,
        RoundMode::Trunc => 4,
    }
}
