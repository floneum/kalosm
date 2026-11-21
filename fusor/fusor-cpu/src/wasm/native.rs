//! The emitted modules run natively under `wasmi`, for the conformance suite.
//! `wasmi` has its own linear memory, so every launch copies the bound buffers
//! in, runs, and copies them back out.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module, Store, TypedFunc};

use super::emit;
use crate::emit::{Program, RawBuf};
use crate::helpers;

type Entry = TypedFunc<(i32, i32, i32, i32, i32, i32, i32, i32), ()>;

#[derive(Clone)]
pub struct Kernel(
    Arc<Mutex<Inner>>,
    Option<(crate::jit::JitKernel, Arc<Program>)>,
);

struct Inner {
    store: Store<()>,
    memory: Memory,
    run: Entry,
    frame: usize,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Kernel(wasmi)")
    }
}

/// Where the frame starts in the module's memory; the buffer table and the
/// buffers follow it.
const FRAME: usize = 64;
const PAGE: usize = 65536;

fn align(x: usize) -> usize {
    (x + 63) & !63
}

pub(crate) fn compile(prog: &Arc<Program>) -> Result<Kernel, String> {
    static CACHE: OnceLock<Mutex<HashMap<Arc<Program>, Kernel>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(prog)
        .cloned()
    {
        return Ok(hit);
    }
    let mut kernel = instantiate(prog)?;
    // `FUSOR_WASM_DIFF` runs the Cranelift kernel beside every launch and
    // reports the first buffer the two disagree on.
    if std::env::var_os("FUSOR_WASM_DIFF").is_some() {
        kernel.1 = Some((crate::jit::compile(prog)?, Arc::clone(prog)));
    }
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(Arc::clone(prog), kernel.clone());
    Ok(kernel)
}

fn instantiate(prog: &Program) -> Result<Kernel, String> {
    let bytes = emit::module(prog)?;
    let engine = Engine::default();
    let module = Module::new(&engine, &bytes)
        .map_err(|e| format!("the wasm emitter produced an invalid module: {e}"))?;
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).map_err(|e| e.to_string())?;
    let mut linker = <Linker<()>>::new(&engine);
    let err = |e: wasmi::Error| e.to_string();
    let link = |e: wasmi::errors::LinkerError| e.to_string();
    linker.define("env", "memory", memory).map_err(link)?;
    let m = memory;
    linker
        .func_wrap(
            "env",
            "fusor2_jit_read",
            move |mut caller: Caller<'_, ()>,
                  ptr: i32,
                  bytes: i32,
                  index: i32,
                  mask: i32,
                  fill: i32,
                  elem: i32|
                  -> i32 {
                let data = m.data_mut(&mut caller);
                let ptr = data.as_mut_ptr().wrapping_add(ptr as u32 as usize);
                helpers::jit_read(
                    ptr,
                    bytes as u32 as usize,
                    index as u32,
                    mask as u32,
                    fill as u32,
                    elem as u32,
                ) as i32
            },
        )
        .map_err(link)?;
    linker
        .func_wrap(
            "env",
            "fusor2_jit_write",
            move |mut caller: Caller<'_, ()>,
                  ptr: i32,
                  bytes: i32,
                  index: i32,
                  value: i32,
                  mask: i32,
                  elem: i32| {
                let data = m.data_mut(&mut caller);
                let ptr = data.as_mut_ptr().wrapping_add(ptr as u32 as usize);
                helpers::jit_write(
                    ptr,
                    bytes as u32 as usize,
                    index as u32,
                    value as u32,
                    mask as u32,
                    elem as u32,
                );
            },
        )
        .map_err(link)?;
    linker
        .func_wrap(
            "env",
            "fusor2_jit_un",
            |code: i32, ty: i32, bits: i32| -> i32 {
                helpers::jit_un(code as u32, ty as u32, bits as u32) as i32
            },
        )
        .map_err(link)?;
    linker
        .func_wrap(
            "env",
            "fusor2_jit_bin",
            |code: i32, ty: i32, a: i32, b: i32| -> i32 {
                helpers::jit_bin(code as u32, ty as u32, a as u32, b as u32) as i32
            },
        )
        .map_err(link)?;
    linker
        .func_wrap(
            "env",
            "fusor2_jit_cast",
            |from: i32, to: i32, bits: i32| -> i32 {
                helpers::jit_cast(from as u32, to as u32, bits as u32) as i32
            },
        )
        .map_err(link)?;
    linker
        .func_wrap("env", "fusor2_jit_round", |mode: i32, bits: i32| -> i32 {
            helpers::jit_round(mode as u32, bits as u32) as i32
        })
        .map_err(link)?;
    linker
        .func_wrap("env", "fusor2_jit_narrow", |elem: i32, bits: i32| -> i32 {
            helpers::jit_narrow(elem as u32, bits as u32) as i32
        })
        .map_err(link)?;
    linker
        .func_wrap("env", "fusor2_jit_unpack", |bits: i32, high: i32| -> i32 {
            helpers::jit_unpack(bits as u32, high as u32) as i32
        })
        .map_err(link)?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(err)?;
    let run = instance
        .get_typed_func::<(i32, i32, i32, i32, i32, i32, i32, i32), ()>(&store, "run")
        .map_err(err)?;
    Ok(Kernel(
        Arc::new(Mutex::new(Inner {
            store,
            memory,
            run,
            frame: emit::frame_bytes(prog),
        })),
        None,
    ))
}

impl Kernel {
    pub fn run(&self, bufs: &[RawBuf], gid: [u32; 3], grid: [u32; 3]) {
        let Some((jit, prog)) = &self.1 else {
            return self.run_wasm(bufs, gid, grid);
        };
        let snapshot = |bufs: &[RawBuf]| -> Vec<Vec<u8>> {
            bufs.iter()
                .map(|b| unsafe { std::slice::from_raw_parts(b.ptr, b.bytes) }.to_vec())
                .collect()
        };
        let before = snapshot(bufs);
        jit.run(bufs, gid, grid);
        let native = snapshot(bufs);
        for (buf, bytes) in bufs.iter().zip(&before) {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr, buf.bytes) };
        }
        self.run_wasm(bufs, gid, grid);
        let wasm = snapshot(bufs);
        for (i, (n, w)) in native.iter().zip(&wasm).enumerate() {
            if let Some(at) = n.chunks(4).zip(w.chunks(4)).position(|(a, b)| a != b) {
                let word = |v: &[u8]| u32::from_le_bytes(v[at * 4..at * 4 + 4].try_into().unwrap());
                eprintln!(
                    "[wasm diff] buffer {i} word {at}: cranelift {:#x} ({}) wasm {:#x} ({}) gid {gid:?} grid {grid:?}\n{prog:?}",
                    word(n),
                    f32::from_bits(word(n)),
                    word(w),
                    f32::from_bits(word(w)),
                );
                break;
            }
        }
    }

    fn run_wasm(&self, bufs: &[RawBuf], gid: [u32; 3], grid: [u32; 3]) {
        self.run_each(bufs, grid, std::iter::once(gid));
    }

    /// Every workgroup of `gids` in one copy of the buffers in and out.
    pub fn run_each(&self, bufs: &[RawBuf], grid: [u32; 3], gids: impl Iterator<Item = [u32; 3]>) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Inner {
            store,
            memory,
            run,
            frame,
        } = &mut *inner;
        let table = align(FRAME + *frame);
        // One copy per distinct host buffer: a kernel may bind the same
        // buffer twice (an in-place scatter reads and writes it).
        let mut offsets: Vec<usize> = Vec::with_capacity(bufs.len());
        let mut at = align(table + bufs.len() * 8);
        for (i, buf) in bufs.iter().enumerate() {
            match bufs[..i].iter().position(|earlier| earlier.ptr == buf.ptr) {
                Some(same) => offsets.push(offsets[same]),
                None => {
                    offsets.push(at);
                    at = align(at + buf.bytes.max(4));
                }
            }
        }
        let pages = at.div_ceil(PAGE) as u64;
        let have = memory.size(&*store);
        if have < pages {
            memory
                .grow(&mut *store, pages - have)
                .expect("wasmi memory grows for a launch");
        }
        let data = memory.data_mut(&mut *store);
        for (i, (buf, offset)) in bufs.iter().zip(&offsets).enumerate() {
            let entry = table + i * 8;
            data[entry..entry + 4].copy_from_slice(&(*offset as u32).to_le_bytes());
            data[entry + 4..entry + 8].copy_from_slice(&(buf.bytes as u32).to_le_bytes());
            if buf.bytes > 0 {
                // SAFETY: `buf` is a live host buffer of `bytes` bytes and the
                // destination range was sized for it above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.ptr,
                        data.as_mut_ptr().add(*offset),
                        buf.bytes,
                    )
                };
            }
        }
        for gid in gids {
            memory.data_mut(&mut *store)[FRAME..FRAME + *frame].fill(0);
            run.call(
                &mut *store,
                (
                    table as i32,
                    gid[0] as i32,
                    gid[1] as i32,
                    gid[2] as i32,
                    grid[0] as i32,
                    grid[1] as i32,
                    grid[2] as i32,
                    FRAME as i32,
                ),
            )
            .expect("the wasm kernel trapped");
        }
        let data = memory.data(&*store);
        for (buf, offset) in bufs.iter().zip(&offsets) {
            if buf.bytes > 0 {
                // SAFETY: as above, in the other direction.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(*offset), buf.ptr, buf.bytes)
                };
            }
        }
    }
}
