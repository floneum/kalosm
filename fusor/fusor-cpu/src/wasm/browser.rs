//! The emitted modules run in the browser against the host module's own
//! memory: the generated module imports `memory` and the eight helpers from
//! `wasm_bindgen::exports()`, so a kernel reads and writes host buffers in
//! place. Its `run` export goes into the host's function table when one is
//! exported, which makes a launch a plain indirect call; otherwise it is
//! called through `Function.apply`.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use js_sys::{Array, Function, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::{JsCast, JsValue};

use super::emit::{self, HELPERS};
use crate::emit::{Program, RawBuf};
use crate::helpers;

type Entry = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, i32, i32);

#[derive(Clone)]
pub struct Kernel(Arc<Inner>);

struct Inner {
    call: Call,
    frame: UnsafeCell<Box<[u8]>>,
}

enum Call {
    Table(Entry),
    Js(Function),
}

// SAFETY: wasm32 here is single-threaded, so no two launches of one kernel
// ever overlap and the frame is never shared.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Kernel(wasm)")
    }
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
    let kernel = instantiate(prog)?;
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(Arc::clone(prog), kernel.clone());
    Ok(kernel)
}

fn js(error: JsValue) -> String {
    format!("{error:?}")
}

fn instantiate(prog: &Program) -> Result<Kernel, String> {
    let bytes = emit::module(prog)?;
    let module =
        WebAssembly::Module::new(&Uint8Array::from(bytes.as_slice()).into()).map_err(js)?;
    let exports = wasm_bindgen::exports();
    let env = Object::new();
    Reflect::set(&env, &"memory".into(), &wasm_bindgen::memory()).map_err(js)?;
    for (name, ..) in HELPERS {
        let helper = Reflect::get(&exports, &JsValue::from_str(name)).map_err(js)?;
        if !helper.is_function() {
            return Err(format!("the host module does not export {name}"));
        }
        Reflect::set(&env, &JsValue::from_str(name), &helper).map_err(js)?;
    }
    let imports = Object::new();
    Reflect::set(&imports, &"env".into(), &env).map_err(js)?;
    let instance = WebAssembly::Instance::new(&module, &imports).map_err(js)?;
    let run: Function = Reflect::get(&instance.exports(), &"run".into())
        .map_err(js)?
        .dyn_into()
        .map_err(|_| "the kernel module's run export is not a function".to_string())?;
    let call = match function_table(&exports) {
        Some(table) => {
            let index = table.grow(1).map_err(js)?;
            table.set(index, &run).map_err(js)?;
            // SAFETY: on wasm32 a function pointer is its table index, and the
            // entry's wasm type is exactly `Entry`'s.
            Call::Table(unsafe { std::mem::transmute::<usize, Entry>(index as usize) })
        }
        None => Call::Js(run),
    };
    Ok(Kernel(Arc::new(Inner {
        call,
        frame: UnsafeCell::new(vec![0u8; emit::frame_bytes(prog)].into_boxed_slice()),
    })))
}

/// The host module's function table, when its build exported it
/// (`-C link-arg=--export-table`). The externref table wasm-bindgen exports
/// cannot hold a function.
fn function_table(exports: &JsValue) -> Option<WebAssembly::Table> {
    let table = Reflect::get(exports, &"__indirect_function_table".into()).ok()?;
    table
        .is_instance_of::<WebAssembly::Table>()
        .then(|| table.unchecked_into())
}

impl Kernel {
    pub fn run(&self, bufs: &[RawBuf], gid: [u32; 3], grid: [u32; 3]) {
        // SAFETY: single-threaded; the frame is scratch for this call alone.
        let frame = unsafe { (*self.0.frame.get()).as_mut_ptr() } as i32;
        let args = [
            bufs.as_ptr() as i32,
            gid[0] as i32,
            gid[1] as i32,
            gid[2] as i32,
            grid[0] as i32,
            grid[1] as i32,
            grid[2] as i32,
            frame,
        ];
        match &self.0.call {
            // SAFETY: the table entry has `Entry`'s type (see `instantiate`).
            Call::Table(entry) => unsafe {
                entry(
                    args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                )
            },
            Call::Js(function) => {
                let list = Array::new();
                for arg in args {
                    list.push(&JsValue::from(arg));
                }
                function
                    .apply(&JsValue::UNDEFINED, &list)
                    .expect("the wasm kernel trapped");
            }
        }
    }
}

// The helpers a generated module imports. Exported from the host module under
// the names `HELPERS` lists.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_read(
    ptr: *const u8,
    bytes: usize,
    index: u32,
    mask: u32,
    fill: u32,
    elem: u32,
) -> u32 {
    helpers::jit_read(ptr, bytes, index, mask, fill, elem)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_write(
    ptr: *mut u8,
    bytes: usize,
    index: u32,
    value: u32,
    mask: u32,
    elem: u32,
) {
    helpers::jit_write(ptr, bytes, index, value, mask, elem)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_un(code: u32, ty: u32, bits: u32) -> u32 {
    helpers::jit_un(code, ty, bits)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_bin(code: u32, ty: u32, a: u32, b: u32) -> u32 {
    helpers::jit_bin(code, ty, a, b)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_cast(from: u32, to: u32, bits: u32) -> u32 {
    helpers::jit_cast(from, to, bits)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_round(mode: u32, bits: u32) -> u32 {
    helpers::jit_round(mode, bits)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_narrow(elem: u32, bits: u32) -> u32 {
    helpers::jit_narrow(elem, bits)
}
#[unsafe(no_mangle)]
pub(crate) extern "C" fn fusor2_jit_unpack(bits: u32, high: u32) -> u32 {
    helpers::jit_unpack(bits, high)
}
