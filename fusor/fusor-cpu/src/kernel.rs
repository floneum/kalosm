//! The runnable form of a compiled [`Program`](crate::emit::Program), chosen
//! per build: native Cranelift code, or a wasm module generated from the same
//! tape and run either by the browser (wasm32) or by `wasmi` (the
//! `wasm-emit` feature, which lets the native conformance suite exercise the
//! wasm emitter).

use std::sync::Arc;

use crate::emit::Program;

#[cfg(all(not(target_arch = "wasm32"), not(feature = "wasm-emit")))]
pub use crate::jit::JitKernel as Kernel;
#[cfg(target_arch = "wasm32")]
pub use crate::wasm::browser::Kernel;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasm-emit"))]
pub use crate::wasm::native::Kernel;

pub(crate) fn compile(prog: &Arc<Program>) -> Result<Kernel, String> {
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "wasm-emit")))]
    {
        crate::jit::compile(prog)
    }
    #[cfg(all(not(target_arch = "wasm32"), feature = "wasm-emit"))]
    {
        crate::wasm::native::compile(prog)
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm::browser::compile(prog)
    }
}
