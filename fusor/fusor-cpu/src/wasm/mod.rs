//! The wasm form of a compiled kernel: the same [`Program`](crate::emit::Program)
//! the Cranelift emitter lowers, generated as a wasm module instead. The
//! browser instantiates it against the host's own linear memory, so kernels
//! read and write the host's buffers directly; natively the `wasm-emit`
//! feature runs the same modules under `wasmi` so the conformance suite
//! verifies the emitter.

#[cfg(target_arch = "wasm32")]
pub(crate) mod browser;
#[cfg(any(target_arch = "wasm32", feature = "wasm-emit"))]
pub(crate) mod emit;
#[cfg(all(not(target_arch = "wasm32"), feature = "wasm-emit"))]
pub(crate) mod native;
