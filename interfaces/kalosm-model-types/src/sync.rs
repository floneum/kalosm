use std::future::Future;

/// A trait that is not `Send` on wasm32 targets, but is on other targets.
#[cfg(target_arch = "wasm32")]
pub trait WasmNotSend {}

/// A trait that is not `Send` on wasm32 targets, but is on other targets.
#[cfg(not(target_arch = "wasm32"))]
pub trait WasmNotSend: std::marker::Send {}

#[cfg(target_arch = "wasm32")]
impl<T> WasmNotSend for T {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: std::marker::Send> WasmNotSend for T {}

/// A trait that is not `Send` or `Sync` on wasm32 targets, but is on other targets.
#[cfg(target_arch = "wasm32")]
pub trait WasmNotSendSync {}

/// A trait that is not `Send` or `Sync` on wasm32 targets, but is on other targets.
#[cfg(not(target_arch = "wasm32"))]
pub trait WasmNotSendSync: std::marker::Send + std::marker::Sync {}

#[cfg(target_arch = "wasm32")]
impl<T> WasmNotSendSync for T {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: std::marker::Send + std::marker::Sync> WasmNotSendSync for T {}

/// A future that is not `Send` on wasm32 targets, but is on other targets.
pub trait FutureWasmNotSend: Future + WasmNotSend {}
impl<T: Future + WasmNotSend> FutureWasmNotSend for T {}
