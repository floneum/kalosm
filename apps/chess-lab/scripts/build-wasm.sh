#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# SIMD for the CPU play search (all current browsers support wasm simd128); the
# function table is exported so Fusor's compiled CPU kernels can be called as
# plain function pointers.
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128 -C link-arg=--export-table" cargo build --manifest-path fusor-chess/Cargo.toml --target wasm32-unknown-unknown --lib --release --locked
"${WASM_BINDGEN:-wasm-bindgen}" fusor-chess/target/wasm32-unknown-unknown/release/rookie_fusor.wasm --target web --out-dir src/wasm --out-name rookie_fusor
