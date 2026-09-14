# Browser shader support

`wgpu/` is the published **wgpu 29.0.4** crate, with its MIT/Apache licenses.
Only `src/backend/webgpu.rs` differs from the registry source; the complete
change is recorded in `wgpu-webgpu.patch`. Generated bindings and native
backends are unchanged.

The bridge maps and requests advertised `subgroups` and
`chromium-experimental-subgroup-matrix` features, reads subgroup widths and
matrix configurations, and emits the browser's required subgroup enable line.
Missing properties preserve conservative defaults.

`naga-browser/` vendors **Naga 29.0.3** from
[ealmloff/wgpu, c4f55cabaadc22b942c1dd8894bf85f17f2718dc](https://github.com/ealmloff/wgpu/tree/c4f55cabaadc22b942c1dd8894bf85f17f2718dc/naga),
including the existing cooperative-load barrier ordering fix. Its source,
build script and licenses are copied from that revision; Cargo workspace
inheritance is expanded into a standalone manifest, without upstream test
fixtures or development dependencies. Two source files change, captured in
`naga-browser.patch`.

The WGSL writer's `BROWSER_SUBGROUP_MATRIX` option serializes validated Naga
matrix types, loads, stores, zero values and multiply-accumulate operations
using Dawn's dialect. It handles both explicit and inferred matrix types.
Array pointers and offsets come from the IR; shader text is not rewritten.
The ordinary Session compiler can therefore use hardware matrices on the web.
The native dialect and Metal emitter remain available without this option.

Device initialization validates each advertised, supported matrix kind through
this same browser serialization path before admitting Session candidates.
Rejected kinds retain a diagnostic and fall back to portable matrices without
disabling subgroup collectives. TrainingProgram uses those validated caps and
also catches failures compiling its complete kernels.
`TrainingProgram::acceleration_fallback()` exposes the reason. Fixed programs
currently emit Dawn WGSL directly after validating their native-dialect twin.

All three Cargo roots patch both crates. Keep the patches with the vendored
source so they can be reviewed and rebased; remove them when an upstream
release supplies equivalent support. Dawn's experimental API can change
independently of wgpu. Browser regression checks live in
`../webgpu-runner/tests/training.cjs` and require the opt-in `training-checks`
feature, which is omitted from normal release builds.

API reference: [Dawn subgroup matrices](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/dawn/features/subgroup_matrix.md).
