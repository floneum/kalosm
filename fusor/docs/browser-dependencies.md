# Browser shader dependencies

All three Cargo roots pin `wgpu` and `naga` to
[`ealmloff/wgpu@24a7d3577213fc147637fe7fb20e0e845a74df8a`](https://github.com/ealmloff/wgpu/commit/24a7d3577213fc147637fe7fb20e0e845a74df8a).
The dependency code lives in that fork, not in this repository's source tree.

The revision extends the existing `c4f55cabaadc22b942c1dd8894bf85f17f2718dc`
cooperative-load ordering fix. Three source files implement browser subgroup
feature requests, subgroup width/matrix configuration discovery, and Dawn WGSL
matrix serialization. The wgpu package remains 29.0.4 and resolves its native
backend dependencies from the registry, preserving the previously tested graph.

Device initialization probes advertised matrix kinds. Unsupported capabilities
and rejected shaders fall back to the portable implementation; the ordinary
Session executor and TrainingProgram use the same validated capabilities.
Browser capability checks are in `webgpu-runner/tests/training.cjs`.

The fork passed `cargo fmt --all --check` and
`cargo clippy -p wgpu -p naga --tests`. Fusor's native model oracles and browser
regressions validate the application paths. First Cargo resolution fetches the
pinned Git dependency; subsequent offline builds use Cargo's cache.
