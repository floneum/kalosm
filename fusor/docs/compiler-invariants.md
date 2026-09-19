# Compiler construction and selection

The e-graph contains equivalent, well-formed alternatives. A rewrite constructs
an equivalent value and unions it with the original. It must establish the
preconditions for its own transformation before inserting nodes. There is no
production validation pass, invalid-member blacklist, or numerical admission
check after extraction.

Rules read shapes, types, numeric contracts and device capabilities through
`Facts` and `Builder`. They do not read timing observations or the cost model.
Schedule constructors describe supported combinations. Cooperative geometry and
staging depth are paired because their scratch requirements are coupled;
split-K is a graph rewrite with explicit partial and combine operations.
Promotion derives the reduction domain for its new accumulator footprint.
Contraction variants restore the original output axes before union.

Extraction chooses among these alternatives and constructs an executable DAG.
A cyclic combination of otherwise equivalent members is not an extraction.
Materialization, layout and binding choices must compose into an executable
plan; those obligations are part of plan construction. The cost model and
timing cache rank choices. They cannot declare an incorrect kernel acceptable,
remove a miscompile with a blacklist, or introduce variants absent from the
graph. The timing cache stores only measured durations and chosen combinations.

Independent checks live in compiler tests. `compiler-tests` enables plan and
kernel invariant checks and the differential member harness. The conformance
binary enables it by default; ordinary `fusor` users and browser benchmark
builds do not. Browser regression builds opt in through `training-checks`.
Naga still computes the module analysis required by its shader backend; its
optional validation flags are enabled only in compiler test builds.

Constructor tests check every generated cooperative schedule against resource
limits, promotion under scratch pressure, absent CPU schedules, and output
shapes across contraction families. Numerical conformance samples each member
independently of tuning budgets and the timing cache. Small schedule domains are
exhaustive; larger domains sample boundaries and the midpoint. A test invariant
failure is reported, never converted into a slower fallback.

Run from `fusor/`:

```sh
cargo test --release --workspace --lib --tests
cargo run --release -p fusor-conformance
cargo clippy --workspace --all-targets -- -D warnings
```
