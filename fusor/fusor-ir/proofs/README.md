# Machine-checked map-fusion model

`MapFusion.lean` contains a Lean proof, not a handwritten proof note.
It imports `Std` only, with no Lake or Mathlib dependency; `lean-toolchain` pins Lean 4.24.0.

Run from this directory:

```sh
lean -DwarningAsError=true MapFusion.lean
```

The path-filtered proof workflow runs the same command; `#print axioms` puts the theorem audit in CI output.

## Universal theorem and model

`eval_subst` is universal over values, semantics, expressions, and substitutions.
`single_splice` proves one splice for every fixed semantics; `map_fusion` lifts equality pointwise
over any shared index type and operand valuations.
`Value` is any Lean type. `Semantics` fixes total pure unary, binary, and ternary functions by
`Nat` operator tags; the functions are shared on both sides and no operator law is assumed.
`Expr Value arity` has `Fin`-bounded arguments, generic literals, uniforms, coordinates, and ordered children.

The splice starts from arbitrary original values `C : Fin (n+1) → Value`.
`insertSlot` maps retained fused ports into `C`'s numbering; `retainedSlot` maps original ports other than `k` back.
Lean proves these maps inverse on every retained port rather than assuming the renumbering is correct.
Fused operands append producer inputs after retained consumer inputs.
At `k`, the original environment reads the producer value; elsewhere it reads `C` directly.
Operand valuations may encode any fixed buffer/address reads at each point; the rewrite preserves them.
The producer and its shared outputs are not mutated. Both maps use the same point, coordinates, and uniforms.

Selecting the producer value at `k` assumes storage/reload is identity at its value type; no rounding or cast is erased.
This is not a verified typed refinement of Rust and proves neither the rule guards nor `ScalarExpr::compose` refinement.
It excludes floating-point lowering, effects, allocation, extraction, scheduling, binding, backend realizability,
and all other `MAP_INTO_MAP` branches.

Rust correspondence: [`rules.rs`](../src/rules.rs) (`shift_args`), [`scalar.rs`](../src/scalar.rs)
(`compose`, `map_children`), and [`rules/fusion.rs`](../src/rules/fusion.rs)
(`splice`, `absorb_chain`, `map_into_map`).
