# Hand proof: one equal-space `MAP_INTO_MAP` splice

## Scope and model

Here an **interaction graph** means a pure, acyclic, typed, port-labelled dataflow graph. Each node has ordered input ports; an edge records its source and the indexed read-address map used at that port. A `Launch::Map` node evaluates its scalar body at each point of its index space. This is a local graph/dataflow definition, **not** Interaction Nets or Girard's interaction-net theory.

This proof covers one splice only:

- the consumer and producer are both `Launch::Map` nodes over the same full index space `I`;
- consumer port `k` reads the producer through a dense `Alias` edge, at the same point of `I`;
- both bodies are finite, well-typed scalar expressions; every `Arg` index is in range, and each `IndexOf` axis is in range for `I`;
- every operand address map and read is defined and in bounds at each `t ∈ I`; address and argument-index arithmetic is representable, without overflow;
- producer and consumer use one immutable input and uniform-value snapshot;
- scalar primitives are deterministic, pure, and congruent for equal, equally typed children; index coordinates have the same context before and after the splice;
- removing the producer's storage/reload boundary is the identity at its value type. In particular, this proof does not erase a rounding or cast at that boundary.

Backend contraction and excess-precision behavior are outside this model. The proof does not cover widening or broadcasted index spaces, `Logical::Map` normalization, folds, contractions, or any other `MAP_INTO_MAP` branch. It proves neither that a candidate can be extracted nor that it can be scheduled, bound, or legally lowered by a backend.

The only observations proved equal are typed output tensor values at each point. Allocation, timing, and error behavior are not observations in this theorem.

Write `eval(e, A, t, U)` for the recursively defined value of a scalar expression under operand values `A`, coordinate `t`, and immutable uniform snapshot `U`. For each operand, its edge label determines the value read at `t`; equality below is at the operand's declared type. The index-space and edge-address assumptions ensure both maps use the same coordinate at each corresponding point.

## One-port substitution

Let the consumer have `n` operands `O = [o₀, …, oₙ₋₁]` and body `e_c`. Let the producer have `m` operands `P = [p₀, …, pₘ₋₁]` and body `e_p`. Assume `0 ≤ k < n`; producer body arguments are in `0..m`, consumer body arguments in `0..n`. At a fixed point `t ∈ I`, write `V_j(t)` for the value read by consumer operand `o_j`, and `Q_i(t)` for the value read by producer operand `p_i`.

The fused operand list and its producer-argument base are

```text
O'   = [o₀, …, oₖ₋₁, oₖ₊₁, …, oₙ₋₁, p₀, …, pₘ₋₁]
base = n - 1
```

By definition of edge evaluation, `values(O, t) = [V₀(t), …, Vₙ₋₁(t)]`. The equal-space splice preserves the retained consumer edges and copies producer operands with their access and layout labels unchanged. Thus

```text
values(O', t) = [V_0(t), ..., V_{k-1}(t), V_{k+1}(t), ..., V_{n-1}(t), Q_0(t), ..., Q_{m-1}(t)].
```

This identity is pointwise: retaining the same indexed edge at the same `t` preserves its read, and appending an unchanged producer operand reads the same `Q_i(t)` as in the producer map.

For a producer expression, let `shift_base` replace `Arg(i)` by `Arg(base+i)` and leave `Lit`, `Uniform`, and `IndexOf` unchanged. Define the consumer substitution `σ` on its argument leaves by

```text
σ(Arg(j)) = Arg(j)                 when j < k
σ(Arg(k)) = shift_base(e_p)
σ(Arg(j)) = Arg(j - 1)             when j > k
```

and extend it recursively over every other expression constructor. The fused body is `e'_c = σ(e_c)`. Each consumer argument is thus replaced by the port that now carries its value; every occurrence of `Arg(k)` receives the same shifted producer body.

Let `Γ_c` and `Γ_p` be the consumer and producer operand-type vectors. Launch typing gives `type(e_p) = Γ_c[k]`; the fused vector has types `Γ' = Γ_c` with entry `k` removed, followed by `Γ_p`. Each clause of `σ` therefore replaces a consumer argument with an expression of the same type, so the recursive substitution is well-typed.

### Lemma 1: producer arguments shift correctly

For every well-typed producer expression `e_p` and every `t ∈ I`,

```text
eval(shift_base(e_p), values(O', t), t, U)
  = eval(e_p, [Q₀(t), …, Qₘ₋₁(t)], t, U).
```

Proof by structural induction on `e_p`. For `Arg(i)`, the range assumption gives `i < m`; `Arg(base+i)` in `O'` denotes `p_i`, hence has value `Q_i(t)` and the same type. `Lit` is unchanged. `Uniform` reads the same immutable snapshot `U`. `IndexOf(a)` reads the same coordinate `t[a]`, since `I` and its index context are unchanged. For every compound constructor—unary, binary, comparison, select, cast, bitcast, round, dot, or splat—the induction hypotheses give equal, equally typed values for corresponding children. Determinism and congruence of that primitive, with its operator, result type, and ordered child positions unchanged, give equal parent values. These cases exhaust the scalar-expression constructors.

### Lemma 2: the substituted consumer body preserves its value

At the fixed point `t`, the values of the new operands corresponding to old consumer operands are

```text
values(O', t)[j]     = V_j(t)       for j < k
values(O', t)[j - 1] = V_j(t)       for j > k
```

The dense `Alias` edge reads the producer at `t`; the identity-boundary assumption therefore gives `V_k(t) = eval(e_p, [Q₀(t), …, Qₘ₋₁(t)], t, U)`. By Lemma 1, `shift_base(e_p)` in the fused operand environment has exactly `V_k(t)`.

For every consumer argument leaf, `σ` consequently evaluates to the value of that original argument: the two retained-port cases follow from the displayed operand identities, and the removed-port case follows from Lemma 1. For `Lit`, `Uniform`, and `IndexOf`, substitution leaves the leaf unchanged and it observes the same literal, snapshot, or coordinate. Structural induction on `e_c` now gives

```text
eval(e'_c, values(O', t), t, U) = eval(e_c, values(O, t), t, U).
```

At a compound node, corresponding children have equal values and types by induction; congruence of the unchanged, ordered primitive gives equal results. Repeated occurrences of any argument are covered independently by the leaf case, so repeated uses of the producer value remain equal.

### Theorem: one splice preserves the map result

The maps have the same full index space `I`. Lemma 2 holds at every `t ∈ I`; therefore the original and fused maps are extensionally equal at every output point. Each replacement has the type of the argument it replaces, so induction over constructors also preserves the body's result type. This establishes the semantic correctness of this one graph alternative under the stated assumptions.

The graph is append-only: adding the fused consumer does not mutate the producer, its operand edges, or any other consumer of it. A sibling consumer continues to read the original producer. Downstream context is preserved by topological induction: if a node's input values and indexed read addresses agree, its unchanged pure deterministic operation produces the same value. Thus sharing the producer does not require rewriting its other consumers. For a finite sequence of `r` splices, induction on `r` gives preservation if every individual step satisfies these same assumptions: zero steps is identity, and each next valid splice preserves the current graph's value by this theorem. This is not a proof of all cases the implementation's greedy loop may encounter.

## Correspondence obligations in Rust

These paths identify the implementation points to compare with the model; this handwritten argument does not prove that the Rust code refines the model.

| Model obligation | Fusor source |
| --- | --- |
| Build `O'` and the port-indexed scalar substitutions, including the `base = n - 1` producer argument shift | [`rules.rs`](../src/rules.rs): `splice_args`, `shift_args`, and `operand_dtypes` |
| Define expression substitution and rebuild children with the original scalar constructor/order | [`scalar.rs`](../src/scalar.rs): `ScalarExpr::compose`, `rewrite`, and `map_children` |
| Enforce alias access, substitution-domain coverage, and dense producer reads; preserve producer operands unchanged in the equal-space branch | [`rules/fusion.rs`](../src/rules/fusion.rs): `splice`, `covers_for_substitution`, and `reads_producer_densely` |
| Apply repeated splices and compose each splice into the current body | [`rules/fusion.rs`](../src/rules/fusion.rs): `absorb_chain` |
| Restrict the entry point to a `Launch::Map`, build the alternative, and record its equality assertion | [`rules/fusion.rs`](../src/rules/fusion.rs): `map_into_map` |
| Represent the pure, acyclic graph and append-only equality alternative | [`egraph.rs`](../src/egraph.rs) and [`ir/mod.rs`](../src/ir/mod.rs) |

`map_view` also recognizes `Logical::Map`; that producer spelling is excluded here. The entry point may greedily apply multiple eligible splices, but this proof covers only the single equal-space `Launch::Map` step. `Union` records the implementation's equality assertion; it is not itself a proof. Rust regression tests and the `compiler-tests` differential harness are useful evidence, not a machine-checked proof or a verified refinement argument. Even a proof of semantic equality does not show that extraction selects the alternative or that a particular schedule, binding, or backend lowering is realizable and legal.
