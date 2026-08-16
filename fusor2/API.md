# The fusor2 public API

The target, stated once: **a model crate ports from fusor1 to fusor2 by
changing its imports.** Same shape — const generics plus type parameters,
infallible calls, a flat crate root. Not the same items: fusor1's warts stay
out, and fusor2's better ideas stay in.

This file is the intended surface. Anything `pub` that is not listed here is
either on its way off the surface or is an oversight; either way, name it here
before adding it back.

---

## 1. The root

```rust
use fusor2::{Tensor, Device, Graph, Session, QMatrix, VarBuilder, Dim, Dtype, Error, Result};
```

| Item | Why it is public |
| --- | --- |
| `Tensor<const R: usize, T: Element = f32>` | The headline type. Rank and dtype in the type, every op infallible. Matches fusor1's `Tensor<R, D, B>` in *shape* minus the `B: Fusion` slot, which had nothing to say once the e-graph owned materialization. |
| `Element` | The dtype witness (`f32`, `f16`, `bf16`, `u32`, `i32`). Needed to write a `T`-generic layer. fusor1 spelled it `SimdElement`; that alias survives at `tensor::SimdElement`. |
| `Axis`, `Minus1`, `Minus2` | Axis selectors: `x.sum::<2>(Minus1)`. Strictly better than fusor1's `D`/`Dim` marker zoo — `usize` also implements `Axis`, so `x.sum::<2>(1)` reads as it should. **Kept.** |
| `cat`, `stack` | Free functions over an `IntoIterator`, as fusor1 had them. `Tensor::cat` is the associated spelling. |
| `Device` | One device type: backend + session + graph. What constructors take **and** what `Tensor::device()` returns. |
| `Graph` | The program under construction. A model names it when loading weights and when scoping a training step. |
| `Session` | Device + cost model + extractor + plan cache. Models do not name it; conformance, benches and examples do, and it is the object `resolve` belongs to. |
| `QMatrix` | A block-quantized weight matrix. `kalosm-llama`, `rbert` and `rwhisper` all load one. |
| `VarBuilder`, `ShardedVarBuilder` | GGUF weight loading. Re-exported from `fusor2-gguf` so a model needs one dependency, not two. |
| `Dim` | An extent: `Const` or `Sym`. Load-bearing — the whole decode-with-one-plan trick is a symbolic sequence length. |
| `Dtype` | A runtime dtype. Named at every loader boundary, where the dtype is data. |
| `Error`, `Result` | The error type of the fallible boundary (loading, readback, the `Dyn` layer). |
| `ToVec` | The readback trait behind `pollster::block_on(t.as_slice())?.to_vec()`. Kept for `betlang-train`, which imports it by that name. |

## 2. Modules

| Module | Contents that are public, and why |
| --- | --- |
| `tensor` | `Dyn` — the runtime-rank, `Result`-returning tensor. The **escape hatch**, reachable by `Tensor::into_dyn`/`as_dyn`, for code where a rank or a dtype is genuinely data. Also `Extent`, `RoundMode`, `IndexOp`/`TensorIndex`, `arange`/`arange_step`, `TensorSlice`, `ToVec`, `Element`/`Axis` (re-exported to the root). |
| `device` | `Device`, `Cpu`, `Gpu`, `KernelProfile`, `KernelProfileRow`. The profile types are how a trainer reads per-kernel timings. |
| `session` | `Session`, `Backend` (the CPU/GPU *selector* — `Session::new(Backend::gpu_blocking()?)`), `wrong_member_count`, the tuning constants. |
| `graph` | `Graph`, `GraphRef`, `Gradients`. |
| `layers` | `Linear`, `RmsNorm`, `LayerNorm`, `LayerNormNd`, `Embedding`, `ConvNd`. Every model crate uses these. |
| `cache` | `KvCache`, `TensorCache`, `MaskCache`, `AttentionMask`, `MaskKind`, `RopeCache`. The decode-loop state every model threads. |
| `quantized` | `QMatrix`. |
| `composite` | The op library at the `Dyn` layer: attention, rope, conv, pool, normalization, loss, upsample. Public because the conformance suite drives it directly and because it is where a new op is added. The **model-facing** entry points become methods on `Tensor` (§4). |
| `autograd` | The differentiable const-rank tensor, the tape, `with_backwards`, `BackwardTarget`, `GradientSlot`. `betlang-train`'s first `use` line. |
| `optim` | `AdamW`, `cosine_decay`, `global_norm`, `clip_global_norm`. No in-workspace consumer; `betlang-train` is the out-of-workspace one. |
| `sampling` | `StandardSamplerParams`, `Mirostat2Sampler`, `top_k_pairs`, `GpuSampledToken`. `kalosm-llama` calls `top_k_pairs`. |

## 3. Removed from the public surface

Landed in this change:

| Removed | Where it went, and why |
| --- | --- |
| the `typed-api` feature | **Deleted.** It swapped what `fusor2::Tensor` and `fusor2::Device` *named*, which made it non-additive: `--all-features` was not a valid configuration of the workspace (133 errors, measured). One root now, unconditionally. The `mod root` indirection, the two `#[cfg]` re-export arms and the two type-identity `const`s that policed them are gone with it. |
| `fusor2::Tensor` = the runtime-rank tensor | It is `tensor::Dyn` now, and `fusor2::Tensor` is the const-rank one. The struct was **renamed**, not aliased: a `pub(crate) type Tensor = Dyn` keeps the ~40 in-crate `use crate::tensor::Tensor` lines meaning what they meant, and puts nothing back on the public surface. |
| `session::Device` | Renamed `session::Backend`. There were two types named `Device` and `Tensor::device()` returned the *wrong one* — a different type than the `&Device` its own constructors take, so `Tensor::zeros(&x.device(), shape)` did not compile. That was a bug, not a naming preference. |
| `Tensor::device() -> session::Device` | Returns `Device`. `Tensor::backend()` is the selector, if a caller genuinely wants to branch on CPU/GPU. |
| `mod broadcast` | Private. Its `broadcast_as`/`expand` are inherent methods and stay public; `align`, `expand_to` and `result_shape` had **no caller anywhere** and are deleted. |
| `mod ops` | `pub(crate)`. It holds inherent `impl` blocks — those stay public — plus internals nothing outside named. |
| `Gradients` (root) | `graph::Gradients`. |
| `TensorSlice` (root) | `tensor::TensorSlice`. It is the argument type of a readback, not a name a caller writes. |
| `Typed` (root) | Deleted from the root. It was the pre-rename spelling of the const-rank tensor; the root *is* that tensor now. `tensor::Typed` remains as the compatibility alias. |
| `Persistence`, `QFmt`, `QLayout` (root) | IR-level detail. No consumer imports them: the conformance suite takes them from `fusor2_ir::dtype` directly, and models never spell them (a `QMatrix` gets its format from `Dtype::Q(fmt)` destructuring and from `RawTensorBytes`, both inferred). |
| `RoundMode` (root) | `tensor::RoundMode`. One op takes it. |
| `SymId` (root) | Off the root. Nothing in or out of this workspace imports it; `Dim` is the type callers name. |

One item went the other way. `composite::attention::MaskKind` was a *private*
`use` of `fusor2_ir::ir::launch::MaskKind`, so `attention_masked` took an
argument no caller could spell — `segment-anything-rs` had two lines that
could not compile against any version of this crate. It is a `pub use` now.

## 4. Not being added, from fusor1

The point of the port is fusor1's *ergonomics*, not its inventory.

- **The `*_fused` families.** Not merely absent from the const-rank surface:
  **deleted from the crate**, `Dyn` layer included. A macro op unions the sugar
  node with its expansion in the same call, so **how many kernels it launches
  is the extractor's answer, not the caller's**, and every one of these names
  was a one-line delegation to the natural spelling. Offering the alias would
  ask a model author for a decision that has no effect.

  Deleted outright, each a pure delegation: `rope_fused` (→`rope_interleaved`),
  `rope_normal_fused` (→`rope`), `softmax_last_dim_fused` and `softmax_last`
  (→`softmax_last_dim`), `softmax_slow_last` (→`softmax_slow_last_dim`),
  `rms_norm_fused_no_bias` (→`rms_norm`), `layer_norm_last_dim_fused`
  (→`layer_norm`). Every one had a conformance case; each case's natural-
  spelling twin was already registered, so no numeric coverage was dropped.

  Kept, having lost only a suffix that named a kernel rather than an operand:
  - `rms_norm_fused` is **`rms_norm_with_bias`** — `bias` is a real operand,
    not a fusion hint.
  - `rms_norm_residual_fused` is **`rms_norm_residual`** — the add is *inside*
    the norm's expansion, a different node than `(x + r).rms_norm(..)`.
  - `rope_normal_pair_fused` is **`rope_pair`** and `rope_pair_fused` is
    **`rope_interleaved_pair`** (`_with_position` likewise). These rotate `q`
    and `k` in **one** node and hand back two views, which is not something the
    single form composes to.
- **`dispatch_pair` / `dispatch_triple` / `dispatch_quad`.** Manual fusion
  scheduling. The resolver does this.
- **`as_cpu_mut`, `unwrap_gpu`, backend juggling.** fusor2 has no CPU/GPU
  tensor split at all — one tensor, the session owns the backend. That is an
  improvement and it stays.
- **`to_concrete` / `into_concrete`.** They exist today as the identity, so the
  trainer's spellings resolve. They are not part of the intended surface and
  are documented as no-ops; nothing new should call them.
- **`softmax_slow`, `softmax_slow_last_dim`.** The
  "expansion with no sugar node over it" is a compiler-testing spelling, not a
  model-facing one. It stays at the `Dyn` layer.
- **The rank-witness traits** — `LastRank`, `NextRank`, `SmallerRank`,
  `LargerRank`, `MaxRank`, `ShapeWithOneHole` — and the rank ceiling of 21. A
  rank-changing op takes its output rank as an ordinary const parameter and
  validates it once, with a panic that names the op.
- **`B: Fusion<R, D>`.** The e-graph decides materialization.

## 5. The ergonomic target: methods, not module paths

The tell that the port is unfinished is a model reaching three module levels
deep for something fusor1 had as a method:

```rust
// today
let scores = fusor2::composite::attention::attention_masked(&q, &k, &v, scale, mask)?;
let (q, k) = fusor2::composite::rope::rope_pair(&q, &k, &cos, &sin, offset)?;
let up = fusor2::composite::upsample::upsample_nearest2d(&x, 4)?;

// intended
let scores = q.attention_masked(&k, &v, scale, mask);
let (q, k) = q.rope_pair(&k, &cos, &sin, offset);
let up = x.upsample_nearest2d(4);
```

These become inherent methods on `Tensor<R, T>` — one rank-checked signature
each, no `Result`. The free functions stay at the `Dyn` layer underneath them.

## 6. The ops the models needed — delivered

Derived by intersecting every method call in the four model crates'
**compiled** source (`models/kalosm-llama/src/raw/vision/` is not a module —
it is unported fusor1 code sitting in the tree, and it was excluded) with the
op surface of `Dyn`, then subtracting what `Tensor<R, T>` already exposed.
All of it now lives on the const-rank tensor, in
`tensor/typed/{ops,composite,construct}.rs`. Each method wraps the `Dyn`
implementation of the same name and re-implements no math: the claim that a
method and its free function are the *same node* is asserted directly, in
`typed::composite::tests::a_method_and_its_free_function_are_the_same_node`.

Shape and readback (`typed/ops.rs`):

- `extents() -> [Dim; R]`, `extent(axis) -> Dim` — **symbolic-safe**. `shape()`
  panics on a `Dim::Sym`, and the whole one-plan decode loop is a `Dim::Sym`
  sequence length.
- `elem_count() -> Option<u64>` — the total form; `elements() -> usize` is the
  panicking one. The old `elem_count` was an alias of `elements` and had no
  caller in or out of this workspace.
- `set_bytes(Vec<u8>)` / `set_elements(&[T])` — the per-step token and position
  leaves. The node id is unchanged, which is the whole reason one resolved plan
  survives a decode step.
- `to_vec_f32` / `to_vec_u32` / `to_vec_i32` / `to_bytes` — readback that
  *converts*; `to_flat()` is same-dtype. `to_vec_f32_async` / `to_flat_async`
  return `Result` rather than panicking: an `await` point is exactly where a
  caller does have somewhere to put the error.
- `reshape_dims([Dim; O])`, `broadcast_dims`, `resize_dims` — the symbolic
  spellings.

Views and indexing (`typed/ops.rs`): `flatten_last_n`, `flatten_first_n`,
`flatten`, `squeeze_dims`, `unsqueeze_dims`, `repeat`, `resize`, `pad_axis`,
`pad_with_zeros`, `windows`, `embedding`, `gather_last`, `slice_assign`, and
`Tensor<1, T>::top_k`.

Ops as methods (`typed/composite.rs`) — this is §5's ergonomic target, landed:
`softmax`, `softmax_last_dim`, `log_softmax`, `rms_norm`, `rms_norm_no_weight`,
`rms_norm_residual`, `layer_norm`, `attention`, `attention_causal`,
`attention_masked`, `rope`, `rope_interleaved`, `rope_pair`,
`rope_interleaved_pair`, `rope_at`, `rope_pair_at`,
`rope_interleaved_pair_at`, `pool`, `pool_max`,
`pool_min`, `pool_avg`, `upsample_nearest2d`, `upsample_bilinear`, and
`q_mat_mul` — on the **activation**, as the reference has it, rather than the
inverted `QMatrix::q_mat_mul(&act)` receiver, which reads backwards in a
forward pass. `base_inverse_frequency` stays a free function at
`composite::rope`: it takes no tensor, so there is no receiver for it.

The rope family is a **square**, not a list: pairing (`rope` halves,
`rope_interleaved` pairs `2i, 2i+1`) is architecture data a checkpoint fixes,
while the offset form (a host `u64`, or a device `positions` vector so the plan
survives a decode step) is the *loop's* choice. `rope_interleaved_pair_at` was
the missing fourth corner — `kalosm-llama` picks its pairing from
`general.architecture` and its offset form from whether it is prefilling, so it
reaches all four.

`QMatrix::q_mat_mul` promotes a rank-3-or-higher activation the way it already
promoted a rank-1 one: a weight is rank 2, `matmul_t` shares no batch rank with
it, so the leading axes fold into the row axis and are restored afterwards.
`kalosm-llama` and `rwhisper` each carried a byte-identical private helper
because the method stopped at rank 2; the promotion builds the views they
built, and a symbolic leading extent is reported rather than `expect`ed.

Construction (`typed/construct.rs`): `Tensor::new(&device, array)` — the
reference's spelling, which is why wrapping a runtime-rank value moved to
`from_dyn`/`try_from_dyn` (`new`/`try_new` had exactly one call site in this
workspace, a test) — plus `full`, `zeros_dims`, `param`, `leaf`, `arange`,
`arange_step`, and `from_raw_bytes(device, dtype, [Dim; R], bytes)`. `leaf` is
the decode loop's constructor: a step-local input buffer, minted once and
refilled with `set_elements` per step, so the node id — and the resolved plan
over it — outlives the step. It is not `param` (nothing registers it as a
weight) and it carries no name, because `Graph::leaf` discards the one it is
handed. That last
one is the weight-load constructor: a GGUF entry's dtype is *data* read from
the file while its rank is program structure the model knows, and it casts to
`T` when the two differ, so a checkpoint loads without the cast being written
at every site. `QMatrix::to_tensor` and `QMatrix::rows_at` are const-rank
because a `QMatrix` *is* rank 2.

## 7. Layers and caches are generic

Both families now carry the rank and element parameters the reference gives
them, each defaulted so the bare name still means the common case and no model
writes a turbofish it did not write before.

| Type | Parameters | The reference's |
| --- | --- | --- |
| `Linear<T = f32>` | element only — the weight is `[out, in]` by definition | `Linear<T: SimdElement>` |
| `RmsNorm<const N = 1, T = f32>` | `N` is the **weight's** rank | `RmsNorm<const N: usize, T>` |
| `LayerNorm<const N = 1, T = f32>` | same | `LayerNorm<const N: usize, D>` |
| `LayerNormNd<const N = 1, T = f32>` | `N` is the affine parameters', which may be shaped like the normalized group | `LayerNormNd<D = f32>`, which pins them to rank 1 |
| `Embedding<T = f32>` | element only — a table is `[vocab, dim]` | `Embedding<T>` |
| `ConvNd<const W = 4, T = f32>` | `W` is the weight's rank, so `W - 2` is the spatial rank | none; the reference has no `ConvNd` |
| `TensorCache<const R = 4, T = f32>` | the cached values' | `TensorCache<const R: usize, D>` |
| `KvCache<const R = 4, T = f32>` | same | `KvCache<D>`, holding two `TensorCache<4, D>` |
| `AttentionMask<T = f32>`, `MaskCache<T = f32>` | element only — a materialized mask is `[Lq, Lk]` | `MaskCache<D>` |
| `RopeCache<T = f32>` | element only — the table is `[rows, head_dim / 2]` | none; the reference's is f32-only |

Every `forward` is rank-generic over the *activation* and infallible, matching
the tensor surface: `Linear::forward<const R>(&Tensor<R, T>) -> Tensor<R, T>`.
The runtime-rank bodies are unchanged and stay reachable where the rank
arithmetic is genuinely the lowering's business —
`linear::forward_dyn`, `layer_norm::forward_nd_dyn`,
`TensorCache::append_dyn` — which is also where the negative cases (a
disagreeing extent, a rank mismatch) are still asserted as `Result`s rather
than as panics.

Two accessors stay runtime-rank on purpose: `TensorCache::pending` and
`KvCache::pending_into` feed a *resolve batch*, which collects roots of every
rank a step produced and is the one genuinely heterogeneous list in the crate.

## 8. A test-isolation note that is really a compiler finding

`Device::cpu()` is one per process **and so is its graph**: every test that
builds a value on it adds nodes to one shared e-graph that is never reset.
That is not inert. Extraction picks among cost-*identical* class members, and
`Launch::Launch::Scatter`'s four `ScatterMode`s are exactly that — an `OpDef`'s `work` is
a function of operand and output facts only, so `mode` is invisible to the cost
model and all four price the same. One of them,
`ScatterMode::OneHotContract`, has **no lowering on either backend**: both
`fusor2-cpu` and `fusor2-gpu` return `Error::Plan("OneHotContract is present
only as the candidate the cost model rejects; it lowers through Launch::Contract, not
here")`. So `fusor2-tile`'s `SCATTER_ONE_HOT_CONTRACT` rule mints a candidate
that the cost model provably cannot reject and that hard-errors if selected,
and which member the tie-break lands on shifts with unrelated graph content.

Measured, not inferred: adding seven ambient-graph tests to a clean `HEAD`
worktree — **no library change**, only test bodies in `HEAD`'s own `Dyn`
spellings — turns `trainer_surface`'s f16 convolution backward red with that
exact error. The comment on `ScatterMode` claiming the cost model rejects it
("1.2 GB of traffic against `WgPrivateMerge`'s 96 KB") is not implemented and,
as `work` is typed today, cannot be.

This is pre-existing and is **not fixed here** — it is a compiler-crate change
that needs a perf-verified decision about the scatter candidate set, not an API
one. What is done here is that a unit test of the const-rank *wrapper* no
longer depends on global extraction state: `Device::private()` gives it its own
session and graph. Tests genuinely about the ambient device keep using
`Device::cpu()` under `test_device_lock`.
