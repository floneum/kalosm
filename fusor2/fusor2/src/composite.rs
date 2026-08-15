//! Macro ops. Every constructor here mints the sugar node **and unions its
//! `defn` expansion into the same chain in the same call**, so there is
//! nothing to recognize later: recognition ordering, sole-consumer gates and
//! `spike_no_recognition` all evaporate, and the structural attributes a
//! pattern match would have to re-derive (`MaskKind::Causal`) stay on the
//! node.

pub mod activations;
pub mod attention;
pub mod conv;
pub mod loss;
pub mod normalization;
pub mod pool;
pub mod quantized;
pub mod rope;
pub mod upsample;

use fusor2_autograd::tape::GraphTape;
use fusor2_ir::egraph::Id;
use fusor2_ir::facts::{ValueFacts, Work};
use fusor2_ir::ir::level1::{AccessPlan, Effect, L1, MaskKind, Operand};
use fusor2_ir::ir::{Op, OpDef, OpDefId, OpDefRegistry, OpTag, VerifyCtx};
use fusor2_ir::shape::{Dim, Layout, SlidingWindow, SymId};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

use crate::graph::GraphRef;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// Which reduction a normalization performs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormKind {
    Rms,
    Layer,
}

/// Which reduction a pool performs. A macro attribute rather than a closure
/// so the `Window` structural adjoint can read the tie policy alongside it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PoolReduce {
    Max,
    Min,
    Mean,
}

/// Which loss a `Loss` sugar node stands for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LossKind {
    SoftmaxCrossEntropy,
    BinaryCrossEntropyWithLogits,
    Distillation,
    MeanSquaredError,
}

/// The closed macro-attribute vocabulary. Attributes live in a side table
/// keyed by `AttrId` so `Op` stays `Hash + Eq` and the hash-cons memo is
/// exact; a rule reads them back through
/// [`crate::graph::GraphInner::attrs_of`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MacroAttr {
    Softmax {
        axis: u32,
    },
    Norm {
        kind: NormKind,
        /// `eps` is a `Leaf::Uniform`, never a literal: moving it must not
        /// recompile anything.
        eps: SymId,
        remove_mean: bool,
    },
    Conv {
        padding: SmallVec<[u32; 3]>,
        stride: SmallVec<[u32; 3]>,
        groups: u32,
        spatial: u32,
    },
    Pool {
        windows: SmallVec<[SlidingWindow; 3]>,
        reduce: PoolReduce,
    },
    Upsample {
        scales: SmallVec<[u32; 3]>,
    },
    Attention {
        mask: MaskKind,
        causal: bool,
        /// `H / Hkv`. 1 for plain multi-head attention.
        groups: u32,
        /// Which value this sugar node stands for. Sugar is hash-consed on
        /// its attributes, so two attention macros over the same operands
        /// that compute different things must differ here or they are one
        /// node. Nothing reads it; it exists to keep them apart.
        produce: AttentionOut,
        scale: SymId,
    },
    Rope {
        interleaved: bool,
        paired: bool,
        with_position: bool,
    },
    Loss {
        kind: LossKind,
    },
}

/// Which value an `Attention` sugar node stands for.
///
/// A frontend fact about the macro surface, not an IR node kind: the four
/// attention entry points build four macros over overlapping operand lists,
/// and this is what keeps them from hash-consing into one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttentionOut {
    Output,
    LogSumExp,
    GradQ,
    GradKV,
}

// ---------------------------------------------------------------------------
// The op table
// ---------------------------------------------------------------------------

/// One row of [`MACRO_OPS`]. The discriminant **is** the `OpDefId`, because
/// registration happens once at `Session::new` in table order and `PlanHash`
/// reads registration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MacroOp {
    Softmax = 0,
    Norm = 1,
    Conv = 2,
    Pool = 3,
    Upsample = 4,
    Attention = 5,
    Rope = 6,
    Loss = 7,
}

impl MacroOp {
    pub const ALL: [MacroOp; 8] = [
        MacroOp::Softmax,
        MacroOp::Norm,
        MacroOp::Conv,
        MacroOp::Pool,
        MacroOp::Upsample,
        MacroOp::Attention,
        MacroOp::Rope,
        MacroOp::Loss,
    ];

    pub const fn def_id(self) -> OpDefId {
        OpDefId(self as u32)
    }

    pub const fn name(self) -> &'static str {
        MACRO_OPS[self as usize].name
    }
}

/// Every sugar op, in registration order.
///
/// All eight declare `lower_per_target: &[]` — they are **unrunnable by
/// construction**. They exist so rules can read the attributes a pattern
/// match would otherwise have to re-derive; the `defn` in the same e-class is
/// always the runnable floor, so a plan that selected a sugar node is
/// unbuildable rather than merely unlikely.
pub static MACRO_OPS: &[OpDef] = &[
    OpDef {
        name: "softmax",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_softmax,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "norm",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_norm,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "conv",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_conv,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "pool",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_pool,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "upsample",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_index_only,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "attention",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_attention,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "rope",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_rope,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
    OpDef {
        name: "loss",
        tag: OpTag::Ext,
        verify: verify_witness,
        infer: infer_witness,
        work: work_softmax,
        adjoint: None,
        lower_per_target: &[],
        effect: Effect::Pure,
    },
];

/// Register every macro op into a fresh registry. Called exactly once, by
/// `Session::new`, so ids are assigned in table order.
pub fn register_macro_ops(registry: &mut OpDefRegistry) {
    for (i, def) in MACRO_OPS.iter().enumerate() {
        let id = registry.register(def.clone());
        debug_assert_eq!(
            id,
            OpDefId(i as u32),
            "macro op ids must follow table order; PlanHash reads it"
        );
    }
}

// ---------------------------------------------------------------------------
// Registry rows
// ---------------------------------------------------------------------------

/// The last operand of every sugar node is its `defn`, which is what makes
/// inference total without handing `OpDef::infer` the attribute blob: the
/// definitional expansion already knows the answer.
fn witness(ins: &[ValueFacts]) -> Result<&ValueFacts> {
    ins.last()
        .ok_or_else(|| Error::Shape("a macro op carries its defn as its last operand".into()))
}

fn infer_witness(ins: &[ValueFacts]) -> Result<ValueFacts> {
    witness(ins).cloned()
}

fn verify_witness(cx: &VerifyCtx<'_>) -> Result<()> {
    let w = witness(cx.operands)?;
    if w.dtype != cx.result.dtype || w.shape != cx.result.shape {
        return Err(Error::verify(
            fusor2_ir::ir::Level::L1,
            cx.id,
            "a macro op's facts must equal its defn's",
        ));
    }
    Ok(())
}

fn elements(f: &ValueFacts) -> u64 {
    f.shape
        .iter()
        .map(|d| d.as_const().unwrap_or(1))
        .product::<u64>()
        .max(1)
}

fn rows(f: &ValueFacts) -> u64 {
    let last = f.shape.last().and_then(|d| d.as_const()).unwrap_or(1).max(1);
    (elements(f) / last).max(1)
}

fn work_softmax(_ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let n = elements(out);
    Work {
        macs: n,
        transcendentals: n,
        index_ops: n,
        wg_bytes: 0,
    }
}

fn work_norm(_ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let n = elements(out);
    Work {
        macs: 2 * n,
        transcendentals: rows(out),
        index_ops: n,
        wg_bytes: 0,
    }
}

fn work_conv(ins: &[ValueFacts], out: &ValueFacts) -> Work {
    // ins = [x, weight, (bias), defn]; the weight's element count divided by
    // its output-channel axis is the per-output-element MAC count.
    let n = elements(out);
    let per_out = ins
        .get(1)
        .map(|w| {
            let out_ch = w.shape.first().and_then(|d| d.as_const()).unwrap_or(1).max(1);
            (elements(w) / out_ch).max(1)
        })
        .unwrap_or(1);
    Work {
        macs: n.saturating_mul(per_out),
        transcendentals: 0,
        index_ops: n,
        wg_bytes: 0,
    }
}

fn work_pool(ins: &[ValueFacts], out: &ValueFacts) -> Work {
    // A pool reads every input element exactly once.
    let read = ins.first().map(elements).unwrap_or_else(|| elements(out));
    Work {
        macs: read,
        transcendentals: 0,
        index_ops: read,
        wg_bytes: 0,
    }
}

fn work_index_only(_ins: &[ValueFacts], out: &ValueFacts) -> Work {
    Work {
        macs: 0,
        transcendentals: 0,
        index_ops: elements(out),
        wg_bytes: 0,
    }
}

fn work_attention(ins: &[ValueFacts], out: &ValueFacts) -> Work {
    // ins = [q, k, v, .., defn]; scores are `Lq x Lk` per (batch, head) and
    // the value contraction costs the same again.
    let q = ins.first().map(elements).unwrap_or_else(|| elements(out));
    let k_len = ins
        .get(1)
        .and_then(|f| f.shape.get(f.shape.len().saturating_sub(2)).copied())
        .and_then(|d| d.as_const())
        .unwrap_or(1)
        .max(1);
    let head_dim = ins
        .first()
        .and_then(|f| f.shape.last().copied())
        .and_then(|d| d.as_const())
        .unwrap_or(1)
        .max(1);
    let scores = (q.saturating_mul(k_len) / head_dim).max(1);
    Work {
        macs: 2u64.saturating_mul(q).saturating_mul(k_len),
        transcendentals: scores,
        index_ops: scores,
        wg_bytes: 0,
    }
}

fn work_rope(_ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let n = elements(out);
    Work {
        macs: 3 * n,
        transcendentals: 0,
        index_ops: n,
        wg_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// The one construction discipline
// ---------------------------------------------------------------------------

/// Build one macro op.
///
/// Every macro op in this crate goes through exactly this function, and it
/// always does the same five things:
///
/// 1. build the `defn` — the definitional core-L0 expansion — **first**, so
///    its id is below the sugar's and the union's operand 0 is the defn (the
///    adjoint walk descends operand 0, and only the defn has an adjoint);
/// 2. `mark_defn` it, so it is never evicted;
/// 3. intern `attrs` into the `AttrId` side table;
/// 4. add the sugar `L1::Ext { def, ops, attrs }`, whose last operand is the
///    defn — its shape witness;
/// 5. `union(defn, sugar)` and return a [`Tensor`] over the **union root**.
///
/// No macro op may return the sugar id or the defn id alone: a caller holding
/// either would pin one member of the class and defeat late selection.
pub(crate) fn macro_op(
    graph: &GraphRef,
    def: MacroOp,
    attrs: MacroAttr,
    ops: &[Id],
    build_defn: impl FnOnce(&mut GraphTape<'_>) -> Result<Id>,
) -> Result<Tensor> {
    let attrs = graph.intern_attrs(attrs);
    let (defn, sugar) = graph.with_egraph(|g| {
        let defn = {
            let mut tape = GraphTape::new(g);
            build_defn(&mut tape)?
        };
        g.mark_defn(defn);

        let mut operands: Vec<Operand> = Vec::with_capacity(ops.len() + 1);
        for src in ops.iter().copied().chain(std::iter::once(defn)) {
            let facts = g.facts(src);
            operands.push(Operand {
                src,
                layout: Layout::contiguous(&facts.shape),
                access: AccessPlan::Alias,
            });
        }
        let sugar = g.add(Op::L1(L1::Ext {
            def: def.def_id(),
            ops: operands,
            attrs,
        }))?;
        Ok((defn, sugar))
    })?;
    // `union_stable`: the first build returns exactly the union root the
    // plain `union` would (identical extraction inputs); a rebuild — a
    // decode loop re-running the same model code next step — gets the same
    // id back instead of the class's *moved* root, so downstream consumers
    // hash-cons and the step graph stays node-identical.
    let root = graph.union_stable(defn, sugar)?;
    Ok(graph.tensor(root))
}

/// Build a plain core expansion with no sugar node — the `*_slow` spellings,
/// whose only semantic difference from the sugared ones is that no rule can
/// read an attribute off them.
pub(crate) fn core_op(
    graph: &GraphRef,
    build: impl FnOnce(&mut GraphTape<'_>) -> Result<Id>,
) -> Result<Tensor> {
    let id = graph.build(build)?;
    Ok(graph.tensor(id))
}

/// A const extent, or an error. Used only where an algorithm genuinely needs
/// the integer (a window size, a kernel extent) rather than where a symbolic
/// extent would do.
pub(crate) fn const_dim(d: Dim, what: &str) -> Result<u64> {
    d.as_const()
        .ok_or_else(|| Error::Shape(format!("{what} needs a decidable extent, got {d}")))
}

/// A rank-1 `u32` index leaf holding `values`.
///
/// Scatter and gather both take a real index tensor. A pad's run, an
/// upsample's repeat and a concatenation's destination rows are all this — one
/// small buffer uploaded once, never a special node kind.
pub(crate) fn index_leaf(graph: &GraphRef, values: &[u32]) -> Result<Id> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    graph.constant_leaf(
        fusor2_ir::dtype::Dtype::U32,
        &[Dim::Const(values.len() as u64)],
        bytes,
    )
}

/// `index_leaf` over the run `start .. start + len`.
pub(crate) fn index_run(graph: &GraphRef, start: u64, len: u64) -> Result<Id> {
    let values: Vec<u32> = (0..len)
        .map(|i| {
            u32::try_from(start + i)
                .map_err(|_| Error::Shape(format!("index {} exceeds a u32", start + i)))
        })
        .collect::<Result<_>>()?;
    index_leaf(graph, &values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::dtype::Dtype;

    fn facts(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|d| Dim::Const(*d)))
    }

    #[test]
    fn ids_follow_table_order() {
        let mut reg = OpDefRegistry::new();
        register_macro_ops(&mut reg);
        for op in MacroOp::ALL {
            let def = reg.get(op.def_id()).expect("registered");
            assert_eq!(def.name, op.name());
        }
        assert_eq!(reg.iter().count(), MACRO_OPS.len());
    }

    #[test]
    fn every_sugar_op_is_unrunnable() {
        for def in MACRO_OPS {
            assert!(
                def.lower_per_target.is_empty(),
                "{} must be unrunnable; the defn is the floor",
                def.name
            );
            assert!(def.adjoint.is_none());
            assert_eq!(def.effect, Effect::Pure);
        }
    }

    /// `verify_l1` rejects an `OpDef` whose `work` is constant in shape. Every
    /// row here must survive that tripwire.
    #[test]
    fn no_work_row_is_constant_in_shape() {
        for def in MACRO_OPS {
            let small = [facts(&[2, 3, 4]), facts(&[2, 3, 4]), facts(&[2, 3, 4])];
            let large = [facts(&[4, 6, 8]), facts(&[4, 6, 8]), facts(&[4, 6, 8])];
            let a = (def.work)(&small, &small[0]);
            let b = (def.work)(&large, &large[0]);
            assert_ne!(a, b, "{}: work() does not vary with shape", def.name);
            assert_ne!(a, Work::default(), "{}: work() is empty", def.name);
        }
    }

    #[test]
    fn inference_reads_the_defn_witness() {
        let ins = [facts(&[8, 8]), facts(&[2, 5])];
        let out = infer_witness(&ins).unwrap();
        assert_eq!(&out.shape[..], &[Dim::Const(2), Dim::Const(5)]);
        assert!(infer_witness(&[]).is_err());
    }
}

#[cfg(test)]
mod constant_identity {
    use super::*;
    use crate::graph::Graph;
    // The backend selector, by module path. The crate root's `Backend` is
    // whichever of the two the `typed-api` feature selects; in-crate code
    // names the one it means so it compiles under either root.
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
    }

    /// Two index vectors of the same length are two different values.
    ///
    /// Rope mints `perm` and `expand` at the same length in the same call,
    /// so this ensures they hash-cons into separate nodes.
    #[test]
    fn two_index_leaves_of_one_length_are_two_nodes() {
        let g = graph();
        let h = g.handle();
        let perm = index_leaf(h, &[2, 3, 0, 1]).unwrap();
        let expand = index_leaf(h, &[0, 1, 0, 1]).unwrap();
        assert_ne!(perm, expand);
        assert_eq!(
            h.leaf_bytes(perm).unwrap(),
            [2u32, 3, 0, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        );
    }

    /// Equal content still shares a node, so constant pooling survives.
    #[test]
    fn equal_index_content_still_hash_conses() {
        let g = graph();
        let h = g.handle();
        assert_eq!(
            index_leaf(h, &[4, 5, 6]).unwrap(),
            index_leaf(h, &[4, 5, 6]).unwrap()
        );
    }

    /// Every leaf name in one graph comes from one allocator. Separate private
    /// counters would hand out overlapping `BufferId`s — one of them starting
    /// at exactly the `u32::MAX` the constant leaves hardcode.
    #[test]
    fn uploads_and_constants_never_share_a_buffer_name() {
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::level0::{L0, LeafKind};

        let g = graph();
        let h = g.handle();
        let mut names = Vec::new();
        let ids = [
            index_leaf(h, &[1, 2, 3, 4]).unwrap(),
            index_leaf(h, &[9, 9, 9, 9]).unwrap(),
            crate::tensor::Tensor::from_slice(
                h,
                Dtype::U32,
                &[Dim::Const(4)],
                &[0u8; 16],
            )
            .unwrap()
            .id(),
        ];
        for id in ids {
            h.with_egraph(|eg| {
                if let fusor2_ir::ir::Op::L0(L0::Leaf(LeafKind::Buffer { name, .. })) =
                    &eg.node(id).op
                {
                    names.push(*name);
                }
                Ok(())
            })
            .unwrap();
        }
        let mut sorted = names.clone();
        sorted.sort_by_key(|b| b.0);
        sorted.dedup_by_key(|b| b.0);
        assert_eq!(sorted.len(), names.len(), "buffer names collided: {names:?}");
    }
}
