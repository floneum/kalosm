//! Macro ops. Every constructor here mints the sugar node and unions its
//! `defn` expansion into the same chain in the same call, so there is nothing
//! to recognize later and the structural attributes (`MaskKind::Causal`) stay
//! on the node.

pub(crate) mod activations;
pub(crate) mod attention;
pub(crate) mod conv;
pub(crate) mod loss;
pub(crate) mod normalization;
pub(crate) mod pool;
pub(crate) mod quantized;
pub(crate) mod rope;
pub(crate) mod upsample;

pub use attention::{
    attention, attention_causal, attention_grads, attention_lse, attention_masked,
    attention_with_lse,
};
pub use conv::{conv, grouped_conv, pad_with_zeros};
pub use loss::{binary_cross_entropy_with_logits, distillation_loss, mse, softmax_cross_entropy};
pub use pool::{PoolSize, pool, pool_avg, pool_max, pool_min};
pub use rope::{
    base_inverse_frequency, rope, rope_interleaved, rope_interleaved_pair,
    rope_interleaved_pair_with_position, rope_interleaved_with_position, rope_pair,
    rope_pair_with_position, rope_with_position, rotate_half,
};
pub use upsample::{upsample_bilinear, upsample_nearest, upsample_nearest2d};

use fusor_autograd::tape::GraphTape;
use fusor_ir::egraph::Id;
use fusor_ir::facts::{ValueFacts, Work};
use fusor_ir::ir::launch::{AccessPlan, Effect, Launch, MaskKind, Operand};
use fusor_ir::ir::{Op, OpDef, OpDefId, OpDefRegistry, OpTag, VerifyCtx};
use fusor_ir::shape::{Dim, Layout, SlidingWindow, SymId};
use fusor_ir::{Error, Result};
use smallvec::SmallVec;

use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// Which reduction a normalization performs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum NormKind {
    Rms,
    Layer,
}

/// Which reduction a pool performs. A macro attribute rather than a closure
/// so the `Window` structural adjoint can read the tie policy alongside it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PoolReduce {
    /// Maximum pooling.
    Max,
    /// Minimum pooling.
    Min,
    /// Average pooling.
    Mean,
}

/// The closed macro-attribute vocabulary. Attributes live in a side table
/// keyed by `AttrId` so `Op` stays `Hash + Eq` and the hash-cons memo is
/// exact; a rule reads them back through
/// [`crate::graph::GraphInner::attrs_of`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MacroAttr {
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
}

/// Which value an `Attention` sugar node stands for.
///
/// A frontend fact about the macro surface, not an IR node kind: the four
/// attention entry points build four macros over overlapping operand lists,
/// and this is what keeps them from hash-consing into one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AttentionOut {
    Output,
    LogSumExp,
    GradQ,
    GradKV,
}

/// One row of [`MACRO_OPS`]. The discriminant **is** the `OpDefId`, because
/// registration happens once at `Session::new` in table order and `PlanHash`
/// reads registration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(crate) enum MacroOp {
    Softmax = 0,
    Norm = 1,
    Conv = 2,
    Pool = 3,
    Upsample = 4,
    Attention = 5,
    Rope = 6,
}

impl MacroOp {
    pub(crate) const fn def_id(self) -> OpDefId {
        OpDefId(self as u32)
    }
}

/// Every sugar op, in registration order.
///
/// All seven declare `lower_per_target: &[]` — unrunnable by construction.
/// They exist so rules can read attributes off them; the `defn` in the same
/// e-class is always the runnable floor.
pub(crate) static MACRO_OPS: &[OpDef] = &[
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
];

/// Register every macro op into a fresh registry. Called exactly once, by
/// `Session::new`, so ids are assigned in table order.
pub(crate) fn register_macro_ops(registry: &mut OpDefRegistry) {
    for (i, def) in MACRO_OPS.iter().enumerate() {
        let id = registry.register(def.clone());
        debug_assert_eq!(
            id,
            OpDefId(i as u32),
            "macro op ids must follow table order; PlanHash reads it"
        );
    }
}

/// The last operand of every sugar node is its `defn`; inference reads it as
/// the shape witness.
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
            fusor_ir::ir::Level::Launch,
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
    let last = f
        .shape
        .last()
        .and_then(|d| d.as_const())
        .unwrap_or(1)
        .max(1);
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
            let out_ch = w
                .shape
                .first()
                .and_then(|d| d.as_const())
                .unwrap_or(1)
                .max(1);
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

/// Build one macro op.
///
/// Every macro op in this crate goes through exactly this function, and it
/// always does the same five things:
///
/// 1. build the `defn` — the definitional core-Logical expansion — **first**, so
///    its id is below the sugar's and the union's operand 0 is the defn (the
///    adjoint walk descends operand 0, and only the defn has an adjoint);
/// 2. `mark_defn` it, so it is never evicted;
/// 3. intern `attrs` into the `AttrId` side table;
/// 4. add the sugar `Launch::Ext { def, ops, attrs }`, whose last operand is the
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
        let sugar = g.add(Op::Launch(Launch::Ext {
            def: def.def_id(),
            ops: operands,
            attrs,
        }))?;
        Ok((defn, sugar))
    })?;
    // `union_stable`: a rebuild — a decode loop re-running the same model
    // code next step — gets the same id back instead of the class's *moved*
    // root, so downstream consumers hash-cons and the step graph stays
    // node-identical.
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
/// the integer (a window size, a kernel extent).
pub(crate) fn const_dim(d: Dim, what: &str) -> Result<u64> {
    d.as_const()
        .ok_or_else(|| Error::Shape(format!("{what} needs a decidable extent, got {d}")))
}

/// A rank-1 `u32` index leaf holding `values`: one small buffer uploaded
/// once, fed to scatter and gather as a real index tensor.
pub(crate) fn index_leaf(graph: &GraphRef, values: &[u32]) -> Result<Id> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    graph.constant_leaf(
        fusor_ir::dtype::Dtype::U32,
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
