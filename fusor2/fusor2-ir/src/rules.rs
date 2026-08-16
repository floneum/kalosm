//! Every level-generic rewrite rule, and the one table the driver is handed.
//!
//! Guards may read only [`crate::egraph::Facts`] — legality, never
//! profitability.
//!
//! Rule order carries no semantics; the fixed order below exists only for
//! reproducibility.

pub mod algebra;
pub mod fusion;
pub mod layout;
pub mod lower_floor;
pub mod promote;
pub mod rebase;
pub mod sink;
pub mod specialize;
pub mod tuple;

use crate::dtype::Dtype;
use crate::egraph::{Builder, Id, Rule};
use crate::ir::Op;
use crate::ir::logical::Logical;
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand};
use crate::scalar::ScalarExpr;
use crate::shape::{Dim, Layout, StrideSpec};

/// Position of a rule in whatever slice was handed to the driver. `RuleId`
/// is positional, not global: a target concatenates [`CORE_RULES`] with its
/// own `Target::rules()` and the driver indexes the concatenation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(pub u16);

/// Every rule `fusor2-ir` owns, in a fixed order. A target's `rules()` is
/// concatenated onto this; nothing here mentions lane or subgroup geometry.
pub static CORE_RULES: &[Rule] = &[
    // Logical algebra
    algebra::STRIP,
    algebra::RECOGNIZE_CONTRACT,
    algebra::CONTRACT_REASSOC,
    algebra::CONST_FOLD_MAP,
    algebra::IDENTITY_ELIM,
    algebra::WIDEN_STORE_CAST,
    algebra::UNIT_FOLD_COLLAPSE,
    // Launch fusion
    fusion::ABSORB,
    fusion::MAP_INTO_CONTRACT,
    fusion::MAP_INTO_MAP,
    fusion::FOLD_POST_EPILOGUE,
    fusion::FORM_KREGION,
    // Launch fold algebra — the carrier laws. `HOIST` and `RETARGET` are two
    // entries sharing one dependence query: the driver's fired set is per
    // `(RuleId, Id)`, so one merged rule could fire at most once per node
    // and the second answer would be unreachable.
    promote::PROMOTE,
    rebase::HOIST,
    rebase::RETARGET,
    tuple::TUPLE,
    tuple::TUPLE_SIBLING,
    // Launch sinking
    sink::SINK_EPILOGUE,
    sink::FOLD_VIEWS_INTO_INDEX,
    sink::FOLD_VIEWS_INTO_FOLD_INDEX,
    // Launch operand access
    layout::OPERAND_ALIAS,
    layout::OPERAND_GATHER,
    layout::OPERAND_PACK,
    layout::OPERAND_UNFLATTEN,
    // the M0 correctness floor
    lower_floor::LOWER_MAP,
    lower_floor::LOWER_FOLD,
    lower_floor::LOWER_CONTRACT_GENERIC,
    lower_floor::LOWER_RESTRIDE,
    lower_floor::LOWER_WINDOW,
    lower_floor::LOWER_GATHER,
    lower_floor::LOWER_SCATTER,
    lower_floor::LOWER_DEQUANT,
    lower_floor::LOWER_PROJECT,
    // shape specialization
    specialize::SPECIALIZE_DIM,
];

/// Look a core rule up by the name its `rule!` declaration stringified.
pub fn rule_id(name: &str) -> Option<RuleId> {
    CORE_RULES
        .iter()
        .position(|r| r.name == name)
        .map(|i| RuleId(i as u16))
}

/// The core rule at `id`. Panics when `id` is out of range, which can only
/// happen if a caller mixes ids minted against a different slice.
pub fn rule(id: RuleId) -> &'static Rule {
    &CORE_RULES[id.0 as usize]
}

/// An operand read straight out of its producer's dense row-major layout.
pub(crate) fn alias_operand_of(src: Id, shape: &[Dim]) -> Operand {
    Operand {
        src,
        layout: Layout::contiguous(shape),
        access: AccessPlan::Alias,
    }
}

/// The identity scalar body, `Arg(0)`.
pub(crate) fn ident_expr(dtype: Dtype) -> ScalarExpr {
    ScalarExpr::arg(0, dtype)
}

/// The access predicate `map_into_fold` and `map_into_contract` guard on:
/// `Alias`, `Unflatten` and `Gather` are legal in any space; a `Pack` is
/// legal only when the packed layout has the consuming space's rank.
pub(crate) fn access_legal_in(a: &AccessPlan, space: &IndexSpace) -> bool {
    match a {
        AccessPlan::Alias | AccessPlan::Unflatten(_) | AccessPlan::Gather => true,
        AccessPlan::Pack { into } => into.rank() == space.rank(),
    }
}

/// The elementwise producer shape a fusion rule inlines.
///
/// Equality in this e-graph is **not congruent**, so an `Logical::Map` and the
/// `Launch::Map` it was lowered to are one class but the consuming Launch node's
/// operand still names whichever id the frontend built. Both spellings
/// denote the same value, so both are inlinable; this normalizes them.
pub(crate) struct MapView {
    pub space: IndexSpace,
    pub body: ScalarExpr,
    pub ops: Vec<Operand>,
}

/// Read `id` as an elementwise producer, in either spelling.
pub(crate) fn map_view(b: &Builder<'_>, id: Id) -> Option<MapView> {
    match b.node(id).op.clone() {
        Op::Launch(Launch::Map {
            space, body, ops, ..
        }) => Some(MapView { space, body, ops }),
        Op::Logical(Logical::Map { expr, ins, outs }) if outs == 1 => {
            let space = IndexSpace::new(b.facts_of(id).shape.iter().copied());
            let ops = ins
                .iter()
                .map(|&s| alias_operand_of(s, &b.facts_of(s).shape))
                .collect();
            Some(MapView {
                space,
                body: expr,
                ops,
            })
        }
        _ => None,
    }
}

/// Renumber `Arg(i)` to `Arg(i + by)` throughout `e`, given each argument's
/// element type.
pub(crate) fn shift_args(e: &ScalarExpr, by: u32, arg_dtypes: &[Dtype]) -> ScalarExpr {
    let args: Vec<ScalarExpr> = arg_dtypes
        .iter()
        .enumerate()
        .map(|(i, d)| ScalarExpr::arg(i as u32 + by, *d))
        .collect();
    e.compose(&args)
}

/// Element type each operand of `ops` presents to a scalar body.
pub(crate) fn operand_dtypes(b: &Builder<'_>, ops: &[Operand]) -> Vec<Dtype> {
    ops.iter().map(|o| b.facts_of(o.src).dtype).collect()
}

/// Apply a relative restride spec vector to a dense row-major input shape.
/// Returns `None` when a stride or offset is not decidable.
pub(crate) fn composed_layout(specs: &[StrideSpec], in_shape: &[Dim]) -> Option<Layout> {
    let in_strides = Layout::row_major_strides(in_shape);
    let mut shape: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut strides: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut offset: u64 = 0;
    for s in specs {
        shape.push(s.size);
        let base = in_strides.get(s.input_dim as usize)?.as_const()?;
        // Accumulated for every spec, including a stride-0 one: an axis being
        // broadcast says nothing about where in the input it starts.
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
        if s.multiplier == 0 {
            strides.push(Dim::Const(0));
            continue;
        }
        strides.push(Dim::Const(base.checked_mul(u64::from(s.multiplier))?));
    }
    Layout::from_parts(Dim::Const(offset), &shape, &strides).ok()
}

/// The plain affine layout a whole view spine denotes over its base, or
/// `None` when the composition is not expressible as one stride vector.
///
/// Composing a chain substitutes each stage's strides into the next, so a
/// narrow → reshape → transpose spine collapses to
/// `offset + Σ stride_j · i_j` over the base buffer. Everything must be
/// const-decidable and every stage's bounds proof [`BoundsProof::Static`]:
/// a `RuntimeMask` view masks reads the composed layout could not, and
/// dropping a mask is a wrong value.
///
/// A spec that walks past its input axis's extent (an axis-merging reshape)
/// is only affine when the stage it reads is dense row-major from that axis
/// inward, so the composition declines elsewhere.
pub(crate) fn composed_spine_layout(
    b: &Builder<'_>,
    spine: &crate::egraph::ViewSpine,
) -> Option<Layout> {
    use crate::ir::logical::Logical;
    let base_shape = b.facts_of(spine.base).shape.clone();
    let mut shape: Vec<u64> = base_shape
        .iter()
        .map(|d| d.as_const())
        .collect::<Option<_>>()?;
    let mut strides: Vec<u64> = Layout::row_major_strides(&base_shape)
        .iter()
        .map(|d| d.as_const())
        .collect::<Option<_>>()?;
    let mut offset: u64 = 0;
    for view in &spine.views {
        let Op::Logical(Logical::Restride { specs, bounds, .. }) = &b.node(*view).op else {
            return None;
        };
        if *bounds != crate::shape::BoundsProof::Static {
            return None;
        }
        // Whether the *current* stage is one dense row-major block, which is
        // the only stage an axis-overrunning spec addresses correctly: there
        // the stage is a flat window and `k · multiplier · in_stride` is flat
        // addressing, whatever axis boundaries the walk crosses.
        let dense = {
            let mut want = 1u64;
            let mut ok = true;
            for i in (0..shape.len()).rev() {
                // An extent-1 axis is unobservable whatever its stride says.
                if shape[i] <= 1 {
                    continue;
                }
                if strides[i] != want {
                    ok = false;
                    break;
                }
                want = want.saturating_mul(shape[i]);
            }
            ok
        };
        let mut nshape: Vec<u64> = Vec::with_capacity(specs.len());
        let mut nstrides: Vec<u64> = Vec::with_capacity(specs.len());
        for s in specs {
            let idim = s.input_dim as usize;
            let in_ext = *shape.get(idim)?;
            let in_stride = *strides.get(idim)?;
            let size = s.size.as_const()?;
            let off = s.offset.as_const()?;
            offset = offset.checked_add(off.checked_mul(in_stride)?)?;
            if s.multiplier == 0 {
                nshape.push(size);
                nstrides.push(0);
                continue;
            }
            let span = u64::from(s.multiplier)
                .checked_mul(size.saturating_sub(1))?
                .checked_add(off)?;
            if span >= in_ext.max(1) && !dense {
                return None;
            }
            nshape.push(size);
            nstrides.push(in_stride.checked_mul(u64::from(s.multiplier))?);
        }
        shape = nshape;
        strides = nstrides;
    }
    let shape: Vec<Dim> = shape.into_iter().map(Dim::Const).collect();
    let strides: Vec<Dim> = strides.into_iter().map(Dim::Const).collect();
    Layout::from_parts(Dim::Const(offset), &shape, &strides).ok()
}

/// Whether a spec vector is the identity view of `in_shape`.
pub(crate) fn is_identity_specs(specs: &[StrideSpec], in_shape: &[Dim]) -> bool {
    specs.len() == in_shape.len()
        && specs.iter().enumerate().all(|(i, s)| {
            s.multiplier == 1
                && s.input_dim as usize == i
                && s.offset.known_eq(Dim::Const(0))
                && s.size.known_eq(in_shape[i])
        })
}

/// A minimal in-crate [`crate::ir::Semantics`] plus graph constructors, so
/// every rule module can build a fixture without depending on
/// `CoreSemantics`.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::scalar::ScalarKind;
    use crate::device::{Caps, CoopKind, DeviceKind, Limits, SubgroupWidths};
    use crate::dtype::{NumericContract, Persistence};
    use crate::egraph::EGraph;
    use crate::error::{Error, Result};
    use crate::facts::{ValueFacts, Work};
    use crate::carrier::Carrier;
    use crate::ir::logical::{BufferId, EinSpec, LeafKind, ScatterCombine};
    use crate::ir::launch::{ContractSide, Effect, Family, ScheduleDomain};
    use crate::ir::{Children, Semantics, VerifyCtx};
    use crate::shape::{BoundsProof, Dims};
    use smallvec::smallvec;
    use std::sync::Arc;

    /// Inference and child order mirroring what `CoreSemantics` must do.
    pub struct TestSem;

    fn meet(ins: &[ValueFacts]) -> NumericContract {
        ins.iter()
            .fold(NumericContract::RELAXED, |acc, f| acc.meet(f.numeric))
    }

    fn persistence(ins: &[ValueFacts]) -> Persistence {
        if ins.iter().all(|f| f.persistence == Persistence::Persistent) && !ins.is_empty() {
            Persistence::Persistent
        } else {
            Persistence::Step
        }
    }

    fn has_round(e: &ScalarExpr) -> bool {
        match e.kind() {
            ScalarKind::Round { .. } => true,
            ScalarKind::Un { x, .. }
            | ScalarKind::Cast { x, .. }
            | ScalarKind::Bitcast { x, .. }
            | ScalarKind::Splat { x, .. } => has_round(x),
            ScalarKind::Bin { a, b, .. }
            | ScalarKind::Cmp { a, b, .. }
            | ScalarKind::Dot { a, b } => has_round(a) || has_round(b),
            ScalarKind::Select { c, t, f } => has_round(c) || has_round(t) || has_round(f),
            _ => false,
        }
    }

    fn facts(dtype: Dtype, shape: Dims, ins: &[ValueFacts]) -> ValueFacts {
        ValueFacts {
            dtype,
            shape,
            numeric: meet(ins),
            persistence: persistence(ins),
            outs: 1,
        }
    }

    pub fn children_of(op: &Op) -> Children {
        match op {
            Op::Union(a, b) => smallvec![*a, *b],
            Op::Logical(o) => match o {
                Logical::Leaf(_) => Children::new(),
                Logical::Map { ins, .. } | Logical::Fold { ins, .. } => ins.iter().copied().collect(),
                Logical::Restride { x, .. }
                | Logical::Window { x, .. }
                | Logical::Dequant { x, .. }
                | Logical::Project { x, .. } => smallvec![*x],
                Logical::Contract { a, b, .. } => smallvec![*a, *b],
                Logical::Gather { x, idx, .. } => smallvec![*x, *idx],
                Logical::Scatter {
                    base, idx, upd, ..
                } => smallvec![*base, *idx, *upd],
            },
            Op::Launch(o) => match o {
                Launch::Map { ops, .. }
                | Launch::Fold { ops, .. }
                | Launch::Gather { ops, .. }
                | Launch::Scatter { ops, .. }
                | Launch::Ext { ops, .. } => ops.iter().map(|p| p.src).collect(),
                Launch::Contract { a, b, .. } => {
                    a.ops.iter().chain(b.ops.iter()).map(|p| p.src).collect()
                }
                Launch::Region { members, .. } => members.iter().copied().collect(),
            },
        }
    }

    fn infer_logical(o: &Logical, ins: &[ValueFacts]) -> Result<ValueFacts> {
        Ok(match o {
            Logical::Leaf(k) => match k {
                LeafKind::Buffer { dtype, shape, .. } => {
                    facts(*dtype, shape.clone(), &[])
                }
                LeafKind::Param { dtype, shape, .. } => {
                    let mut f = facts(*dtype, shape.clone(), &[]);
                    f.persistence = Persistence::Persistent;
                    f
                }
                LeafKind::Const { value, shape } => facts(value.dtype(), shape.clone(), &[]),
                LeafKind::Uniform { dtype, .. } => facts(*dtype, Dims::new(), &[]),
                LeafKind::Quantized {
                    fmt, shape, ..
                } => {
                    let mut f =
                        facts(Dtype::Q(*fmt), shape.iter().copied().collect(), &[]);
                    f.persistence = Persistence::Persistent;
                    f
                }
            },
            Logical::Map { expr, ins: _, outs } => {
                let shape = ins
                    .first()
                    .map(|f| f.shape.clone())
                    .ok_or_else(|| Error::Shape("map with no operand".into()))?;
                let mut f = facts(expr.dtype(), shape, ins);
                f.outs = *outs;
                // A rounding body is the QAT fake-quant path: its value may
                // not be reassociated or contracted.
                if has_round(expr) {
                    f.numeric = f.numeric.meet(NumericContract::STRICT);
                }
                f
            }
            Logical::Fold {
                axis, acc, carrier, ..
            } => {
                let src = ins.first().ok_or_else(|| Error::Shape("fold".into()))?;
                let mut shape = src.shape.clone();
                if (*axis as usize) >= shape.len() {
                    return Err(Error::Shape("fold axis out of range".into()));
                }
                shape.remove(*axis as usize);
                if let Some(d) = carrier
                    .out_dim()
                    .ok_or_else(|| Error::Shape("symbolic carrier lane count".into()))?
                {
                    shape.push(d);
                }
                facts(*acc, shape, ins)
            }
            Logical::Contract { spec, acc, .. } => {
                let a = &ins[0];
                let b = &ins[1];
                let mut shape: Dims = Dims::new();
                for l in &spec.out {
                    let d = spec
                        .a
                        .iter()
                        .position(|x| x == l)
                        .map(|i| a.shape[i])
                        .or_else(|| spec.b.iter().position(|x| x == l).map(|i| b.shape[i]))
                        .ok_or_else(|| Error::Shape("unbound contract label".into()))?;
                    shape.push(d);
                }
                facts(*acc, shape, ins)
            }
            Logical::Restride { specs, .. } => {
                facts(ins[0].dtype, specs.iter().map(|s| s.size).collect(), ins)
            }
            Logical::Window { specs, .. } => {
                let mut shape = ins[0].shape.clone();
                for w in specs {
                    let n = shape[w.axis as usize]
                        .as_const()
                        .ok_or_else(|| Error::Shape("symbolic window axis".into()))?;
                    shape[w.axis as usize] =
                        Dim::Const((n - u64::from(w.window)) / u64::from(w.step) + 1);
                    shape.push(Dim::Const(u64::from(w.window)));
                }
                facts(ins[0].dtype, shape, ins)
            }
            Logical::Gather { axis, .. } => {
                let mut shape = ins[0].shape.clone();
                shape[*axis as usize] = *ins[1]
                    .shape
                    .first()
                    .ok_or_else(|| Error::Shape("gather index rank".into()))?;
                facts(ins[0].dtype, shape, ins)
            }
            Logical::Scatter { .. } => facts(ins[0].dtype, ins[0].shape.clone(), ins),
            Logical::Dequant { .. } => facts(Dtype::F32, ins[0].shape.clone(), ins),
            Logical::Project { .. } => facts(ins[0].dtype, ins[0].shape.clone(), ins),
        })
    }

    fn infer_launch(o: &Launch, ins: &[ValueFacts]) -> Result<ValueFacts> {
        Ok(match o {
            Launch::Map { space, body, .. } => {
                facts(body.dtype(), space.dims.clone(), ins)
            }
            Launch::Fold {
                space,
                axis,
                acc,
                carrier,
                vec_axes,
                ..
            } => {
                if (*axis as usize) >= space.dims.len() {
                    return Err(Error::Shape("kfold axis out of range".into()));
                }
                let mut shape: Dims = space
                    .dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != *axis as usize && !vec_axes.contains(&(*i as u32)))
                    .map(|(_, d)| *d)
                    .collect();
                if let Some(d) = carrier
                    .out_dim()
                    .ok_or_else(|| Error::Shape("symbolic carrier lane count".into()))?
                {
                    shape.push(d);
                }
                facts(*acc, shape, ins)
            }
            Launch::Contract {
                m, n, batch, post, ..
            } => {
                let mut shape: Dims = Dims::new();
                if !batch.known_eq(Dim::ONE) {
                    shape.push(*batch);
                }
                shape.push(*m);
                shape.push(*n);
                facts(post.dtype(), shape, ins)
            }
            Launch::Gather { space, .. } | Launch::Scatter { space, .. } => {
                facts(ins[0].dtype, space.dims.clone(), ins)
            }
            Launch::Region { .. } => {
                let last = ins
                    .last()
                    .ok_or_else(|| Error::Shape("empty region".into()))?;
                facts(last.dtype, last.shape.clone(), ins)
            }
            Launch::Ext { .. } => return Err(Error::Legality("no test Ext registry".into())),
        })
    }

    impl Semantics for TestSem {
        fn children(&self, op: &Op) -> Children {
            children_of(op)
        }
        fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts> {
            match op {
                Op::Logical(o) => infer_logical(o, ins),
                Op::Launch(o) => infer_launch(o, ins),
                Op::Union(..) => Err(Error::Legality("union inferred by the graph".into())),
            }
        }
        fn work(&self, _op: &Op, _ins: &[ValueFacts], out: &ValueFacts) -> Work {
            Work {
                macs: out.elements().unwrap_or(1),
                ..Work::default()
            }
        }
        fn verify(&self, _cx: &VerifyCtx<'_>) -> Result<()> {
            Ok(())
        }
        fn effect(&self, _op: &Op) -> Effect {
            Effect::Pure
        }
    }

    pub fn graph() -> EGraph {
        EGraph::new(Arc::new(TestSem))
    }

    pub fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: true,
            coop: smallvec![CoopKind {
                operand: Dtype::F32,
                acc: Dtype::F32,
                m: 8,
                n: 8,
                k: 8,
            }],
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: smallvec![4, 8],
            threads: 1,
        }
    }

    pub fn buffer(g: &mut EGraph, dtype: Dtype, shape: &[Dim]) -> Id {
        let n = g.len() as u32;
        g.add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(n),
            dtype,
            shape: shape.iter().copied().collect(),
        })))
        .unwrap()
    }

    pub fn map(g: &mut EGraph, expr: ScalarExpr, ins: &[Id]) -> Id {
        g.add(Op::Logical(Logical::Map {
            expr,
            ins: ins.iter().copied().collect(),
            outs: 1,
        }))
        .unwrap()
    }

    /// The single-slot binop carrier every plain reduction is.
    pub fn binop_carrier(op: crate::scalar::BinOp, acc: Dtype) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, acc).unwrap(), acc)
    }

    pub fn fold(g: &mut EGraph, carrier: Carrier, axis: u32, acc: Dtype, x: Id) -> Id {
        g.add(Op::Logical(Logical::Fold {
            carrier,
            axis,
            acc,
            ins: smallvec![x],
        }))
        .unwrap()
    }

    pub fn restride(g: &mut EGraph, specs: &[StrideSpec], x: Id) -> Id {
        g.add(Op::Logical(Logical::Restride {
            specs: specs.iter().copied().collect(),
            bounds: BoundsProof::Static,
            x,
        }))
        .unwrap()
    }

    pub fn contract(g: &mut EGraph, spec: EinSpec, acc: Dtype, a: Id, b: Id) -> Id {
        g.add(Op::Logical(Logical::Contract {
            spec,
            acc,
            a,
            b,
            outs: 1,
        }))
        .unwrap()
    }

    pub fn scatter(g: &mut EGraph, axis: u32, base: Id, idx: Id, upd: Id) -> Id {
        g.add(Op::Logical(Logical::Scatter {
            axis,
            combine: ScatterCombine::Add,
            base,
            idx,
            upd,
            unique: false,
        }))
        .unwrap()
    }

    pub fn kmap(g: &mut EGraph, space: &[Dim], body: ScalarExpr, ops: Vec<Operand>) -> Id {
        g.add(Op::Launch(Launch::Map {
            space: IndexSpace::new(space.iter().copied()),
            body,
            ops,
            sched: ScheduleDomain::Point,
        }))
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kfold(
        g: &mut EGraph,
        space: &[Dim],
        axis: u32,
        carrier: Carrier,
        acc: Dtype,
        post: ScalarExpr,
        ops: Vec<Operand>,
    ) -> Id {
        g.add(Op::Launch(Launch::Fold {
            space: IndexSpace::new(space.iter().copied()),
            axis,
            vec_axes: smallvec![],
            carrier,
            acc,
            post: smallvec![post],
            ops,
            sched: ScheduleDomain::Point,
        }))
        .unwrap()
    }

    pub fn kcontract(
        g: &mut EGraph,
        m: Dim,
        n: Dim,
        k: Dim,
        post: ScalarExpr,
        a: Operand,
        b: Operand,
    ) -> Id {
        g.add(Op::Launch(Launch::Contract {
            m,
            n,
            k,
            batch: Dim::ONE,
            family: Family::Sgemm,
            post,
            acc: Dtype::F32,
            a: ContractSide::one(ident_expr(Dtype::F32), a),
            b: ContractSide::one(ident_expr(Dtype::F32), b),
            sched: ScheduleDomain::Point,
        }))
        .unwrap()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::ScalarKind;
    use crate::egraph::{RuleTag, SaturationBudget, Saturate};
    use crate::ir::{Level, OpTag};
    use crate::rules::test_support as ts;
    use crate::saturate::CoreSaturate;
    use rustc_hash::FxHashSet;

    #[test]
    fn core_rules_are_exactly_the_named_set() {
        let expected = [
            "STRIP",
            "RECOGNIZE_CONTRACT",
            "CONTRACT_REASSOC",
            "CONST_FOLD_MAP",
            "IDENTITY_ELIM",
            "WIDEN_STORE_CAST",
            "UNIT_FOLD_COLLAPSE",
            "ABSORB",
            "MAP_INTO_CONTRACT",
            "MAP_INTO_MAP",
            "FOLD_POST_EPILOGUE",
            "FORM_KREGION",
            "PROMOTE",
            "HOIST",
            "RETARGET",
            "TUPLE",
            "TUPLE_SIBLING",
            "SINK_EPILOGUE",
            "FOLD_VIEWS_INTO_INDEX",
            "FOLD_VIEWS_INTO_FOLD_INDEX",
            "OPERAND_ALIAS",
            "OPERAND_GATHER",
            "OPERAND_PACK",
            "OPERAND_UNFLATTEN",
            "LOWER_MAP",
            "LOWER_FOLD",
            "LOWER_CONTRACT_GENERIC",
            "LOWER_RESTRIDE",
            "LOWER_WINDOW",
            "LOWER_GATHER",
            "LOWER_SCATTER",
            "LOWER_DEQUANT",
            "LOWER_PROJECT",
            "SPECIALIZE_DIM",
        ];
        let got: Vec<&str> = CORE_RULES.iter().map(|r| r.name).collect();
        assert_eq!(got, expected);
        assert_eq!(CORE_RULES.len(), 34);
        assert!(
            !CORE_RULES.iter().any(|r| r.name.contains("FLASH")),
            "a flash recognizer is back in the table"
        );
        for (i, r) in CORE_RULES.iter().enumerate() {
            assert_eq!(rule_id(r.name), Some(RuleId(i as u16)));
            assert_eq!(rule(RuleId(i as u16)).name, r.name);
        }
    }

    #[test]
    fn every_l0_op_but_leaf_has_a_lowering_floor_rule() {
        let floors: FxHashSet<OpTag> = CORE_RULES
            .iter()
            .filter(|r| r.tag == RuleTag::StrictlyLowering)
            .map(|r| r.head)
            .collect();
        for tag in [
            OpTag::Map,
            OpTag::Fold,
            OpTag::Contract,
            OpTag::Restride,
            OpTag::Window,
            OpTag::Gather,
            OpTag::Scatter,
            OpTag::Dequant,
            OpTag::Project,
        ] {
            assert!(floors.contains(&tag), "no floor rule for {tag:?}");
        }
        assert!(!floors.contains(&OpTag::Leaf));
        // Every lowering-floor rule descends from Logical.
        assert!(
            CORE_RULES
                .iter()
                .filter(|r| r.tag == RuleTag::StrictlyLowering)
                .all(|r| r.level == Level::Logical)
        );
    }

    /// Elementwise-into-elementwise is `ScalarExpr::compose` — a tree
    /// substitution — and no `Logical::Map`-headed rule produces a second
    /// `Logical::Map`; `fusion::MAP_INTO_MAP` fuses at Launch instead.
    #[test]
    fn elementwise_into_elementwise_needs_no_rule() {
        let inner = ScalarExpr::un(crate::scalar::UnOp::Exp, ScalarExpr::arg(0, Dtype::F32));
        let outer = ScalarExpr::un(crate::scalar::UnOp::Sqrt, ScalarExpr::arg(0, Dtype::F32));
        let fused = outer.compose(&[inner.clone()]);
        match fused.kind() {
            ScalarKind::Un { op, x } => {
                assert_eq!(*op, crate::scalar::UnOp::Sqrt);
                assert_eq!(x, &inner);
            }
            other => panic!("expected sqrt(exp(x)), got {other:?}"),
        }

        // No rule fires on a Map whose sole operand is another Map in a way
        // that produces a third Map: the Map-headed rules are the three
        // algebraic ones plus the lowering floor.
        let map_headed: Vec<&str> = CORE_RULES
            .iter()
            .filter(|r| r.head == OpTag::Map)
            .map(|r| r.name)
            .collect();
        assert_eq!(
            map_headed,
            ["CONST_FOLD_MAP", "IDENTITY_ELIM", "WIDEN_STORE_CAST", "LOWER_MAP"]
        );

        // And the graph agrees: saturating Map(Map(x)) mints no fused Map.
        let mut g = ts::graph();
        let caps = ts::caps();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(8)]);
        let m1 = ts::map(&mut g, inner, &[x]);
        let m2 = ts::map(&mut g, outer, &[m1]);
        let before = g.chain(m2).len();
        CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, SaturationBudget::default())
            .unwrap();
        let members = g.chain(m2);
        // Only the Map lowering joined the class; no second Logical Map appeared.
        let l0_maps = members
            .iter()
            .filter(|&&m| matches!(g.node(m).op, Op::Logical(Logical::Map { .. })))
            .count();
        assert_eq!(before, 1);
        assert_eq!(l0_maps, 1, "a Map-into-Map alternative was minted");

        // At Launch it *is* a rule, and it fires: the class holds a one-operand
        // `Map` whose body is the composed expression, reading `x` directly.
        let fused_kmap = members.iter().copied().find(|&m| {
            matches!(&g.node(m).op, Op::Launch(Launch::Map { ops, .. }) if ops.len() == 1 && ops[0].src == x)
        });
        let fused_kmap = fused_kmap.expect("MAP_INTO_MAP did not fuse a two-map chain");
        let Op::Launch(Launch::Map { body, .. }) = &g.node(fused_kmap).op else {
            unreachable!()
        };
        assert_eq!(body, &fused, "the fused body is not the composed one");
    }
}
