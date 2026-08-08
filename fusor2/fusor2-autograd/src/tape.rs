//! [`GraphTape`]: the [`Tape`] implementation over a `&mut EGraph`. Every
//! method appends L0 nodes to the same graph the forward lives in, which makes
//! checkpointing an extraction decision.

use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::dtype::{Dtype, Splat};
use fusor2_ir::egraph::EGraph;
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::carrier::Carrier;
use fusor2_ir::ir::level0::{EinSpec, L0, LeafKind, ScatterCombine};
use fusor2_ir::ir::{Children, Node, Op};
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor2_ir::shape::{BoundsProof, Dim, Dims, StrideSpec, broadcast_specs};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

/// A thin writer over the live e-graph.
pub struct GraphTape<'a> {
    graph: &'a mut EGraph,
}

impl<'a> GraphTape<'a> {
    pub fn new(graph: &'a mut EGraph) -> Self {
        Self { graph }
    }

    pub fn graph(&self) -> &EGraph {
        self.graph
    }

    pub fn graph_mut(&mut self) -> &mut EGraph {
        self.graph
    }

    pub fn node(&self, v: Val) -> &Node {
        self.graph.node(v)
    }

    pub fn children_of(&self, v: Val) -> Children {
        self.graph.node(v).children.clone()
    }

}

/// Construction helpers every adjoint uses, on any tape.
///
/// These live on an extension trait rather than on [`GraphTape`] because
/// [`fusor2_ir::autograd::AdjointFn`] receives `&mut dyn Tape`, and the blanket
/// impl over `T: Tape + ?Sized` makes them callable there.
pub trait TapeExt: Tape {
    fn shape_of(&self, v: Val) -> Dims {
        self.facts(v).shape.clone()
    }

    fn dtype_of(&self, v: Val) -> Dtype {
        self.facts(v).dtype
    }

    fn rank_of(&self, v: Val) -> usize {
        self.facts(v).shape.len()
    }

    /// `Arg(i)` typed like operand `i`.
    fn arg_like(&self, v: Val, slot: u32) -> ScalarExpr {
        ScalarExpr::arg(slot, self.dtype_of(v))
    }

    /// Elementwise unary over one operand.
    fn unary(&mut self, op: UnOp, v: Val) -> Result<Val> {
        let body = ScalarExpr::un(op, self.arg_like(v, 0));
        self.map(body, &[v])
    }

    /// Elementwise binary over two same-shaped operands.
    fn binary(&mut self, op: BinOp, a: Val, b: Val) -> Result<Val> {
        let body = ScalarExpr::bin(op, self.arg_like(a, 0), self.arg_like(b, 1));
        self.map(body, &[a, b])
    }

    /// Elementwise comparison; 1.0/0.0 in the operand dtype (no bool at L0).
    fn compare(&mut self, op: CmpOp, a: Val, b: Val) -> Result<Val> {
        let body = ScalarExpr::cmp(op, self.arg_like(a, 0), self.arg_like(b, 1));
        self.map(body, &[a, b])
    }

    /// `where_cond`.
    fn select(&mut self, c: Val, t: Val, f: Val) -> Result<Val> {
        let body = ScalarExpr::select(
            self.arg_like(c, 0),
            self.arg_like(t, 1),
            self.arg_like(f, 2),
        );
        self.map(body, &[c, t, f])
    }

    /// Numeric conversion. Differentiable both directions with no special case.
    fn cast(&mut self, to: Dtype, v: Val) -> Result<Val> {
        if self.dtype_of(v) == to {
            return Ok(v);
        }
        let body = ScalarExpr::cast(to, self.arg_like(v, 0));
        self.map(body, &[v])
    }

    /// `v * k` with `k` a compile-time literal in `v`'s dtype.
    fn mul_scalar(&mut self, v: Val, k: f32) -> Result<Val> {
        let dtype = self.dtype_of(v);
        let body = ScalarExpr::bin(
            BinOp::Mul,
            self.arg_like(v, 0),
            ScalarExpr::lit(splat_of(dtype, k)?),
        );
        self.map(body, &[v])
    }

    /// A constant tensor shaped and typed like `v`.
    fn splat_like(&mut self, v: Val, value: f32) -> Result<Val> {
        let dtype = self.dtype_of(v);
        let shape = self.shape_of(v);
        self.add(L0::Leaf(LeafKind::Const {
            value: splat_of(dtype, value)?,
            shape,
        }))
    }

    /// `Fold{Add}` over `axis`.
    fn sum_axis(&mut self, v: Val, axis: u32) -> Result<Val> {
        let acc = accum_dtype(self.dtype_of(v));
        self.fold_binop(fusor2_ir::scalar::BinOp::Add, axis, acc, v)
    }

    /// Insert a stride-0 axis of `extent` at position `axis`.
    fn broadcast_axis(&mut self, v: Val, axis: u32, extent: Dim) -> Result<Val> {
        let shape = self.shape_of(v);
        let rank = shape.len();
        let axis = axis as usize;
        if axis > rank {
            return Err(Error::Shape(format!(
                "broadcast_axis {axis} past rank {rank}"
            )));
        }
        let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::with_capacity(rank + 1);
        for (j, d) in shape.iter().copied().enumerate().take(axis) {
            specs.push(StrideSpec::dim(j as u32, d));
        }
        specs.push(StrideSpec::broadcast(extent));
        for (j, d) in shape.iter().copied().enumerate().skip(axis) {
            specs.push(StrideSpec::dim(j as u32, d));
        }
        self.restride(&specs, v)
    }

    /// Right-aligned broadcast into `shape`, via [`broadcast_specs`]. Callers
    /// must insert this before [`Tape::map`]: broadcast is illegal inside a
    /// `Map` body (verify_l0 clause 2).
    fn broadcast_to(&mut self, v: Val, shape: &[Dim]) -> Result<Val> {
        let src = self.shape_of(v);
        if src.len() == shape.len() && src.iter().zip(shape).all(|(a, b)| a.known_eq(*b)) {
            return Ok(v);
        }
        let specs = broadcast_specs(&src, shape)?;
        self.restride(&specs, v)
    }

    /// Reshape to `shape` by reading `v` row-major. Only legal when the
    /// element counts agree decidably.
    fn reshape(&mut self, v: Val, shape: &[Dim]) -> Result<Val> {
        let src = self.shape_of(v);
        if src.len() == shape.len() && src.iter().zip(shape).all(|(a, b)| a.known_eq(*b)) {
            return Ok(v);
        }
        let want = const_numel(shape).ok_or_else(|| {
            Error::Shape("reshape needs decidable extents on the target shape".into())
        })?;
        let have = const_numel(&src)
            .ok_or_else(|| Error::Shape("reshape needs decidable source extents".into()))?;
        if want != have {
            return Err(Error::Shape(format!(
                "reshape {src:?} -> {shape:?} changes element count {have} -> {want}"
            )));
        }
        // The innermost source axis carries stride 1, so a full row-major
        // re-slicing is expressible as multipliers against it.
        let inner = src.len().saturating_sub(1) as u32;
        let mut mult: u64 = 1;
        let mut specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
            StrideSpec::broadcast(Dim::Const(1));
            shape.len()
        ];
        for axis in (0..shape.len()).rev() {
            let size = shape[axis];
            specs[axis] = StrideSpec::dim_with(inner, size, mult as u32);
            mult *= size.as_const().unwrap_or(1);
        }
        self.restride(&specs, v)
    }

    /// Permute axes: `perm[j]` is the source axis that becomes axis `j`.
    fn permute(&mut self, v: Val, perm: &[u32]) -> Result<Val> {
        let shape = self.shape_of(v);
        if perm.len() != shape.len() {
            return Err(Error::Shape(format!(
                "permute rank {} against shape rank {}",
                perm.len(),
                shape.len()
            )));
        }
        if perm.iter().enumerate().all(|(j, p)| j as u32 == *p) {
            return Ok(v);
        }
        let specs: SmallVec<[StrideSpec; 6]> = perm
            .iter()
            .map(|&p| StrideSpec::dim(p, shape[p as usize]))
            .collect();
        self.restride(&specs, v)
    }

    /// Gather rows of `x` along `axis` with a rank-1 index tensor.
    fn gather(&mut self, axis: u32, x: Val, idx: Val) -> Result<Val> {
        self.add(L0::Gather { axis, x, idx })
    }

    /// `Scatter{Set}`. Used by the adjoint of a `Scatter{Set}`, which inherits
    /// the primal's uniqueness proof.
    fn scatter_set(
        &mut self,
        axis: u32,
        base: Val,
        idx: Val,
        upd: Val,
        unique: bool,
    ) -> Result<Val> {
        self.add(L0::Scatter {
            axis,
            combine: ScatterCombine::Set,
            base,
            idx,
            upd,
            unique,
        })
    }

    /// A constant tensor of `value`, dense `dtype`, shape `shape`.
    fn zeros_shaped(&mut self, dtype: Dtype, shape: &[Dim]) -> Result<Val> {
        self.add(L0::Leaf(LeafKind::Const {
            value: splat_of(dtype, 0.0)?,
            shape: shape.iter().copied().collect(),
        }))
    }

    /// Flatten `v` to rank 1.
    fn flatten(&mut self, v: Val) -> Result<Val> {
        let shape = self.shape_of(v);
        let n = const_numel(&shape)
            .ok_or_else(|| Error::Shape("cannot flatten a symbolic shape".into()))?;
        self.reshape(v, &[Dim::Const(n)])
    }
}

impl<T: Tape + ?Sized> TapeExt for T {}

impl Tape for GraphTape<'_> {
    fn add(&mut self, op: L0) -> Result<Val> {
        self.graph.add(Op::L0(op))
    }

    fn facts(&self, v: Val) -> &ValueFacts {
        self.graph.facts(v)
    }

    fn zeros_like(&mut self, v: Val) -> Result<Val> {
        self.splat_like(v, 0.0)
    }

    fn map(&mut self, expr: ScalarExpr, ins: &[Val]) -> Result<Val> {
        if ins.is_empty() {
            return Err(Error::Shape(
                "Map needs at least one operand to fix its index space".into(),
            ));
        }
        // verify_l0 clause 2: no implicit broadcasting inside a Map. The caller
        // inserts `restride` first.
        let first = self.shape_of(ins[0]);
        for (slot, v) in ins.iter().enumerate().skip(1) {
            let s = self.shape_of(*v);
            if s.len() != first.len() || !s.iter().zip(&first).all(|(a, b)| a.known_eq(*b)) {
                return Err(Error::Shape(format!(
                    "Map operand {slot} has shape {s:?}, expected {first:?}; \
                     broadcast is illegal inside a Map body"
                )));
            }
        }
        self.add(L0::Map {
            expr,
            ins: ins.iter().copied().collect(),
            outs: 1,
        })
    }

    fn contract(&mut self, a: Val, b: Val, spec: EinSpec, acc: Dtype) -> Result<Val> {
        crate::contract::verify_spec(&spec)?;
        self.add(L0::Contract {
            spec,
            acc,
            a,
            b,
            outs: 1,
        })
    }

    fn fold(&mut self, carrier: Carrier, axis: u32, acc: Dtype, x: Val) -> Result<Val> {
        self.add(L0::Fold {
            carrier,
            axis,
            acc,
            ins: smallvec::smallvec![x],
        })
    }

    fn restride(&mut self, specs: &[StrideSpec], x: Val) -> Result<Val> {
        let shape = self.shape_of(x);
        let bounds = bounds_proof(specs, &shape);
        self.add(L0::Restride {
            specs: specs.iter().copied().collect(),
            bounds,
            x,
        })
    }

    fn scatter_add(&mut self, axis: u32, base: Val, idx: Val, upd: Val) -> Result<Val> {
        // Never `Set`: duplicates must accumulate. The embedding table
        // receiving one token twice gets the summed gradient.
        self.add(L0::Scatter {
            axis,
            combine: ScatterCombine::Add,
            base,
            idx,
            upd,
            unique: false,
        })
    }

    fn accumulate(&mut self, a: Val, b: Val) -> Result<Val> {
        if a == b {
            // A fan-in of two identical ids hash-conses into one scale.
            return self.mul_scalar(a, 2.0);
        }
        self.binary(BinOp::Add, a, b)
    }
}

/// `BoundsProof::Static` when every extent is decidable and the composed reach
/// provably lands inside the source; `RuntimeMask` otherwise.
pub fn bounds_proof(specs: &[StrideSpec], src: &[Dim]) -> BoundsProof {
    let Some(strides) = const_row_major(src) else {
        return BoundsProof::RuntimeMask;
    };
    let Some(numel) = const_numel(src) else {
        return BoundsProof::RuntimeMask;
    };
    let mut reach: u128 = 0;
    for spec in specs {
        let Some(size) = spec.size.as_const() else {
            return BoundsProof::RuntimeMask;
        };
        let Some(offset) = spec.offset.as_const() else {
            return BoundsProof::RuntimeMask;
        };
        let Some(&stride) = strides.get(spec.input_dim as usize) else {
            return BoundsProof::RuntimeMask;
        };
        if size == 0 {
            continue;
        }
        let step = stride as u128 * spec.multiplier as u128;
        reach += step * (size as u128 - 1) + stride as u128 * offset as u128;
    }
    if reach < numel as u128 {
        BoundsProof::Static
    } else {
        BoundsProof::RuntimeMask
    }
}

/// Row-major strides when every extent is decidable.
pub fn const_row_major(shape: &[Dim]) -> Option<Vec<u64>> {
    let mut out = vec![0u64; shape.len()];
    let mut acc: u64 = 1;
    for axis in (0..shape.len()).rev() {
        out[axis] = acc;
        acc = acc.checked_mul(shape[axis].as_const()?)?;
    }
    Some(out)
}

/// Element count when every extent is decidable.
pub fn const_numel(shape: &[Dim]) -> Option<u64> {
    shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
}

/// A typed literal of `value` in `dtype`.
pub fn splat_of(dtype: Dtype, value: f32) -> Result<Splat> {
    Ok(match dtype {
        Dtype::F32 => Splat::F32(value),
        Dtype::F16 => Splat::F16(half::f16::from_f32(value).to_bits()),
        Dtype::BF16 => Splat::BF16(half::bf16::from_f32(value).to_bits()),
        Dtype::U32 => Splat::U32(value as u32),
        Dtype::I32 => Splat::I32(value as i32),
        Dtype::Q(f) => {
            return Err(Error::Dtype(format!(
                "no dense literal in quantized format {f:?}"
            )));
        }
    })
}

/// A `ScalarExpr` literal of `value` in `dtype`.
pub fn lit(value: f32, dtype: Dtype) -> Result<ScalarExpr> {
    Ok(ScalarExpr::lit(splat_of(dtype, value)?))
}

/// Accumulator width for a fold over `dtype`. Narrow floats accumulate in f32,
/// the `NumericContract` floor.
pub const fn accum_dtype(dtype: Dtype) -> Dtype {
    dtype.compute_dtype()
}

#[cfg(test)]
pub(crate) mod testing {
    //! The e-graph this crate's tests build against, plus a naive L0
    //! interpreter that checks every adjoint against a central difference of
    //! the forward it differentiates.
    //!
    //! The `Semantics` is [`CoreSemantics`], so `infer_fold` types a float
    //! fold's output as `acc` and `verify_l0` runs on every graph an adjoint
    //! builds. [`SumArenaPlanner`] suffices because this crate builds L0 only.

    use super::*;
    use fusor2_ir::device::{Caps, DeviceKind, Limits};
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use std::sync::Arc;

    pub fn graph() -> EGraph {
        EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)))
    }

    /// A capability set no adjoint reads: this crate builds L0 only.
    pub fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Cpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: None,
            f16: true,
            bf16: true,
            coop: SmallVec::new(),
            atomic_f32: false,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: smallvec::smallvec![4, 8],
            threads: 1,
        }
    }


    // The interpreter is dense, row-major, f32-valued and single-threaded.

    use fusor2_ir::egraph::Id;
    use fusor2_ir::scalar::{CmpOp, ScalarKind};
    use rustc_hash::FxHashMap;

    /// Values bound to the graph's leaves, keyed by node id.
    pub type Env = FxHashMap<Id, Vec<f32>>;

    pub fn shape_of(g: &EGraph, id: Id) -> Vec<usize> {
        g.facts(id)
            .shape
            .iter()
            .map(|d| d.as_const().expect("test graphs use constant extents") as usize)
            .collect()
    }

    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product::<usize>().max(1)
    }

    pub fn strides(shape: &[usize]) -> Vec<usize> {
        let mut out = vec![1usize; shape.len()];
        let mut acc = 1usize;
        for a in (0..shape.len()).rev() {
            out[a] = acc;
            acc *= shape[a];
        }
        out
    }

    pub fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
        let st = strides(shape);
        let mut out = vec![0usize; shape.len()];
        for a in 0..shape.len() {
            out[a] = linear / st[a];
            linear %= st[a];
        }
        out
    }

    /// Evaluate one scalar body. `coords` backs `IndexOf`.
    pub fn eval_scalar(e: &ScalarExpr, args: &[f32], coords: &[usize]) -> f32 {
        let go = |x: &ScalarExpr| eval_scalar(x, args, coords);
        match e.kind() {
            ScalarKind::Arg(i) => args[*i as usize],
            ScalarKind::Lit(l) => match l.0 {
                Splat::F32(v) => v,
                Splat::F16(b) => half::f16::from_bits(b).to_f32(),
                Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
                Splat::U32(v) => v as f32,
                Splat::I32(v) => v as f32,
            },
            ScalarKind::Uniform(_) => 0.0,
            ScalarKind::IndexOf(axis) => coords[*axis as usize] as f32,
            ScalarKind::Un { op, x } => {
                let v = go(x);
                match op {
                    UnOp::Exp
                    | UnOp::ApproximateExp
                    | UnOp::LessApproximateExp => v.exp(),
                    UnOp::Exp2 => v.exp2(),
                    UnOp::Log => v.ln(),
                    UnOp::Log2 => v.log2(),
                    UnOp::Sqrt => v.sqrt(),
                    UnOp::InverseSqrt => 1.0 / v.sqrt(),
                    UnOp::Sin => v.sin(),
                    UnOp::Cos => v.cos(),
                    UnOp::Tan => v.tan(),
                    UnOp::Tanh => v.tanh(),
                    UnOp::Asin => v.asin(),
                    UnOp::Acos => v.acos(),
                    UnOp::Atan => v.atan(),
                    UnOp::Sinh => v.sinh(),
                    UnOp::Cosh => v.cosh(),
                    UnOp::Asinh => v.asinh(),
                    UnOp::Acosh => v.acosh(),
                    UnOp::Atanh => v.atanh(),
                    UnOp::Abs => v.abs(),
                    UnOp::Neg => -v,
                    UnOp::Unpack2x16Float => v,
                }
            }
            ScalarKind::Bin { op, a, b } => {
                let (x, y) = (go(a), go(b));
                match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    BinOp::Rem => x % y,
                    BinOp::Pow => x.powf(y),
                    BinOp::Min => x.min(y),
                    BinOp::Max => x.max(y),
                    BinOp::BitAnd => ((x as u32) & (y as u32)) as f32,
                    BinOp::BitOr => ((x as u32) | (y as u32)) as f32,
                    BinOp::BitXor => ((x as u32) ^ (y as u32)) as f32,
                    BinOp::Shr => ((x as u32) >> (y as u32)) as f32,
                    BinOp::Shl => ((x as u32) << (y as u32)) as f32,
                    BinOp::LogicalAnd => f32::from(x != 0.0 && y != 0.0),
                    BinOp::LogicalOr => f32::from(x != 0.0 || y != 0.0),
                }
            }
            ScalarKind::Cmp { op, a, b } => {
                let (x, y) = (go(a), go(b));
                let t = match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                };
                f32::from(t)
            }
            ScalarKind::Select { c, t, f } => {
                if go(c) != 0.0 {
                    go(t)
                } else {
                    go(f)
                }
            }
            ScalarKind::Cast { to, x } => match to {
                Dtype::F16 => half::f16::from_f32(go(x)).to_f32(),
                Dtype::BF16 => half::bf16::from_f32(go(x)).to_f32(),
                Dtype::U32 => (go(x).max(0.0) as u32) as f32,
                Dtype::I32 => (go(x) as i32) as f32,
                _ => go(x),
            },
            ScalarKind::Bitcast { x, .. } => go(x),
            ScalarKind::Round { mode, x } => {
                let v = go(x);
                match mode {
                    fusor2_ir::dtype::RoundMode::Floor => v.floor(),
                    fusor2_ir::dtype::RoundMode::Ceil => v.ceil(),
                    fusor2_ir::dtype::RoundMode::Trunc => v.trunc(),
                    _ => v.round(),
                }
            }
            ScalarKind::Dot { a, b } => go(a) * go(b),
            ScalarKind::Splat { x, .. } => go(x),
        }
    }

    /// Evaluate `id` under `env`, memoizing every visited node.
    pub fn eval(g: &EGraph, id: Id, env: &Env) -> Vec<f32> {
        let mut memo: FxHashMap<Id, Vec<f32>> = FxHashMap::default();
        eval_memo(g, id, env, &mut memo)
    }

    fn eval_memo(g: &EGraph, id: Id, env: &Env, memo: &mut FxHashMap<Id, Vec<f32>>) -> Vec<f32> {
        if let Some(v) = memo.get(&id) {
            return v.clone();
        }
        let out = eval_node(g, id, env, memo);
        memo.insert(id, out.clone());
        out
    }

    fn eval_node(g: &EGraph, id: Id, env: &Env, memo: &mut FxHashMap<Id, Vec<f32>>) -> Vec<f32> {
        let shape = shape_of(g, id);
        let n = numel(&shape);
        match &g.node(id).op {
            Op::Union(a, _) => eval_memo(g, *a, env, memo),
            Op::L1(_) => panic!("the test interpreter is L0-only"),
            Op::L0(l0) => match l0 {
                L0::Leaf(LeafKind::Const { value, .. }) => {
                    let v = match value {
                        Splat::F32(v) => *v,
                        Splat::F16(b) => half::f16::from_bits(*b).to_f32(),
                        Splat::BF16(b) => half::bf16::from_bits(*b).to_f32(),
                        Splat::U32(v) => *v as f32,
                        Splat::I32(v) => *v as f32,
                    };
                    vec![v; n]
                }
                L0::Leaf(_) => env
                    .get(&id)
                    .unwrap_or_else(|| panic!("no binding for leaf {id}"))
                    .clone(),
                L0::Map { expr, ins, .. } => {
                    let vals: Vec<Vec<f32>> =
                        ins.iter().map(|v| eval_memo(g, *v, env, memo)).collect();
                    let mut args = vec![0.0f32; vals.len()];
                    (0..n)
                        .map(|e| {
                            for (slot, v) in vals.iter().enumerate() {
                                args[slot] = v[e];
                            }
                            eval_scalar(expr, &args, &unravel(e, &shape))
                        })
                        .collect()
                }
                L0::Fold {
                    carrier, axis, ins, ..
                } => {
                    let x = &ins[0];
                    let src = eval_memo(g, *x, env, memo);
                    let xshape = shape_of(g, *x);
                    let xst = strides(&xshape);
                    let axis = *axis as usize;
                    let extent = xshape[axis];
                    (0..n)
                        .map(|e| {
                            let oc = unravel(e, &shape);
                            let mut xc = Vec::with_capacity(xshape.len());
                            xc.extend_from_slice(&oc[..axis]);
                            xc.push(0);
                            xc.extend_from_slice(&oc[axis..]);
                            let base: usize =
                                xc.iter().zip(&xst).map(|(c, s)| c * s).sum::<usize>();
                            // Run the carrier rather than switching on a name:
                            // seed from the identity and absorb.
                            let mut acc = carrier.identity_f32();
                            for k in 0..extent {
                                let v = src[base + k * xst[axis]];
                                acc = carrier
                                    .absorb(&acc, &[v])
                                    .expect("the host evaluator covers this carrier");
                            }
                            acc[0]
                        })
                        .collect()
                }
                L0::Restride { specs, x, .. } => {
                    let src = eval_memo(g, *x, env, memo);
                    let xst = strides(&shape_of(g, *x));
                    (0..n)
                        .map(|e| {
                            let oc = unravel(e, &shape);
                            let mut lin = 0usize;
                            for (pos, s) in specs.iter().enumerate() {
                                let base = xst[s.input_dim as usize];
                                lin += oc[pos] * s.multiplier as usize * base;
                                lin += s.offset.as_const().unwrap_or(0) as usize * base;
                            }
                            src[lin]
                        })
                        .collect()
                }
                L0::Window { specs, x } => {
                    let src = eval_memo(g, *x, env, memo);
                    let xshape = shape_of(g, *x);
                    let xst = strides(&xshape);
                    let rank = xshape.len();
                    (0..n)
                        .map(|e| {
                            let oc = unravel(e, &shape);
                            let mut lin = 0usize;
                            let mut coord: Vec<usize> = oc[..rank].to_vec();
                            for (i, w) in specs.iter().enumerate() {
                                let a = w.axis as usize;
                                coord[a] = oc[a] * w.step as usize + oc[rank + i];
                            }
                            for (d, c) in coord.iter().enumerate() {
                                lin += c * xst[d];
                            }
                            src[lin]
                        })
                        .collect()
                }
                L0::Contract { spec, a, b, .. } => {
                    let av = eval_memo(g, *a, env, memo);
                    let bv = eval_memo(g, *b, env, memo);
                    let ash = shape_of(g, *a);
                    let bsh = shape_of(g, *b);
                    let ast = strides(&ash);
                    let bst = strides(&bsh);
                    let mut extent: FxHashMap<u8, usize> = FxHashMap::default();
                    for (i, l) in spec.a.iter().enumerate() {
                        extent.insert(l.0, ash[i]);
                    }
                    for (i, l) in spec.b.iter().enumerate() {
                        extent.insert(l.0, bsh[i]);
                    }
                    let contracted: Vec<u8> = extent
                        .keys()
                        .copied()
                        .filter(|l| !spec.out.iter().any(|o| o.0 == *l))
                        .collect();
                    let cshape: Vec<usize> = contracted.iter().map(|l| extent[l]).collect();
                    let csize = numel(&cshape);
                    (0..n)
                        .map(|e| {
                            let oc = unravel(e, &shape);
                            let mut assign: FxHashMap<u8, usize> = FxHashMap::default();
                            for (i, l) in spec.out.iter().enumerate() {
                                assign.insert(l.0, oc[i]);
                            }
                            let mut acc = 0.0f32;
                            for k in 0..csize {
                                let kc = unravel(k, &cshape);
                                for (i, l) in contracted.iter().enumerate() {
                                    assign.insert(*l, kc[i]);
                                }
                                let ai: usize = spec
                                    .a
                                    .iter()
                                    .enumerate()
                                    .map(|(i, l)| assign[&l.0] * ast[i])
                                    .sum();
                                let bi: usize = spec
                                    .b
                                    .iter()
                                    .enumerate()
                                    .map(|(i, l)| assign[&l.0] * bst[i])
                                    .sum();
                                acc += av[ai] * bv[bi];
                            }
                            acc
                        })
                        .collect()
                }
                L0::Gather { axis, x, idx } => {
                    let src = eval_memo(g, *x, env, memo);
                    let ind = eval_memo(g, *idx, env, memo);
                    let xst = strides(&shape_of(g, *x));
                    let axis = *axis as usize;
                    (0..n)
                        .map(|e| {
                            let mut oc = unravel(e, &shape);
                            oc[axis] = ind[oc[axis]] as usize;
                            src[oc.iter().zip(&xst).map(|(c, s)| c * s).sum::<usize>()]
                        })
                        .collect()
                }
                L0::Scatter {
                    axis,
                    combine,
                    base,
                    idx,
                    upd,
                    ..
                } => {
                    let mut out = eval_memo(g, *base, env, memo);
                    let ind = eval_memo(g, *idx, env, memo);
                    let uv = eval_memo(g, *upd, env, memo);
                    let ushape = shape_of(g, *upd);
                    let ost = strides(&shape);
                    let axis = *axis as usize;
                    for (e, v) in uv.iter().enumerate() {
                        let mut oc = unravel(e, &ushape);
                        oc[axis] = ind[oc[axis]] as usize;
                        let t: usize = oc.iter().zip(&ost).map(|(c, s)| c * s).sum();
                        match combine {
                            ScatterCombine::Set => out[t] = *v,
                            ScatterCombine::Add => out[t] += *v,
                        }
                    }
                    out
                }
                L0::Dequant { x, .. } | L0::Project { x, .. } => eval_memo(g, *x, env, memo),
            },
        }
    }


    /// `sum(forward)` — the scalar loss a unit seed differentiates.
    pub fn sum_forward(g: &EGraph, root: Id, env: &Env) -> f32 {
        eval(g, root, env).iter().sum()
    }

    /// Compare every produced gradient against a central difference of
    /// `sum(forward)` at `h = 1e-3`.
    pub fn check_gradients(
        g: &EGraph,
        root: Id,
        wrt: &[Id],
        grads: &[Id],
        env: &Env,
        rtol: f32,
    ) {
        const H: f32 = 1e-3;
        for (k, w) in wrt.iter().enumerate() {
            let gid = grads[k];
            let analytic = eval(g, gid, env);
            let len = env[w].len();
            assert_eq!(analytic.len(), len, "gradient of {w} has the wrong extent");
            for (j, a) in analytic.iter().enumerate() {
                let mut lo = env.clone();
                lo.get_mut(w).unwrap()[j] -= H;
                let mut hi = env.clone();
                hi.get_mut(w).unwrap()[j] += H;
                let numeric = (sum_forward(g, root, &hi) - sum_forward(g, root, &lo)) / (2.0 * H);
                let tol = rtol * numeric.abs().max(1.0);
                assert!(
                    (a - numeric).abs() <= tol,
                    "d(sum)/d{w}[{j}]: analytic {a} vs finite difference {numeric}"
                );
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::testing::graph;
    use super::*;
    use fusor2_ir::ir::level0::BufferId;

    fn param(g: &mut EGraph, shape: &[u64]) -> Val {
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(0),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    #[test]
    fn zeros_like_hash_conses() {
        let mut g = graph();
        let a = param(&mut g, &[3, 4]);
        let mut t = GraphTape::new(&mut g);
        let z0 = t.zeros_like(a).unwrap();
        let z1 = t.zeros_like(a).unwrap();
        assert_eq!(z0, z1, "one zero seed per (dtype, shape)");
    }

    #[test]
    fn map_rejects_a_broadcast_operand() {
        let mut g = graph();
        let a = param(&mut g, &[3, 4]);
        let b = param(&mut g, &[4]);
        let mut t = GraphTape::new(&mut g);
        let body = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        assert!(t.map(body, &[a, b]).is_err());
    }

    #[test]
    fn accumulate_of_one_id_is_a_scale() {
        let mut g = graph();
        let a = param(&mut g, &[2]);
        let mut t = GraphTape::new(&mut g);
        let two = t.accumulate(a, a).unwrap();
        match &t.graph().node(two).op {
            Op::L0(L0::Map { ins, .. }) => assert_eq!(ins.len(), 1),
            other => panic!("expected a single-operand Map, got {other:?}"),
        }
    }

    #[test]
    fn broadcast_axis_inserts_a_stride_zero_spec() {
        let mut g = graph();
        let a = param(&mut g, &[4]);
        let mut t = GraphTape::new(&mut g);
        let b = t.broadcast_axis(a, 0, Dim::Const(3)).unwrap();
        assert_eq!(t.shape_of(b), Dims::from_slice(&[Dim::Const(3), Dim::Const(4)]));
        match &t.graph().node(b).op {
            Op::L0(L0::Restride { specs, .. }) => {
                assert!(specs[0].is_broadcast());
                assert!(!specs[1].is_broadcast());
            }
            other => panic!("expected Restride, got {other:?}"),
        }
    }

    #[test]
    fn bounds_are_static_for_an_in_range_const_view() {
        let src = [Dim::Const(3), Dim::Const(4)];
        let specs = [StrideSpec::dim(0, Dim::Const(3)), StrideSpec::dim(1, Dim::Const(4))];
        assert_eq!(bounds_proof(&specs, &src), BoundsProof::Static);
    }

    #[test]
    fn bounds_need_a_runtime_mask_under_a_symbolic_extent() {
        let src = [Dim::Sym(fusor2_ir::shape::SymId(0))];
        let specs = [StrideSpec::dim(0, src[0])];
        assert_eq!(bounds_proof(&specs, &src), BoundsProof::RuntimeMask);
    }

    #[test]
    fn reshape_round_trips_element_counts() {
        let mut g = graph();
        let a = param(&mut g, &[2, 3]);
        let mut t = GraphTape::new(&mut g);
        let flat = t.reshape(a, &[Dim::Const(6)]).unwrap();
        assert_eq!(t.shape_of(flat), Dims::from_slice(&[Dim::Const(6)]));
        assert!(t.reshape(a, &[Dim::Const(7)]).is_err());
    }
}
