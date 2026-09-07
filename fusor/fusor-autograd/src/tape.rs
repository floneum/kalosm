//! [`GraphTape`]: the [`Tape`] implementation over a `&mut EGraph`. Every
//! method appends Logical nodes to the *same* graph the forward lives in, which is
//! what makes checkpointing an extraction decision.

use fusor_ir::autograd::{Tape, Val};
use fusor_ir::carrier::Carrier;
use fusor_ir::dtype::{Dtype, Splat};
use fusor_ir::egraph::EGraph;
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::logical::{EinSpec, LeafKind, Logical, ScatterCombine};
use fusor_ir::ir::{Children, Node, Op};
use fusor_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor_ir::shape::{BoundsProof, Dim, Dims, StrideSpec, broadcast_specs};
use fusor_ir::{Error, Result};
use smallvec::SmallVec;

/// A tape that is not a tape: a thin writer over the live e-graph.
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
/// [`fusor_ir::autograd::AdjointFn`] receives `&mut dyn Tape`; the blanket
/// impl over `T: Tape + ?Sized` is what makes these callable there.
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

    /// Elementwise comparison; 1.0/0.0 in the operand dtype (no bool at Logical).
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

    /// Numeric conversion. Differentiable in both directions.
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
        self.add(Logical::Leaf(LeafKind::Const {
            value: splat_of(dtype, value)?,
            shape,
        }))
    }

    /// `Fold{Add}` over `axis` — the sum reduction every broadcast backward
    /// and `mean` is built from.
    fn sum_axis(&mut self, v: Val, axis: u32) -> Result<Val> {
        let acc = accum_dtype(self.dtype_of(v));
        self.fold_binop(fusor_ir::scalar::BinOp::Add, axis, acc, v)
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
        self.add(Logical::Gather { axis, x, idx })
    }

    /// `Scatter{Set}`. Only ever used by the adjoint of a `Scatter{Set}`,
    /// which inherits the primal's uniqueness proof.
    fn scatter_set(
        &mut self,
        axis: u32,
        base: Val,
        idx: Val,
        upd: Val,
        unique: bool,
    ) -> Result<Val> {
        self.add(Logical::Scatter {
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
        self.add(Logical::Leaf(LeafKind::Const {
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
    fn add(&mut self, op: Logical) -> Result<Val> {
        self.graph.add(Op::Logical(op))
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
        // verify_l0 clause 2: no implicit broadcasting inside a Map. The
        // caller inserts `restride` first; this is the assertion that says so.
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
        self.add(Logical::Map {
            expr,
            ins: ins.iter().copied().collect(),
            outs: 1,
        })
    }

    fn contract(&mut self, a: Val, b: Val, spec: EinSpec, acc: Dtype) -> Result<Val> {
        crate::contract::verify_spec(&spec)?;
        self.add(Logical::Contract {
            spec,
            acc,
            a,
            b,
            outs: 1,
        })
    }

    fn fold(&mut self, carrier: Carrier, axis: u32, acc: Dtype, x: Val) -> Result<Val> {
        self.add(Logical::Fold {
            carrier,
            axis,
            acc,
            ins: smallvec::smallvec![x],
        })
    }

    fn restride(&mut self, specs: &[StrideSpec], x: Val) -> Result<Val> {
        let shape = self.shape_of(x);
        let bounds = bounds_proof(specs, &shape);
        self.add(Logical::Restride {
            specs: specs.iter().copied().collect(),
            bounds,
            x,
        })
    }

    fn scatter_add(&mut self, axis: u32, base: Val, idx: Val, upd: Val) -> Result<Val> {
        // Never `Set`: duplicates must accumulate. The embedding table
        // receiving one token twice gets the summed gradient.
        self.add(Logical::Scatter {
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

/// `BoundsProof::Static` when every extent is decidable and the composed
/// reach provably lands inside the source; `RuntimeMask` otherwise. There is
/// no third case and no user `assume`.
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
    shape
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
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

/// Accumulator width for a fold over `dtype`. Narrow floats accumulate in
/// f32 — the `NumericContract` floor, not a preference.
pub const fn accum_dtype(dtype: Dtype) -> Dtype {
    dtype.compute_dtype()
}
