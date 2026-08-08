//! L1 -> L2 lowering helpers both backends share: the `ScalarExpr` walker,
//! the `AddressMap` walk, and the small typed-constant utilities they lean
//! on. Everything builds through [`TileBuilder`], so the two backends cannot
//! drift on the shape of the terms; the one real backend difference —
//! uniform-scalar access, plus the GPU's finite-literal clamping — is the
//! [`ScalarEnv`] each backend implements, monomorphized per call site.

use fusor2_ir::Result;
use fusor2_ir::dtype::{Dtype, NumericContract, QLayout, Splat};
use fusor2_ir::egraph::Id;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level1::AddressMap;
use fusor2_ir::ir::level2::{
    ElementType, ScalarElement, TileBinaryOp, TileCompareOp, TileExpr,
};
use fusor2_ir::scalar::ScalarExpr;
use fusor2_ir::shape::SymId;
use fusor2_ir::target::LowerCtx;

use crate::build::TileBuilder;

/// L0/L1 dtype to L2 element. Quantized weights bind as plain `u32` storage;
/// their decode is arithmetic over those words, never a buffer type.
pub const fn scalar_element(dtype: Dtype) -> ScalarElement {
    match dtype {
        Dtype::F32 => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::I32 => ScalarElement::I32,
        Dtype::U32 | Dtype::Q(_) => ScalarElement::U32,
    }
}

/// The typed zero of an element type, per lane for a vector.
pub fn zero_of(b: &mut TileBuilder, elem: ElementType) -> TileExpr {
    match elem {
        ElementType::Scalar(s) => b.zero_scalar(s),
        ElementType::Vector { scalar, lanes } => {
            let z = b.zero_scalar(scalar);
            b.vec(scalar, vec![z; lanes as usize])
        }
        ElementType::CoopMatrix { scalar, .. } => b.zero_scalar(scalar),
    }
}

/// The typed one of an element type's scalar.
pub fn one_of(b: &mut TileBuilder, elem: ElementType) -> TileExpr {
    let scalar = match elem {
        ElementType::Scalar(s) => s,
        ElementType::Vector { scalar, .. } | ElementType::CoopMatrix { scalar, .. } => scalar,
    };
    match scalar {
        ScalarElement::F32 => b.lit_f32(1.0),
        ScalarElement::F16 => b.lit(fusor2_ir::ir::level2::TileLiteral::F16(
            half::f16::ONE.to_bits(),
        )),
        ScalarElement::BF16 => b.lit(fusor2_ir::ir::level2::TileLiteral::BF16(
            half::bf16::ONE.to_bits(),
        )),
        ScalarElement::U32 => b.lit_u32(1),
        ScalarElement::I32 => b.lit_i32(1),
        ScalarElement::Bool => b.lit_bool(true),
    }
}

/// What a backend supplies the shared [`eval_scalar`] walker: how a runtime
/// scalar is read, and how a literal is spelled. Literal clamping (the GPU's
/// no-infinite-literal obligation) lives in the sink, never as a branch in
/// the shared walk — the CPU's literal bits must not change.
pub trait ScalarEnv {
    /// Read a runtime scalar, typically out of the uniform block.
    fn uniform(&mut self, b: &mut TileBuilder, sym: SymId, dtype: Dtype) -> Result<TileExpr>;
    /// A literal, after any target-required clamping.
    fn literal(&mut self, b: &mut TileBuilder, value: Splat) -> TileExpr;
}

/// Translate a [`ScalarExpr`] body into L2.
///
/// `args` are the already-loaded operand values; `coords` are the index
/// space coordinates `IndexOf` reads. Comparisons return 1.0/0.0 in the
/// operand's own dtype, matching L0 semantics — L2's `Bool` exists only
/// between the compare and the select.
pub fn eval_scalar<E: ScalarEnv>(
    b: &mut TileBuilder,
    env: &mut E,
    expr: &ScalarExpr,
    args: &[TileExpr],
    coords: &[TileExpr],
) -> Result<TileExpr> {
    use fusor2_ir::scalar::ScalarKind as K;
    let relaxed = NumericContract::RELAXED;
    Ok(match expr.kind() {
        K::Arg(i) => args.get(*i as usize).cloned().ok_or_else(|| {
            Error::Plan(format!("body reads Arg({i}) with {} operands", args.len()))
        })?,
        K::Lit(l) => env.literal(b, l.0),
        K::Uniform(sym) => env.uniform(b, *sym, expr.dtype())?,
        K::IndexOf(axis) => {
            let c = coords.get(*axis as usize).cloned().ok_or_else(|| {
                Error::Plan(format!("body reads IndexOf({axis}) outside the index space"))
            })?;
            b.cast(c, ElementType::Scalar(ScalarElement::U32))
        }
        K::Un { op, x } => {
            let v = eval_scalar(b, env, x, args, coords)?;
            b.unary(*op, v, relaxed)
        }
        K::Bin { op, a, b: rhs } => {
            let l = eval_scalar(b, env, a, args, coords)?;
            let r = eval_scalar(b, env, rhs, args, coords)?;
            b.binary(*op, l, r, relaxed)
        }
        K::Cmp { op, a, b: rhs } => {
            let l = eval_scalar(b, env, a, args, coords)?;
            let r = eval_scalar(b, env, rhs, args, coords)?;
            let elem = l.element();
            let c = b.compare(*op, l, r);
            let one = one_of(b, elem);
            let zero = zero_of(b, elem);
            b.select(c, one, zero)
        }
        K::Select { c, t, f } => {
            let cv = eval_scalar(b, env, c, args, coords)?;
            let tv = eval_scalar(b, env, t, args, coords)?;
            let fv = eval_scalar(b, env, f, args, coords)?;
            let elem = cv.element();
            let zero = zero_of(b, elem);
            let nonzero = b.compare(TileCompareOp::Ne, cv, zero);
            b.select(nonzero, tv, fv)
        }
        K::Cast { to, x } => {
            let v = eval_scalar(b, env, x, args, coords)?;
            b.cast(v, ElementType::Scalar(scalar_element(*to)))
        }
        K::Bitcast { to, x } => {
            let v = eval_scalar(b, env, x, args, coords)?;
            b.bitcast(v, ElementType::Scalar(scalar_element(*to)))
        }
        // A rounding mode is a real primitive, so the trainer's
        // 14-chained-comparison `round_small` deletes. `Round` is its own L2
        // node rather than the `(x + 2^23) - 2^23` trick, so there is no
        // arithmetic identity for Metal's default fast math to fold away and
        // QAT cannot be silently disabled.
        K::Round { mode, x } => {
            let v = eval_scalar(b, env, x, args, coords)?;
            b.round(*mode, v)
        }
        K::Dot { a, b: rhs } => {
            let l = eval_scalar(b, env, a, args, coords)?;
            let r = eval_scalar(b, env, rhs, args, coords)?;
            b.dot(l, r)
        }
        K::Splat { lanes, x } => {
            let v = eval_scalar(b, env, x, args, coords)?;
            let scalar = match v.element() {
                ElementType::Scalar(s) => s,
                ElementType::Vector { scalar, .. } => scalar,
                ElementType::CoopMatrix { scalar, .. } => scalar,
            };
            b.vec(scalar, vec![v; *lanes as usize])
        }
    })
}

/// `flat` run through one operand's [`AddressMap`]. The caller has already
/// stated its own error for an operand with no decidable map.
pub fn map_address(
    b: &mut TileBuilder,
    map: &AddressMap,
    flat: TileExpr,
    space_total: u64,
) -> TileExpr {
    if map.is_identity_over(space_total) {
        return flat;
    }
    let mut acc: Option<TileExpr> = (map.offset != 0).then(|| b.lit_u32(map.offset));
    for (i, t) in map.terms.iter().enumerate() {
        let mut e = flat.clone();
        if t.divisor > 1 {
            let d = b.lit_u32(t.divisor);
            e = b.binary(TileBinaryOp::Div, e, d, NumericContract::RELAXED);
        }
        if map.needs_modulo(i, space_total) {
            let m = b.lit_u32(t.modulus);
            e = b.binary(TileBinaryOp::Rem, e, m, NumericContract::RELAXED);
        }
        if t.stride != 1 {
            let s = b.lit_u32(t.stride);
            e = b.mul(e, s);
        }
        acc = Some(match acc {
            Some(a) => b.add(a, e),
            None => e,
        });
    }
    acc.unwrap_or_else(|| b.lit_u32(0))
}

/// The storage layout a quantized value carries, read off its `LeafKind`.
/// Layout is a priced operand attribute, never a device branch, so it is
/// recovered from the leaf rather than assumed.
pub fn qlayout_of(cx: &LowerCtx<'_>, value: Id) -> Option<QLayout> {
    let class = cx.graph.class_of(value);
    cx.graph
        .class_ids(class)
        .into_iter()
        .find_map(|m| match &cx.graph.node(m).op {
            fusor2_ir::ir::Op::L0(fusor2_ir::ir::level0::L0::Leaf(
                fusor2_ir::ir::level0::LeafKind::Quantized { layout, .. },
            )) => Some(*layout),
            _ => None,
        })
}

/// The literal a `Leaf::Const` operand folds to, if it is one. A `Const` is
/// folded into the kernel — no buffer, no binding, no traffic — which is
/// exactly what `LeafRole::Free` means in the plan, so `derive_bindings`
/// never emits a binding for one and loading it would look up a key that
/// deliberately does not exist.
pub fn const_operand(b: &mut TileBuilder, cx: &LowerCtx<'_>, src: Id) -> Option<TileExpr> {
    use fusor2_ir::ir::level2::TileLiteral;
    let fusor2_ir::ir::Op::L0(fusor2_ir::ir::level0::L0::Leaf(
        fusor2_ir::ir::level0::LeafKind::Const { value, .. },
    )) = &cx.graph.node(cx.selected(src)).op
    else {
        return None;
    };
    Some(match *value {
        Splat::F32(v) => b.lit_f32(v),
        Splat::F16(v) => b.lit(TileLiteral::F16(v)),
        Splat::BF16(v) => b.lit(TileLiteral::BF16(v)),
        Splat::U32(v) => b.lit_u32(v),
        Splat::I32(v) => b.lit_i32(v),
    })
}
