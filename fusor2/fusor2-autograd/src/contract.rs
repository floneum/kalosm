//! The one analytic non-elementwise adjoint:
//! `d(Contract) = (grad x b -> a, a x grad -> b)`, expressed by reusing the
//! primal spec's [`fusor2_ir::ir::level0::EinSpec::d_lhs`] and `d_rhs`. It
//! holds regardless of tile geometry, which is exactly why it is stated at L0
//! and not restated per lowering.
//!
//! Because transposed-rhs is a *spec* and not an op, this single rule
//! subsumes `mat_mul`, `mat_mul_transposed_rhs`, every batched form and —
//! through the macro `defn` expansions — `conv`/`grouped_conv`'s `dInput`,
//! `dWeight` and `dBias`. There is no `replay_*` combinator anywhere.
//!
//! Owned by W5.

use fusor2_ir::autograd::{Grads, Tape, Val};
use fusor2_ir::ir::Node;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level0::{EinSpec, L0, Label};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

pub fn contract_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::L0(L0::Contract { spec, acc, outs, .. }) = &node.op else {
        return Err(Error::Plan(format!(
            "contract_adjoint called on a non-Contract node: {:?}",
            node.op
        )));
    };
    if *outs != 1 {
        return Err(Error::Plan(
            "multi-output Contract has no per-slot adjoint".into(),
        ));
    }
    let (a, b) = match ins {
        [a, b] => (*a, *b),
        other => {
            return Err(Error::Plan(format!(
                "Contract takes two operands, got {}",
                other.len()
            )));
        }
    };

    // A block-quantized operand gets `None`, which is what [`Grads`] means by
    // "a parent that does not require grad". The route is not trainable: an
    // adjoint contraction for it would produce a dense f32 tensor over the
    // weight's element grid, and nothing can apply that to a block-quantized
    // buffer — which is precisely why QAT keeps a separate f32 master copy
    // rather than a quantized backward kernel.
    //
    // Stated here rather than left to fall out of a lowering failure. It used
    // to hold only because `L1::KQContract` could not lower the `d_rhs` spec;
    // the moment that contraction found any lowering at all — the generic
    // floor does — `q_mat_mul_backward_reaches_the_activation_only`'s "a
    // gradient was produced for a quantized weight" assert fired.
    let da = if tape.facts(a).dtype.is_quantized() {
        None
    } else {
        let d_lhs = spec.d_lhs();
        verify_spec(&d_lhs)?;
        Some(tape.contract(grad, b, d_lhs, *acc)?)
    };
    let db = if tape.facts(b).dtype.is_quantized() {
        None
    } else {
        let d_rhs = spec.d_rhs();
        verify_spec(&d_rhs)?;
        Some(tape.contract(a, grad, d_rhs, *acc)?)
    };
    Ok(smallvec::smallvec![da, db])
}

/// `verify_l0` rule 4, restated locally: every label appears in at least two
/// of `{a, b, out}`, and no operand repeats a label.
///
/// This duplicates `fusor2_ir::contract_spec::verify_spec` on purpose — the
/// adjoint must reject an inconsistent derived partition *before* adding the
/// node, and the tape cannot depend on a verifier that may not have run.
pub fn verify_spec(spec: &EinSpec) -> Result<()> {
    for (name, labels) in [("a", &spec.a), ("b", &spec.b), ("out", &spec.out)] {
        let mut seen: SmallVec<[Label; 8]> = SmallVec::new();
        for l in labels.iter() {
            if seen.contains(l) {
                return Err(Error::Shape(format!(
                    "contraction operand {name} repeats label {}",
                    l.0
                )));
            }
            seen.push(*l);
        }
    }
    let mut all: SmallVec<[Label; 12]> = SmallVec::new();
    for l in spec.a.iter().chain(&spec.b).chain(&spec.out) {
        if !all.contains(l) {
            all.push(*l);
        }
    }
    for l in all {
        let count = [&spec.a, &spec.b, &spec.out]
            .into_iter()
            .filter(|side| side.contains(&l))
            .count();
        if count < 2 {
            return Err(Error::Shape(format!(
                "contraction label {} appears in only one of {{a, b, out}}",
                l.0
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::GraphTape;
    use crate::tape::testing::graph;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level0::{BufferId, LeafKind};
    use fusor2_ir::shape::{Dim, Dims};

    fn labels(v: &[u8]) -> SmallVec<[Label; 6]> {
        v.iter().copied().map(Label).collect()
    }

    fn matmul_spec() -> EinSpec {
        // [m, k] x [k, n] -> [m, n]
        EinSpec {
            a: labels(&[0, 2]),
            b: labels(&[2, 1]),
            out: labels(&[0, 1]),
        }
    }

    fn param(g: &mut fusor2_ir::egraph::EGraph, shape: &[u64], name: u32) -> Val {
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(name),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    #[test]
    fn a_matmul_spec_and_both_of_its_adjoint_specs_verify() {
        let s = matmul_spec();
        verify_spec(&s).unwrap();
        verify_spec(&s.d_lhs()).unwrap();
        verify_spec(&s.d_rhs()).unwrap();
    }

    #[test]
    fn a_label_appearing_only_once_is_rejected() {
        let bad = EinSpec {
            a: labels(&[0, 2]),
            b: labels(&[2, 1]),
            out: labels(&[0, 1, 3]),
        };
        assert!(verify_spec(&bad).is_err());
    }

    #[test]
    fn adjoint_shapes_match_the_primal_operands() {
        let mut g = graph();
        let a = param(&mut g, &[4, 3], 0);
        let b = param(&mut g, &[3, 5], 1);
        let spec = matmul_spec();
        let y = g
            .add(Op::L0(L0::Contract {
                spec: spec.clone(),
                acc: Dtype::F32,
                a,
                b,
                outs: 1,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[4, 5], 2);
        let mut tape = GraphTape::new(&mut g);
        let grads = contract_adjoint(&mut tape, &node, grad, &[a, b], y).unwrap();
        let da = grads[0].unwrap();
        let db = grads[1].unwrap();
        assert_eq!(
            tape.facts(da).shape,
            Dims::from_slice(&[Dim::Const(4), Dim::Const(3)])
        );
        assert_eq!(
            tape.facts(db).shape,
            Dims::from_slice(&[Dim::Const(3), Dim::Const(5)])
        );
    }

    /// A block-quantized weight receives `None`, and the activation beside it
    /// still receives its gradient.
    ///
    /// The invariant is stated by the adjoint, not inherited from a lowering
    /// that happens to fail: any lowering of `a x grad -> b` computes a dense
    /// f32 tensor over the weight's element grid, which no optimizer can apply
    /// to a block-quantized buffer.
    #[test]
    fn a_quantized_operand_gets_no_gradient() {
        use fusor2_ir::dtype::{QFmt, QLayout};

        // `matmul_t`: [m, k] x [n, k] -> [m, n].
        let spec = EinSpec {
            a: labels(&[0, 2]),
            b: labels(&[1, 2]),
            out: labels(&[0, 1]),
        };
        let mut g = graph();
        let a = param(&mut g, &[4, 32], 0);
        let w = g
            .add(Op::L0(L0::Leaf(LeafKind::Quantized {
                name: BufferId(1),
                fmt: QFmt::Q8_0,
                layout: QLayout::Native,
                shape: [Dim::Const(5), Dim::Const(32)].into_iter().collect(),
            })))
            .unwrap();
        let y = g
            .add(Op::L0(L0::Contract {
                spec,
                acc: Dtype::F32,
                a,
                b: w,
                outs: 1,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[4, 5], 2);
        let mut tape = GraphTape::new(&mut g);
        let grads = contract_adjoint(&mut tape, &node, grad, &[a, w], y).unwrap();
        let da = grads[0].expect("the activation is still differentiable");
        assert_eq!(
            tape.facts(da).shape,
            Dims::from_slice(&[Dim::Const(4), Dim::Const(32)])
        );
        assert!(
            grads[1].is_none(),
            "a quantized weight is not trainable through this route"
        );
    }

    #[test]
    fn a_transposed_rhs_spec_lands_db_in_the_rhs_layout_with_no_extra_restride() {
        // [m, k] x [n, k] -> [m, n]: `mat_mul_transposed_rhs` is a spec.
        let spec = EinSpec {
            a: labels(&[0, 2]),
            b: labels(&[1, 2]),
            out: labels(&[0, 1]),
        };
        let mut g = graph();
        let a = param(&mut g, &[4, 3], 0);
        let b = param(&mut g, &[5, 3], 1);
        let y = g
            .add(Op::L0(L0::Contract {
                spec: spec.clone(),
                acc: Dtype::F32,
                a,
                b,
                outs: 1,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[4, 5], 2);
        let mut tape = GraphTape::new(&mut g);
        let grads = contract_adjoint(&mut tape, &node, grad, &[a, b], y).unwrap();
        let db = grads[1].unwrap();
        assert_eq!(
            tape.facts(db).shape,
            Dims::from_slice(&[Dim::Const(5), Dim::Const(3)]),
            "d_rhs comes out in the rhs's own layout"
        );
        assert!(
            matches!(tape.node(db).op, Op::L0(L0::Contract { .. })),
            "no extra Restride is inserted"
        );
    }
}

#[cfg(test)]
mod numeric {
    //! Rank-2/3/4 contraction adjoints against central differences.

    use super::*;
    use crate::backward::backward_into;
    use crate::tape::testing::{Env, caps, check_gradients, graph};
    use fusor2_ir::dtype::{Dtype, Splat};
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::ir::level0::{BufferId, LeafKind};
    use fusor2_ir::shape::Dim;
    use rustc_hash::FxHashMap;

    fn param(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn ones(g: &mut EGraph, shape: &[u64]) -> Id {
        g.add(Op::L0(L0::Leaf(LeafKind::Const {
            value: Splat::F32(1.0),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37 + seed).sin()).collect()
    }

    fn check(a_shape: &[u64], b_shape: &[u64], spec: EinSpec, out_shape: &[u64]) {
        let mut g = graph();
        let a = param(&mut g, a_shape);
        let b = param(&mut g, b_shape);
        let y = g
            .add(Op::L0(L0::Contract {
                spec,
                acc: Dtype::F32,
                a,
                b,
                outs: 1,
            }))
            .unwrap();
        assert_eq!(
            g.facts(y).shape.len(),
            out_shape.len(),
            "inferred rank disagrees with the expectation"
        );
        let seed = ones(&mut g, out_shape);
        let grads = backward_into(&mut g, &caps(), y, seed, &[a, b]).unwrap();
        let mut env: Env = FxHashMap::default();
        env.insert(a, ramp(a_shape.iter().product::<u64>() as usize, 0.1));
        env.insert(b, ramp(b_shape.iter().product::<u64>() as usize, 1.7));
        check_gradients(&g, y, &[a, b], &grads, &env, 3e-3);
    }

    fn labels(v: &[u8]) -> SmallVec<[Label; 6]> {
        v.iter().copied().map(Label).collect()
    }

    #[test]
    fn rank_2_matmul() {
        check(
            &[3, 4],
            &[4, 2],
            EinSpec {
                a: labels(&[0, 2]),
                b: labels(&[2, 1]),
                out: labels(&[0, 1]),
            },
            &[3, 2],
        );
    }

    #[test]
    fn rank_2_transposed_rhs() {
        check(
            &[3, 4],
            &[2, 4],
            EinSpec {
                a: labels(&[0, 2]),
                b: labels(&[1, 2]),
                out: labels(&[0, 1]),
            },
            &[3, 2],
        );
    }

    #[test]
    fn rank_3_batched() {
        check(
            &[2, 3, 4],
            &[2, 4, 5],
            EinSpec {
                a: labels(&[0, 1, 3]),
                b: labels(&[0, 3, 2]),
                out: labels(&[0, 1, 2]),
            },
            &[2, 3, 5],
        );
    }

    #[test]
    fn rank_4_two_batch_axes() {
        check(
            &[2, 2, 3, 4],
            &[2, 2, 4, 2],
            EinSpec {
                a: labels(&[0, 1, 2, 4]),
                b: labels(&[0, 1, 4, 3]),
                out: labels(&[0, 1, 2, 3]),
            },
            &[2, 2, 3, 2],
        );
    }
}
