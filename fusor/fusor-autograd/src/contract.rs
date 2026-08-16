//! The one analytic non-elementwise adjoint:
//! `d(Contract) = (grad x b -> a, a x grad -> b)`, expressed by reusing the
//! primal spec's [`fusor_ir::ir::logical::EinSpec::d_lhs`] and `d_rhs`. It
//! holds regardless of tile geometry, stated at Logical and not restated per lowering.
//!
//! Because transposed-rhs is a *spec* and not an op, this rule subsumes
//! `mat_mul`, `mat_mul_transposed_rhs`, every batched form and — through the
//! macro `defn` expansions — `conv`/`grouped_conv`'s `dInput`, `dWeight` and
//! `dBias`.

use fusor_ir::autograd::{Grads, Tape, Val};
use fusor_ir::ir::Node;
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{EinSpec, Label, Logical};
use fusor_ir::{Error, Result};
use smallvec::SmallVec;

pub(crate) fn contract_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Contract {
        spec, acc, outs, ..
    }) = &node.op
    else {
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

    // A block-quantized operand gets `None`: an adjoint contraction for it
    // would produce a dense f32 tensor over the weight's element grid, and
    // nothing can apply that to a block-quantized buffer.
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
/// The adjoint must reject an inconsistent derived partition *before* adding
/// the node; the tape cannot depend on a verifier that may not have run.
pub(crate) fn verify_spec(spec: &EinSpec) -> Result<()> {
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
