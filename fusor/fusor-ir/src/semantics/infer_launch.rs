//! Total inference for the Launch op family. A Launch node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::{Dtype, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::launch::Launch;

/// Infer the result facts of a Launch node from its operands' facts.
pub fn infer_launch(op: &Launch, ins: &[ValueFacts]) -> Result<ValueFacts> {
    match op {
        Launch::StreamFold {
            producer,
            fold,
            operand,
            ..
        } => {
            if !op.stream_compatible() {
                return Err(Error::Legality(
                    "streamed Fold recipes or generated read are incompatible".into(),
                ));
            }
            let (_, _, inputs) = stream_inputs(producer, fold, *operand, ins)?;
            infer_launch(fold, &inputs)
        }
        Launch::Map { space, body, .. } => {
            Ok(ValueFacts::step(body.dtype(), space.dims.clone(), ins))
        }

        // Promoted axes stay in the output shape as carrier lanes, so
        // `PROMOTE` never changes a node's `ValueFacts`.
        Launch::Fold {
            space,
            axis,
            acc,
            carrier,
            vec_axes,
            ..
        } => {
            if *axis as usize >= space.rank() {
                return Err(Error::Shape(format!(
                    "Fold axis {axis} out of range for a rank-{} index space",
                    space.rank()
                )));
            }
            let shape = space.fold_shape(*axis, vec_axes, carrier).ok_or_else(|| {
                Error::Shape("a multi-slot carrier needs a constant Vector extent".into())
            })?;
            Ok(ValueFacts::step(*acc, shape, ins))
        }

        Launch::Contract { output, post, .. } => {
            Ok(ValueFacts::step(post.dtype(), output.dims.clone(), ins))
        }
        // `QuantizedRows` decodes what it gathers: float-typed and step-lived,
        // never the leaf's `Q(fmt)` dtype or persistence.
        Launch::Gather {
            space,
            mode: crate::ir::launch::GatherMode::QuantizedRows,
            ..
        } => Ok(ValueFacts::step(Dtype::F32, space.dims.clone(), ins)),
        Launch::Gather { space, .. } => Ok(ValueFacts {
            dtype: ins.first().map_or(Dtype::F32, |f| f.dtype),
            persistence: ins.first().map_or(Persistence::Step, |f| f.persistence),
            ..ValueFacts::step(Dtype::F32, space.dims.clone(), ins)
        }),

        // A scatter's value is its base, so its shape comes from operand 0,
        // never from `space` (which may be the update domain).
        Launch::Scatter { space, .. } => match ins.first() {
            Some(base) => Ok(ValueFacts {
                numeric: ValueFacts::meet(ins),
                ..base.view(base.shape.clone())
            }),
            None => Ok(ValueFacts::step(Dtype::F32, space.dims.clone(), ins)),
        },

        // A composite is its last member's value, the last of `ins`.
        Launch::Slab { members, .. } | Launch::Group { members, .. } => {
            if members.len() < 2 {
                return Err(Error::Shape("a Slab needs at least two members".into()));
            }
            let facts = ins
                .last()
                .ok_or_else(|| Error::Shape("a Slab's last member has no inferred facts".into()))?;
            let mut out = facts.clone();
            out.outs = 1;
            Ok(out)
        }
    }
}

/// A streamed fold's operand facts split: the producer's operand count, its
/// inferred value, and the fold's operand facts with that value spliced in.
pub fn stream_inputs(
    producer: &Launch,
    fold: &Launch,
    operand: u32,
    ins: &[ValueFacts],
) -> Result<(usize, ValueFacts, Vec<ValueFacts>)> {
    let count = super::children::children_launch(producer).len();
    if count > ins.len()
        || operand as usize > ins.len() - count
        || count + super::children::children_launch(fold).len() != ins.len() + 1
    {
        return Err(Error::Shape(
            "streamed Fold operand facts are incomplete".into(),
        ));
    }
    let produced = infer_launch(producer, &ins[..count])?;
    let mut inputs = ins[count..].to_vec();
    inputs.insert(operand as usize, produced.clone());
    Ok((count, produced, inputs))
}
