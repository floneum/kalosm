//! Total inference for the Launch op family. A Launch node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::launch::Launch;

/// Infer the result facts of a Launch node from its operands' facts.
pub fn infer_launch(op: &Launch, ins: &[ValueFacts]) -> Result<ValueFacts> {
    match op {
        Launch::RmsNorm {
            output,
            width,
            block,
            epsilon,
            ops,
            ..
        } => {
            let mut args = Vec::new();
            epsilon.collect_args(&mut args);
            if *width == 0
                || width.div_ceil((*block).max(1))
                    > crate::ir::launch::MAX_NORMALIZATION_VALUES_PER_LANE
                || !matches!(*block, 32 | 64 | 128 | 256 | 512 | 1024)
                || output.dims.last().and_then(|d| d.as_const()) != Some(u64::from(*width))
                || output
                    .iterations()
                    .is_none_or(|n| n == 0 || n > u64::from(u32::MAX))
                || !matches!(ops.len(), 2 | 3)
                || ins.len() != ops.len()
                || ins.iter().any(|f| f.dtype != Dtype::F32)
                || epsilon.dtype() != Dtype::F32
                || !args.is_empty()
                || epsilon.reads_index_of()
                || ops[0].layout.shape() != output.dims.as_slice()
                || ops[1].layout.shape() != [crate::shape::Dim::Const(u64::from(*width))]
                || ops
                    .get(2)
                    .is_some_and(|r| r.layout.shape() != output.dims.as_slice())
            {
                return Err(Error::Shape(
                    "unsupported fused RMS normalization geometry".into(),
                ));
            }
            Ok(ValueFacts::step(Dtype::F32, output.dims.clone(), ins))
        }
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
        Launch::Gather {
            space,
            mode,
            post,
            ops,
            ..
        } => {
            use crate::ir::launch::GatherMode;
            use crate::scalar::ScalarKind;
            if ops.len() < 2 || ins.len() != ops.len() {
                return Err(Error::Shape(
                    "gather requires source and index operands".into(),
                ));
            }
            let gathered = match mode {
                GatherMode::QuantizedRows => Dtype::F32,
                GatherMode::RowPerGroup => ins[0].dtype,
            };
            let mut valid = true;
            post.walk(&mut |expr| match expr.kind() {
                ScalarKind::Arg(0) => valid &= expr.dtype() == gathered,
                ScalarKind::Arg(i) => {
                    valid &= (*i as usize)
                        .checked_add(1)
                        .and_then(|i| ins.get(i))
                        .is_some_and(|f| f.dtype == expr.dtype())
                }
                ScalarKind::IndexOf(axis) => valid &= (*axis as usize) < space.rank(),
                _ => {}
            });
            if !valid {
                return Err(Error::Dtype(
                    "gather epilogue reads an absent or incompatible value".into(),
                ));
            }
            let mut facts = ValueFacts::step(post.dtype(), space.dims.clone(), ins);
            if *mode == GatherMode::RowPerGroup
                && ops.len() == 2
                && matches!(post.kind(), ScalarKind::Arg(0))
            {
                facts.persistence = ins[0].persistence;
            }
            Ok(facts)
        }

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
