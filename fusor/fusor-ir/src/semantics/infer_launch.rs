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
        Launch::DecodeAttention {
            output,
            kv_heads,
            keys,
            block,
            subgroup,
            chunk,
            ops,
            scale,
            prefix,
            ..
        } => {
            use crate::shape::Dim;
            let dims = &output.dims;
            let channels = ops
                .first()
                .and_then(|o| o.layout.shape().get(3))
                .and_then(|d| d.as_const())
                .unwrap_or(0);
            let mut scalar_args = Vec::new();
            scale.collect_args(&mut scalar_args);
            prefix.collect_args(&mut scalar_args);
            let (splits, width) = if *chunk == 0 {
                (Dim::ONE, Dim::Const(channels))
            } else {
                (keys.div_ceil(u64::from(*chunk)), Dim::Const(channels + 2))
            };
            let b = dims.first().and_then(|d| d.as_const()).unwrap_or(0);
            let h = dims.get(1).and_then(|d| d.as_const()).unwrap_or(0);
            if dims.len() != 4
                || channels == 0
                || channels > 256
                || b == 0
                || h == 0
                || dims[2] != splits
                || dims[3] != width
                || *kv_heads == 0
                || !subgroup.is_power_of_two()
                || !block.is_multiple_of((*subgroup).max(1))
                || !h.is_multiple_of(u64::from(*kv_heads))
                || keys.as_const() == Some(0)
                || (*chunk == 0 && keys.as_const().is_none())
                || !matches!(*block, 32 | 64 | 128 | 256 | 512)
                || ins.len() != 3
                || ops.len() != 3
                || ins.iter().any(|f| f.dtype != Dtype::F32)
                || scale.dtype() != Dtype::F32
                || !matches!(prefix.dtype(), Dtype::F32 | Dtype::U32)
                || !scalar_args.is_empty()
                || scale.reads_index_of()
                || prefix.reads_index_of()
                || output.iterations().is_some_and(|n| n > u64::from(u32::MAX))
                || keys.as_const().is_some_and(|keys| {
                    b.saturating_mul(u64::from(*kv_heads))
                        .saturating_mul(keys)
                        .saturating_mul(channels)
                        > u64::from(u32::MAX)
                })
            {
                return Err(Error::Shape(
                    "unsupported decode attention geometry or dtype".into(),
                ));
            }
            let q = [dims[0], dims[1], Dim::ONE, Dim::Const(channels)];
            let kv = [
                dims[0],
                Dim::Const(u64::from(*kv_heads)),
                *keys,
                Dim::Const(channels),
            ];
            for (i, expected) in [q.as_slice(), kv.as_slice(), kv.as_slice()]
                .into_iter()
                .enumerate()
            {
                if ops[i].layout.shape() != expected {
                    return Err(Error::Shape(
                        "decode attention operand layout differs from its geometry".into(),
                    ));
                }
            }
            Ok(ValueFacts::step(Dtype::F32, output.dims.clone(), ins))
        }
        Launch::DecodeAttentionMerge {
            output,
            splits,
            ops,
            ..
        } => {
            use crate::shape::Dim;
            let d = output
                .dims
                .iter()
                .map(|d| d.as_const())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Shape("attention merge needs fixed output extents".into()))?;
            if d.len() != 4
                || d[0] == 0
                || d[1] == 0
                || d[2] != 1
                || d[3] == 0
                || d[3] > 256
                || splits.as_const() == Some(0)
                || ops.len() != 1
                || ins.len() != 1
                || ins[0].dtype != Dtype::F32
                || splits.as_const().is_some_and(|s| {
                    d[0].saturating_mul(d[1])
                        .saturating_mul(s)
                        .saturating_mul(d[3] + 2)
                        > u64::from(u32::MAX)
                })
                || ops[0].layout.shape()
                    != [
                        Dim::Const(d[0]),
                        Dim::Const(d[1]),
                        *splits,
                        Dim::Const(d[3] + 2),
                    ]
            {
                return Err(Error::Shape("unsupported attention merge geometry".into()));
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
