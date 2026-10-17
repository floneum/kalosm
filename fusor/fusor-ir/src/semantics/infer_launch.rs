//! Total inference for the Launch op family. A Launch node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::{Dtype, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::launch::Launch;

/// Infer the result facts of a Launch node from its operands' facts.
pub fn infer_launch(op: &Launch, ins: &[ValueFacts]) -> Result<ValueFacts> {
    match op {
        Launch::DecodeRope {
            output,
            head_dim,
            ops,
            ..
        } => {
            let width = u64::from(*head_dim);
            let dims = &output.dims;
            if width == 0
                || width > 256
                || !width.is_multiple_of(2)
                || dims.len() != 4
                || dims[2] != crate::shape::Dim::ONE
                || dims[3].as_const() != Some(width)
                || output
                    .iterations()
                    .is_none_or(|n| n == 0 || n > u64::from(u32::MAX))
                || ops.len() != 4
                || ins.len() != 4
                || ins[..3].iter().any(|f| f.dtype != Dtype::F32)
                || ins[3].dtype != Dtype::U32
                || ops[0].layout.shape() != dims.as_slice()
                || ops[1].layout.shape().len() != 2
                || ops[1].layout.shape() != ops[2].layout.shape()
                || ops[1].layout.shape()[0]
                    .as_const()
                    .is_none_or(|n| n == 0 || n > u64::from(u32::MAX) / (width / 2))
                || ops[1].layout.shape()[1].as_const() != Some(width / 2)
                || ops[3].layout.shape() != [crate::shape::Dim::ONE]
            {
                return Err(Error::Shape(
                    "unsupported decode rotary embedding geometry".into(),
                ));
            }
            Ok(ValueFacts::step(Dtype::F32, dims.clone(), ins))
        }
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
                || *width > 8192
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
            heads_per_group,
            chunk,
            ops,
            scale,
            prefix,
            ..
        } => {
            let dims = output
                .dims
                .iter()
                .map(|d| d.as_const())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Shape("decode attention needs fixed extents".into()))?;
            let mut scalar_args = Vec::new();
            scale.collect_args(&mut scalar_args);
            prefix.collect_args(&mut scalar_args);
            let (splits, width) = if *chunk == 0 {
                (1, 128)
            } else {
                (u64::from(keys.div_ceil(*chunk)), 130)
            };
            if dims.len() != 4
                || dims[0] == 0
                || dims[1] == 0
                || dims[2] != splits
                || dims[3] != width
                || *kv_heads == 0
                || !matches!(*heads_per_group, 1 | 2 | 4)
                || !dims[1].is_multiple_of(u64::from(*kv_heads))
                || !(dims[1] / u64::from(*kv_heads)).is_multiple_of(u64::from(*heads_per_group))
                || *keys == 0
                || !matches!(*block, 32 | 64 | 128 | 256 | 512)
                || ins.len() != 3
                || ops.len() != 3
                || ins.iter().any(|f| f.dtype != Dtype::F32)
                || scale.dtype() != Dtype::F32
                || prefix.dtype() != Dtype::F32
                || !scalar_args.is_empty()
                || scale.reads_index_of()
                || prefix.reads_index_of()
                || output.iterations().is_none_or(|n| n > u64::from(u32::MAX))
            {
                return Err(Error::Shape(
                    "unsupported decode attention geometry or dtype".into(),
                ));
            }
            let q = [dims[0], dims[1], 1, 128];
            let kv = [dims[0], u64::from(*kv_heads), u64::from(*keys), 128];
            if kv
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d))
                .is_none_or(|n| n > u64::from(u32::MAX))
            {
                return Err(Error::Shape(
                    "decode attention exceeds u32 addressing".into(),
                ));
            }
            for (i, expected) in [q.as_slice(), kv.as_slice(), kv.as_slice()]
                .into_iter()
                .enumerate()
            {
                if ops[i]
                    .layout
                    .shape()
                    .iter()
                    .map(|d| d.as_const())
                    .collect::<Option<Vec<_>>>()
                    .as_deref()
                    != Some(expected)
                {
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
            let d = output
                .dims
                .iter()
                .map(|d| d.as_const())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Shape("attention merge needs fixed extents".into()))?;
            if d.len() != 4
                || d[0] == 0
                || d[1] == 0
                || d[2] != 1
                || d[3] != 128
                || *splits == 0
                || ops.len() != 1
                || ins.len() != 1
                || ins[0].dtype != Dtype::F32
                || d[0]
                    .saturating_mul(d[1])
                    .saturating_mul(u64::from(*splits))
                    .saturating_mul(130)
                    > u64::from(u32::MAX)
                || ops[0]
                    .layout
                    .shape()
                    .iter()
                    .map(|d| d.as_const())
                    .collect::<Option<Vec<_>>>()
                    != Some(vec![d[0], d[1], u64::from(*splits), 130])
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
