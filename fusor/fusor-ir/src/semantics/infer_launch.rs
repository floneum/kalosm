//! Total inference for the Launch op family. A Launch node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::launch::Launch;
use crate::shape::Dims;

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
            let count = super::children::children_launch(producer).len();
            if count > ins.len()
                || *operand as usize > ins.len() - count
                || count + super::children::children_launch(fold).len() != ins.len() + 1
            {
                return Err(Error::Shape(
                    "streamed Fold operand facts are incomplete".into(),
                ));
            }
            let produced = infer_launch(producer, &ins[..count])?;
            let mut inputs = ins[count..].to_vec();
            inputs.insert(*operand as usize, produced);
            infer_launch(fold, &inputs)
        }
        Launch::Map { space, body, .. } => Ok(ValueFacts {
            dtype: body.dtype(),
            shape: space.dims.clone(),
            numeric: meet(ins),
            persistence: Persistence::Step,
            outs: 1,
        }),

        // The reduced axis leaves the shape and the carrier's lane count is
        // appended when it exceeds one — the convention slot readback is an
        // ordinary `Restride` of. Promoted axes leave the *iteration* domain
        // but stay in the output shape as carrier lanes, which is exactly why
        // `PROMOTE` does not change a node's `ValueFacts` at all.
        Launch::Fold {
            space,
            axis,
            acc,
            carrier,
            vec_axes,
            ..
        } => {
            let axis = *axis as usize;
            if axis >= space.rank() {
                return Err(Error::Shape(format!(
                    "Fold axis {axis} out of range for a rank-{} index space",
                    space.rank()
                )));
            }
            let mut shape: Dims = space
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis && !vec_axes.contains(&(*i as u32)))
                .map(|(_, d)| *d)
                .collect();
            if let Some(d) = carrier.out_dim().ok_or_else(|| {
                Error::Shape("a multi-slot carrier needs a constant Vector extent".into())
            })? {
                shape.push(d);
            }
            Ok(ValueFacts {
                dtype: *acc,
                shape,
                numeric: meet(ins),
                persistence: Persistence::Step,
                outs: 1,
            })
        }

        Launch::Contract { output, post, .. } => Ok(ValueFacts {
            dtype: post.dtype(),
            shape: output.dims.clone(),
            numeric: meet(ins),
            persistence: Persistence::Step,
            outs: 1,
        }),
        // `QuantizedRows` reads the quantized leaf but *decodes* every
        // element it gathers, so its value is float-typed and step-lived —
        // inheriting the leaf's `Q(fmt)` dtype is exactly the double-decode
        // this mode exists to avoid, and inheriting the leaf's persistence
        // would cache a value that changes with every step's indices.
        Launch::Gather {
            space,
            mode: crate::ir::launch::GatherMode::QuantizedRows,
            ..
        } => Ok(ValueFacts {
            dtype: Dtype::F32,
            shape: space.dims.clone(),
            numeric: meet(ins),
            persistence: Persistence::Step,
            outs: 1,
        }),
        Launch::Gather { space, .. } => Ok(ValueFacts {
            dtype: ins.first().map_or(Dtype::F32, |f| f.dtype),
            shape: space.dims.clone(),
            numeric: meet(ins),
            persistence: ins.first().map_or(Persistence::Step, |f| f.persistence),
            outs: 1,
        }),

        // A scatter's value is its **base** with the updates applied, so its
        // shape comes from operand 0 — never from `space`. The two disagree:
        // `fusor_tile::rules::scatter` mints the *update* iteration domain
        // (`[index_count, ...]`), so reading the shape off `space` sized a
        // 1024-row table's buffer at the 300 tokens that wrote into it, and
        // every element past the update count came back undefined. `infer_logical`
        // already says `Scatter` returns the base facts; this has to agree.
        Launch::Scatter { space, .. } => match ins.first() {
            Some(base) => Ok(ValueFacts {
                dtype: base.dtype,
                shape: base.shape.clone(),
                numeric: meet(ins),
                persistence: base.persistence,
                outs: 1,
            }),
            None => Ok(ValueFacts {
                dtype: Dtype::F32,
                shape: space.dims.clone(),
                numeric: meet(ins),
                persistence: Persistence::Step,
                outs: 1,
            }),
        },

        // A slab is its last member's value; the children are the members in
        // order, so that is the last of `ins`.
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

fn meet(ins: &[ValueFacts]) -> NumericContract {
    ins.iter()
        .map(|f| f.numeric)
        .reduce(NumericContract::meet)
        .unwrap_or(NumericContract::RELAXED)
}
