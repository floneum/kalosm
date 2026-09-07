//! Total inference for the Launch op family. A Launch node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::OpDefRegistry;
use crate::ir::launch::Launch;
use crate::shape::{Dim, Dims};

/// Infer the result facts of a Launch node from its operands' facts.
///
/// `Launch::Ext` is the one variant this cannot answer alone — its row lives in
/// the open [`OpDefRegistry`], which only [`crate::CoreSemantics`] holds. Use
/// [`infer_launch_with`] when you have the registry; this function reports a
/// typed error rather than guessing.
pub fn infer_launch(op: &Launch, ins: &[ValueFacts]) -> Result<ValueFacts> {
    infer_launch_inner(op, ins, None)
}

/// [`infer_launch`] with the extension registry, so `Launch::Ext` resolves.
pub fn infer_launch_with(
    op: &Launch,
    ins: &[ValueFacts],
    registry: &OpDefRegistry,
) -> Result<ValueFacts> {
    infer_launch_inner(op, ins, Some(registry))
}

fn infer_launch_inner(
    op: &Launch,
    ins: &[ValueFacts],
    registry: Option<&OpDefRegistry>,
) -> Result<ValueFacts> {
    match op {
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

        // A quantized contraction has no batch axis: its weight side is a
        // single `[n, k]` matrix.
        Launch::Contract {
            m, n, batch, post, ..
        } => Ok(contract_facts(*m, *n, *batch, post, ins)),
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

        Launch::Region {
            members, live_outs, ..
        } => {
            let first = *live_outs.first().ok_or_else(|| {
                Error::Shape("a Region must declare at least one live output".into())
            })? as usize;
            if first >= members.len() {
                return Err(Error::Shape(format!(
                    "Region live_out {first} names no member (there are {})",
                    members.len()
                )));
            }
            let facts = ins.get(first).ok_or_else(|| {
                Error::Shape(format!("Region member {first} has no inferred facts"))
            })?;
            let mut out = facts.clone();
            out.outs = live_outs.len() as u8;
            Ok(out)
        }

        Launch::Ext { def, .. } => {
            let registry = registry.ok_or_else(|| {
                Error::Shape(
                    "Launch::Ext inference needs the OpDefRegistry; call infer_launch_with".into(),
                )
            })?;
            let d = registry
                .get(*def)
                .ok_or_else(|| Error::Shape(format!("no OpDef registered as {def:?}")))?;
            (d.infer)(ins)
        }
    }
}

/// `[batch, m, n]`, dropping a unit batch so a plain `[m, n]` matmul does not
/// grow a leading axis nothing reads.
fn contract_facts(
    m: Dim,
    n: Dim,
    batch: Dim,
    post: &crate::scalar::ScalarExpr,
    ins: &[ValueFacts],
) -> ValueFacts {
    let mut shape: Dims = Dims::new();
    if !batch.known_eq(Dim::Const(1)) {
        shape.push(batch);
    }
    shape.push(m);
    shape.push(n);
    ValueFacts {
        dtype: post.dtype(),
        shape,
        numeric: meet(ins),
        persistence: Persistence::Step,
        outs: 1,
    }
}

fn meet(ins: &[ValueFacts]) -> NumericContract {
    ins.iter()
        .map(|f| f.numeric)
        .reduce(NumericContract::meet)
        .unwrap_or(NumericContract::RELAXED)
}
