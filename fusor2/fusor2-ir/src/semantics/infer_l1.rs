//! Total inference for the L1 op family. An L1 node's result shape is its
//! index space minus the reduced axes, and its dtype is the epilogue's.

use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::OpDefRegistry;
use crate::ir::level1::L1;
use crate::shape::{Dim, Dims};

/// Infer the result facts of an L1 node from its operands' facts.
///
/// `L1::Ext` is the one variant this cannot answer alone — its row lives in
/// the open [`OpDefRegistry`], which only [`crate::CoreSemantics`] holds. Use
/// [`infer_l1_with`] when you have the registry; this function reports a
/// typed error rather than guessing.
pub fn infer_l1(op: &L1, ins: &[ValueFacts]) -> Result<ValueFacts> {
    infer_l1_inner(op, ins, None)
}

/// [`infer_l1`] with the extension registry, so `L1::Ext` resolves.
pub fn infer_l1_with(op: &L1, ins: &[ValueFacts], registry: &OpDefRegistry) -> Result<ValueFacts> {
    infer_l1_inner(op, ins, Some(registry))
}

fn infer_l1_inner(
    op: &L1,
    ins: &[ValueFacts],
    registry: Option<&OpDefRegistry>,
) -> Result<ValueFacts> {
    match op {
        L1::KMap { space, body, .. } => Ok(ValueFacts {
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
        L1::KFold {
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
                    "KFold axis {axis} out of range for a rank-{} index space",
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
        L1::KContract {
            m, n, batch, post, ..
        } => Ok(contract_facts(*m, *n, *batch, post, ins)),
        // `QuantizedRows` reads the quantized leaf but *decodes* every
        // element it gathers, so its value is float-typed and step-lived —
        // inheriting the leaf's `Q(fmt)` dtype is exactly the double-decode
        // this mode exists to avoid, and inheriting the leaf's persistence
        // would cache a value that changes with every step's indices.
        L1::KGather {
            space,
            mode: crate::ir::level1::GatherMode::QuantizedRows,
            ..
        } => Ok(ValueFacts {
            dtype: Dtype::F32,
            shape: space.dims.clone(),
            numeric: meet(ins),
            persistence: Persistence::Step,
            outs: 1,
        }),
        L1::KGather { space, .. } => Ok(ValueFacts {
            dtype: ins.first().map_or(Dtype::F32, |f| f.dtype),
            shape: space.dims.clone(),
            numeric: meet(ins),
            persistence: ins.first().map_or(Persistence::Step, |f| f.persistence),
            outs: 1,
        }),

        // A scatter's value is its **base** with the updates applied, so its
        // shape comes from operand 0 — never from `space`. The two disagree:
        // `fusor2_tile::rules::scatter` mints the *update* iteration domain
        // (`[index_count, ...]`), so reading the shape off `space` sized a
        // 1024-row table's buffer at the 300 tokens that wrote into it, and
        // every element past the update count came back undefined. `infer_l0`
        // already says `Scatter` returns the base facts; this has to agree.
        L1::KScatter { space, .. } => match ins.first() {
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

        L1::KRegion {
            members, live_outs, ..
        } => {
            let first = *live_outs.first().ok_or_else(|| {
                Error::Shape("a KRegion must declare at least one live output".into())
            })? as usize;
            if first >= members.len() {
                return Err(Error::Shape(format!(
                    "KRegion live_out {first} names no member (there are {})",
                    members.len()
                )));
            }
            let facts = ins.get(first).ok_or_else(|| {
                Error::Shape(format!("KRegion member {first} has no inferred facts"))
            })?;
            let mut out = facts.clone();
            out.outs = live_outs.len() as u8;
            Ok(out)
        }

        L1::Ext { def, .. } => {
            let registry = registry.ok_or_else(|| {
                Error::Shape("L1::Ext inference needs the OpDefRegistry; call infer_l1_with".into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::Id;
    use crate::carrier::{ArgRemap, Carrier};
    use crate::scalar::BinOp;
    use crate::ir::level1::{
        AccessPlan, ContractSide, Family, IndexSpace, Operand, ScheduleDomain,
    };
    use crate::scalar::ScalarExpr;
    use crate::shape::Layout;
    use smallvec::smallvec;

    fn f32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|&d| Dim::Const(d)))
    }
    fn dims(v: &[u64]) -> Dims {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }
    fn operand(src: u32) -> Operand {
        Operand {
            src: Id(src),
            layout: Layout::contiguous(&[Dim::Const(1)]),
            access: AccessPlan::Alias,
        }
    }

    #[test]
    fn kmap_takes_the_index_space() {
        let op = L1::KMap {
            space: IndexSpace::new(dims(&[4, 8])),
            body: ScalarExpr::arg(0, Dtype::F16),
            ops: vec![operand(0)],
            sched: ScheduleDomain::Point,
        };
        let facts = infer_l1(&op, &[f32s(&[4, 8])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[4, 8])[..]);
        assert_eq!(facts.dtype, Dtype::F16);
    }

    fn binop(op: BinOp) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32).unwrap(), Dtype::F32)
    }

    #[test]
    fn kfold_drops_the_axis_and_appends_the_carrier() {
        let three = binop(BinOp::Add)
            .tuple(&binop(BinOp::Max), &ArgRemap::identity(1))
            .carrier
            .tuple(&binop(BinOp::Mul), &ArgRemap::identity(1))
            .carrier;
        let op = L1::KFold {
            space: IndexSpace::new(dims(&[4, 8, 16])),
            axis: 1,
            vec_axes: smallvec![],
            carrier: three.clone(),
            acc: Dtype::F32,
            post: (0..3).map(|i| ScalarExpr::arg(i, Dtype::F32)).collect(),
            ops: vec![operand(0)],
            sched: ScheduleDomain::Point,
        };
        let facts = infer_l1(&op, &[f32s(&[4, 8, 16])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[4, 16, 3])[..]);

        let bad_axis = L1::KFold {
            space: IndexSpace::new(dims(&[4])),
            axis: 7,
            vec_axes: smallvec![],
            carrier: binop(BinOp::Add),
            acc: Dtype::F32,
            post: smallvec![ScalarExpr::arg(0, Dtype::F32)],
            ops: vec![operand(0)],
            sched: ScheduleDomain::Point,
        };
        assert!(infer_l1(&bad_axis, &[f32s(&[4])]).is_err());
    }

    /// **Promotion does not change the node's facts.** A free axis moving from
    /// the iteration domain into the accumulator's data space leaves the output
    /// shape byte-identical, which is the check that catches a botched
    /// renumbering.
    #[test]
    fn promoting_an_axis_leaves_the_shape_identical() {
        let plain = L1::KFold {
            space: IndexSpace::new(dims(&[4, 8, 16])),
            axis: 2,
            vec_axes: smallvec![],
            carrier: binop(BinOp::Add),
            acc: Dtype::F32,
            post: smallvec![ScalarExpr::arg(0, Dtype::F32)],
            ops: vec![operand(0)],
            sched: ScheduleDomain::Point,
        };
        let promoted = L1::KFold {
            space: IndexSpace::new(dims(&[4, 8, 16])),
            axis: 2,
            vec_axes: smallvec![1],
            carrier: binop(BinOp::Add).promote(Dim::Const(8)).unwrap(),
            acc: Dtype::F32,
            post: smallvec![ScalarExpr::arg(0, Dtype::F32)],
            ops: vec![operand(0)],
            sched: ScheduleDomain::Point,
        };
        let a = infer_l1(&plain, &[f32s(&[4, 8, 16])]).unwrap();
        let b = infer_l1(&promoted, &[f32s(&[4, 8, 16])]).unwrap();
        assert_eq!(&a.shape[..], &dims(&[4, 8])[..]);
        assert_eq!(a, b, "PROMOTE must not change ValueFacts");
        assert_eq!(&promoted.iter_space().dims[..], &dims(&[4, 16])[..]);
    }

    #[test]
    fn kcontract_drops_a_unit_batch() {
        let make = |batch: Dim| L1::KContract {
            m: Dim::Const(3),
            n: Dim::Const(5),
            k: Dim::Const(4),
            batch,
            family: Family::Sgemm,
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(0)),
            b: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(1)),
            sched: ScheduleDomain::Point,
        };
        let ins = [f32s(&[3, 4]), f32s(&[4, 5])];
        assert_eq!(
            &infer_l1(&make(Dim::Const(1)), &ins).unwrap().shape[..],
            &dims(&[3, 5])[..]
        );
        assert_eq!(
            &infer_l1(&make(Dim::Const(2)), &ins).unwrap().shape[..],
            &dims(&[2, 3, 5])[..]
        );
    }

    #[test]
    fn kregion_reports_its_arity() {
        let region = L1::KRegion {
            members: smallvec![Id(1), Id(2), Id(3)],
            live_outs: smallvec![1, 2],
            sched: ScheduleDomain::Point,
        };
        let facts = infer_l1(&region, &[f32s(&[1]), f32s(&[7]), f32s(&[9])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[7])[..]);
        assert_eq!(facts.outs, 2);

        // An out-of-range live_out is an error, not an index panic.
        let bad = L1::KRegion {
            members: smallvec![Id(1)],
            live_outs: smallvec![4],
            sched: ScheduleDomain::Point,
        };
        assert!(infer_l1(&bad, &[f32s(&[1])]).is_err());
    }

    #[test]
    fn ext_without_a_registry_is_an_error_not_a_panic() {
        let op = L1::Ext {
            def: crate::ir::OpDefId(0),
            ops: vec![],
            attrs: crate::ir::AttrId(0),
        };
        assert!(infer_l1(&op, &[]).is_err());
        assert!(infer_l1_with(&op, &[], &OpDefRegistry::new()).is_err());
    }

}
