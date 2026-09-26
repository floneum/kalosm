//! Reshape long contractions into independent partial sums before scheduling.
//! This pass changes logical work, not the physical layouts of kernel operands.
use super::plan::{BLOCK, Value};
use fusor_ir::{
    Result,
    carrier::Carrier,
    dtype::Dtype,
    egraph::Id,
    error::Error,
    ir::logical::{Label, Logical},
    scalar::BinOp,
    shape::{BoundsProof, Dim, StrideSpec},
};
use rustc_hash::FxHashMap;

pub(super) fn split_contractions(
    values: &mut Vec<Value>,
    map: &mut FxHashMap<Id, usize>,
    max_bytes: u64,
    first_unused_id: u32,
) -> Result<()> {
    let old = std::mem::take(values);
    let old_map = map.clone();
    // Keep compiler-generated IDs outside the entire source graph, including
    // nodes that are not reachable from this program's roots.
    let mut next = map
        .keys()
        .map(|id| id.0)
        .max()
        .unwrap_or(0)
        .max(first_unused_id.saturating_sub(1));
    let mut fresh = || -> Result<Id> {
        next = next
            .checked_add(1)
            .ok_or_else(|| Error::Plan("program workload id overflow".into()))?;
        Ok(Id(next))
    };
    for mut value in old.iter().cloned() {
        if let Logical::Contract {
            mut spec,
            a,
            b,
            acc: Dtype::F32,
            outs: 1,
        } = value.op.clone()
        {
            let av = &old[old_map[&a]];
            let candidate = spec
                .a
                .iter()
                .enumerate()
                .find(|(_, l)| !spec.out.contains(l) && spec.b.contains(l));
            if let Some((axis, label)) = candidate {
                let label = *label;
                let width = av.shape[axis];
                // Aim for roughly 128 independent 32x16 matrix tiles. Keep at
                // least 64 reduction elements per chunk to amortize its setup;
                // a non-divisible axis simply keeps the unsplit implementation.
                let mut splits = (128u32.div_ceil(value.len().div_ceil(512)))
                    .next_power_of_two()
                    .clamp(2, 16);
                while splits > 1 && (!width.is_multiple_of(splits) || width / splits < 64) {
                    splits /= 2;
                }
                let partial_len = value.len().checked_mul(splits);
                let valid = width >= 256
                    && value.len() <= 32768
                    && splits > 1
                    && partial_len.is_some_and(|n| {
                        n <= u32::MAX - 2 * BLOCK && u64::from(n) * 4 <= max_bytes
                    });
                if valid {
                    let split_label = (0..=u8::MAX).map(Label).find(|l| {
                        !spec.a.contains(l) && !spec.b.contains(l) && !spec.out.contains(l)
                    });
                    if let Some(split_label) = split_label {
                        let mut operands = vec![];
                        for (input, labels) in [(a, &mut spec.a), (b, &mut spec.b)] {
                            let source = &old[old_map[&input]];
                            let axis = labels.iter().position(|l| *l == label).unwrap();
                            let mut shape = source.shape.clone();
                            shape[axis] /= splits;
                            shape.insert(axis, splits);
                            let specs = source
                                .shape
                                .iter()
                                .enumerate()
                                .flat_map(|(i, n)| {
                                    if i == axis {
                                        vec![
                                            StrideSpec::dim_with(
                                                i as u32,
                                                Dim::Const(splits.into()),
                                                width / splits,
                                            ),
                                            StrideSpec::dim(
                                                i as u32,
                                                Dim::Const((width / splits).into()),
                                            ),
                                        ]
                                    } else {
                                        vec![StrideSpec::dim(i as u32, Dim::Const((*n).into()))]
                                    }
                                })
                                .collect();
                            let id = fresh()?;
                            values.push(Value {
                                id,
                                shape,
                                dtype: source.dtype,
                                op: Logical::Restride {
                                    x: input,
                                    specs,
                                    bounds: BoundsProof::Static,
                                },
                                offset: None,
                                forwarded: false,
                            });
                            labels.insert(axis, split_label);
                            operands.push(id);
                        }
                        spec.out.insert(0, split_label);
                        let mut shape = value.shape.clone();
                        shape.insert(0, splits);
                        let partial = fresh()?;
                        values.push(Value {
                            id: partial,
                            shape,
                            dtype: value.dtype,
                            op: Logical::Contract {
                                spec,
                                a: operands[0],
                                b: operands[1],
                                acc: Dtype::F32,
                                outs: 1,
                            },
                            offset: None,
                            forwarded: false,
                        });
                        value.op = Logical::Fold {
                            axis: 0,
                            acc: Dtype::F32,
                            ins: vec![partial].into(),
                            carrier: Carrier::binop(
                                BinOp::Add,
                                Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                                Dtype::F32,
                            ),
                        };
                    }
                }
            }
        }
        values.push(value);
    }
    map.clear();
    for (i, v) in values.iter().enumerate() {
        map.insert(v.id, i);
    }
    for (alias, i) in old_map {
        map.insert(alias, map[&old[i].id]);
    }
    Ok(())
}
