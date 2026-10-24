//! Semantic launch descriptions for joining measured GPU time across allocation
//! and shader-body identities. Composite launches retain their members: their
//! timestamp is never arbitrarily divided between matrix and scalar work.
use fusor_ir::{
    dtype::{Dtype, QLayout},
    egraph::{EGraph, Id},
    extract::Plan,
    ir::{Op, launch::Launch},
    shape::Dim,
};
use serde_json::{Value, json};

/// Describe one resolved plan. Byte counts describe logical operand volume,
/// not measured memory traffic or cache misses. Unknown dimensions stay null.
pub fn describe(
    plan: &Plan,
    graph: &EGraph,
    resolve: &dyn Fn(Dim) -> Option<u64>,
    identities: &[(u128, [u32; 3])],
) -> Value {
    assert_eq!(
        plan.launches.len(),
        identities.len(),
        "incomplete plan identities"
    );
    let launches: Vec<_> = plan
        .launches
        .iter()
        .zip(identities)
        .enumerate()
        .map(|(index, (dispatch, (body, grid)))| {
            json!({"index": index, "body": format!("{body:032x}"), "grid": grid,
            "operation": member(dispatch.root, plan, graph, resolve)})
        })
        .collect();
    json!({"plan": format!("{:032x}", plan.hash.0), "arena_bytes": plan.arena_bytes, "launches": launches})
}

fn member(id: Id, plan: &Plan, graph: &EGraph, resolve: &dyn Fn(Dim) -> Option<u64>) -> Value {
    let node = graph.node(id);
    let schedule = plan.extraction.theta.get(&id).map(|p| format!("{p:?}"));
    let Op::Launch(op) = &node.op else {
        return json!({"kind": format!("{:?}", node.op.tag())});
    };
    let mut out = match op {
        Launch::RmsNorm {
            output,
            width,
            block,
            ops,
            ..
        } => json!({
            "kind": "rms_norm", "shape": output.dims.iter().map(|d| resolve(*d)).collect::<Vec<_>>(),
            "width": width, "acc": "F32", "block": block, "residual": ops.len() == 3,
            "workgroup_storage_bound": u64::from(*block) * 4
        }),
        Launch::DecodeAttention {
            output,
            kv_heads,
            keys,
            chunk,
            block,
            subgroup,
            ops,
            ..
        } => json!({
            "kind": "decode_attention", "shape": output.dims.iter().map(|d| resolve(*d)).collect::<Vec<_>>(),
            "kv_heads": kv_heads, "capacity": resolve(*keys), "valid_prefix": "runtime uniform",
            "chunk": chunk, "block": block, "subgroup": subgroup, "acc": "F32", "workgroup_storage_bound": u64::from(*block / *subgroup) * (ops[0].layout.shape()[3].as_const().unwrap_or(0) + 2) * 4,
            "operands": ops.iter().map(|o| json!({
                "shape": o.layout.shape().iter().map(|d| resolve(*d)).collect::<Vec<_>>(),
                "strides": o.layout.strides().iter().map(|d| resolve(*d)).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        }),
        Launch::DecodeAttentionMerge { output, splits, .. } => json!({
            "kind": "decode_attention_merge", "shape": output.dims.iter().map(|d| resolve(*d)).collect::<Vec<_>>(),
            "splits": resolve(*splits), "acc": "F32", "workgroup_storage_bound": 0
        }),
        Launch::Contract {
            m,
            n,
            k,
            batch,
            family,
            a,
            b,
            acc,
            post,
            ..
        } => {
            let dims = [*m, *n, *k, *batch].map(resolve);
            let macs = dims.into_iter().try_fold(1u64, |v, d| v.checked_mul(d?));
            let side = |s: &fusor_ir::ir::launch::ContractSide| {
                s.ops
                    .iter()
                    .map(|o| {
                        let dtype = graph.facts(o.src).dtype;
                        let shape: Vec<_> = o.layout.shape().iter().map(|d| resolve(*d)).collect();
                        let strides: Vec<_> =
                            o.layout.strides().iter().map(|d| resolve(*d)).collect();
                        let unique = shape.iter().zip(&strides).try_fold(1u64, |v, (d, s)| {
                            v.checked_mul(if (*s)? == 0 { 1 } else { (*d)? })
                        });
                        let quant_layout = graph.class_ids(graph.class_of(o.src)).into_iter().find_map(|id| {
                            match &graph.node(id).op {
                                Op::Logical(fusor_ir::ir::logical::Logical::Leaf(
                                    fusor_ir::ir::logical::LeafKind::Quantized { layout, .. }
                                )) => Some(*layout),
                                _ => None,
                            }
                        });
                        let physical_bytes = unique.and_then(|n| match dtype {
                            Dtype::Q(fmt) => n.div_ceil(u64::from(fmt.block_elements()))
                                .checked_mul(u64::from(fmt.block_bytes(quant_layout?))),
                            _ => n.checked_mul(dtype.byte_size()),
                        });
                        let bytes = unique.and_then(|n| match dtype {
                            Dtype::Q(fmt) => n
                                .div_ceil(u64::from(fmt.block_elements()))
                                .checked_mul(u64::from(fmt.block_bytes(QLayout::Native))),
                            _ => n.checked_mul(dtype.byte_size()),
                        });
                        json!({"dtype": format!("{dtype:?}"), "shape": shape, "strides": strides,
                        "logical_bytes_native_quant": bytes,
                        "selected_quant_layout": quant_layout.map(|l| format!("{l:?}")),
                        "selected_storage_bytes": physical_bytes, "access": format!("{:?}", o.access)})
                    })
                    .collect::<Vec<_>>()
            };
            json!({"kind": "contract", "family": format!("{family:?}"), "m": dims[0], "n": dims[1],
                "k": dims[2], "batch": dims[3], "macs": macs, "acc": format!("{acc:?}"), "a": side(a), "b": side(b),
                "a_pre": format!("{:?}", a.pre), "b_pre": format!("{:?}", b.pre), "post": format!("{post:?}")})
        }
        Launch::Map { space, body, .. } => {
            json!({"kind": "LaunchMap", "shape": space.dims.iter().map(|d| resolve(*d)).collect::<Vec<_>>(),
                "expression": format!("{body:?}")})
        }
        Launch::Fold {
            space,
            axis,
            carrier,
            post,
            ..
        } => {
            json!({"kind": "fold", "shape": space.dims.iter().map(|d| resolve(*d)).collect::<Vec<_>>(), "axis": axis,
                "carrier": format!("{carrier:?}"), "post": format!("{post:?}")})
        }
        Launch::Group { members, .. } | Launch::Slab { members, .. } => json!({
            "kind": if matches!(op, Launch::Group { .. }) { "group" } else { "slab" },
            "members": members.iter().map(|id| member(*id, plan, graph, resolve)).collect::<Vec<_>>()
        }),
        _ => {
            json!({"kind": format!("{:?}", node.op.tag()), "shape": graph.facts(id).shape.iter().map(|d| resolve(*d)).collect::<Vec<_>>()})
        }
    };
    out["schedule"] = json!(schedule);
    out
}
