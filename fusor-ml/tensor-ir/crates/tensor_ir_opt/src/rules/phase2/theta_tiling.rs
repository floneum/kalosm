use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{DispatchNode, SimdNode, TensorIr, add_list, extract_list};
use crate::types::{BinaryOp, BinderKind, ScalarValue, VarRef, slots};

pub(super) fn build(tile_size: u32) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        format!("theta-tile-{tile_size}"),
        SimpleEclassSearcher::new(move |egraph, eclass| {
            egraph[eclass]
                .iter()
                .any(|node| theta_is_tileable(egraph, node, tile_size))
        }),
        crate::applier::AdaptedApplier(ThetaTilingApplier { tile_size }),
    )
    .unwrap()
}

fn theta_count_value(egraph: &EGraph<TensorIr, TensorAnalysis>, count: Id) -> Option<u32> {
    match &egraph[count].data.constant {
        Some(ScalarValue::U32(v)) => Some(*v),
        Some(ScalarValue::I32(v)) if *v > 0 => Some((*v).cast_unsigned()),
        _ => None,
    }
}

fn theta_is_tileable(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
    tile_size: u32,
) -> bool {
    // Tiling reorders iterations via `actual_k = outer*tile + inner`.
    // Sound iff:
    //   - count is a positive constant divisible by the candidate tile,
    //   - init is not a `Pack` (Pack => running reduction with coupled state),
    //   - the body doesn't already reference an outer iter binder (`iter(1)`),
    //   - the body has no nested `Theta`.
    let TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }) = node
    else {
        return false;
    };
    let Some(count_val) = theta_count_value(egraph, *count) else {
        return false;
    };
    count_val > tile_size
        && count_val.is_multiple_of(tile_size)
        && !egraph[*init].data.contains_pack
        && !egraph[*update].data.var_dep.contains(&VarRef::iter(1))
        && !egraph[*update].data.contains_theta
}

struct ThetaTilingApplier {
    tile_size: u32,
}

impl crate::applier::TypedApplier for ThetaTilingApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| theta_is_tileable(egraph, n, self.tile_size))
            .cloned();

        let Some(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
            ..
        })) = node
        else {
            return vec![];
        };
        let count_val = theta_count_value(egraph, count).expect("tileable theta has const count");

        let outer_count_val = count_val / self.tile_size;
        let outer_count = egraph.add(TensorIr::Const(ScalarValue::U32(outer_count_val)));
        let inner_count = egraph.add(TensorIr::Const(ScalarValue::U32(self.tile_size)));
        let tile_lit = egraph.add(TensorIr::Const(ScalarValue::U32(self.tile_size)));
        // Inside the inner Theta's body, the outer Theta's iter is at depth 1
        // and the inner Theta's iter is at depth 0.
        let outer_k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(1))));
        let inner_k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
        let outer_times_tile = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [outer_k_var, tile_lit]));
        let actual_k = egraph.add(TensorIr::BinOp(
            BinaryOp::Add,
            [outer_times_tile, inner_k_var],
        ));

        // Substitute the original body's reference to its (single) iter
        // (`Bound { Iter, 0 }`) with the recomposed `actual_k`. The new shift/
        // subst machinery is scope-aware: if `update` itself contains nested
        // Thetas, depths are tracked correctly during the descent.
        let remapped_update = binding::subst(
            egraph,
            update,
            BinderKind::Theta,
            slots::THETA_ITER,
            0,
            actual_k,
        );
        // Inner Theta's `init` carries the outer Theta's accumulator
        // value — `VarRef::acc(0)` (i.e. `Bound { Theta, Acc, 0 }`)
        // one level inside the outer. There's no separate
        // register-blocked accumulator variant.
        let acc_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
        // Packed Thetas: the outer's init is a `Pack(N)` tuple, so the
        // update body also produces a `Pack(N)`. The inner Theta's init
        // must be a Pack(N) of Extract-from-acc to keep codegen's structural
        // arity check happy (`lower_theta` asserts init/update arity match).
        // For scalar accumulators this just evaluates to `Var(acc_ref)`.
        let inner_init = wrap_inner_init_for_packed(egraph, init, acc_var);
        let inner_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
            children: [inner_init, inner_count, remapped_update],
        }));
        let outer_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
            children: [init, outer_count, inner_theta],
        }));

        egraph.union(eclass, outer_theta);
        vec![outer_theta]
    }
}

/// If the outer Theta's init is a `Pack(N)` tuple, the update body is also
/// a `Pack(N)`. Codegen's `lower_theta` asserts that init and update have
/// matching structural arity. So when we split such a Theta, we must build
/// the inner Theta's init as a `Pack(N)` of `Extract(i, acc_var)` instead of
/// a bare `Var(acc)` — which would have arity 1 and trip the assertion.
///
/// For scalar (non-packed) Thetas, this returns `acc_var` unchanged.
fn wrap_inner_init_for_packed(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    outer_init: Id,
    acc_var: Id,
) -> Id {
    let arity = egraph[outer_init].iter().find_map(|node| {
        if let TensorIr::Dispatch(DispatchNode::Pack { children_list }) = node {
            Some(extract_list(egraph, *children_list).len())
        } else {
            None
        }
    });
    let Some(arity) = arity else {
        return acc_var;
    };
    let elements: Vec<Id> = (0..arity)
        .map(|i| {
            egraph.add(TensorIr::Dispatch(DispatchNode::Extract {
                index: u32::try_from(i).expect("pack arity fits in u32"),
                tuple: acc_var,
            }))
        })
        .collect();
    let list = add_list(egraph, &elements);
    egraph.add(TensorIr::Dispatch(DispatchNode::Pack {
        children_list: list,
    }))
}
