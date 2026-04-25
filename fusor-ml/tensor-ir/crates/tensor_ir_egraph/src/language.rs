use std::{collections::HashSet, fmt};

use egg::{Id, Language};

use crate::types::{
    BinaryOp, BinderInfo, BinderKind, BufferRef, DType, HasBinder, MemTier, ReduceOp, ScalarValue,
    Shape, Strides, TernaryOp, UnaryOp, VarRef,
};

/// Domain-specific nodes for high-level tensor operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HighLevelNode {
    Input {
        id: u32,
        shape: Shape,
        dtype: DType,
    },
    Restride {
        new_shape: Shape,
        strides: Strides,
        offset: i64,
        expr: Id,
    },
    Elementwise {
        index_space: Shape,
        num_inputs: u32,
        children_list: Id,
    },
    Reduce {
        axis: u32,
        op: ReduceOp,
        expr: Id,
    },
    /// Input-buffer slot number. Inside an `Elementwise`/`Reduce` body this
    /// denotes "the element of input `i` at the current iteration index".
    Param(u32),
    /// Current output/index-space dimension value inside an elementwise body.
    Index(u32),
    /// Input-buffer slot number with explicit per-axis index expressions.
    IndexedParam {
        index: u32,
        children_list: Id,
    },
}

/// Domain-specific nodes for GPU dispatch mapping.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DispatchNode {
    /// GPU dispatch kernel. `children_list` is a flat list laid out as
    /// `[inputs (num_inputs ids), output_pair_1 (value, addr), ...]`, so
    /// the number of outputs is `(list.len() - num_inputs) / 2` and is
    /// derived from the structure. Whether the body is "composite"
    /// (contains a nested Theta chain) is read from analysis data
    /// (`TensorData::contains_theta`) instead of stored as a tag.
    Dispatch {
        workgroups: u32,
        num_inputs: u32,
        children_list: Id,
    },
    Token,
    Seq(Id),
    Pipeline(Id),
    Pack {
        children_list: Id,
    },
    Extract {
        index: u32,
        tuple: Id,
    },
}

/// Domain-specific nodes for SIMD thread execution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdNode {
    Var(VarRef),
    Load {
        tier: MemTier,
        children: [Id; 2],
    },
    Shuffle([Id; 2]),
    ReduceSimd {
        op: ReduceOp,
        src: Id,
    },
    /// Functional fixpoint loop. `update` runs `count` times; the body sees
    /// the iter index as `Var(Bound{Iter,0})` and the accumulator as
    /// `Var(Bound{Acc,0})`. Whether this Theta behaves as an iteration,
    /// reduction, running-reduction, or coop-load is determined by the
    /// structure of `init` and `update` — not encoded as a tag.
    Theta {
        children: [Id; 3],
    },
    Store {
        tier: MemTier,
        children: [Id; 3],
    },
    StoreIf {
        tier: MemTier,
        children: [Id; 4],
    },
    Barrier {
        /// Threadgroup buffers covered by this barrier.
        regions: Vec<BufferRef>,
        state: Id,
    },
}

/// Unified language for the tensor IR e-graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorIr {
    HighLevel(HighLevelNode),
    Dispatch(DispatchNode),
    Simd(SimdNode),

    // Operations (shared across levels)
    BinOp(BinaryOp, [Id; 2]),
    UnOp(UnaryOp, Id),
    TernOp(TernaryOp, [Id; 3]),
    Const(ScalarValue),

    // Structural List (for representing arrays of items cleanly)
    Nil,
    Cons([Id; 2]),
}

fn matches_high_level(lhs: &HighLevelNode, rhs: &HighLevelNode) -> bool {
    match (lhs, rhs) {
        (
            HighLevelNode::Input {
                id: lhs_id,
                shape: lhs_shape,
                dtype: lhs_dtype,
            },
            HighLevelNode::Input {
                id: rhs_id,
                shape: rhs_shape,
                dtype: rhs_dtype,
            },
        ) => lhs_id == rhs_id && lhs_shape == rhs_shape && lhs_dtype == rhs_dtype,
        (
            HighLevelNode::Restride {
                new_shape: lhs_shape,
                strides: lhs_strides,
                offset: lhs_offset,
                ..
            },
            HighLevelNode::Restride {
                new_shape: rhs_shape,
                strides: rhs_strides,
                offset: rhs_offset,
                ..
            },
        ) => lhs_shape == rhs_shape && lhs_strides == rhs_strides && lhs_offset == rhs_offset,
        (
            HighLevelNode::Elementwise {
                index_space: lhs_index_space,
                num_inputs: lhs_num_inputs,
                ..
            },
            HighLevelNode::Elementwise {
                index_space: rhs_index_space,
                num_inputs: rhs_num_inputs,
                ..
            },
        ) => lhs_index_space == rhs_index_space && lhs_num_inputs == rhs_num_inputs,
        (
            HighLevelNode::Reduce {
                axis: lhs_axis,
                op: lhs_op,
                ..
            },
            HighLevelNode::Reduce {
                axis: rhs_axis,
                op: rhs_op,
                ..
            },
        ) => lhs_axis == rhs_axis && lhs_op == rhs_op,
        (HighLevelNode::Param(lhs_p), HighLevelNode::Param(rhs_p)) => lhs_p == rhs_p,
        (HighLevelNode::Index(lhs_i), HighLevelNode::Index(rhs_i)) => lhs_i == rhs_i,
        (
            HighLevelNode::IndexedParam {
                index: lhs_index, ..
            },
            HighLevelNode::IndexedParam {
                index: rhs_index, ..
            },
        ) => lhs_index == rhs_index,
        _ => false,
    }
}

fn matches_dispatch(lhs: &DispatchNode, rhs: &DispatchNode) -> bool {
    match (lhs, rhs) {
        (
            DispatchNode::Dispatch {
                workgroups: lhs_workgroups,
                num_inputs: lhs_num_inputs,
                ..
            },
            DispatchNode::Dispatch {
                workgroups: rhs_workgroups,
                num_inputs: rhs_num_inputs,
                ..
            },
        ) => lhs_workgroups == rhs_workgroups && lhs_num_inputs == rhs_num_inputs,
        (DispatchNode::Token, DispatchNode::Token)
        | (DispatchNode::Seq(_), DispatchNode::Seq(_))
        | (DispatchNode::Pipeline(_), DispatchNode::Pipeline(_))
        | (DispatchNode::Pack { .. }, DispatchNode::Pack { .. }) => true,
        (
            DispatchNode::Extract {
                index: lhs_index, ..
            },
            DispatchNode::Extract {
                index: rhs_index, ..
            },
        ) => lhs_index == rhs_index,
        _ => false,
    }
}

fn matches_simd(lhs: &SimdNode, rhs: &SimdNode) -> bool {
    match (lhs, rhs) {
        (SimdNode::Var(lhs_var), SimdNode::Var(rhs_var)) => lhs_var == rhs_var,
        (SimdNode::Load { tier: lhs_tier, .. }, SimdNode::Load { tier: rhs_tier, .. })
        | (SimdNode::Store { tier: lhs_tier, .. }, SimdNode::Store { tier: rhs_tier, .. })
        | (SimdNode::StoreIf { tier: lhs_tier, .. }, SimdNode::StoreIf { tier: rhs_tier, .. }) => {
            lhs_tier == rhs_tier
        }
        (SimdNode::Shuffle(_), SimdNode::Shuffle(_)) => true,
        (SimdNode::Theta { .. }, SimdNode::Theta { .. }) => true,
        (SimdNode::ReduceSimd { op: lhs_op, .. }, SimdNode::ReduceSimd { op: rhs_op, .. }) => {
            lhs_op == rhs_op
        }
        (
            SimdNode::Barrier {
                regions: lhs_regions,
                ..
            },
            SimdNode::Barrier {
                regions: rhs_regions,
                ..
            },
        ) => lhs_regions == rhs_regions,
        _ => false,
    }
}

const fn high_level_children(node: &HighLevelNode) -> &[Id] {
    match node {
        HighLevelNode::Input { .. } | HighLevelNode::Param(_) | HighLevelNode::Index(_) => &[],
        HighLevelNode::IndexedParam { children_list, .. } => std::slice::from_ref(children_list),
        HighLevelNode::Restride { expr, .. } | HighLevelNode::Reduce { expr, .. } => {
            std::slice::from_ref(expr)
        }
        HighLevelNode::Elementwise { children_list, .. } => std::slice::from_ref(children_list),
    }
}

const fn dispatch_children(node: &DispatchNode) -> &[Id] {
    match node {
        DispatchNode::Token => &[],
        DispatchNode::Seq(list) | DispatchNode::Pipeline(list) => std::slice::from_ref(list),
        DispatchNode::Dispatch { children_list, .. } | DispatchNode::Pack { children_list } => {
            std::slice::from_ref(children_list)
        }
        DispatchNode::Extract { tuple, .. } => std::slice::from_ref(tuple),
    }
}

const fn high_level_children_mut(node: &mut HighLevelNode) -> &mut [Id] {
    match node {
        HighLevelNode::Input { .. } | HighLevelNode::Param(_) | HighLevelNode::Index(_) => &mut [],
        HighLevelNode::IndexedParam { children_list, .. } => std::slice::from_mut(children_list),
        HighLevelNode::Restride { expr, .. } | HighLevelNode::Reduce { expr, .. } => {
            std::slice::from_mut(expr)
        }
        HighLevelNode::Elementwise { children_list, .. } => std::slice::from_mut(children_list),
    }
}

const fn dispatch_children_mut(node: &mut DispatchNode) -> &mut [Id] {
    match node {
        DispatchNode::Token => &mut [],
        DispatchNode::Seq(list) | DispatchNode::Pipeline(list) => std::slice::from_mut(list),
        DispatchNode::Dispatch { children_list, .. } | DispatchNode::Pack { children_list } => {
            std::slice::from_mut(children_list)
        }
        DispatchNode::Extract { tuple, .. } => std::slice::from_mut(tuple),
    }
}

impl Language for TensorIr {
    type Discriminant = std::mem::Discriminant<Self>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(self)
    }

    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::HighLevel(lhs), Self::HighLevel(rhs)) => matches_high_level(lhs, rhs),
            (Self::Dispatch(lhs), Self::Dispatch(rhs)) => matches_dispatch(lhs, rhs),
            (Self::Simd(lhs), Self::Simd(rhs)) => matches_simd(lhs, rhs),
            (Self::BinOp(a_op, _), Self::BinOp(b_op, _)) => a_op == b_op,
            (Self::UnOp(a_op, _), Self::UnOp(b_op, _)) => a_op == b_op,
            (Self::TernOp(a_op, _), Self::TernOp(b_op, _)) => a_op == b_op,
            (Self::Const(a_v), Self::Const(b_v)) => a_v == b_v,
            (Self::Nil, Self::Nil) | (Self::Cons(_), Self::Cons(_)) => true,
            _ => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::HighLevel(hl) => high_level_children(hl),
            Self::Dispatch(dp) => dispatch_children(dp),
            Self::Simd(s) => match s {
                SimdNode::Var(_) => &[],
                SimdNode::Load { children, .. } => children,
                SimdNode::Shuffle(ids) => ids,
                SimdNode::ReduceSimd { src, .. } => std::slice::from_ref(src),
                SimdNode::Theta { children, .. } => children,
                SimdNode::Store { children, .. } => children,
                SimdNode::StoreIf { children, .. } => children,
                SimdNode::Barrier { state, .. } => std::slice::from_ref(state),
            },
            Self::BinOp(_, ids) | Self::Cons(ids) => ids,
            Self::UnOp(_, id) => std::slice::from_ref(id),
            Self::TernOp(_, ids) => ids,
            Self::Const(_) | Self::Nil => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::HighLevel(hl) => high_level_children_mut(hl),
            Self::Dispatch(dp) => dispatch_children_mut(dp),
            Self::Simd(s) => match s {
                SimdNode::Var(_) => &mut [],
                SimdNode::Load { children, .. } => children,
                SimdNode::Shuffle(ids) => ids,
                SimdNode::ReduceSimd { src, .. } => std::slice::from_mut(src),
                SimdNode::Theta { children, .. } => children,
                SimdNode::Store { children, .. } => children,
                SimdNode::StoreIf { children, .. } => children,
                SimdNode::Barrier { state, .. } => std::slice::from_mut(state),
            },
            Self::BinOp(_, ids) | Self::Cons(ids) => ids,
            Self::UnOp(_, id) => std::slice::from_mut(id),
            Self::TernOp(_, ids) => ids,
            Self::Const(_) | Self::Nil => &mut [],
        }
    }
}

/// Walk a list-shaped e-class, returning the head ids in order. Returns
/// `None` if any class along the chain contains no `Cons`/`Nil` node —
/// e-class merging can leave a list class holding non-list nodes alongside
/// the list ones, and some call sites (e.g. analysis passes run while the
/// graph is mid-construction) will see a malformed list that shouldn't be
/// crashed on.
#[must_use]
pub fn try_extract_list<A: egg::Analysis<TensorIr>>(
    egraph: &egg::EGraph<TensorIr, A>,
    list_id: Id,
) -> Option<Vec<Id>> {
    let mut curr = list_id;
    let mut res = Vec::new();
    loop {
        let list_node = egraph[curr]
            .nodes
            .iter()
            .find(|node| matches!(node, TensorIr::Cons(_) | TensorIr::Nil))?;
        match list_node {
            TensorIr::Cons([head, tail]) => {
                res.push(*head);
                curr = *tail;
            }
            TensorIr::Nil => break,
            _ => unreachable!("find() filter admits only Cons/Nil"),
        }
    }
    Some(res)
}

/// Walk a list-shaped e-class, returning the head ids in order.
///
/// # Panics
///
/// Panics if any e-class along the chain has no `Cons`/`Nil` node. Use
/// [`try_extract_list`] when the caller can tolerate a malformed list.
pub fn extract_list<A: egg::Analysis<TensorIr>>(
    egraph: &egg::EGraph<TensorIr, A>,
    list_id: Id,
) -> Vec<Id> {
    try_extract_list(egraph, list_id).unwrap_or_else(|| {
        panic!(
            "Expected Cons/Nil list at e-class {list_id:?}, found nodes: {:?}",
            egraph[list_id].nodes
        )
    })
}

pub fn add_list<A: egg::Analysis<TensorIr>>(
    egraph: &mut egg::EGraph<TensorIr, A>,
    items: &[Id],
) -> Id {
    let mut curr = egraph.add(TensorIr::Nil);
    for &item in items.iter().rev() {
        curr = egraph.add(TensorIr::Cons([item, curr]));
    }
    curr
}

/// Returns true when `children` has the value/address dispatch layout and no
/// two output pairs address the same element of the dispatch output buffer.
#[must_use]
pub fn dispatch_children_have_unique_output_addrs<A: egg::Analysis<TensorIr>>(
    egraph: &egg::EGraph<TensorIr, A>,
    num_inputs: u32,
    children: &[Id],
) -> bool {
    let num_inputs = num_inputs as usize;
    let Some(body_len) = children.len().checked_sub(num_inputs) else {
        return false;
    };
    if body_len == 0 || !body_len.is_multiple_of(2) {
        return false;
    }

    let mut seen = HashSet::new();
    for output in 0..(body_len / 2) {
        let addr = children[num_inputs + output * 2 + 1];
        if !seen.insert(egraph.find(addr)) {
            return false;
        }
    }
    true
}

/// Add a value/address dispatch only when its structural output pairs are
/// well formed. Multi-output dispatches share one output buffer, so repeated
/// output addresses would create ambiguous stores and are rejected here.
pub fn try_add_value_addr_dispatch<A: egg::Analysis<TensorIr>>(
    egraph: &mut egg::EGraph<TensorIr, A>,
    workgroups: u32,
    num_inputs: u32,
    children: &[Id],
) -> Option<Id> {
    if !dispatch_children_have_unique_output_addrs(egraph, num_inputs, children) {
        return None;
    }
    let children_list = add_list(egraph, children);
    Some(egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups,
        num_inputs,
        children_list,
    })))
}

impl HasBinder for TensorIr {
    fn binder_info(&self) -> Option<BinderInfo> {
        match self {
            Self::Simd(SimdNode::Theta { .. }) => Some(BinderInfo {
                kind: BinderKind::Theta,
                // Children order is [init, count, update]. Only `update` lives
                // inside the new scope.
                body_mask: 0b100,
            }),
            Self::Dispatch(DispatchNode::Dispatch { .. }) => Some(BinderInfo {
                kind: BinderKind::Dispatch,
                // Children order is [children_list]; the list head is the kernel
                // body, which sees the Dispatch thread-index bindings.
                body_mask: 0b001,
            }),
            _ => None,
        }
    }
}

/// Compact single-line label suitable for GraphViz via `EGraph::dot`.
impl fmt::Display for TensorIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighLevel(hl) => match hl {
                HighLevelNode::Input { id, dtype, .. } => write!(f, "Input#{id}:{dtype}"),
                HighLevelNode::Restride { .. } => write!(f, "Restride"),
                HighLevelNode::Elementwise { num_inputs, .. } => {
                    write!(f, "Elementwise×{num_inputs}")
                }
                HighLevelNode::Reduce { axis, op, .. } => write!(f, "Reduce/{op}@{axis}"),
                HighLevelNode::Param(i) => write!(f, "Param({i})"),
                HighLevelNode::Index(i) => write!(f, "Index({i})"),
                HighLevelNode::IndexedParam { index, .. } => write!(f, "IndexedParam({index})"),
            },
            Self::Dispatch(dp) => match dp {
                DispatchNode::Dispatch { workgroups, .. } => {
                    write!(f, "Dispatch[wg={workgroups}]")
                }
                DispatchNode::Token => write!(f, "Token"),
                DispatchNode::Seq(_) => write!(f, "Seq"),
                DispatchNode::Pipeline(_) => write!(f, "Pipeline"),
                DispatchNode::Pack { .. } => write!(f, "Pack"),
                DispatchNode::Extract { index, .. } => write!(f, "Extract[{index}]"),
            },
            Self::Simd(s) => match s {
                SimdNode::Var(v) => write!(f, "Var({v})"),
                SimdNode::Load { tier, .. } => write!(f, "Load@{tier}"),
                SimdNode::Shuffle(_) => write!(f, "Shuffle"),
                SimdNode::ReduceSimd { op, .. } => write!(f, "ReduceSimd/{op}"),
                SimdNode::Theta { .. } => write!(f, "Theta"),
                SimdNode::Store { tier, .. } => write!(f, "Store@{tier}"),
                SimdNode::StoreIf { tier, .. } => write!(f, "StoreIf@{tier}"),
                SimdNode::Barrier { regions, .. } => write!(f, "Barrier[{}]", regions.len()),
            },
            Self::BinOp(op, _) => write!(f, "{op}"),
            Self::UnOp(op, _) => write!(f, "{op}"),
            Self::TernOp(op, _) => write!(f, "{op}"),
            Self::Const(v) => write!(f, "{v}"),
            Self::Nil => write!(f, "Nil"),
            Self::Cons(_) => write!(f, "::"),
        }
    }
}

#[must_use]
/// # Panics
///
/// Panics if `list_id` does not point to a `Cons`/`Nil` list.
pub fn extract_recexpr_list(nodes: &[TensorIr], list_id: Id) -> Vec<Id> {
    let mut curr = list_id;
    let mut res = Vec::new();
    loop {
        match &nodes[usize::from(curr)] {
            TensorIr::Cons([head, tail]) => {
                res.push(*head);
                curr = *tail;
            }
            TensorIr::Nil => break,
            _ => panic!("Expected Cons/Nil list"),
        }
    }
    res
}
