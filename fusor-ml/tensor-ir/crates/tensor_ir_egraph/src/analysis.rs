use egg::{Analysis, DidMerge, EGraph, Id, Language};

use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr};
use crate::types::{
    AddressProfile, BinaryOp, BinderKind, DType, DepSet, MemTier, ReduceOp, ScalarValue, Shape,
    Strides, TernaryOp, UnaryOp, VarDepSet, VarRef, index_level_from_slot,
};

/// Analysis facts for the generic whole-expression composite-dispatch lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompositeDispatchInfo {
    /// True when this e-class contains a supported high-level tensor tree with
    /// fully literal tensor shapes, so it can lower as one composite dispatch.
    pub lowerable: bool,
}

/// Analysis fact for tensors of the form `exp(x - reduce_max(x))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpMaxInfo {
    pub score: Id,
    pub axis: u32,
}

/// Analysis fact for tensors normalized by an additive reduction along an axis.
///
/// The canonical producer is `numerator / reduce_sum(numerator)`, with the
/// denominator broadcast back over `axis`.  `exp_max` is populated when the
/// numerator itself is known to be `exp(x - max(x))`, letting generic online
/// reduction lowering use the stable softmax recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizedWeightInfo {
    pub numerator: Id,
    pub axis: u32,
    pub exp_max: Option<ExpMaxInfo>,
}

/// Shape-level facts for a concrete `DispatchNode::Dispatch` e-class.
///
/// Surfaces the structural shape choices the extractor needs to compare
/// dispatch variants side-by-side (plain vs. cooperative, register-blocked
/// vs. scalar, tiled vs. untiled). Populated bottom-up from body analysis so
/// cost evaluation doesn't have to re-walk subtrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchShapeFacts {
    /// True iff the body contains a `ReduceSimd` — the cooperative-split /
    /// SIMD-reduction variant. Implies a shuffle tree + barrier in codegen.
    pub cooperative: bool,
    /// Per-lane output register block size. Equals `(children.len() - num_inputs) / 2`
    /// for a well-formed Dispatch; 1 for plain, 2/4 for register-blocked forms.
    pub reg_block: u8,
    /// Outermost Theta iteration count when literal and present. Used as the
    /// tile-K estimate when sizing threadgroup footprint.
    pub tile_k: Option<u32>,
    /// Rough estimate of threadgroup (shared) memory bytes this dispatch
    /// consumes. Derived from whether any body subtree contains TG loads
    /// combined with `tile_k` and the body's dtype. Conservative zero when
    /// the body has no TG loads.
    pub tg_bytes: u32,
}

/// Per-e-class analysis data computed during equality saturation.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor shape (for high-level nodes).
    pub shape: Option<Shape>,
    /// Element data type.
    pub dtype: Option<DType>,
    /// Storage bytes per scalar element. Derived from `dtype` whenever it is
    /// known; cached here so consumers (cost model, footprint accounting,
    /// codegen) don't have to repeat the lookup or carry `DType` through.
    pub dtype_bytes: Option<u32>,
    /// Canonical strides for tensor-typed eclasses. Populated for `Input`
    /// (row-major) and `Restride` (the explicit strides), propagated through
    /// `Reduce` (axis dropped). Lets phase3/skeleton stop walking the e-graph
    /// to reconstruct strides. `None` if unknown or eclass members disagree.
    pub stride_profile: Option<Strides>,
    /// Dependence set: which GPU hierarchy levels this value depends on.
    pub dep: DepSet,
    /// If this is a known constant scalar.
    pub constant: Option<ScalarValue>,
    /// Which named Var nodes this value transitively depends on.
    pub var_dep: VarDepSet,
    /// Which named Var nodes are free in this subtree after local binders.
    pub free_var_dep: VarDepSet,
    /// Static bound on a scalar/address expression's runtime value. Populated
    /// for `Const`/`Add`/`Mul` subtrees that don't depend on a `Var`. Lets
    /// skeleton avoid re-walking address expressions to size threadgroup
    /// buffers. `None` if any subterm is `Var`-dependent or eclass members
    /// disagree.
    pub address_profile: Option<AddressProfile>,
    /// Generic composite-dispatch eligibility facts, derived from analysis
    /// instead of workload-specific structural matching.
    pub composite_dispatch: CompositeDispatchInfo,
    /// True if this subtree reaches (or is) a `Theta` node. Replaces
    /// ad-hoc recursive walks in phase-2/4 rule guards. Propagated
    /// bottom-up; merged via OR on e-class union.
    pub contains_theta: bool,
    /// True if this subtree reaches (or is) a `Pack` node. Used by
    /// phase-2 merge/tile rules to reject running-reduction Theta init
    /// tuples without walking. Propagated bottom-up; merged via OR.
    pub contains_pack: bool,
    /// True if this subtree reaches (or is) a `ReduceSimd` node. Consumed
    /// by `DispatchShapeFacts::cooperative` without walking the body.
    /// Propagated bottom-up; merged via OR.
    pub contains_reduce_simd: bool,
    /// True if this subtree reaches (or is) a `Load`/`Store` against a
    /// `Threadgroup` tier. Drives the threadgroup-footprint branch of
    /// `DispatchShapeFacts::tg_bytes`. Propagated bottom-up; merged via OR.
    pub contains_tg_load: bool,
    /// Shape-level facts for `DispatchNode::Dispatch` e-classes, surfaced
    /// for the extractor's cost model. `None` for non-Dispatch subtrees.
    pub dispatch_shape: Option<DispatchShapeFacts>,
    /// Number of nested `HighLevel::Reduce` frames wrapping this
    /// e-class (propagated bottom-up; 0 for non-reducing nodes). Used
    /// by `recursive-to-dispatch` to reject compound-reduction trees
    /// (e.g. softmax) that the composable `reduce_lowering` +
    /// `exp_algebra` + `theta_merge` path handles better. Monotonic
    /// max on e-class union.
    pub reduction_depth: u8,
    /// True when this e-class contains a scalar binary op whose operands may
    /// be swapped without changing semantics. Rewrites use this as a cheap
    /// analysis gate before materializing alternate operand orders.
    pub has_commutative_binop: bool,
    /// True when this e-class contains a scalar binary op whose nested groups
    /// may be rotated without changing semantics.
    pub has_associative_binop: bool,
    /// True when this e-class contains `exp(x - reduce_max(x))` along an axis.
    pub exp_max: Option<ExpMaxInfo>,
    /// True when this e-class is a tensor normalized by a reduction along an
    /// axis, for example softmax probabilities.
    pub normalized_weight: Option<NormalizedWeightInfo>,
}

impl Default for TensorData {
    fn default() -> Self {
        Self {
            shape: None,
            dtype: None,
            dtype_bytes: None,
            stride_profile: None,
            dep: DepSet::EMPTY,
            constant: None,
            var_dep: VarDepSet::empty(),
            free_var_dep: VarDepSet::empty(),
            address_profile: None,
            composite_dispatch: CompositeDispatchInfo::default(),
            contains_theta: false,
            contains_pack: false,
            contains_reduce_simd: false,
            contains_tg_load: false,
            dispatch_shape: None,
            reduction_depth: 0,
            has_commutative_binop: false,
            has_associative_binop: false,
            exp_max: None,
            normalized_weight: None,
        }
    }
}

/// E-graph analysis for the tensor IR.
///
/// Propagates:
/// - Shape and dtype from tensor nodes
/// - Dependence sets for stride-zero detection
/// - Constant folding for scalar expressions
#[derive(Debug, Clone, Default)]
pub struct TensorAnalysis;

fn empty_tensor_data() -> TensorData {
    TensorData {
        dep: DepSet::EMPTY,
        var_dep: VarDepSet::empty(),
        free_var_dep: VarDepSet::empty(),
        ..Default::default()
    }
}

fn child_dep_union(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> DepSet {
    ids.iter()
        .fold(DepSet::EMPTY, |acc, id| acc.union(egraph[*id].data.dep))
}

fn child_var_dep_union(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> VarDepSet {
    ids.iter().fold(VarDepSet::empty(), |acc, id| {
        acc.union(&egraph[*id].data.var_dep)
    })
}

fn child_free_var_dep_union(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> VarDepSet {
    ids.iter().fold(VarDepSet::empty(), |acc, id| {
        acc.union(&egraph[*id].data.free_var_dep)
    })
}

/// True if any of `ids` (or its subtrees) already contains a `Theta` node.
fn child_contains_theta(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> bool {
    ids.iter().any(|id| egraph[*id].data.contains_theta)
}

/// True if any of `ids` (or its subtrees) already contains a `Pack` node.
fn child_contains_pack(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> bool {
    ids.iter().any(|id| egraph[*id].data.contains_pack)
}

/// True if any of `ids` (or its subtrees) already contains a `ReduceSimd` node.
fn child_contains_reduce_simd(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> bool {
    ids.iter().any(|id| egraph[*id].data.contains_reduce_simd)
}

/// True if any of `ids` (or its subtrees) already touches threadgroup memory.
fn child_contains_tg_load(egraph: &EGraph<TensorIr, TensorAnalysis>, ids: &[Id]) -> bool {
    ids.iter().any(|id| egraph[*id].data.contains_tg_load)
}

/// Direct check for threadgroup `Load`/`Store`/`StoreIf` at this node.
const fn node_is_tg_memory(node: &TensorIr) -> bool {
    match node {
        TensorIr::Simd(SimdNode::Load { tier, .. })
        | TensorIr::Simd(SimdNode::Store { tier, .. })
        | TensorIr::Simd(SimdNode::StoreIf { tier, .. }) => {
            matches!(tier, MemTier::Threadgroup(_))
        }
        _ => false,
    }
}

/// Pick the outermost `Theta` iteration count literal reachable through a
/// single e-node choice per e-class. Used to estimate tile-K footprint for
/// threadgroup buffer sizing without enumerating all extracted variants.
fn outer_theta_count(egraph: &EGraph<TensorIr, TensorAnalysis>, root: Id) -> Option<u32> {
    let canonical = egraph.find(root);
    for node in egraph[canonical].iter() {
        if let TensorIr::Simd(SimdNode::Theta {
            children: [_, count, _],
        }) = node
            && let Some(ScalarValue::U32(v)) = &egraph[*count].data.constant
        {
            return Some(*v);
        }
    }
    None
}

fn copy_child_tensor_data(child: &TensorData) -> TensorData {
    TensorData {
        dep: child.dep,
        dtype: child.dtype,
        var_dep: child.var_dep.clone(),
        free_var_dep: child.free_var_dep.clone(),
        ..Default::default()
    }
}

fn unary_output_dtype(op: UnaryOp, input: Option<DType>) -> Option<DType> {
    match op {
        UnaryOp::CastF16 => Some(DType::F16),
        UnaryOp::CastF32 => Some(DType::F32),
        UnaryOp::CastI32 => Some(DType::I32),
        UnaryOp::CastU32 => Some(DType::U32),
        UnaryOp::CastBool | UnaryOp::Not => Some(DType::Bool),
        _ => input,
    }
}

fn scalar_expr_dtype(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    params: &[DType],
) -> Option<DType> {
    if let Some(dtype) = egraph[id].data.dtype {
        return Some(dtype);
    }
    egraph[id].iter().find_map(|node| match node {
        TensorIr::HighLevel(HighLevelNode::Param(index)) => params.get(*index as usize).copied(),
        TensorIr::HighLevel(HighLevelNode::Index(_)) => Some(DType::U32),
        TensorIr::ShapeParam(_) => Some(DType::U32),
        TensorIr::HighLevel(HighLevelNode::IndexedParam { index, .. }) => {
            params.get(*index as usize).copied()
        }
        TensorIr::Const(value) => Some(match value {
            ScalarValue::F16(_) => DType::F16,
            ScalarValue::F32(_) => DType::F32,
            ScalarValue::I32(_) => DType::I32,
            ScalarValue::U32(_) => DType::U32,
            ScalarValue::Bool(_) => DType::Bool,
        }),
        TensorIr::BinOp(op, args) => match op {
            BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Neq => Some(DType::Bool),
            _ => scalar_expr_dtype(egraph, args[0], params),
        },
        TensorIr::UnOp(op, arg) => unary_output_dtype(*op, scalar_expr_dtype(egraph, *arg, params)),
        TensorIr::TernOp(op, args) => match op {
            TernaryOp::Fma => scalar_expr_dtype(egraph, args[0], params),
            TernaryOp::Select => scalar_expr_dtype(egraph, args[1], params),
        },
        _ => None,
    })
}

fn list_tensor_data(egraph: &EGraph<TensorIr, TensorAnalysis>, list_id: Id) -> TensorData {
    copy_child_tensor_data(&egraph[list_id].data)
}

fn make_high_level_data(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &HighLevelNode,
) -> TensorData {
    let has_nonzero_shape = |shape: &Shape| !matches!(shape.static_numel(), Some(0));
    match node {
        HighLevelNode::Input { shape, dtype, .. } => TensorData {
            shape: Some(shape.clone()),
            dtype: Some(*dtype),
            stride_profile: Some(Strides::row_major_for_shape(shape)),
            composite_dispatch: CompositeDispatchInfo {
                lowerable: has_nonzero_shape(shape),
            },
            ..empty_tensor_data()
        },
        HighLevelNode::Restride {
            new_shape,
            strides,
            expr,
            ..
        } => {
            let child = &egraph[*expr].data;
            TensorData {
                shape: Some(new_shape.clone()),
                dtype: child.dtype,
                stride_profile: Some(strides.clone()),
                dep: child.dep,
                var_dep: child.var_dep.clone(),
                free_var_dep: child.free_var_dep.clone(),
                composite_dispatch: CompositeDispatchInfo {
                    lowerable: child.composite_dispatch.lowerable && has_nonzero_shape(new_shape),
                },
                reduction_depth: child.reduction_depth,
                ..Default::default()
            }
        }
        HighLevelNode::Elementwise {
            index_space,
            num_inputs,
            children_list,
        } => {
            let mut data = list_tensor_data(egraph, *children_list);
            data.shape = Some(index_space.clone());
            data.stride_profile = Some(Strides::row_major_for_shape(index_space));
            // During mid-construction passes (e.g. `build_dispatch_program_from_extracted`
            // adding nodes one at a time) `children_list` may briefly resolve to a
            // class whose first form isn't Cons/Nil because of e-class merges.
            // Surface default composite-dispatch / reduction-depth facts in that
            // case; a later rebuild will see the correct list form.
            if let Some(children) = crate::language::try_extract_list(egraph, *children_list) {
                let inputs_lowerable = children[..children.len().saturating_sub(1)]
                    .iter()
                    .all(|child| egraph[*child].data.composite_dispatch.lowerable);
                let input_count = (*num_inputs as usize).min(children.len());
                let input_dtypes = children[..input_count]
                    .iter()
                    .map(|child| egraph[*child].data.dtype)
                    .collect::<Option<Vec<_>>>();
                data.dtype = input_dtypes.as_deref().and_then(|input_dtypes| {
                    children
                        .last()
                        .and_then(|body| scalar_expr_dtype(egraph, *body, input_dtypes))
                });
                data.composite_dispatch = CompositeDispatchInfo {
                    lowerable: has_nonzero_shape(index_space) && inputs_lowerable,
                };
                data.reduction_depth = children[..children.len().saturating_sub(1)]
                    .iter()
                    .map(|c| egraph[*c].data.reduction_depth)
                    .max()
                    .unwrap_or(0);
                data.exp_max = detect_exp_max(egraph, index_space, *num_inputs, &children);
                data.normalized_weight =
                    detect_normalized_weight(egraph, index_space, *num_inputs, &children);
            }
            data
        }
        HighLevelNode::Reduce { axis, expr, .. } => {
            let child = &egraph[*expr].data;
            let reduced_shape = child
                .shape
                .as_ref()
                .filter(|shape| (*axis as usize) < shape.rank())
                .map(|shape| shape.remove_axis(*axis as usize));
            let stride_profile = child
                .stride_profile
                .as_ref()
                .filter(|s| (*axis as usize) < s.0.len())
                .map(|s| s.remove_axis(*axis as usize));
            TensorData {
                shape: reduced_shape.clone(),
                dtype: child.dtype,
                stride_profile,
                dep: child.dep,
                var_dep: child.var_dep.clone(),
                free_var_dep: child.free_var_dep.clone(),
                composite_dispatch: CompositeDispatchInfo {
                    lowerable: child.composite_dispatch.lowerable
                        && reduced_shape.as_ref().is_some_and(has_nonzero_shape),
                },
                reduction_depth: child.reduction_depth.saturating_add(1),
                ..Default::default()
            }
        }
        HighLevelNode::Param(_) | HighLevelNode::Index(_) | HighLevelNode::IndexedParam { .. } => {
            empty_tensor_data()
        }
    }
}

fn detect_exp_max(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    index_space: &Shape,
    num_inputs: u32,
    children: &[Id],
) -> Option<ExpMaxInfo> {
    if num_inputs != 2 || children.len() < 3 {
        return None;
    }
    let score = children[0];
    let max_bcast = children[1];
    let body = *children.last()?;
    if !is_exp_of_param_sub(egraph, body, 0, 1) {
        return None;
    }
    let axis = match_broadcast_reduce(egraph, max_bcast, index_space, ReduceOp::Max, score)?;
    Some(ExpMaxInfo {
        score: egraph.find(score),
        axis,
    })
}

fn detect_normalized_weight(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    index_space: &Shape,
    num_inputs: u32,
    children: &[Id],
) -> Option<NormalizedWeightInfo> {
    if num_inputs != 2 || children.len() < 3 {
        return None;
    }
    let numerator = children[0];
    let denominator = children[1];
    let body = *children.last()?;
    if !is_param_binop(egraph, body, BinaryOp::Div, 0, 1) {
        return None;
    }
    let axis = match_broadcast_reduce(egraph, denominator, index_space, ReduceOp::Add, numerator)?;
    Some(NormalizedWeightInfo {
        numerator: egraph.find(numerator),
        axis,
        exp_max: egraph[egraph.find(numerator)].data.exp_max,
    })
}

fn match_broadcast_reduce(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    broadcast: Id,
    index_space: &Shape,
    op: ReduceOp,
    source: Id,
) -> Option<u32> {
    let canonical = egraph.find(broadcast);
    egraph[canonical].iter().find_map(|node| {
        let TensorIr::HighLevel(HighLevelNode::Restride {
            new_shape,
            strides,
            expr,
            ..
        }) = node
        else {
            return None;
        };
        if new_shape != index_space {
            return None;
        }
        let zero_stride_axis = strides
            .0
            .iter()
            .enumerate()
            .find_map(|(axis, stride)| (stride.as_const() == Some(0)).then_some(axis as u32))?;
        egraph[egraph.find(*expr)].iter().find_map(|node| {
            let TensorIr::HighLevel(HighLevelNode::Reduce {
                axis,
                op: node_op,
                expr: node_source,
            }) = node
            else {
                return None;
            };
            (*axis == zero_stride_axis
                && *node_op == op
                && egraph.find(*node_source) == egraph.find(source))
            .then_some(*axis)
        })
    })
}

fn is_exp_of_param_sub(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    lhs_param: u32,
    rhs_param: u32,
) -> bool {
    let canonical = egraph.find(id);
    egraph[canonical].iter().any(|node| {
        let TensorIr::UnOp(UnaryOp::Exp, inner) = node else {
            return false;
        };
        is_param_binop(egraph, *inner, BinaryOp::Sub, lhs_param, rhs_param)
    })
}

fn is_param_binop(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    op: BinaryOp,
    lhs_param: u32,
    rhs_param: u32,
) -> bool {
    let canonical = egraph.find(id);
    egraph[canonical].iter().any(|node| {
        matches!(
            node,
            TensorIr::BinOp(node_op, [lhs, rhs])
                if *node_op == op
                    && is_param(egraph, *lhs, lhs_param)
                    && is_param(egraph, *rhs, rhs_param)
        )
    })
}

fn is_param(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, param: u32) -> bool {
    let canonical = egraph.find(id);
    egraph[canonical]
        .iter()
        .any(|node| matches!(node, TensorIr::HighLevel(HighLevelNode::Param(p)) if *p == param))
}

const fn dispatch_list_id(node: &DispatchNode) -> Option<Id> {
    match node {
        DispatchNode::Dispatch { children_list, .. } | DispatchNode::Pack { children_list } => {
            Some(*children_list)
        }
        DispatchNode::Seq(list_id) | DispatchNode::Pipeline(list_id) => Some(*list_id),
        DispatchNode::Token | DispatchNode::Extract { .. } => None,
    }
}

fn make_dispatch_data(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &DispatchNode,
) -> TensorData {
    if let Some(list_id) = dispatch_list_id(node) {
        let mut data = list_tensor_data(egraph, list_id);
        // Surface shape facts for the extractor cost model. Only full
        // `Dispatch` nodes (not Pack/Seq/Pipeline) carry shape semantics.
        if let DispatchNode::Dispatch {
            children_list,
            num_inputs,
            ..
        } = node
        {
            data.dispatch_shape = Some(make_dispatch_shape(egraph, *children_list, *num_inputs));
            if let Some(children) = crate::language::try_extract_list(egraph, *children_list) {
                let first_value_index = (*num_inputs as usize).min(children.len());
                if let Some(value_id) = children.get(first_value_index) {
                    data.dtype = egraph[*value_id].data.dtype;
                }
            }
        }
        return data;
    }

    match node {
        DispatchNode::Token => empty_tensor_data(),
        DispatchNode::Extract { tuple, .. } => copy_child_tensor_data(&egraph[*tuple].data),
        DispatchNode::Dispatch { .. }
        | DispatchNode::Pack { .. }
        | DispatchNode::Seq(_)
        | DispatchNode::Pipeline(_) => {
            unreachable!("list-like dispatch nodes handled above")
        }
    }
}

/// Derive `DispatchShapeFacts` for a `Dispatch` e-node from its children list.
///
/// When the analysis runs mid-construction — e.g. `build_dispatch_program_from_extracted`
/// adding a Dispatch whose `children_list` has been merged with non-list nodes by
/// downstream rules — the list may not be walkable. In that case we surface default
/// shape facts and let the caller reject the malformed dispatch, rather than
/// panicking the whole build.
fn make_dispatch_shape(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    children_list: Id,
    num_inputs: u32,
) -> DispatchShapeFacts {
    let Some(children) = crate::language::try_extract_list(egraph, children_list) else {
        return DispatchShapeFacts::default();
    };
    let inputs = usize::try_from(num_inputs).unwrap_or(0).min(children.len());
    let body_children = &children[inputs..];
    // Output pairs are (value, addr); reg_block is the number of value slots.
    let reg_block = u8::try_from((body_children.len() / 2).max(1)).unwrap_or(u8::MAX);

    // Consult any one value subtree for cooperative/tg-load reachability;
    // well-formed dispatches have all value children sharing body structure,
    // so the first one is representative.
    let cooperative = body_children
        .iter()
        .step_by(2)
        .any(|value| egraph[*value].data.contains_reduce_simd);
    let has_tg_load = body_children
        .iter()
        .step_by(2)
        .any(|value| egraph[*value].data.contains_tg_load);

    let tile_k = body_children
        .iter()
        .step_by(2)
        .find_map(|value| outer_theta_count(egraph, *value));
    let dtype_bytes = egraph[children_list].data.dtype_bytes.unwrap_or(4).max(1);
    // Conservative TG footprint: tile_k × dtype_bytes when the body touches
    // TG memory. Zero when no TG loads exist, which keeps plain dispatches at
    // zero footprint in cost accounting.
    let tg_bytes = if has_tg_load {
        tile_k.unwrap_or(0).saturating_mul(dtype_bytes)
    } else {
        0
    };

    DispatchShapeFacts {
        cooperative,
        reg_block,
        tile_k,
        tg_bytes,
    }
}

fn make_simd_child_data(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    children: &[Id],
    dep: DepSet,
) -> TensorData {
    TensorData {
        dep,
        var_dep: child_var_dep_union(egraph, children),
        free_var_dep: child_free_var_dep_union(egraph, children),
        ..Default::default()
    }
}

const fn simd_child_slice(node: &SimdNode) -> Option<&[Id]> {
    match node {
        SimdNode::Load { children, .. } => Some(children),
        SimdNode::Store { children, .. } => Some(children),
        SimdNode::StoreIf { children, .. } => Some(children),
        SimdNode::Var(_)
        | SimdNode::Shuffle(_)
        | SimdNode::ReduceSimd { .. }
        | SimdNode::Theta { .. }
        | SimdNode::Barrier { .. } => None,
    }
}

fn make_simd_data(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &SimdNode) -> TensorData {
    if let Some(children) = simd_child_slice(node) {
        return make_simd_child_data(egraph, children, child_dep_union(egraph, children));
    }

    match node {
        SimdNode::Var(var) => {
            // Dispatch-bound thread refs carry a per-level `dep` so
            // stride-zero / shuffle promotion analyses see the dependency.
            let dep = match var {
                VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot,
                    ..
                } => index_level_from_slot(*slot).map_or(DepSet::EMPTY, DepSet::from_index_level),
                _ => DepSet::EMPTY,
            };
            TensorData {
                dep,
                var_dep: VarDepSet::singleton(*var),
                free_var_dep: VarDepSet::singleton(*var),
                ..Default::default()
            }
        }
        SimdNode::Shuffle(ids) => make_simd_child_data(egraph, ids, DepSet::LANE),
        SimdNode::ReduceSimd { src, .. } | SimdNode::Barrier { state: src, .. } => {
            copy_child_tensor_data(&egraph[*src].data)
        }
        SimdNode::Theta {
            children: [init, count, update],
            ..
        } => {
            // init/count execute outside the binder; update executes inside,
            // so the body's free `Bound` refs ascend (drop depth-0, decrement
            // higher depths) before joining init/count.
            let init_data = &egraph[*init].data;
            let count_data = &egraph[*count].data;
            let update_data = &egraph[*update].data;
            let dep = init_data.dep.union(count_data.dep).union(update_data.dep);
            // Both `var_dep` and `free_var_dep` record *free* Bound refs
            // relative to this node. Ascent converts the body's depth-0 refs
            // (bound here) into nothing, and deeper refs get decremented.
            let var_dep = init_data
                .var_dep
                .union(&count_data.var_dep)
                .union(&update_data.var_dep.ascend_theta());
            let free_var_dep = init_data
                .free_var_dep
                .union(&count_data.free_var_dep)
                .union(&update_data.free_var_dep.ascend_theta());
            TensorData {
                dep,
                dtype: init_data.dtype,
                var_dep,
                free_var_dep,
                ..Default::default()
            }
        }
        SimdNode::Load { .. } | SimdNode::Store { .. } | SimdNode::StoreIf { .. } => {
            unreachable!("simd child nodes handled above")
        }
    }
}

impl Analysis<TensorIr> for TensorAnalysis {
    type Data = TensorData;

    fn make(egraph: &mut EGraph<TensorIr, Self>, enode: &TensorIr) -> Self::Data {
        let mut data = match enode {
            TensorIr::HighLevel(node) => make_high_level_data(egraph, node),
            TensorIr::Dispatch(node) => make_dispatch_data(egraph, node),
            TensorIr::Nil => TensorData::default(),
            TensorIr::Cons([head, tail]) => {
                let h = &egraph[*head].data;
                let t = &egraph[*tail].data;
                TensorData {
                    dep: h.dep.union(t.dep),
                    var_dep: h.var_dep.clone().union(&t.var_dep),
                    free_var_dep: h.free_var_dep.clone().union(&t.free_var_dep),
                    ..Default::default()
                }
            }
            TensorIr::Const(v) => TensorData {
                dtype: Some(match v {
                    ScalarValue::F16(_) => DType::F16,
                    ScalarValue::F32(_) => DType::F32,
                    ScalarValue::I32(_) => DType::I32,
                    ScalarValue::U32(_) => DType::U32,
                    ScalarValue::Bool(_) => DType::Bool,
                }),
                constant: Some(v.clone()),
                address_profile: match v {
                    ScalarValue::U32(u) => Some(AddressProfile::from_const(*u)),
                    _ => None,
                },
                ..empty_tensor_data()
            },
            TensorIr::ShapeParam(_) => TensorData {
                dtype: Some(DType::U32),
                ..empty_tensor_data()
            },
            TensorIr::BinOp(name, args) => {
                let dep = child_dep_union(egraph, args);
                let constant = try_fold_scalar_op(*name, args, egraph);
                let address_profile = combine_address_profile(*name, args, egraph);
                TensorData {
                    dep,
                    dtype: match name {
                        BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge
                        | BinaryOp::Eq
                        | BinaryOp::Neq => Some(DType::Bool),
                        _ => args.first().and_then(|arg| egraph[*arg].data.dtype),
                    },
                    constant,
                    var_dep: child_var_dep_union(egraph, args),
                    free_var_dep: child_free_var_dep_union(egraph, args),
                    address_profile,
                    has_commutative_binop: name.is_commutative(),
                    has_associative_binop: name.is_associative(),
                    ..Default::default()
                }
            }
            TensorIr::UnOp(name, id) => {
                let mut data = copy_child_tensor_data(&egraph[*id].data);
                data.dtype = unary_output_dtype(*name, egraph[*id].data.dtype);
                data
            }
            TensorIr::TernOp(name, args) => TensorData {
                dep: child_dep_union(egraph, args),
                dtype: match name {
                    TernaryOp::Fma => egraph[args[0]].data.dtype,
                    TernaryOp::Select => egraph[args[1]].data.dtype,
                },
                var_dep: child_var_dep_union(egraph, args),
                free_var_dep: child_free_var_dep_union(egraph, args),
                ..Default::default()
            },
            TensorIr::Simd(node) => make_simd_data(egraph, node),
        };
        // Derived facts: keep `dtype_bytes` in lockstep with `dtype` so
        // individual branches don't have to thread it explicitly.
        data.dtype_bytes = data.dtype.map(DType::byte_size);
        // Structural presence flags. Propagated bottom-up so rules can
        // gate O(1) on `contains_theta` / `contains_pack` /
        // `contains_reduce_simd` / `contains_tg_load` instead of walking
        // subtrees.
        let child_ids = enode.children();
        let children_contain_theta = child_contains_theta(egraph, child_ids);
        let children_contain_pack = child_contains_pack(egraph, child_ids);
        let children_contain_reduce_simd = child_contains_reduce_simd(egraph, child_ids);
        let children_contain_tg_load = child_contains_tg_load(egraph, child_ids);
        data.contains_theta =
            children_contain_theta || matches!(enode, TensorIr::Simd(SimdNode::Theta { .. }));
        data.contains_pack =
            children_contain_pack || matches!(enode, TensorIr::Dispatch(DispatchNode::Pack { .. }));
        data.contains_reduce_simd = children_contain_reduce_simd
            || matches!(enode, TensorIr::Simd(SimdNode::ReduceSimd { .. }));
        data.contains_tg_load = children_contain_tg_load || node_is_tg_memory(enode);
        data
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let mut changed = false;

        // Union dep sets
        let new_dep = a.dep.union(b.dep);
        if new_dep != a.dep {
            a.dep = new_dep;
            changed = true;
        }

        // Merge shapes: keep if both agree, take known over unknown
        if let (None, Some(_)) = (&a.shape, &b.shape) {
            a.shape = b.shape;
            changed = true;
        }

        // Merge dtypes
        if let (None, Some(_)) = (&a.dtype, &b.dtype) {
            a.dtype = b.dtype;
            a.dtype_bytes = a.dtype.map(DType::byte_size);
            changed = true;
        }

        // Merge stride profiles: take known over unknown; clear if they
        // disagree (eclass members describe layouts the consumer can't
        // canonicalize without picking one).
        match (&a.stride_profile, &b.stride_profile) {
            (None, Some(_)) => {
                a.stride_profile = b.stride_profile;
                changed = true;
            }
            (Some(av), Some(bv)) if av != bv => {
                a.stride_profile = None;
                changed = true;
            }
            _ => {}
        }

        // Merge constants: take known over unknown
        if let (None, Some(_)) = (&a.constant, &b.constant) {
            a.constant = b.constant;
            changed = true;
        }

        // Merge var_dep: union
        let new_var_dep = a.var_dep.union(&b.var_dep);
        if new_var_dep != a.var_dep {
            a.var_dep = new_var_dep;
            changed = true;
        }

        let new_free_var_dep = a.free_var_dep.union(&b.free_var_dep);
        if new_free_var_dep != a.free_var_dep {
            a.free_var_dep = new_free_var_dep;
            changed = true;
        }

        // Merge address profiles: take known over unknown; clear if eclass
        // members disagree (consumer can re-derive from a chosen extraction).
        match (a.address_profile, b.address_profile) {
            (None, Some(_)) => {
                a.address_profile = b.address_profile;
                changed = true;
            }
            (Some(av), Some(bv)) if av != bv => {
                a.address_profile = None;
                changed = true;
            }
            _ => {}
        }

        let composite_lowerable = a.composite_dispatch.lowerable || b.composite_dispatch.lowerable;
        if composite_lowerable != a.composite_dispatch.lowerable {
            a.composite_dispatch.lowerable = composite_lowerable;
            changed = true;
        }

        // Structural presence flags: monotonic OR. Once an e-class is
        // known to contain a Theta/Pack/ReduceSimd/TgLoad, unioning with
        // a form that doesn't doesn't change that fact.
        if b.contains_theta && !a.contains_theta {
            a.contains_theta = true;
            changed = true;
        }
        if b.contains_pack && !a.contains_pack {
            a.contains_pack = true;
            changed = true;
        }
        if b.contains_reduce_simd && !a.contains_reduce_simd {
            a.contains_reduce_simd = true;
            changed = true;
        }
        if b.contains_tg_load && !a.contains_tg_load {
            a.contains_tg_load = true;
            changed = true;
        }
        // Dispatch shape: keep the first-populated variant; clear if
        // e-class members disagree so the extractor doesn't get a
        // misleading signal. The dispatch rewrite rules unify Dispatch
        // variants with *different* shapes into one e-class, so "cleared
        // when disagreeing" correctly flags the multi-variant case.
        match (a.dispatch_shape, b.dispatch_shape) {
            (None, Some(_)) => {
                a.dispatch_shape = b.dispatch_shape;
                changed = true;
            }
            (Some(av), Some(bv)) if av != bv => {
                a.dispatch_shape = None;
                changed = true;
            }
            _ => {}
        }
        // Reduction depth: monotonic max. We represent the highest
        // nested-Reduce depth any member of the e-class has.
        if b.reduction_depth > a.reduction_depth {
            a.reduction_depth = b.reduction_depth;
            changed = true;
        }
        if b.has_commutative_binop && !a.has_commutative_binop {
            a.has_commutative_binop = true;
            changed = true;
        }
        if b.has_associative_binop && !a.has_associative_binop {
            a.has_associative_binop = true;
            changed = true;
        }
        if a.exp_max.is_none() && b.exp_max.is_some() {
            a.exp_max = b.exp_max;
            changed = true;
        }
        if a.normalized_weight.is_none() && b.normalized_weight.is_some() {
            a.normalized_weight = b.normalized_weight;
            changed = true;
        }

        // DidMerge(a_changed, b_changed) - we report conservatively
        DidMerge(changed, changed)
    }
}

/// Combine child `AddressProfile`s for a `BinOp`. Only `Add` and `Mul` carry
/// useful information; other ops produce no profile.
fn combine_address_profile(
    op: BinaryOp,
    args: &[Id],
    egraph: &EGraph<TensorIr, TensorAnalysis>,
) -> Option<AddressProfile> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Mul) || args.len() != 2 {
        return None;
    }
    let lhs = egraph[args[0]].data.address_profile?;
    let rhs = egraph[args[1]].data.address_profile?;
    Some(match op {
        BinaryOp::Add => AddressProfile::saturating_sum(lhs, rhs),
        BinaryOp::Mul => AddressProfile::saturating_product(lhs, rhs),
        _ => unreachable!(),
    })
}

/// Try to constant-fold a scalar op.
fn try_fold_scalar_op(
    name: BinaryOp,
    args: &[Id],
    egraph: &EGraph<TensorIr, TensorAnalysis>,
) -> Option<ScalarValue> {
    let consts: Vec<&ScalarValue> = args
        .iter()
        .filter_map(|id| egraph[*id].data.constant.as_ref())
        .collect();
    if consts.len() != args.len() {
        return None;
    }
    fold_bin_op(name, &consts)
}

/// Fold a binary operation over constant arguments.
fn fold_bin_op(op: BinaryOp, args: &[&ScalarValue]) -> Option<ScalarValue> {
    use ordered_float::OrderedFloat;

    let [a, b] = args else {
        return None;
    };

    match (op, a, b) {
        // f16 (tracked as f32 values rounded by backend literal emission)
        (BinaryOp::Add, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0 + b.0)))
        }
        (BinaryOp::Sub, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0 - b.0)))
        }
        (BinaryOp::Mul, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0 * b.0)))
        }
        (BinaryOp::Div, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0 / b.0)))
        }
        (BinaryOp::Pow, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0.powf(b.0))))
        }
        (BinaryOp::Max, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0.max(b.0))))
        }
        (BinaryOp::Min, ScalarValue::F16(a), ScalarValue::F16(b)) => {
            Some(ScalarValue::F16(OrderedFloat(a.0.min(b.0))))
        }

        // i32
        (BinaryOp::Add, ScalarValue::I32(a), ScalarValue::I32(b)) => Some(ScalarValue::I32(a + b)),
        (BinaryOp::Sub, ScalarValue::I32(a), ScalarValue::I32(b)) => Some(ScalarValue::I32(a - b)),
        (BinaryOp::Mul, ScalarValue::I32(a), ScalarValue::I32(b)) => Some(ScalarValue::I32(a * b)),
        (BinaryOp::Div, ScalarValue::I32(a), ScalarValue::I32(b)) if *b != 0 => {
            Some(ScalarValue::I32(a / b))
        }
        (BinaryOp::Mod, ScalarValue::I32(a), ScalarValue::I32(b)) if *b != 0 => {
            Some(ScalarValue::I32(a % b))
        }
        (BinaryOp::Max, ScalarValue::I32(a), ScalarValue::I32(b)) => {
            Some(ScalarValue::I32(*a.max(b)))
        }
        (BinaryOp::Min, ScalarValue::I32(a), ScalarValue::I32(b)) => {
            Some(ScalarValue::I32(*a.min(b)))
        }

        // u32
        (BinaryOp::Add, ScalarValue::U32(a), ScalarValue::U32(b)) => {
            Some(ScalarValue::U32(a.wrapping_add(*b)))
        }
        (BinaryOp::Sub, ScalarValue::U32(a), ScalarValue::U32(b)) => {
            Some(ScalarValue::U32(a.wrapping_sub(*b)))
        }
        (BinaryOp::Mul, ScalarValue::U32(a), ScalarValue::U32(b)) => {
            Some(ScalarValue::U32(a.wrapping_mul(*b)))
        }
        (BinaryOp::Div, ScalarValue::U32(a), ScalarValue::U32(b)) if *b != 0 => {
            Some(ScalarValue::U32(a / b))
        }
        (BinaryOp::Mod, ScalarValue::U32(a), ScalarValue::U32(b)) if *b != 0 => {
            Some(ScalarValue::U32(a % b))
        }

        // f32
        (BinaryOp::Add, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0 + b.0)))
        }
        (BinaryOp::Sub, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0 - b.0)))
        }
        (BinaryOp::Mul, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0 * b.0)))
        }
        (BinaryOp::Div, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0 / b.0)))
        }
        (BinaryOp::Pow, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0.powf(b.0))))
        }
        (BinaryOp::Max, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0.max(b.0))))
        }
        (BinaryOp::Min, ScalarValue::F32(a), ScalarValue::F32(b)) => {
            Some(ScalarValue::F32(OrderedFloat(a.0.min(b.0))))
        }

        _ => None,
    }
}
