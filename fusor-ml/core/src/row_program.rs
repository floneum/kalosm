//! Generic lowering for fused row programs.
//!
//! A row program is a cluster of same-axis reductions and elementwise
//! expressions over one axis of an index space. Each phase is either a
//! scalar reduction (folding an expression over the axis into a per-row
//! value visible to later phases) or a staged per-element value (evaluated
//! once at each axis position, optionally folding a private inner axis — an
//! inline dot product). The output either maps one element per index-space
//! position (softmax, RMS norm) or folds the axis once more with a free
//! output dimension appended to the row shape (attention's `Σ p·v`).
//!
//! One workgroup runs one row. Map-style programs stride the axis in chunks;
//! programs with element phases or a reducing output pin one lane per axis
//! position (the axis must fit one workgroup), which lets the score of a
//! decode-attention row live in a register across every phase. The axis
//! length may be dynamic: kernels compile per block bucket and read the
//! active length from a trailing u32 params input, so a growing KV cache
//! stays in one cached kernel.

use std::{any::TypeId, hash::Hash};

use fusor_tile_ir as tile_ir;
use rustc_hash::FxHasher;
use tile_ir::{
    ElementType, ScalarElement,
    tile::{Mask, Tile, WorkgroupTile},
};

use crate::{
    compute_graph::NodeIndex,
    mir::{
        inputs::MirValue,
        kernel_backend::{self, DirectKernel},
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_direct::{
        ValueTile, apply_unary_function_chain, declare_value, eval_nary_expr, layout_index,
        output_dims_from_flat, tile_u32,
    },
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryScalar, UnaryFunctionChain},
    reduce::{ReduceFunction, ReduceOp, ReduceOperation, max_fn, sum_fn},
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::{MaybeQData, distribute_workgroups},
};

/// One reduction phase: `expression` (over the external inputs and the
/// slots of earlier phases) folded along the row axis, then `post_chain`
/// applied to the combined value once per row.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) struct RowReduce {
    pub(crate) expression: NaryExpr,
    pub(crate) function: ReduceFunction,
    pub(crate) post_chain: UnaryFunctionChain,
}

/// The private inner axis of an element phase: the expression is evaluated
/// `len` times with the fold coordinate as `DimIndex(rank)` and accumulated
/// with `function` — an inline dot product per axis position.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) struct RowFold {
    pub(crate) len: usize,
    pub(crate) function: ReduceFunction,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) enum RowOutput {
    /// One output element per index-space position; output shape == `shape`.
    Map(NaryExpr),
    /// One output element per row; output shape == row dims.
    Scalar(NaryExpr),
    /// Fold the axis once more with one free output dimension appended to
    /// the row shape: `combine` sees the free coordinate as `DimIndex(rank)`
    /// and element-phase slots at the folded axis position. Output shape =
    /// row dims ++ `[free_dim]`.
    Reduce {
        combine: NaryExpr,
        function: ReduceFunction,
        free_dim: usize,
    },
}

/// Dynamic-axis configuration: the kernel is compiled for the `block`
/// capacity bucket and reads the active axis length from a trailing u32
/// params input, so per-token axis growth (the KV cache) reuses one kernel.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) struct DynamicAxis {
    /// Workgroup size and per-tile capacity; axis lengths beyond it stream
    /// through the online tile loop.
    pub(crate) block: u32,
    /// For each input, the input dimension whose extent tracks the axis
    /// (the KV dim of K/V) — normalized out of the kernel cache key.
    pub(crate) input_axis_dims: Vec<Option<usize>>,
    /// A row dimension whose coordinate bounds the active axis length:
    /// `effective_len = min(axis_len, coord + 1)` — causal attention skips
    /// every tile past the query position.
    pub(crate) axis_bound_dim: Option<usize>,
}

/// One ordered row-program step. All non-output steps produce slots for later
/// expressions; the final step must be [`RowStep::Output`] and defines the
/// tensor write contract.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) enum RowStep {
    /// Fold over the row axis into a per-row scalar slot.
    Reduce(RowReduce),
    /// A staged per-element value: evaluated once at each axis position,
    /// with one lane pinned per position (the axis must fit the workgroup).
    Element {
        expression: NaryExpr,
        fold: Option<RowFold>,
        datatype: DataTypeEnum,
    },
    Output(RowOutput),
}

/// The algebraic shape the online-streaming lowering requires: a staged
/// score element, a max step over a scaled score `e`, an exp-sum step
/// shifted by that max, a staged probability element, and a linear combine.
/// The exp shift is what licenses streaming — rescaling the running sum and
/// accumulators by `exp(M_old − M_new)` keeps them exact across tiles.
struct OnlineSoftmax<'a> {
    /// Element step: the raw per-position score (slot `n`).
    score: (&'a NaryExpr, &'a Option<RowFold>),
    /// The scaled/masked score expression `e` over slot `n` (max step body).
    scaled: &'a NaryExpr,
    max_identity: NaryScalar,
    /// The per-position weight in `combine = p · weight`.
    weight: &'a NaryExpr,
}

fn slot_expr(input_count: usize, phase: usize) -> NaryExpr {
    NaryExpr::IndexedInput {
        input_idx: input_count + phase,
        indices: vec![],
    }
}

fn unary_op_child<'a>(expr: &'a NaryExpr, op: &NaryOp) -> Option<&'a NaryExpr> {
    match expr {
        NaryExpr::Op { children, function } if function.op == *op && children.len() == 1 => {
            Some(&children[0])
        }
        _ => None,
    }
}

fn binary_op_children<'a>(expr: &'a NaryExpr, op: &NaryOp) -> Option<(&'a NaryExpr, &'a NaryExpr)> {
    match expr {
        NaryExpr::Op { children, function } if function.op == *op && children.len() == 2 => {
            Some((&children[0], &children[1]))
        }
        _ => None,
    }
}

/// Match the terminal-output online-softmax shape (see [`OnlineSoftmax`]). The
/// attention constructor builds exactly this; the lowering re-derives it so
/// the step expressions stay the single source of truth.
fn match_online_softmax<'a>(steps: &'a [RowStep], input_count: usize) -> Option<OnlineSoftmax<'a>> {
    let [
        RowStep::Element {
            expression: score,
            fold: score_fold,
            ..
        },
        RowStep::Reduce(max_phase),
        RowStep::Reduce(sum_phase),
        RowStep::Element {
            expression: prob,
            fold: None,
            ..
        },
        RowStep::Output(RowOutput::Reduce {
            combine, function, ..
        }),
    ] = steps
    else {
        return None;
    };
    if max_phase.function.op != ReduceOp::Max
        || sum_phase.function.op != ReduceOp::Sum
        || !max_phase.post_chain.functions.is_empty()
        || !sum_phase.post_chain.functions.is_empty()
    {
        return None;
    }
    let scaled = &max_phase.expression;
    // sum phase: exp(e − m)
    let (shift_lhs, shift_rhs) = binary_op_children(
        unary_op_child(&sum_phase.expression, &NaryOp::Exp)?,
        &NaryOp::Sub,
    )?;
    if shift_lhs != scaled || *shift_rhs != slot_expr(input_count, 1) {
        return None;
    }
    // prob element: exp(e − m) / l
    let (num, denom) = binary_op_children(prob, &NaryOp::Div)?;
    if num != &sum_phase.expression || *denom != slot_expr(input_count, 2) {
        return None;
    }
    // combine: p · weight, summed
    if function.op != ReduceOp::Sum {
        return None;
    }
    let (p_ref, weight) = binary_op_children(combine, &NaryOp::Mul)?;
    if *p_ref != slot_expr(input_count, 3) {
        return None;
    }
    // The weight and scaled score may only reference tensor inputs, dims,
    // and the score slot — never later phase slots (those are consumed by
    // the streaming structure itself).
    for later in 1..4 {
        if weight.uses_input(input_count + later) || scaled.uses_input(input_count + later) {
            return None;
        }
    }
    if weight.uses_input(input_count) {
        return None;
    }
    Some(OnlineSoftmax {
        score: (score, score_fold),
        scaled,
        max_identity: max_phase.function.initial_value,
        weight,
    })
}

/// Slot convention for every expression in the program: indices below
/// `inputs.len()` are tensor reads; index `inputs.len() + p` is step `p`'s
/// value for non-output steps (a per-row scalar for reduce steps, a
/// per-element value for element steps). The final output step never produces
/// a reusable slot.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowProgramOperation {
    pub(crate) inputs: Vec<NodeIndex>,
    /// The full row-parallel index space (including the axis).
    pub(crate) shape: Box<[usize]>,
    pub(crate) axis: usize,
    pub(crate) steps: Vec<RowStep>,
    pub(crate) output_datatype: DataTypeEnum,
    /// `Some` exactly when the program pins one lane per axis position
    /// (element steps / reducing output); `None` for chunked map programs.
    pub(crate) dynamic_axis: Option<DynamicAxis>,
}

impl RowProgramOperation {
    pub(crate) fn from_reduce(reduce: &ReduceOperation) -> Self {
        let scalar_slot = slot_expr(reduce.inputs.len(), 0);
        Self {
            inputs: reduce.inputs.clone(),
            shape: reduce.shape.clone(),
            axis: reduce.axis,
            steps: vec![
                RowStep::Reduce(RowReduce {
                    expression: reduce.expression.clone(),
                    function: reduce.function.clone(),
                    post_chain: reduce.post_element_wise.clone(),
                }),
                RowStep::Output(RowOutput::Scalar(scalar_slot)),
            ],
            output_datatype: reduce.out_datatype(),
            dynamic_axis: None,
        }
    }

    fn output_step(&self) -> &RowOutput {
        match self.steps.last() {
            Some(RowStep::Output(output)) => output,
            _ => panic!("row program must end with an output step"),
        }
    }

    fn phase_steps(&self) -> &[RowStep] {
        match self.steps.last() {
            Some(RowStep::Output(_)) => &self.steps[..self.steps.len() - 1],
            _ => panic!("row program must end with an output step"),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.shape
            .iter()
            .enumerate()
            .filter_map(|(dim, &size)| (dim != self.axis).then_some(size))
            .product()
    }

    fn row_shape(&self) -> Vec<usize> {
        self.shape
            .iter()
            .enumerate()
            .filter_map(|(dim, &size)| (dim != self.axis).then_some(size))
            .collect()
    }

    pub(crate) fn out_shape(&self) -> Vec<usize> {
        match self.output_step() {
            RowOutput::Map(_) => self.shape.to_vec(),
            RowOutput::Scalar(_) => self.row_shape(),
            RowOutput::Reduce { free_dim, .. } => {
                let mut shape = self.row_shape();
                shape.push(*free_dim);
                shape
            }
        }
    }

    pub(crate) fn phase_count(&self) -> usize {
        self.phase_steps().len()
    }

    fn block(&self, device: &crate::Device) -> u32 {
        match &self.dynamic_axis {
            Some(dynamic) => dynamic.block,
            None => {
                let policy = device.dispatch_policy();
                let max_block = policy.preferred_workgroup_lanes();
                // Size the workgroup to the axis: a k=64
                // softmax runs 64-lane workgroups whose whole-block reduction
                // is subgroup-accelerated, instead of packing four rows into
                // a 256-lane workgroup whose per-row reduction walks the
                // shared-memory tree one barrier per level.
                let k = u32::try_from(self.shape[self.axis]).unwrap_or(max_block);
                k.max(1)
                    .next_power_of_two()
                    .clamp(policy.min_reduction_lanes().min(max_block), max_block)
            }
        }
    }

    /// Whether this program is a plain chunked-map program the horizontal
    /// merge pass can host: static axis, reduce-only phases, and a map or
    /// per-row-scalar output (no free-dim reduce, no staged elements).
    pub(crate) fn mergeable_chunked_map(&self) -> bool {
        self.dynamic_axis.is_none()
            && self
                .phase_steps()
                .iter()
                .all(|step| matches!(step, RowStep::Reduce(_)))
            && matches!(self.output_step(), RowOutput::Map(_) | RowOutput::Scalar(_))
    }

    fn uses_custom_indexing_for_input(&self, input_idx: usize) -> bool {
        let output_uses_custom_indexing = match self.output_step() {
            RowOutput::Map(expr) | RowOutput::Scalar(expr) => {
                expr.uses_custom_indexing_for_input(input_idx)
            }
            RowOutput::Reduce { combine, .. } => combine.uses_custom_indexing_for_input(input_idx),
        };
        output_uses_custom_indexing
            || self.phase_steps().iter().any(|step| match step {
                RowStep::Reduce(reduce) => {
                    reduce.expression.uses_custom_indexing_for_input(input_idx)
                }
                RowStep::Element { expression, .. } => {
                    expression.uses_custom_indexing_for_input(input_idx)
                }
                RowStep::Output(_) => false,
            })
    }
}

impl Operation for RowProgramOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        // With a dynamic axis the kernel is bucketed by `block`; the actual
        // axis extent rides in the params input and must stay out of the
        // key, or every generated token would recompile.
        match &self.dynamic_axis {
            Some(dynamic) => {
                1u8.hash(state);
                dynamic.hash(state);
                for (dim, size) in self.shape.iter().enumerate() {
                    if dim != self.axis {
                        size.hash(state);
                    }
                }
            }
            None => {
                0u8.hash(state);
                self.shape.hash(state);
            }
        }
        self.axis.hash(state);
        self.steps.hash(state);
        self.output_datatype.hash(state);
    }

    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(self.block(device)));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let output: TensorData = inputs.last().unwrap().as_tensor().unwrap().clone();
        distribute_workgroups(
            self.rows() as u32,
            output
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for input in &self.inputs {
            f(*input);
        }
    }

    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        for input in &mut self.inputs {
            f(input);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let mut mir_inputs: Vec<MirValue> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                // Custom-indexed inputs need dense tensor addressing; the
                // quantized matrix path only supports plain row/col reads.
                if self.uses_custom_indexing_for_input(i)
                    && let Some(cached) = nodes.get_result(*idx)
                {
                    return cached.into();
                }
                nodes.get_result_or_qmatrix(*idx).unwrap().into()
            })
            .collect();
        let device = match &mir_inputs[0] {
            MirValue::Tensor(tensor) => tensor.device().clone(),
            MirValue::QMatrix(matrix) => matrix.device().clone(),
            _ => unreachable!("row program inputs are tensors or quantized matrices"),
        };
        if self.dynamic_axis.is_some() {
            mir_inputs
                .push(TensorData::new_splat(&device, &[1], self.shape[self.axis] as u32).into());
        }
        let output_tensor =
            TensorData::new_for_shape(&device, &self.out_shape(), self.output_datatype);
        mir_inputs.push(output_tensor.into());
        mir_inputs
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        build_row_program_kernel(self, graph, workgroup_shape, inputs)
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs.last().unwrap().clone()
    }

    fn name(&self) -> String {
        format!(
            "row_program_{}s_{}",
            self.steps.len(),
            self.shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

struct RowProgramKernelVariant;

/// The shared cache-key recipe, except inputs with a dynamic-axis dimension
/// hash a normalized extent: the KV cache's strides and offset are stable
/// across tokens, so bucketing out the growing dim keeps one kernel per
/// block size.
fn row_program_cache_key(
    operation: &RowProgramOperation,
    workgroup_shape: &WorkgroupShape,
    dispatch_size: [u32; 3],
    inputs: &[MirValue],
    role: u64,
    splits: u32,
) -> kernel_backend::KernelCacheKey {
    kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        11u64.hash(state);
        role.hash(state);
        splits.hash(state);
        kernel_backend::KernelVariantKey::of::<RowProgramKernelVariant>().hash(state);
        TypeId::of::<RowProgramOperation>().hash(state);
        operation.hash_kernel_fields(state);
        workgroup_shape.shape().hash(state);
        dispatch_size.hash(state);
        inputs.len().hash(state);
        for (i, input) in inputs.iter().enumerate() {
            let dynamic_dim = operation
                .dynamic_axis
                .as_ref()
                .and_then(|dynamic| dynamic.input_axis_dims.get(i).copied().flatten());
            std::mem::discriminant(input).hash(state);
            match input {
                MirValue::Tensor(tensor) => {
                    tensor.datatype().hash(state);
                    let layout = tensor.layout();
                    layout.offset().hash(state);
                    for (dim, (size, stride)) in layout
                        .shape()
                        .iter()
                        .zip(layout.strides().iter())
                        .enumerate()
                    {
                        if Some(dim) == dynamic_dim {
                            0usize.hash(state);
                        } else {
                            size.hash(state);
                        }
                        stride.hash(state);
                    }
                }
                MirValue::QMatrix(matrix) => {
                    matrix.datatype().hash(state);
                    matrix.storage_layout().hash(state);
                    matrix.shape().hash(state);
                }
                MirValue::Integer(value) => value.hash(state),
                MirValue::Float(value) => value.to_bits().hash(state),
            }
        }
    })
}

fn raw_tile(value: ValueTile) -> Tile {
    match value {
        ValueTile::F32(tile) | ValueTile::F16(tile) | ValueTile::U32(tile) => tile,
        ValueTile::Bool(_) => unreachable!("row program values are f32/f16/u32"),
    }
}

/// A chunked map program whose axis fits one chunk evaluates every phase and
/// the output at the same coordinates, so each distinct tensor read can be
/// loaded once per lane and kept in a register across the phases instead of
/// being re-read from storage per phase (softmax otherwise reads its input
/// once for the max, again for the exp sum, and a third time for the output).
struct StagedReads {
    /// The original `IndexedInput` leaves, evaluated once each, in order.
    probes: Vec<NaryExpr>,
    /// Phase reduce expressions rewritten over the staged probe slots
    /// (`0..probes.len()`); phase slots follow at `probes.len() + p`.
    phases: Vec<NaryExpr>,
    /// The output expression, rewritten the same way.
    output: NaryExpr,
}

/// Whether `expr` references any phase slot (directly or through an index
/// expression). Reads whose indices depend on a slot cannot be staged before
/// the phases run.
fn uses_any_slot(expr: &NaryExpr, input_count: usize) -> bool {
    match expr {
        NaryExpr::Op { children, .. } => children
            .iter()
            .any(|child| uses_any_slot(child, input_count)),
        NaryExpr::IndexedInput { input_idx, indices } => {
            *input_idx >= input_count
                || indices
                    .iter()
                    .any(|index| uses_any_slot(index, input_count))
        }
        NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => false,
    }
}

/// Collect the distinct tensor-read leaves of `expr` into `probes`,
/// returning `false` when a read cannot be staged.
fn collect_probe_reads(expr: &NaryExpr, input_count: usize, probes: &mut Vec<NaryExpr>) -> bool {
    match expr {
        NaryExpr::Op { children, .. } => children
            .iter()
            .all(|child| collect_probe_reads(child, input_count, probes)),
        NaryExpr::IndexedInput { input_idx, indices } => {
            if *input_idx >= input_count {
                // A phase-slot reference: stays a slot in the rewrite.
                return indices.is_empty();
            }
            if indices
                .iter()
                .any(|index| uses_any_slot(index, input_count))
            {
                return false;
            }
            if !probes.iter().any(|probe| probe == expr) {
                probes.push(expr.clone());
            }
            true
        }
        NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => true,
    }
}

/// Rewrite `expr` over the staged slot space: tensor-read leaves become bare
/// slot references `0..probes.len()`; phase slots shift up by `probes.len()`.
fn rewrite_staged(expr: &NaryExpr, input_count: usize, probes: &[NaryExpr]) -> NaryExpr {
    match expr {
        NaryExpr::Op { children, function } => NaryExpr::Op {
            children: children
                .iter()
                .map(|child| rewrite_staged(child, input_count, probes))
                .collect(),
            function: function.clone(),
        },
        NaryExpr::IndexedInput { input_idx, .. } if *input_idx >= input_count => {
            NaryExpr::IndexedInput {
                input_idx: probes.len() + (input_idx - input_count),
                indices: vec![],
            }
        }
        leaf @ NaryExpr::IndexedInput { .. } => {
            let slot = probes
                .iter()
                .position(|probe| probe == leaf)
                .expect("every stageable tensor read was collected");
            NaryExpr::IndexedInput {
                input_idx: slot,
                indices: vec![],
            }
        }
        other => other.clone(),
    }
}

/// Build the staged rewrite for a single-chunk program: `None` when any read
/// resists staging (slot-dependent indices, non-reduce phases, reducing
/// outputs) or there is nothing to stage.
fn stage_single_chunk_reads(
    phase_steps: &[RowStep],
    output: &RowOutput,
    input_count: usize,
) -> Option<StagedReads> {
    let mut phase_sources = Vec::with_capacity(phase_steps.len());
    for step in phase_steps {
        let RowStep::Reduce(reduce) = step else {
            return None;
        };
        phase_sources.push(&reduce.expression);
    }
    let output_expr = match output {
        RowOutput::Map(expr) | RowOutput::Scalar(expr) => expr,
        RowOutput::Reduce { .. } => return None,
    };
    let mut probes = Vec::new();
    for expr in phase_sources.iter().copied().chain([output_expr]) {
        if !collect_probe_reads(expr, input_count, &mut probes) {
            return None;
        }
    }
    if probes.is_empty() {
        return None;
    }
    Some(StagedReads {
        phases: phase_sources
            .iter()
            .map(|expr| rewrite_staged(expr, input_count, &probes))
            .collect(),
        output: rewrite_staged(output_expr, input_count, &probes),
        probes,
    })
}

/// Evaluate every probe once at `coords` and bind the values to registers.
fn stage_probe_values(
    program: &mut tile_ir::tile::TileBlock<'_>,
    probes: &[NaryExpr],
    coords: &[Tile],
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
    active: Mask,
) -> Vec<(ValueTile, DataTypeEnum)> {
    probes
        .iter()
        .map(|probe| {
            let (value, ty) =
                eval_nary_expr(program, probe, coords, storages, metas, active.clone(), &[]);
            let bound = match value {
                ValueTile::F32(tile) => ValueTile::F32(program.bind(tile)),
                ValueTile::F16(tile) => ValueTile::F16(program.bind(tile)),
                ValueTile::U32(tile) => ValueTile::U32(program.bind(tile)),
                ValueTile::Bool(mask) => ValueTile::Bool(mask),
            };
            (bound, ty)
        })
        .collect()
}

fn f32_literal(value: f32) -> Tile {
    Tile::literal(tile_ir::TileLiteral::f32(value))
}

fn tile_literal_for(value: NaryScalar, target: DataTypeEnum) -> tile_ir::TileLiteral {
    match target {
        DataTypeEnum::F32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::f32(value),
            NaryScalar::F16(value) => tile_ir::TileLiteral::f32(value.to_f32()),
            NaryScalar::U32(value) => tile_ir::TileLiteral::f32(value as f32),
        },
        DataTypeEnum::F16 => match value {
            NaryScalar::F32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value).to_bits())
            }
            NaryScalar::F16(value) => tile_ir::TileLiteral::F16(value.to_bits()),
            NaryScalar::U32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value as f32).to_bits())
            }
        },
        DataTypeEnum::U32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::U32(value as u32),
            NaryScalar::F16(value) => tile_ir::TileLiteral::U32(value.to_f32() as u32),
            NaryScalar::U32(value) => tile_ir::TileLiteral::U32(value),
        },
    }
}

fn tile_reduce_op(op: ReduceOp) -> tile_ir::TileReduceOp {
    match op {
        ReduceOp::Sum => tile_ir::TileReduceOp::Sum,
        ReduceOp::Product => tile_ir::TileReduceOp::Product,
        ReduceOp::Max => tile_ir::TileReduceOp::Max,
        ReduceOp::Min => tile_ir::TileReduceOp::Min,
    }
}

/// The fixed-subgroup proof used to accelerate whole-workgroup reductions:
/// `(token, subgroup width)` when the device reports one subgroup size.
type FixedSubgroups = Option<(tile_ir::tile::SubgroupToken, u32)>;

fn fixed_subgroups(device: &crate::Device) -> FixedSubgroups {
    device
        .subgroup_config()
        .filter(|config| config.is_fixed())
        .map(|config| (config.token(), config.max_size()))
}

/// The per-row-group reduction: subgroup-accelerated when the group spans
/// the whole workgroup on a fixed-subgroup device, the shared-memory tree
/// otherwise. A 1-wide group is the lane's own value (the k=1 bias-grad
/// sum), skipping the scratch round-trip entirely.
fn emit_group_reduce(
    program: &mut tile_ir::tile::TileBlock<'_>,
    subgroups: FixedSubgroups,
    op: tile_ir::TileReduceOp,
    group_size: u32,
    block: u32,
    value: Tile,
) -> Tile {
    if group_size == 1 {
        return program.bind(value);
    }
    match subgroups {
        Some((token, subgroup_size))
            if group_size == block && block.is_multiple_of(subgroup_size) =>
        {
            token.workgroup_reduce(program, op, subgroup_size, value)
        }
        _ => program.group_reduce(op, group_size, value),
    }
}

fn build_row_program_kernel(
    operation: &RowProgramOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let (output, producers) = inputs.split_last()?;
    let output = output.as_tensor()?.clone();
    let (params, producers) = if operation.dynamic_axis.is_some() {
        let (params, producers) = producers.split_last()?;
        (Some(params.as_tensor()?.clone()), producers)
    } else {
        (None, producers)
    };
    let values = producers
        .iter()
        .map(|input| MaybeQData::try_from(input.clone()).ok())
        .collect::<Option<Vec<_>>>()?;
    if !graph.device().f16_supported() {
        let uses_f16 = output.datatype() == DataTypeEnum::F16
            || values.iter().any(|value| match value {
                MaybeQData::Tensor(tensor) => tensor.datatype() == DataTypeEnum::F16,
                MaybeQData::QMatrix(matrix) => {
                    matches!(matrix.datatype(), fusor_gguf::GgmlType::F16)
                }
            });
        if uses_f16 {
            return None;
        }
    }

    let rows: u32 = operation.rows().try_into().ok()?;
    let k: u32 = operation.shape[operation.axis].try_into().ok()?;
    let row_shape = operation.row_shape();
    let axis = operation.axis;
    let rank = operation.shape.len();
    let block = workgroup_shape.x();
    let lanes_own_axis = operation.dynamic_axis.is_some();

    // Long axes with few rows fan out across workgroups: each split runs
    // the online body over one tile, writing its unnormalized accumulator
    // and softmax statistics to scratch; a combine kernel folds the spans
    // with the online monoid.
    let output_kind = operation.output_step().clone();
    let phase_steps = operation.phase_steps().to_vec();
    let free_dim_out = match &output_kind {
        RowOutput::Reduce { free_dim, .. } => Some(*free_dim),
        RowOutput::Map(_) | RowOutput::Scalar(_) => None,
    };
    let tiles = k.div_ceil(block);
    let splits: u32 = match free_dim_out {
        Some(free)
            if lanes_own_axis
                && graph
                    .device()
                    .dispatch_policy()
                    .should_split_for_occupancy(rows, block)
                && tiles > 1
                && tiles <= block
                && (free as u32 + 2) <= block =>
        {
            tiles
        }
        _ => 1,
    };
    // Small-axis chunked programs pack several rows per workgroup: lanes
    // split into `block / k_group` contiguous groups of `k_group` lanes,
    // each group owning one row, reduced with per-group workgroup
    // reductions. Without packing a k=64 reduce runs one row per 256-lane
    // workgroup with 75% of the lanes idle (and a k=1 bias-grad sum runs
    // one workgroup per output scalar).
    let k_group: u32 = if lanes_own_axis || splits > 1 {
        block
    } else {
        let group = k.next_power_of_two().min(block).max(1);
        if block.is_multiple_of(group) {
            group
        } else {
            block
        }
    };
    let rows_per_workgroup = block / k_group;
    let dispatch_rows = rows.div_ceil(rows_per_workgroup);

    let max_dispatch_dim = graph.device().limits().max_compute_workgroups_per_dimension;
    let dispatch_spec = crate::row_dispatch::RowDispatchSpec::distributed(
        dispatch_rows.saturating_mul(splits),
        block,
        max_dispatch_dim,
    );
    let dispatch_size = dispatch_spec.dispatch_size;
    let cache_key =
        row_program_cache_key(operation, workgroup_shape, dispatch_size, inputs, 1, splits);

    let input_count = operation.inputs.len();
    let subgroups = fixed_subgroups(&graph.device());
    // Single-chunk chunked-map programs stage each distinct tensor read once
    // per lane and evaluate every phase (and the output) from the registers
    // instead of re-reading storage per phase.
    let staged_reads = (!lanes_own_axis && k.div_ceil(block) == 1)
        .then(|| stage_single_chunk_reads(&phase_steps, &output_kind, input_count))
        .flatten();
    let axis_bound_dim = operation
        .dynamic_axis
        .as_ref()
        .and_then(|dynamic| dynamic.axis_bound_dim);
    let output_dtype = output.datatype();
    let output_value = MaybeQData::Tensor(output);
    let params_value = params.map(MaybeQData::Tensor);

    let scratch_value = (splits > 1).then(|| {
        MaybeQData::Tensor(TensorData::new_for_shape(
            &graph.device(),
            &[
                rows as usize,
                splits as usize,
                free_dim_out.expect("split row programs reduce") + 2,
            ],
            DataTypeEnum::F32,
        ))
    });
    let online_max_identity = match phase_steps.get(1) {
        Some(RowStep::Reduce(reduce)) => Some(reduce.function.initial_value),
        _ => None,
    };
    let combine = if splits > 1 {
        let scratch_b = scratch_value.clone().expect("split scratch");
        let output_b = output_value.clone();
        let row_shape_b = row_shape.clone();
        let dispatch_spec_b =
            crate::row_dispatch::RowDispatchSpec::distributed(rows, block, max_dispatch_dim);
        let dispatch_b = dispatch_spec_b.dispatch_size;
        let free = free_dim_out.expect("split row programs reduce") as u32;
        let max_identity_scalar =
            online_max_identity.expect("split row programs carry a max phase");
        let key_b =
            row_program_cache_key(operation, workgroup_shape, dispatch_b, inputs, 2, splits);
        let combine = kernel_backend::run_kernel(
            graph.device().kernel_cache(),
            format!("{}_combine", operation.name()),
            key_b,
            dispatch_b,
            move |kb| {
                let (scratch_storage, scratch_meta) = declare_value(kb, &scratch_b, false)?;
                let (output_storage, output_meta) = declare_value(kb, &output_b, true)?;
                crate::row_dispatch::emit_row_grid(
                    kb.program(),
                    dispatch_spec_b,
                    |program, ctx| {
                        let lane = ctx.lane;
                        let row_flat = ctx.row;
                        let in_bounds = ctx.active;
                        let row_dims = output_dims_from_flat(row_flat.clone(), &row_shape_b);
                        let max_identity =
                            Tile::literal(tile_literal_for(max_identity_scalar, DataTypeEnum::F32));

                        // Lane = one span: fold the spans' maxima and rescaled
                        // sums into the row's softmax statistics.
                        let j_active = in_bounds.clone() & lane.clone().lt(splits);
                        let m_index = layout_index(
                            &scratch_meta,
                            &[row_flat.clone(), lane.clone(), tile_u32(free + 1)],
                        );
                        let m_j = program.bind(raw_tile(scratch_storage.load(
                            program,
                            m_index,
                            j_active.clone(),
                        )));
                        let masked_m = Tile::select(j_active.clone(), m_j.clone(), max_identity);
                        let global_max = emit_group_reduce(
                            program,
                            subgroups,
                            tile_reduce_op(ReduceOp::Max),
                            block,
                            block,
                            masked_m,
                        );
                        let global_max = program.bind(global_max);
                        let l_index = layout_index(
                            &scratch_meta,
                            &[row_flat.clone(), lane.clone(), tile_u32(free)],
                        );
                        let l_j =
                            raw_tile(scratch_storage.load(program, l_index, j_active.clone()));
                        let weighted = Tile::select(
                            j_active,
                            l_j * (m_j - global_max.clone()).exp(),
                            f32_literal(0.0),
                        );
                        let denom = emit_group_reduce(
                            program,
                            subgroups,
                            tile_reduce_op(ReduceOp::Sum),
                            block,
                            block,
                            weighted,
                        );
                        let denom = program.bind(denom);

                        // Lane = one free-dim position: rescale and fold the
                        // spans' accumulators.
                        let acc = program.private(ElementType::F32);
                        program.store_local(&acc, f32_literal(0.0));
                        let out_active = in_bounds & lane.clone().lt(free);
                        program.if_then(out_active, |program| {
                            program.loop_range(splits, |program, j| {
                                let m_index = layout_index(
                                    &scratch_meta,
                                    &[row_flat.clone(), j.clone(), tile_u32(free + 1)],
                                );
                                let m =
                                    raw_tile(scratch_storage.load(program, m_index, Mask::all()));
                                let o_index = layout_index(
                                    &scratch_meta,
                                    &[row_flat.clone(), j, lane.clone()],
                                );
                                let o =
                                    raw_tile(scratch_storage.load(program, o_index, Mask::all()));
                                let current = program.load_local(&acc);
                                program.store_local(
                                    &acc,
                                    current + o * (m - global_max.clone()).exp(),
                                );
                            });
                            let value = program.load_local(&acc) / denom.clone();
                            let mut out_coords = row_dims.clone();
                            out_coords.push(lane.clone());
                            let output_index = layout_index(&output_meta, &out_coords);
                            output_storage.store(
                                program,
                                output_index,
                                ValueTile::F32(value).cast_to(output_dtype),
                                Mask::all(),
                            );
                        });
                    },
                );
                Some(())
            },
        )?;
        Some(combine)
    } else {
        None
    };

    let partials = kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        operation.name(),
        cache_key,
        dispatch_size,
        move |kb| {
            let mut storages = Vec::with_capacity(values.len());
            let mut metas = Vec::with_capacity(values.len());
            for value in &values {
                let (storage, meta) = declare_value(kb, value, false)?;
                storages.push(storage);
                metas.push(meta);
            }
            let params_storage = match &params_value {
                Some(value) => Some(declare_value(kb, value, false)?),
                None => None,
            };
            let (output_storage, output_meta) = match &scratch_value {
                Some(scratch) => declare_value(kb, scratch, true)?,
                None => declare_value(kb, &output_value, true)?,
            };

            let phase_handle = kb.program();
            // Element-phase values the reducing output reads across lanes are
            // staged through workgroup memory (the probs of decode attention).
            let staged: Vec<Option<WorkgroupTile>> = phase_steps
                .iter()
                .enumerate()
                .map(|(p, phase)| match (&output_kind, phase) {
                    (RowOutput::Reduce { combine, .. }, RowStep::Element { .. })
                        if combine.uses_input(input_count + p) =>
                    {
                        Some(phase_handle.alloc_workgroup_array(ScalarElement::F32, block))
                    }
                    _ => None,
                })
                .collect();

            crate::row_dispatch::emit_row_grid(phase_handle, dispatch_spec, |program, ctx| {
                let wg_flat = ctx.row;
                // With packing, `lane` becomes the position inside the row's
                // `k_group`-wide lane group; every later axis index and the
                // scalar-store lane test use it unchanged.
                let (row_flat, split_idx, lane) = if rows_per_workgroup > 1 {
                    (
                        program.bind(wg_flat * rows_per_workgroup + ctx.lane.clone() / k_group),
                        tile_u32(0),
                        program.bind(ctx.lane % k_group),
                    )
                } else if splits > 1 {
                    (
                        program.bind(wg_flat.clone() / splits),
                        program.bind(wg_flat % splits),
                        ctx.lane,
                    )
                } else {
                    (wg_flat, tile_u32(0), ctx.lane)
                };
                let in_bounds = row_flat.clone().lt(rows);
                let row_dims = output_dims_from_flat(row_flat.clone(), &row_shape);
                let full_coords = |k_index: Tile| -> Vec<Tile> {
                    let mut coords = Vec::with_capacity(rank);
                    let mut row_dim = 0;
                    for dim in 0..rank {
                        if dim == axis {
                            coords.push(k_index.clone());
                        } else {
                            coords.push(row_dims[row_dim].clone());
                            row_dim += 1;
                        }
                    }
                    coords
                };

                // The active axis length: a params read for dynamic-axis
                // programs, the compiled extent otherwise.
                let axis_len: Tile = match &params_storage {
                    Some((storage, _)) => raw_tile(storage.load(program, tile_u32(0), Mask::all())),
                    None => tile_u32(k),
                };

                let mut slots: Vec<(ValueTile, DataTypeEnum)> = Vec::new();

                if lanes_own_axis {
                    // Online streaming over axis tiles of `block`: lanes own
                    // one axis position per tile; the running max, sum, and
                    // free-dim accumulator are rescaled by
                    // `exp(M_old − M_new)` each tile, so any axis length
                    // streams through one workgroup with exact results.
                    let RowOutput::Reduce { free_dim, .. } = &output_kind else {
                        unreachable!("dynamic-axis row programs have a reducing output")
                    };
                    let mut online_steps = phase_steps.clone();
                    online_steps.push(RowStep::Output(output_kind.clone()));
                    let online = match_online_softmax(&online_steps, input_count)
                        .expect("dynamic-axis row programs are built in online-softmax shape");
                    let probs = staged[3]
                        .as_ref()
                        .expect("the probability element phase is staged");
                    let free = *free_dim as u32;

                    // Causal bound: tiles past the bounding row coordinate
                    // hold no live positions, so the loop ends there.
                    let effective_len: Tile = match axis_bound_dim {
                        Some(dim) => {
                            let row_index = dim - usize::from(axis < dim);
                            program.bind(
                                axis_len
                                    .clone()
                                    .min(row_dims[row_index].clone() + tile_u32(1)),
                            )
                        }
                        None => axis_len,
                    };

                    // A split workgroup owns one `block`-wide span of the
                    // axis; the single-workgroup form owns it all.
                    let (span_start, span_end) = if splits > 1 {
                        let start = program.bind(split_idx.clone() * block);
                        let end = program
                            .bind((start.clone() + tile_u32(block)).min(effective_len.clone()));
                        (start, end)
                    } else {
                        (tile_u32(0), effective_len.clone())
                    };

                    let max_identity =
                        || Tile::literal(tile_literal_for(online.max_identity, DataTypeEnum::F32));
                    let running_max = program.private(ElementType::F32);
                    let running_sum = program.private(ElementType::F32);
                    let acc = program.private(ElementType::F32);
                    let tile_base = program.private(ElementType::U32);
                    let item = program.private(ElementType::U32);
                    program.store_local(&running_max, max_identity());
                    program.store_local(&running_sum, f32_literal(0.0));
                    program.store_local(&acc, f32_literal(0.0));
                    program.store_local(&tile_base, span_start);
                    let out_active = in_bounds.clone() & lane.clone().lt(free);

                    program.loop_forever(|program| {
                        let base = program.load_local(&tile_base);
                        program.break_if(base.clone().ge(span_end.clone()));
                        let kv = program.bind(base.clone() + lane.clone());
                        let kv_active = in_bounds.clone() & kv.clone().lt(effective_len.clone());
                        let coords = full_coords(kv);

                        // Per-position score, optionally an inline fold (the
                        // q·k dot over the head dim).
                        let (score_expr, score_fold) = online.score;
                        let score = match score_fold {
                            Some(RowFold {
                                len,
                                function: fold_fn,
                            }) => {
                                let fold_wire_dtype = fold_fn.datatype();
                                // f32 accumulation for half-precision dots,
                                // rounded once below.
                                let fold_dtype = match fold_wire_dtype {
                                    DataTypeEnum::F16 => DataTypeEnum::F32,
                                    other => other,
                                };
                                let identity = Tile::literal(tile_literal_for(
                                    fold_fn.initial_value,
                                    fold_dtype,
                                ));
                                let reduce_op = tile_reduce_op(fold_fn.op);
                                let [folded] = program.fold(
                                    tile_ir::tile::range(*len as u32),
                                    [identity],
                                    |program, fold_idx, [fold_acc]| {
                                        let mut fold_coords = coords.clone();
                                        fold_coords.push(fold_idx);
                                        let (value, _) = eval_nary_expr(
                                            program,
                                            score_expr,
                                            &fold_coords,
                                            &storages,
                                            &metas,
                                            kv_active.clone(),
                                            &slots,
                                        );
                                        let value = raw_tile(value.cast_to(fold_dtype));
                                        [fold_acc.binary(reduce_op.binary(), value)]
                                    },
                                );
                                if fold_dtype == fold_wire_dtype {
                                    folded
                                } else {
                                    raw_tile(
                                        ValueTile::F32(program.bind(folded))
                                            .cast_to(fold_wire_dtype),
                                    )
                                }
                            }
                            None => {
                                let (value, _) = eval_nary_expr(
                                    program,
                                    score_expr,
                                    &coords,
                                    &storages,
                                    &metas,
                                    kv_active.clone(),
                                    &slots,
                                );
                                raw_tile(value.cast_to(DataTypeEnum::F32))
                            }
                        };
                        let score = program.bind(score);
                        let score_slot = [(ValueTile::F32(score), DataTypeEnum::F32)];

                        // Scaled/masked score, tile max, and the online
                        // rescale of the running state.
                        let (scaled, _) = eval_nary_expr(
                            program,
                            online.scaled,
                            &coords,
                            &storages,
                            &metas,
                            kv_active.clone(),
                            &score_slot,
                        );
                        let scaled = program.bind(Tile::select(
                            kv_active.clone(),
                            raw_tile(scaled.cast_to(DataTypeEnum::F32)),
                            max_identity(),
                        ));
                        let tile_max = emit_group_reduce(
                            program,
                            subgroups,
                            tile_reduce_op(ReduceOp::Max),
                            block,
                            block,
                            scaled.clone(),
                        );
                        let old_max = program.load_local(&running_max);
                        let new_max = program.bind(old_max.clone().max(tile_max));
                        program.store_local(&running_max, new_max.clone());
                        let factor = program.bind((old_max - new_max.clone()).exp());

                        let prob = program.bind(Tile::select(
                            kv_active.clone(),
                            (scaled - new_max).exp(),
                            f32_literal(0.0),
                        ));
                        let tile_sum = emit_group_reduce(
                            program,
                            subgroups,
                            tile_reduce_op(ReduceOp::Sum),
                            block,
                            block,
                            prob.clone(),
                        );
                        let sum = program.load_local(&running_sum);
                        program.store_local(&running_sum, sum * factor.clone() + tile_sum);
                        program.store_workgroup(probs, lane.clone(), prob);
                        program.workgroup_barrier();

                        // Lanes switch to owning one free-dim position each
                        // and fold this tile's staged probabilities against
                        // the weight input.
                        program.if_then(out_active.clone(), |program| {
                            let rescaled = program.load_local(&acc) * factor.clone();
                            program.store_local(&acc, rescaled);
                            program.store_local(&item, tile_u32(0));
                            program.loop_forever(|program| {
                                let j = program.load_local(&item);
                                let kv_j = program.bind(base.clone() + j.clone());
                                program.break_if(
                                    j.clone().ge(block) | kv_j.clone().ge(effective_len.clone()),
                                );
                                let prob_j = program.load_workgroup(probs, j.clone());
                                let mut weight_coords = full_coords(kv_j);
                                weight_coords.push(lane.clone());
                                let (weight, _) = eval_nary_expr(
                                    program,
                                    online.weight,
                                    &weight_coords,
                                    &storages,
                                    &metas,
                                    Mask::all(),
                                    &slots,
                                );
                                let weight = raw_tile(weight.cast_to(DataTypeEnum::F32));
                                let current = program.load_local(&acc);
                                program.store_local(&acc, current + prob_j * weight);
                                program.store_local(&item, j + tile_u32(1));
                            });
                        });
                        program.workgroup_barrier();
                        program.store_local(&tile_base, base + tile_u32(block));
                    });

                    if splits > 1 {
                        // Partials: the unnormalized accumulator plus this
                        // span's softmax statistics — the combine kernel
                        // folds the spans with the online monoid.
                        program.if_then(out_active, |program| {
                            let index = layout_index(
                                &output_meta,
                                &[row_flat.clone(), split_idx.clone(), lane.clone()],
                            );
                            let value = ValueTile::F32(program.load_local(&acc));
                            output_storage.store(program, index, value, Mask::all());
                        });
                        let stat_active = |offset: u32| {
                            in_bounds.clone()
                                & lane.clone().ge(free + offset)
                                & lane.clone().lt(free + offset + 1)
                        };
                        program.if_then(stat_active(0), |program| {
                            let index = layout_index(
                                &output_meta,
                                &[row_flat.clone(), split_idx.clone(), tile_u32(free)],
                            );
                            let value = ValueTile::F32(program.load_local(&running_sum));
                            output_storage.store(program, index, value, Mask::all());
                        });
                        program.if_then(stat_active(1), |program| {
                            let index = layout_index(
                                &output_meta,
                                &[row_flat.clone(), split_idx.clone(), tile_u32(free + 1)],
                            );
                            let value = ValueTile::F32(program.load_local(&running_max));
                            output_storage.store(program, index, value, Mask::all());
                        });
                    } else {
                        // out = O / L — the division the probability phase
                        // declares, applied once at the end.
                        program.if_then(out_active, |program| {
                            let value = program.load_local(&acc) / program.load_local(&running_sum);
                            let mut out_coords = row_dims.clone();
                            out_coords.push(lane.clone());
                            let output_index = layout_index(&output_meta, &out_coords);
                            let value = ValueTile::F32(value).cast_to(output_dtype);
                            output_storage.store(program, output_index, value, Mask::all());
                        });
                    }
                    return;
                }

                // Chunked map program: lanes stride the axis for each phase's
                // fold. Single-chunk programs stage each distinct tensor read
                // in a register up front (`staged_reads`); longer axes
                // re-evaluate per phase — the same trade every multi-pass
                // normalization kernel makes.
                let chunks = k.div_ceil(block);
                let staged_values = staged_reads.as_ref().map(|staged| {
                    let k_index = lane.clone();
                    let active = in_bounds.clone() & k_index.clone().lt(k);
                    let coords = full_coords(k_index);
                    stage_probe_values(program, &staged.probes, &coords, &storages, &metas, active)
                });
                for (phase_index, phase) in phase_steps.iter().enumerate() {
                    let RowStep::Reduce(reduce) = phase else {
                        unreachable!("element phases require a dynamic-axis row program")
                    };
                    let function_dtype = reduce.function.datatype();
                    // Half-precision folds accumulate in f32 and round once
                    // after the group reduction — the same accumulator policy
                    // as the matmul kernels and the composed fused reduce.
                    let phase_dtype = match function_dtype {
                        DataTypeEnum::F16 => DataTypeEnum::F32,
                        other => other,
                    };
                    let identity = || {
                        Tile::literal(tile_literal_for(reduce.function.initial_value, phase_dtype))
                    };
                    let reduce_op = tile_reduce_op(reduce.function.op);
                    let [partial] = program.fold(
                        tile_ir::tile::range(chunks),
                        [identity()],
                        |program, chunk, [acc]| {
                            let k_index = chunk * block + lane.clone();
                            let active = in_bounds.clone() & k_index.clone().lt(k);
                            let coords = full_coords(k_index);
                            let (value, _) = match (&staged_reads, &staged_values) {
                                (Some(staged), Some(staged_values)) => {
                                    let mut extras = staged_values.clone();
                                    extras.extend(slots.iter().cloned());
                                    eval_nary_expr(
                                        program,
                                        &staged.phases[phase_index],
                                        &coords,
                                        &[],
                                        &[],
                                        active.clone(),
                                        &extras,
                                    )
                                }
                                _ => eval_nary_expr(
                                    program,
                                    &reduce.expression,
                                    &coords,
                                    &storages,
                                    &metas,
                                    active.clone(),
                                    &slots,
                                ),
                            };
                            let value = raw_tile(value.cast_to(phase_dtype));
                            let masked = Tile::select(active, value, identity());
                            [acc.binary(reduce_op.binary(), masked)]
                        },
                    );
                    // The per-group workgroup reduction broadcasts: every lane
                    // in the row's `k_group`-wide lane group reads the combined
                    // value, so later phases can use it directly. With one row
                    // per workgroup `k_group == block` and this is the full
                    // workgroup reduction.
                    let combined =
                        emit_group_reduce(program, subgroups, reduce_op, k_group, block, partial);
                    let combined = if phase_dtype == function_dtype {
                        combined
                    } else {
                        raw_tile(ValueTile::F32(program.bind(combined)).cast_to(function_dtype))
                    };
                    let (combined, combined_ty) =
                        apply_unary_function_chain(combined, function_dtype, &reduce.post_chain)
                            .expect("validated row program post chain");
                    let scalar = ValueTile::F32(program.bind(combined)).cast_to(combined_ty);
                    slots.push((scalar, combined_ty));
                }

                let eval_output = |program: &mut tile_ir::tile::TileBlock<'_>,
                                   output_expr: &NaryExpr,
                                   coords: &[Tile],
                                   active: Mask,
                                   slots: &[(ValueTile, DataTypeEnum)]|
                 -> ValueTile {
                    let (value, _) = match (&staged_reads, &staged_values) {
                        (Some(staged), Some(staged_values)) => {
                            let mut extras = staged_values.clone();
                            extras.extend(slots.iter().cloned());
                            eval_nary_expr(
                                program,
                                &staged.output,
                                coords,
                                &[],
                                &[],
                                active,
                                &extras,
                            )
                        }
                        _ => eval_nary_expr(
                            program,
                            output_expr,
                            coords,
                            &storages,
                            &metas,
                            active,
                            slots,
                        ),
                    };
                    value
                };
                match &output_kind {
                    RowOutput::Map(output_expr) => {
                        program.loop_range(chunks, |program, chunk| {
                            let k_index = chunk * block + lane.clone();
                            let active = in_bounds.clone() & k_index.clone().lt(k);
                            let coords = full_coords(k_index);
                            let value =
                                eval_output(program, output_expr, &coords, active.clone(), &slots);
                            let value = value.cast_to(output_dtype);
                            let output_index = layout_index(&output_meta, &coords);
                            output_storage.store(program, output_index, value, active);
                        });
                    }
                    RowOutput::Scalar(output_expr) => {
                        let active = in_bounds & lane.eq(0u32);
                        let coords = full_coords(tile_u32(0));
                        let value =
                            eval_output(program, output_expr, &coords, active.clone(), &slots);
                        let value = value.cast_to(output_dtype);
                        let output_index = layout_index(&output_meta, &row_dims);
                        output_storage.store(program, output_index, value, active);
                    }
                    RowOutput::Reduce { .. } => {
                        unreachable!("reducing output requires a dynamic-axis row program")
                    }
                }
            });
            Some(())
        },
    )?;

    match combine {
        Some(combine) => Some(DirectKernel::sequence(
            "row_program_split",
            vec![partials, combine],
        )),
        None => Some(partials),
    }
}

struct MergedRowProgramKernelVariant;

/// One kernel executing several independent chunked-map row programs: each
/// segment owns a contiguous range of workgroups guarded by a uniform
/// linear-workgroup-id range compare. The guard condition depends only on
/// the workgroup id, so the per-segment workgroup reductions stay in
/// workgroup-uniform control flow. Callers gate segments through
/// [`RowProgramOperation::mergeable_chunked_map`] and bound total bindings.
pub(crate) fn build_merged_row_program_kernel(
    graph: &crate::compute_graph::ComputeGraphInner,
    segments: &[RowProgramOperation],
    segment_inputs: &[Vec<MirValue>],
) -> Option<DirectKernel> {
    use std::hash::Hash;
    let device = graph.device();
    let max_per_dim = device.limits().max_compute_workgroups_per_dimension;
    // Size the shared workgroup to the longest segment axis (matching the
    // unmerged `RowProgramOperation::block`): a merge of k=64 softmax
    // segments runs 64-lane workgroups whose whole-block reductions are
    // subgroup-accelerated instead of walking the shared-memory tree.
    let policy = device.dispatch_policy();
    let max_block = policy.preferred_workgroup_lanes();
    let block = segments
        .iter()
        .map(|op| {
            u32::try_from(op.shape[op.axis])
                .unwrap_or(max_block)
                .max(1)
                .next_power_of_two()
                .clamp(policy.min_reduction_lanes().min(max_block), max_block)
        })
        .max()?;

    struct Segment {
        values: Vec<MaybeQData>,
        output: MaybeQData,
        rows: u32,
        k: u32,
        k_group: u32,
        rows_per_workgroup: u32,
        row_shape: Vec<usize>,
        base: u32,
        groups: u32,
    }
    let mut prepared = Vec::with_capacity(segments.len());
    let mut total_groups = 0u32;
    for (op, inputs) in segments.iter().zip(segment_inputs) {
        if !op.mergeable_chunked_map() {
            return None;
        }
        let (output, producers) = inputs.split_last()?;
        let output = output.as_tensor()?.clone();
        let values = producers
            .iter()
            .map(|input| MaybeQData::try_from(input.clone()).ok())
            .collect::<Option<Vec<_>>>()?;
        if values
            .iter()
            .any(|value| matches!(value, MaybeQData::QMatrix(_)))
        {
            return None;
        }
        if !device.f16_supported()
            && (output.datatype() == DataTypeEnum::F16
                || values.iter().any(|value| {
                    matches!(value, MaybeQData::Tensor(tensor)
                    if tensor.datatype() == DataTypeEnum::F16)
                }))
        {
            return None;
        }
        let rows: u32 = op.rows().try_into().ok()?;
        let k: u32 = op.shape[op.axis].try_into().ok()?;
        let k_group = {
            let group = k.next_power_of_two().min(block).max(1);
            if block.is_multiple_of(group) {
                group
            } else {
                block
            }
        };
        let rows_per_workgroup = block / k_group;
        let groups = rows.div_ceil(rows_per_workgroup);
        prepared.push(Segment {
            values,
            output: MaybeQData::Tensor(output),
            rows,
            k,
            k_group,
            rows_per_workgroup,
            row_shape: op.row_shape(),
            base: total_groups,
            groups,
        });
        total_groups = total_groups.checked_add(groups)?;
    }

    let dispatch_size = distribute_workgroups(total_groups, max_per_dim);
    let subgroups = fixed_subgroups(&device);
    // Single-chunk segments stage each distinct tensor read once per lane
    // (see `stage_single_chunk_reads`); the rewrite is deterministic from
    // the segment's steps, so it stays out of the cache key.
    let staged_segments: Vec<Option<StagedReads>> = segments
        .iter()
        .zip(&prepared)
        .map(|(op, segment)| {
            (segment.k.div_ceil(block) == 1)
                .then(|| {
                    stage_single_chunk_reads(op.phase_steps(), op.output_step(), op.inputs.len())
                })
                .flatten()
        })
        .collect();
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<MergedRowProgramKernelVariant>().hash(state);
        dispatch_size.hash(state);
        block.hash(state);
        crate::compute_graph::resolve::plan_cache::hash_merged_segments(
            state,
            segments.iter(),
            segment_inputs,
        );
    });
    let name = if device.config().trace_decode_names {
        format!(
            "merged_row_program[{}]",
            segments
                .iter()
                .map(|op| op.name())
                .collect::<Vec<_>>()
                .join("; ")
        )
    } else {
        format!("merged_row_program_x{}", segments.len())
    };

    kernel_backend::run_kernel(
        device.kernel_cache(),
        name,
        cache_key,
        dispatch_size,
        move |kb| {
            let mut declared = Vec::with_capacity(prepared.len());
            for segment in &prepared {
                let mut storages = Vec::with_capacity(segment.values.len());
                let mut metas = Vec::with_capacity(segment.values.len());
                for value in &segment.values {
                    let (storage, meta) = declare_value(kb, value, false)?;
                    storages.push(storage);
                    metas.push(meta);
                }
                let output = declare_value(kb, &segment.output, true)?;
                declared.push((storages, metas, output));
            }

            kb.program().program_grid(block, dispatch_size, |program| {
                let full_lane = program.lane();
                let group = program.bind(crate::nary_direct::linear_group(program, dispatch_size));
                for ((op, staged_reads), (segment, (storages, metas, output))) in segments
                    .iter()
                    .zip(&staged_segments)
                    .zip(prepared.iter().zip(&declared))
                {
                    let in_segment = group.clone().ge(segment.base)
                        & group.clone().lt(segment.base + segment.groups);
                    program.if_then(in_segment, |program| {
                        let (output_storage, output_meta) = output;
                        let wg_local = program.bind(group.clone() - segment.base);
                        let (row_flat, lane) = if segment.rows_per_workgroup > 1 {
                            (
                                program.bind(
                                    wg_local.clone() * segment.rows_per_workgroup
                                        + full_lane.clone() / segment.k_group,
                                ),
                                program.bind(full_lane.clone() % segment.k_group),
                            )
                        } else {
                            (wg_local, full_lane.clone())
                        };
                        let in_bounds = row_flat.clone().lt(segment.rows);
                        let row_dims = output_dims_from_flat(row_flat.clone(), &segment.row_shape);
                        let axis = op.axis;
                        let rank = op.shape.len();
                        let full_coords = |k_index: Tile| -> Vec<Tile> {
                            let mut coords = Vec::with_capacity(rank);
                            let mut row_dim = 0;
                            for dim in 0..rank {
                                if dim == axis {
                                    coords.push(k_index.clone());
                                } else {
                                    coords.push(row_dims[row_dim].clone());
                                    row_dim += 1;
                                }
                            }
                            coords
                        };

                        let k = segment.k;
                        let chunks = k.div_ceil(block);
                        let staged_values = staged_reads.as_ref().map(|staged| {
                            let k_index = lane.clone();
                            let active = in_bounds.clone() & k_index.clone().lt(k);
                            let coords = full_coords(k_index);
                            stage_probe_values(
                                program,
                                &staged.probes,
                                &coords,
                                storages,
                                metas,
                                active,
                            )
                        });
                        let mut slots: Vec<(ValueTile, DataTypeEnum)> = Vec::new();
                        for (phase_index, phase) in op.phase_steps().iter().enumerate() {
                            let RowStep::Reduce(reduce) = phase else {
                                unreachable!("merged row programs are reduce-phase only")
                            };
                            let function_dtype = reduce.function.datatype();
                            // Same f32 accumulation policy as the standalone
                            // reduce phases above.
                            let phase_dtype = match function_dtype {
                                DataTypeEnum::F16 => DataTypeEnum::F32,
                                other => other,
                            };
                            let identity = || {
                                Tile::literal(tile_literal_for(
                                    reduce.function.initial_value,
                                    phase_dtype,
                                ))
                            };
                            let reduce_op = tile_reduce_op(reduce.function.op);
                            let [partial] = program.fold(
                                tile_ir::tile::range(chunks),
                                [identity()],
                                |program, chunk, [acc]| {
                                    let k_index = chunk * block + lane.clone();
                                    let active = in_bounds.clone() & k_index.clone().lt(k);
                                    let coords = full_coords(k_index);
                                    let (value, _) = match (&staged_reads, &staged_values) {
                                        (Some(staged), Some(staged_values)) => {
                                            let mut extras = staged_values.clone();
                                            extras.extend(slots.iter().cloned());
                                            eval_nary_expr(
                                                program,
                                                &staged.phases[phase_index],
                                                &coords,
                                                &[],
                                                &[],
                                                active.clone(),
                                                &extras,
                                            )
                                        }
                                        _ => eval_nary_expr(
                                            program,
                                            &reduce.expression,
                                            &coords,
                                            storages,
                                            metas,
                                            active.clone(),
                                            &slots,
                                        ),
                                    };
                                    let value = raw_tile(value.cast_to(phase_dtype));
                                    let masked = Tile::select(active, value, identity());
                                    [acc.binary(reduce_op.binary(), masked)]
                                },
                            );
                            let combined = emit_group_reduce(
                                program,
                                subgroups,
                                reduce_op,
                                segment.k_group,
                                block,
                                partial,
                            );
                            let combined = if phase_dtype == function_dtype {
                                combined
                            } else {
                                raw_tile(
                                    ValueTile::F32(program.bind(combined))
                                        .cast_to(function_dtype),
                                )
                            };
                            let (combined, combined_ty) = apply_unary_function_chain(
                                combined,
                                function_dtype,
                                &reduce.post_chain,
                            )
                            .expect("validated row program post chain");
                            let scalar =
                                ValueTile::F32(program.bind(combined)).cast_to(combined_ty);
                            slots.push((scalar, combined_ty));
                        }

                        let output_dtype = op.output_datatype;
                        let eval_output = |program: &mut tile_ir::tile::TileBlock<'_>,
                                           output_expr: &NaryExpr,
                                           coords: &[Tile],
                                           active: Mask,
                                           slots: &[(ValueTile, DataTypeEnum)]|
                         -> ValueTile {
                            let (value, _) = match (&staged_reads, &staged_values) {
                                (Some(staged), Some(staged_values)) => {
                                    let mut extras = staged_values.clone();
                                    extras.extend(slots.iter().cloned());
                                    eval_nary_expr(
                                        program,
                                        &staged.output,
                                        coords,
                                        &[],
                                        &[],
                                        active,
                                        &extras,
                                    )
                                }
                                _ => eval_nary_expr(
                                    program,
                                    output_expr,
                                    coords,
                                    storages,
                                    metas,
                                    active,
                                    slots,
                                ),
                            };
                            value
                        };
                        match op.output_step() {
                            RowOutput::Map(output_expr) => {
                                program.loop_range(chunks, |program, chunk| {
                                    let k_index = chunk * block + lane.clone();
                                    let active = in_bounds.clone() & k_index.clone().lt(k);
                                    let coords = full_coords(k_index);
                                    let value = eval_output(
                                        program,
                                        output_expr,
                                        &coords,
                                        active.clone(),
                                        &slots,
                                    );
                                    let value = value.cast_to(output_dtype);
                                    let output_index = layout_index(output_meta, &coords);
                                    output_storage.store(program, output_index, value, active);
                                });
                            }
                            RowOutput::Scalar(output_expr) => {
                                let active = in_bounds.clone() & lane.clone().eq(0u32);
                                let coords = full_coords(tile_u32(0));
                                let value = eval_output(
                                    program,
                                    output_expr,
                                    &coords,
                                    active.clone(),
                                    &slots,
                                );
                                let value = value.cast_to(output_dtype);
                                let output_index = layout_index(output_meta, &row_dims);
                                output_storage.store(program, output_index, value, active);
                            }
                            RowOutput::Reduce { .. } => {
                                unreachable!("merged row programs never have a reducing output")
                            }
                        }
                    });
                }
            });
            Some(())
        },
    )
}

/// Scaled dot-product attention as a row program over the KV axis: an
/// element phase stages the q·k score per lane via an inline head-dim fold,
/// a max phase and an exp-sum phase form the softmax statistics, a second
/// element phase stages the probabilities, and the reducing output folds
/// `Σ p·v` with the head dim as the free output dimension. Causal masking
/// is an index-compare select inside the scaled score (plus an axis bound
/// that skips tiles past the query position); an additive mask is a fourth
/// input read at `[q, kv]`. Each query row is one workgroup; KV histories
/// beyond one workgroup bucket stream through the online-softmax tile loop.
/// Returns `None` for shapes the row program cannot host (head dim beyond
/// the largest workgroup bucket, non-float dtypes).
pub(crate) struct AttentionInputs<'a> {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) q_shape: &'a [usize],
    pub(crate) k_shape: &'a [usize],
    pub(crate) v_shape: &'a [usize],
    pub(crate) mask_shape: Option<&'a [usize]>,
    pub(crate) scale: f32,
    pub(crate) input_dtype: DataTypeEnum,
    pub(crate) causal: bool,
}

pub(crate) fn attention_row_program(
    device: &crate::Device,
    inputs: AttentionInputs<'_>,
) -> Option<RowProgramOperation> {
    let AttentionInputs {
        q,
        k,
        v,
        mask,
        q_shape,
        k_shape,
        v_shape,
        mask_shape,
        scale,
        input_dtype,
        causal,
    } = inputs;
    let [batch, num_heads, q_seq_len, head_dim] = *q_shape else {
        return None;
    };
    let [k_batch, num_kv_heads, kv_len, k_head_dim] = *k_shape else {
        return None;
    };
    if !matches!(input_dtype, DataTypeEnum::F32 | DataTypeEnum::F16)
        || (input_dtype == DataTypeEnum::F16 && !device.f16_supported())
        || q_seq_len == 0
        || kv_len == 0
        || head_dim == 0
        || k_batch != batch
        || k_head_dim != head_dim
        || v_shape != k_shape
        || num_kv_heads == 0
        || !num_heads.is_multiple_of(num_kv_heads)
        || (causal && mask.is_some())
    {
        return None;
    }
    if let Some(mask_shape) = mask_shape
        && mask_shape != [q_seq_len, kv_len]
    {
        return None;
    }
    if mask.is_some() != mask_shape.is_some() {
        return None;
    }
    let policy = device.dispatch_policy();
    // The workgroup bucket is one axis tile: small tiles let the split
    // lowering fan decode across workgroups and stream longer axes through
    // the online loop with good occupancy. The floor is one full-width
    // workgroup; the kernel monomorphizes per bucket.
    let needed = head_dim.max(policy.preferred_workgroup_lanes() as usize) as u32;
    let block = policy
        .dynamic_block_buckets()
        .find(|&candidate| candidate >= needed)?;

    let groups = num_heads / num_kv_heads;
    let f32 = DataTypeEnum::F32;
    let dim = NaryExpr::DimIndex;
    let kv_head = || {
        if groups == 1 {
            dim(1)
        } else {
            NaryExpr::Op {
                children: vec![dim(1)],
                function: NaryFunction::unary(
                    Some("kv_head".to_string()),
                    NaryOp::DivConst(NaryScalar::U32(groups as u32)),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                ),
            }
        }
    };
    let binary = |op, a, b| NaryExpr::Op {
        children: vec![a, b],
        function: NaryFunction::binary(None, op, f32, f32, f32),
    };
    let unary = |op, a| NaryExpr::Op {
        children: vec![a],
        function: NaryFunction::unary(None, op, f32, f32),
    };
    let graph_inputs: Vec<NodeIndex> = match mask {
        Some(mask) => vec![q, k, v, mask],
        None => vec![q, k, v],
    };
    let input_count = graph_inputs.len();
    let slot = |p: usize| slot_expr(input_count, p);
    // The fold/free coordinate is one past the rank-4 index space. Loads
    // are cast to the f32 expression types by evaluation, so f16 tensors
    // need no explicit casts.
    let q_read = NaryExpr::indexed_input(0, vec![dim(0), dim(1), dim(2), dim(4)]);
    let k_read = NaryExpr::indexed_input(1, vec![dim(0), kv_head(), dim(3), dim(4)]);
    let v_read = NaryExpr::indexed_input(2, vec![dim(0), kv_head(), dim(3), dim(4)]);

    // The scaled score, with causality or the additive mask folded in. The
    // causal arm masks to the max-phase identity so masked positions vanish
    // from both the max and the exp sum.
    let scaled_score = || {
        let base = unary(NaryOp::MulConst(NaryScalar::F32(scale)), slot(0));
        if causal {
            let bound = NaryExpr::Op {
                children: vec![dim(3), dim(2)],
                function: NaryFunction::binary(
                    Some("causal_bound".to_string()),
                    NaryOp::LessEqual,
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                ),
            };
            NaryExpr::select(
                bound,
                base,
                NaryExpr::Scalar(max_fn(f32).initial_value),
                DataTypeEnum::U32,
                f32,
            )
        } else if mask.is_some() {
            let mask_read = NaryExpr::indexed_input(3, vec![dim(2), dim(3)]);
            binary(NaryOp::Add, base, mask_read)
        } else {
            base
        }
    };
    let shifted_exp = || unary(NaryOp::Exp, binary(NaryOp::Sub, scaled_score(), slot(1)));

    let mut input_axis_dims = vec![None, Some(2), Some(2)];
    if mask.is_some() {
        input_axis_dims.push(Some(1));
    }
    Some(RowProgramOperation {
        inputs: graph_inputs,
        shape: [batch, num_heads, q_seq_len, kv_len].into(),
        axis: 3,
        steps: vec![
            RowStep::Element {
                expression: binary(NaryOp::Mul, q_read, k_read),
                fold: Some(RowFold {
                    len: head_dim,
                    function: sum_fn(f32),
                }),
                datatype: f32,
            },
            RowStep::Reduce(RowReduce {
                expression: scaled_score(),
                function: max_fn(f32),
                post_chain: UnaryFunctionChain::empty(f32),
            }),
            RowStep::Reduce(RowReduce {
                expression: shifted_exp(),
                function: sum_fn(f32),
                post_chain: UnaryFunctionChain::empty(f32),
            }),
            RowStep::Element {
                expression: binary(NaryOp::Div, shifted_exp(), slot(2)),
                fold: None,
                datatype: f32,
            },
            RowStep::Output(RowOutput::Reduce {
                combine: binary(NaryOp::Mul, slot(3), v_read),
                function: sum_fn(f32),
                free_dim: head_dim,
            }),
        ],
        output_datatype: input_dtype,
        dynamic_axis: Some(DynamicAxis {
            block,
            input_axis_dims,
            axis_bound_dim: causal.then_some(2),
        }),
    })
}
