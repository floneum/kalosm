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
    tile::{Mask, Tile},
};

use crate::{
    compute_graph::NodeIndex,
    fold::FoldOperation,
    mir::{
        inputs::MirValue,
        kernel_backend::{self, DirectKernel},
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_direct::{
        ValueTile, apply_unary_function_chain, datatype_element, declare_value, eval_nary_expr,
        layout_index, output_dims_from_flat, tile_u32,
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
    pub(crate) combine: RowCombine,
    pub(crate) post_chain: UnaryFunctionChain,
}

/// How a row phase folds its element into the slot.
///
/// `BuiltIn` is the four closed operators, which have subgroup intrinsics and
/// keep the existing emission byte-for-byte. `General` names the step as an
/// expression over the accumulator and the element, which is what lets a
/// carrier like online softmax be *stated* instead of pattern-matched back out
/// of a fixed step sequence.
#[derive(Debug, Clone, PartialEq, Hash)]
pub(crate) enum RowCombine {
    BuiltIn(ReduceFunction),
    /// `step` reads the accumulator as slot `acc_slot` and the element as
    /// slot `element_slot`, using the same `slot_expr` convention as every
    /// other cross-step reference.
    General {
        init: NaryScalar,
        /// `acc (+) element`, evaluated once per axis position.
        step: NaryExpr,
        /// `acc (+) rhs` over two accumulators, evaluated when partials from
        /// different lanes merge. For a monoid whose step and combine
        /// coincide (sum, max) this is the same expression.
        combine: NaryExpr,
        /// Slot index the accumulator is bound to during evaluation; the
        /// element (or incoming partial) takes `acc_slot + 1`. Emission
        /// appends both to the live slot list, so these are the next two
        /// indices after the phases already computed.
        acc_slot: usize,
        datatype: DataTypeEnum,
    },
}

impl RowCombine {
    pub(crate) fn datatype(&self) -> DataTypeEnum {
        match self {
            RowCombine::BuiltIn(function) => function.datatype(),
            RowCombine::General { datatype, .. } => *datatype,
        }
    }

    pub(crate) fn initial_value(&self) -> NaryScalar {
        match self {
            RowCombine::BuiltIn(function) => function.initial_value,
            RowCombine::General { init, .. } => *init,
        }
    }

    /// The closed operator, when there is one. Emission needs it for the
    /// cross-lane group reduction: subgroup intrinsics are per-operator, so a
    /// general combine can only span lanes once tile-ir grows a group reduce
    /// parameterized by an expression.
    pub(crate) fn built_in(&self) -> Option<&ReduceFunction> {
        match self {
            RowCombine::BuiltIn(function) => Some(function),
            RowCombine::General { .. } => None,
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            RowCombine::BuiltIn(function) => function.name(),
            RowCombine::General { .. } => "fold",
        }
    }
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
#[derive(Debug, Clone, PartialEq)]
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
    /// The joint carrier streamed over the axis.
    ///
    /// A dynamic-axis program *is* one fold, not a phase list: its slots
    /// advance together in a single pass, which is exactly what a `RowStep`
    /// sequence cannot express (phase `i` there sees phase `j < i`'s
    /// *completed* value). So `steps` stays empty for these programs and the
    /// carrier is the whole declaration.
    pub(crate) carrier: FoldOperation,
}

impl Hash for DynamicAxis {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.block.hash(state);
        self.input_axis_dims.hash(state);
        self.axis_bound_dim.hash(state);
        self.carrier.hash_carrier_fields(state);
    }
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

fn slot_expr(input_count: usize, phase: usize) -> NaryExpr {
    NaryExpr::IndexedInput {
        input_idx: input_count + phase,
        indices: vec![],
    }
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
                    combine: RowCombine::BuiltIn(reduce.function.clone()),
                    post_chain: reduce.post_element_wise.clone(),
                }),
                RowStep::Output(RowOutput::Scalar(scalar_slot)),
            ],
            output_datatype: reduce.out_datatype(),
            dynamic_axis: None,
        }
    }

    /// Re-express every built-in phase as the equivalent general combine.
    ///
    /// Sum, Product, Max and Min are monoids whose step and combine coincide,
    /// so each becomes the same binary expression over the accumulator and the
    /// incoming value. Used by `FUSOR_SPIKE_GENERAL_COMBINE` to validate the
    /// general emission path against the closed-operator one.
    pub(crate) fn as_general_combines(&self) -> Option<Self> {
        // Dynamic-axis programs lower through the bespoke online-softmax
        // streaming path, which pattern-matches the closed `Max`/`Sum` phases
        // via `match_online_softmax`. That coupling is exactly what a general
        // combine is meant to retire, but until it does, re-expressing those
        // phases would break the matcher rather than exercise the new path.
        if self.dynamic_axis.is_some() {
            return None;
        }
        let input_count = self.inputs.len();
        let mut steps = self.steps.clone();
        let mut rewrote = false;
        for (phase, step) in steps.iter_mut().enumerate() {
            let RowStep::Reduce(reduce) = step else {
                continue;
            };
            let Some(function) = reduce.combine.built_in() else {
                continue;
            };
            let datatype = function.datatype();
            let op = match function.op {
                ReduceOp::Sum => NaryOp::Add,
                ReduceOp::Product => NaryOp::Mul,
                ReduceOp::Max => NaryOp::Max,
                ReduceOp::Min => NaryOp::Min,
            };
            let body = |acc_index: usize| NaryExpr::Op {
                children: vec![
                    slot_expr(input_count, acc_index),
                    slot_expr(input_count, acc_index + 1),
                ],
                function: NaryFunction::binary(None, op, datatype, datatype, datatype),
            };
            reduce.combine = RowCombine::General {
                init: function.initial_value,
                step: body(phase),
                combine: body(phase),
                acc_slot: phase,
                datatype,
            };
            rewrote = true;
        }
        rewrote.then(|| Self {
            steps,
            ..self.clone()
        })
    }

    /// Whether any phase folds with a general combine. Emission needs a
    /// closed operator for the cross-lane group reduction — subgroup
    /// intrinsics are per-operator — so until tile-ir grows a group reduce
    /// parameterized by an expression, these programs have no direct kernel.
    pub(crate) fn has_general_combine(&self) -> bool {
        if self.dynamic_axis.is_some() {
            // A streaming fold has no phase list; its combine is the carrier's.
            return false;
        }
        self.phase_steps().iter().any(|step| match step {
            RowStep::Reduce(reduce) => reduce.combine.built_in().is_none(),
            _ => false,
        })
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
        if let Some(dynamic) = &self.dynamic_axis {
            let output = dynamic
                .carrier
                .outputs
                .first()
                .expect("a carrier has at least one output");
            return dynamic.carrier.output_shape(output);
        }
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

    /// How many expression evaluations one index-space position costs — the
    /// phase count for a chunked-map program, the carrier width plus its
    /// element for a streaming fold. Used by the cost model and the graph
    /// dump, both of which want one number per program.
    pub(crate) fn work_units(&self) -> usize {
        match &self.dynamic_axis {
            Some(dynamic) => dynamic.carrier.width() + 1,
            None => self.steps.len(),
        }
    }

    fn block(&self, device: &crate::Device) -> u32 {
        match &self.dynamic_axis {
            Some(dynamic) => dynamic.block,
            None => static_axis_block(device, self.shape[self.axis]),
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
        if let Some(dynamic) = &self.dynamic_axis {
            let carrier = &dynamic.carrier;
            return carrier.expression.uses_custom_indexing_for_input(input_idx)
                || carrier.element_fold.as_ref().is_some_and(|fold| {
                    fold.expression.uses_custom_indexing_for_input(input_idx)
                })
                || carrier.carrier.iter().any(|slot| {
                    slot.element
                        .as_ref()
                        .is_some_and(|read| read.uses_custom_indexing_for_input(input_idx))
                });
        }
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
        // The carrier mirrors the same operands; rebinding must reach both or
        // a rematerialized program would fold over the pre-interning nodes.
        if let Some(dynamic) = &mut self.dynamic_axis {
            dynamic.carrier.inputs.clone_from(&self.inputs);
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

/// Wrap a raw tile as a typed value for slot binding.
fn typed_tile(value: Tile, datatype: DataTypeEnum) -> ValueTile {
    match datatype {
        DataTypeEnum::F32 => ValueTile::F32(value),
        DataTypeEnum::F16 => ValueTile::F16(value),
        DataTypeEnum::U32 => ValueTile::U32(value),
    }
}

/// Evaluate a carrier body that reads the accumulator and one incoming value.
///
/// The two are appended to the live slot list, so a body built for phase `p`
/// reads them as slots `p` and `p + 1` — the same `slot_expr` convention every
/// other cross-step reference uses.
#[allow(clippy::too_many_arguments)]
fn emit_carrier_body(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expression: &NaryExpr,
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
    slots: &[(ValueTile, DataTypeEnum)],
    acc: Tile,
    incoming: Tile,
    datatype: DataTypeEnum,
) -> Tile {
    // Storages are passed through so a carrier body indexes slots on the same
    // base as every other row-program expression: `slot_expr(input_count, k)`
    // resolves to `extras[k]`. A carrier body reads no tensor, but giving it a
    // different base would be a silent index footgun.
    let mut extended = slots.to_vec();
    extended.push((typed_tile(acc, datatype), datatype));
    extended.push((typed_tile(incoming, datatype), datatype));
    let (value, _) = eval_nary_expr(
        program,
        expression,
        &[],
        storages,
        metas,
        tile_ir::tile::Mask::from(true),
        &extended,
    );
    raw_tile(value.cast_to(datatype))
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

/// Build the staged rewrite for a program whose axis fits the per-lane
/// register budget: `None` when any read resists staging (slot-dependent
/// indices, non-reduce phases, reducing outputs) or there is nothing to stage.
fn stage_chunk_reads(
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

/// The staged reads of every chunk the lane owns, outer index = chunk. A lane
/// group narrower than the axis strides it `chunks` times, so staging holds
/// one register set per stride — the phases and the output then run entirely
/// out of registers however many strides there are.
type StagedChunks = Vec<Vec<(ValueTile, DataTypeEnum)>>;

/// Evaluate the probes of every chunk the lane owns.
#[allow(clippy::too_many_arguments)]
fn stage_chunk_values(
    program: &mut tile_ir::tile::TileBlock<'_>,
    probes: &[NaryExpr],
    chunks: u32,
    k: u32,
    k_group: u32,
    lane: &Tile,
    in_bounds: &Mask,
    full_coords: impl Fn(Tile) -> Vec<Tile>,
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
) -> StagedChunks {
    (0..chunks)
        .map(|chunk| {
            let k_index = lane.clone() + tile_u32(chunk * k_group);
            let active = in_bounds.clone() & k_index.clone().lt(k);
            let coords = full_coords(k_index);
            stage_probe_values(program, probes, &coords, storages, metas, active)
        })
        .collect()
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

/// Workgroup width for a static-axis (chunked-map) row program.
///
/// With a fixed subgroup width the per-row reduction is a subgroup collective
/// over a lane group narrower than the workgroup ([`lane_group_width`]), so
/// the workgroup is always full width and packs as many rows as fit. Without
/// one every reduction is a shared-memory tree whose barrier count grows with
/// the group, so the workgroup still tracks the axis: a k=64 reduce runs
/// 64-lane workgroups rather than paying a 256-wide tree per row.
fn static_axis_block(device: &crate::Device, axis_len: usize) -> u32 {
    let policy = device.dispatch_policy();
    let max_block = policy.preferred_workgroup_lanes();
    if fixed_subgroups(device).is_some() {
        return max_block;
    }
    let k = u32::try_from(axis_len).unwrap_or(max_block);
    k.max(1)
        .next_power_of_two()
        .clamp(policy.min_reduction_lanes().min(max_block), max_block)
}

/// Lanes per row for a chunked-map program: the workgroup splits into
/// `block / group` contiguous groups, each owning one row.
///
/// Narrowing the group packs more rows per workgroup and, once the group is
/// exactly one subgroup, turns the per-row reduction into a barrier-free
/// subgroup collective — a k=64 softmax goes from one 64-lane workgroup per
/// row with four threadgroup barriers to eight rows per full workgroup with
/// none. The floor on narrowing is the per-thread work budget: each lane then
/// strides the axis `k / group` times.
fn lane_group_width(
    policy: &crate::occupancy::DispatchPolicy,
    subgroups: FixedSubgroups,
    k: u32,
    block: u32,
) -> u32 {
    let mut group = k.next_power_of_two().min(block).max(1);
    if !block.is_multiple_of(group) {
        return block;
    }
    let Some((_, subgroup_width)) = subgroups else {
        return group;
    };
    let budget = policy.work_per_thread(crate::occupancy::RegPressure::ElementwiseFew);
    while group > subgroup_width && k.div_ceil(group / 2) <= budget {
        group /= 2;
    }
    group
}

/// Split a workgroup's lanes into `(row within the workgroup, position in the
/// row's axis span)`.
///
/// When the lane group is subgroup-aligned the split is derived from the
/// subgroup builtins, matching how [`emit_group_reduce`] folds the groups:
/// the mapping from `local_invocation_index` onto subgroups is implementation
/// defined, so a row must not be defined by lane index. The shared-memory
/// tree, in contrast, derives group membership from the lane index itself.
fn split_lane_groups(
    program: &mut tile_ir::tile::TileBlock<'_>,
    subgroups: FixedSubgroups,
    full_lane: Tile,
    k_group: u32,
    block: u32,
) -> (Tile, Tile) {
    match subgroups {
        Some((token, subgroup_width))
            if k_group.is_multiple_of(subgroup_width) && block.is_multiple_of(k_group) =>
        {
            let subgroup_id = token.subgroup_id(program);
            let subgroup_lane = token.subgroup_lane(program);
            let per_group = k_group / subgroup_width;
            if per_group == 1 {
                (program.bind(subgroup_id), program.bind(subgroup_lane))
            } else {
                let row = program.bind(subgroup_id.clone() / per_group);
                let lane = program.bind(subgroup_id % per_group * subgroup_width + subgroup_lane);
                (row, lane)
            }
        }
        _ => (
            program.bind(full_lane.clone() / k_group),
            program.bind(full_lane % k_group),
        ),
    }
}

/// The per-row-group reduction: subgroup-accelerated whenever the group is
/// subgroup-aligned on a fixed-subgroup device (no barrier at all when it is
/// one subgroup), the shared-memory tree otherwise. A 1-wide group is the
/// lane's own value (the k=1 bias-grad sum), skipping the scratch round-trip
/// entirely.
/// Cross-lane reduction for a row phase, dispatching on the combine.
///
/// Built-ins keep the subgroup/tree path unchanged. A general combine has no
/// per-operator intrinsic, so it stages through workgroup memory via
/// [`tile_ir::tile::TileBlock::group_reduce_with`].
#[allow(clippy::too_many_arguments)]
fn emit_group_reduce_combined(
    program: &mut tile_ir::tile::TileBlock<'_>,
    subgroups: FixedSubgroups,
    combine: &RowCombine,
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
    slots: &[(ValueTile, DataTypeEnum)],
    datatype: DataTypeEnum,
    op: tile_ir::TileReduceOp,
    group_size: u32,
    block: u32,
    value: Tile,
) -> Tile {
    match combine {
        RowCombine::BuiltIn(_) => {
            emit_group_reduce(program, subgroups, op, group_size, block, value)
        }
        RowCombine::General {
            combine: combine_expr,
            ..
        } => {
            let slots = slots.to_vec();
            program.group_reduce_with(group_size, value, |program, acc, incoming| {
                emit_carrier_body(
                    program,
                    combine_expr,
                    storages,
                    metas,
                    &slots,
                    acc,
                    incoming,
                    datatype,
                )
            })
        }
    }
}

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
            if group_size.is_multiple_of(subgroup_size) && block.is_multiple_of(group_size) =>
        {
            token.group_reduce(program, op, subgroup_size, group_size, value)
        }
        _ => program.group_reduce(op, group_size, value),
    }
}

/// The element of a streaming fold: the private inner fold (attention's q·k
/// dot over the head dim) when there is one, then the element expression,
/// which reads the folded value as [`crate::fold::fold_value`].
fn emit_fold_element(
    program: &mut tile_ir::tile::TileBlock<'_>,
    carrier: &FoldOperation,
    coords: &[Tile],
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
    mask: Mask,
) -> Tile {
    let mut extras: Vec<(ValueTile, DataTypeEnum)> = Vec::new();
    if let Some(inner) = &carrier.element_fold {
        let wire = inner.function.datatype();
        // f32 accumulation for half-precision dots, rounded once below.
        let accumulate = match wire {
            DataTypeEnum::F16 => DataTypeEnum::F32,
            other => other,
        };
        let identity = Tile::literal(tile_literal_for(inner.function.initial_value, accumulate));
        let op = tile_reduce_op(inner.function.op);
        let [folded] = program.fold(
            tile_ir::tile::range(inner.len as u32),
            [identity],
            |program, index, [acc]| {
                let mut inner_coords = coords.to_vec();
                inner_coords.push(index);
                let (value, _) = eval_nary_expr(
                    program,
                    &inner.expression,
                    &inner_coords,
                    storages,
                    metas,
                    mask.clone(),
                    &[],
                );
                let value = raw_tile(value.cast_to(accumulate));
                [acc.binary(op.binary(), value)]
            },
        );
        let folded = if accumulate == wire {
            folded
        } else {
            raw_tile(ValueTile::F32(program.bind(folded)).cast_to(wire))
        };
        extras.push((typed_tile(program.bind(folded), wire), wire));
    }
    let (value, _) = eval_nary_expr(
        program,
        &carrier.expression,
        coords,
        storages,
        metas,
        mask,
        &extras,
    );
    raw_tile(value.cast_to(DataTypeEnum::F32))
}

/// Evaluate one carrier body. `init`, `step`, `combine` and the outputs are
/// ordinary [`NaryExpr`]s over the slot convention, so evaluation is the same
/// `eval_nary_expr` every other expression in the compiler goes through, with
/// the bindings supplied as extras.
fn eval_carrier_expr(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expression: &NaryExpr,
    storages: &[crate::nary_direct::Storage2],
    metas: &[crate::nary_direct::TensorMeta],
    bindings: &[(ValueTile, DataTypeEnum)],
    datatype: DataTypeEnum,
) -> Tile {
    let (value, _) = eval_nary_expr(
        program,
        expression,
        &[],
        storages,
        metas,
        Mask::from(true),
        bindings,
    );
    raw_tile(value.cast_to(datatype))
}

fn carrier_bindings(carrier: &FoldOperation, values: &[Tile]) -> Vec<(ValueTile, DataTypeEnum)> {
    carrier
        .carrier
        .iter()
        .zip(values)
        .map(|(slot, value)| (typed_tile(value.clone(), slot.datatype), slot.datatype))
        .collect()
}

/// Streaming lowering for a joint carrier: one workgroup per row makes one
/// pass over the axis in `block`-wide tiles, and every tile advances the whole
/// carrier at once.
///
/// Within a tile the lanes change role, which is what a joint carrier over a
/// free dimension forces:
///
/// 1. lane = one axis position. Each lane absorbs its own element into a
///    *fresh* carrier (`step` applied to `init`) and the block's partial
///    carriers are joined across lanes with `combine`
///    ([`tile_ir::tile::TileBlock::group_reduce_with_vec`]), giving the tile's
///    scalar statistics. The elements are staged through workgroup memory.
/// 2. lane = one position of the free dimension. The free-dimension slots fold
///    the staged elements with the scalar slots pinned at their completed tile
///    values, then the tile's carrier is joined onto the running one.
///
/// Long axes over few rows fan out across workgroups: each split writes its
/// partial carrier record to scratch and a combine kernel folds the spans with
/// the same `combine`. That is [`FoldOperation::split`] instantiated in tile
/// space, with ragged runtime-bounded spans an `init` (identity) carrier
/// absorbs harmlessly.
fn build_streaming_fold_kernel(
    operation: &RowProgramOperation,
    dynamic: &DynamicAxis,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let carrier = dynamic.carrier.clone();
    let (output, producers) = inputs.split_last()?;
    let output = output.as_tensor()?.clone();
    let (params, producers) = producers.split_last()?;
    let params = params.as_tensor()?.clone();
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
    let width = carrier.width();
    let (slot_offsets, record) = carrier.slot_offsets();
    let output_spec = carrier.outputs.first()?.clone();
    let free_dim = carrier.output_free_dim(&output_spec);
    // Lanes serve free-dimension positions in the second half of a tile, so a
    // free dimension wider than the workgroup would silently drop positions.
    if free_dim.is_some_and(|free| free as u32 > block) {
        return None;
    }
    let lanes_used = free_dim.unwrap_or(1) as u32;
    let scalar_slots: Vec<usize> = carrier
        .carrier
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.free_dim.is_none().then_some(index))
        .collect();

    // Long axes with few rows fan out across workgroups: each split folds one
    // span and writes its carrier record; a combine kernel joins the spans.
    let tiles = k.div_ceil(block);
    let splits: u32 = if free_dim.is_some()
        && graph
            .device()
            .dispatch_policy()
            .should_split_for_occupancy(rows, block)
        && tiles > 1
        && tiles <= block
        && record as u32 <= block
    {
        tiles
    } else {
        1
    };

    let max_dispatch_dim = graph.device().limits().max_compute_workgroups_per_dimension;
    let dispatch_spec = crate::row_dispatch::RowDispatchSpec::distributed(
        rows.saturating_mul(splits),
        block,
        max_dispatch_dim,
    );
    let dispatch_size = dispatch_spec.dispatch_size;
    let cache_key =
        row_program_cache_key(operation, workgroup_shape, dispatch_size, inputs, 1, splits);

    let axis_bound_dim = dynamic.axis_bound_dim;
    let output_dtype = output.datatype();
    let output_value = MaybeQData::Tensor(output);
    let params_value = MaybeQData::Tensor(params);
    let scratch_value = (splits > 1).then(|| {
        MaybeQData::Tensor(TensorData::new_for_shape(
            &graph.device(),
            &[rows as usize, splits as usize, record],
            DataTypeEnum::F32,
        ))
    });

    let combine_kernel = match &scratch_value {
        Some(scratch) => Some(build_fold_span_combine_kernel(
            operation,
            &carrier,
            graph,
            workgroup_shape,
            inputs,
            scratch.clone(),
            output_value.clone(),
            &row_shape,
            rows,
            splits,
            lanes_used,
            output_dtype,
        )?),
        None => None,
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
            let (params_storage, _) = declare_value(kb, &params_value, false)?;
            let (output_storage, output_meta) = match &scratch_value {
                Some(scratch) => declare_value(kb, scratch, true)?,
                None => declare_value(kb, &output_value, true)?,
            };

            let phase_handle = kb.program();
            // The bridge between the two lane roles: every lane's element,
            // read back by the lanes serving free-dimension positions.
            let staged = phase_handle.alloc_workgroup_array(ScalarElement::F32, block);

            crate::row_dispatch::emit_row_grid(phase_handle, dispatch_spec, |program, ctx| {
                let (row_flat, split_idx, lane) = if splits > 1 {
                    (
                        program.bind(ctx.row.clone() / splits),
                        program.bind(ctx.row % splits),
                        ctx.lane,
                    )
                } else {
                    (ctx.row, tile_u32(0), ctx.lane)
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

                // The active axis length rides in the params input, so a
                // growing KV cache reuses one compiled kernel.
                let axis_len: Tile =
                    raw_tile(params_storage.load(program, tile_u32(0), Mask::all()));
                // Causal bound: tiles past the bounding row coordinate hold no
                // live positions, so the loop ends there. Uniform per
                // workgroup, which keeps the tile loop's break uniform.
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
                let (span_start, span_end) = if splits > 1 {
                    let start = program.bind(split_idx.clone() * block);
                    let end =
                        program.bind((start.clone() + tile_u32(block)).min(effective_len.clone()));
                    (start, end)
                } else {
                    (tile_u32(0), effective_len.clone())
                };

                // The running carrier and, per tile, the free-dimension slots'
                // own partial. Both are one private per lane: a free-dimension
                // slot's lane *is* its free coordinate.
                let running: Vec<_> = carrier
                    .carrier
                    .iter()
                    .map(|slot| program.private(datatype_element(slot.datatype)))
                    .collect();
                let tile_slots: Vec<_> = carrier
                    .carrier
                    .iter()
                    .map(|slot| program.private(datatype_element(slot.datatype)))
                    .collect();
                let tile_base = program.private(ElementType::U32);
                let item = program.private(ElementType::U32);

                let init: Vec<Tile> = carrier
                    .init
                    .iter()
                    .zip(&carrier.carrier)
                    .map(|(expression, slot)| {
                        let value = eval_carrier_expr(
                            program,
                            expression,
                            &storages,
                            &metas,
                            &[],
                            slot.datatype,
                        );
                        program.bind(value)
                    })
                    .collect();
                for (local, value) in running.iter().zip(&init) {
                    program.store_local(local, value.clone());
                }
                program.store_local(&tile_base, span_start);
                let out_active = in_bounds.clone() & lane.clone().lt(lanes_used);

                program.loop_forever(|program| {
                    let base = program.load_local(&tile_base);
                    program.break_if(base.clone().ge(span_end.clone()));
                    let kv = program.bind(base.clone() + lane.clone());
                    let kv_active = in_bounds.clone() & kv.clone().lt(effective_len.clone());
                    let coords = full_coords(kv);

                    // Lane = one axis position: absorb this lane's element into
                    // a fresh carrier, then join the block's partials.
                    let element = emit_fold_element(
                        program,
                        &carrier,
                        &coords,
                        &storages,
                        &metas,
                        kv_active.clone(),
                    );
                    let element = program.bind(element);
                    let mut bindings = carrier_bindings(&carrier, &init);
                    bindings.push((ValueTile::F32(element.clone()), DataTypeEnum::F32));
                    // Only the scalar slots take part in the cross-lane join;
                    // the free-dimension ones are folded in the second lane
                    // role, so staging them would burn workgroup memory on
                    // values nothing reads.
                    let per_lane: Vec<Tile> = scalar_slots
                        .iter()
                        .map(|&index| {
                            let value = eval_carrier_expr(
                                program,
                                &carrier.step[index],
                                &storages,
                                &metas,
                                &bindings,
                                carrier.carrier[index].datatype,
                            );
                            program.bind(Tile::select(
                                kv_active.clone(),
                                value,
                                init[index].clone(),
                            ))
                        })
                        .collect();
                    let scalar_joined = {
                        let carrier = carrier.clone();
                        let storages = storages.clone();
                        let metas = metas.clone();
                        let scalar_slots = scalar_slots.clone();
                        let init = init.clone();
                        program.group_reduce_with_vec(
                            block,
                            per_lane,
                            move |program, acc, incoming| {
                                let expand = |partial: &[Tile]| -> Vec<Tile> {
                                    let mut full = init.clone();
                                    for (slot, value) in scalar_slots.iter().zip(partial) {
                                        full[*slot] = value.clone();
                                    }
                                    full
                                };
                                let mut bindings = carrier_bindings(&carrier, &expand(&acc));
                                bindings
                                    .extend(carrier_bindings(&carrier, &expand(&incoming)));
                                scalar_slots
                                    .iter()
                                    .map(|&index| {
                                        eval_carrier_expr(
                                            program,
                                            &carrier.combine[index],
                                            &storages,
                                            &metas,
                                            &bindings,
                                            carrier.carrier[index].datatype,
                                        )
                                    })
                                    .collect()
                            },
                        )
                    };
                    let mut joined = init.clone();
                    for (slot, value) in scalar_slots.iter().zip(scalar_joined) {
                        joined[*slot] = program.bind(value);
                    }

                    // Stage the elements for the free-dimension pass.
                    program.store_workgroup(&staged, lane.clone(), element);
                    program.workgroup_barrier();

                    // Bound, not lazy loads: the carrier is written back slot
                    // by slot, so a re-evaluated `load_local` would see a
                    // sibling slot's *new* value.
                    let old: Vec<Tile> = running
                        .iter()
                        .map(|local| {
                            let value = program.load_local(local);
                            program.bind(value)
                        })
                        .collect();
                    let mut join_bindings = carrier_bindings(&carrier, &old);
                    join_bindings.extend(carrier_bindings(&carrier, &joined));

                    if carrier.has_free_dim() {
                        // Lane = one free-dimension position. The scalar slots
                        // hold their completed tile values, so a slot's step
                        // absorbs each staged element exactly once.
                        program.if_then(out_active.clone(), |program| {
                            for (index, slot) in carrier.carrier.iter().enumerate() {
                                if slot.free_dim.is_some() {
                                    program.store_local(&tile_slots[index], init[index].clone());
                                }
                            }
                            program.store_local(&item, tile_u32(0));
                            program.loop_forever(|program| {
                                let j = program.load_local(&item);
                                let kv_j = program.bind(base.clone() + j.clone());
                                program.break_if(
                                    j.clone().ge(block) | kv_j.clone().ge(effective_len.clone()),
                                );
                                let staged_element = program.load_workgroup(&staged, j.clone());
                                let mut inner_coords = full_coords(kv_j);
                                inner_coords.push(lane.clone());
                                for (index, slot) in carrier.carrier.iter().enumerate() {
                                    let Some(slot_element) = &slot.element else {
                                        continue;
                                    };
                                    let (value, _) = eval_nary_expr(
                                        program,
                                        slot_element,
                                        &inner_coords,
                                        &storages,
                                        &metas,
                                        Mask::all(),
                                        &[],
                                    );
                                    let value = raw_tile(value.cast_to(slot.datatype));
                                    let mut values = joined.clone();
                                    values[index] = {
                                        let value = program.load_local(&tile_slots[index]);
                                        program.bind(value)
                                    };
                                    let mut bindings = carrier_bindings(&carrier, &values);
                                    bindings.push((
                                        ValueTile::F32(staged_element.clone()),
                                        DataTypeEnum::F32,
                                    ));
                                    bindings
                                        .push((typed_tile(value, slot.datatype), slot.datatype));
                                    let next = eval_carrier_expr(
                                        program,
                                        &carrier.step[index],
                                        &storages,
                                        &metas,
                                        &bindings,
                                        slot.datatype,
                                    );
                                    let next = program.bind(next);
                                    program.store_local(&tile_slots[index], next);
                                }
                                program.store_local(&item, j + tile_u32(1));
                            });
                            // Join the tile's free-dimension slots onto the
                            // running carrier with the same `combine`.
                            for (index, slot) in carrier.carrier.iter().enumerate() {
                                if slot.free_dim.is_none() {
                                    continue;
                                }
                                let mut values = join_bindings.clone();
                                values[width + index] = (
                                    typed_tile(
                                        program.load_local(&tile_slots[index]),
                                        slot.datatype,
                                    ),
                                    slot.datatype,
                                );
                                let next = eval_carrier_expr(
                                    program,
                                    &carrier.combine[index],
                                    &storages,
                                    &metas,
                                    &values,
                                    slot.datatype,
                                );
                                let next = program.bind(next);
                                program.store_local(&running[index], next);
                            }
                        });
                    }
                    let scalar_next: Vec<Tile> = scalar_slots
                        .iter()
                        .map(|&index| {
                            let next = eval_carrier_expr(
                                program,
                                &carrier.combine[index],
                                &storages,
                                &metas,
                                &join_bindings,
                                carrier.carrier[index].datatype,
                            );
                            program.bind(next)
                        })
                        .collect();
                    for (&index, value) in scalar_slots.iter().zip(scalar_next) {
                        program.store_local(&running[index], value);
                    }
                    program.workgroup_barrier();
                    program.store_local(&tile_base, base + tile_u32(block));
                });

                match &scratch_value {
                    Some(_) => {
                        // Partials: this span's carrier record, folded by the
                        // combine kernel with the same monoid.
                        for (index, slot) in carrier.carrier.iter().enumerate() {
                            let offset = slot_offsets[index] as u32;
                            if slot.free_dim.is_some() {
                                program.if_then(out_active.clone(), |program| {
                                    let position = program.bind(lane.clone() + tile_u32(offset));
                                    let store_index = layout_index(
                                        &output_meta,
                                        &[row_flat.clone(), split_idx.clone(), position],
                                    );
                                    let value = ValueTile::F32(program.load_local(&running[index]));
                                    output_storage.store(program, store_index, value, Mask::all());
                                });
                            } else {
                                let active = in_bounds.clone()
                                    & lane.clone().ge(offset)
                                    & lane.clone().lt(offset + 1);
                                program.if_then(active, |program| {
                                    let store_index = layout_index(
                                        &output_meta,
                                        &[
                                            row_flat.clone(),
                                            split_idx.clone(),
                                            tile_u32(offset),
                                        ],
                                    );
                                    let value = ValueTile::F32(program.load_local(&running[index]));
                                    output_storage.store(program, store_index, value, Mask::all());
                                });
                            }
                        }
                    }
                    None => {
                        program.if_then(out_active, |program| {
                            let values: Vec<Tile> = running
                                .iter()
                                .map(|local| {
                                    let value = program.load_local(local);
                                    program.bind(value)
                                })
                                .collect();
                            let bindings = carrier_bindings(&carrier, &values);
                            let value = eval_carrier_expr(
                                program,
                                &output_spec.expression,
                                &storages,
                                &metas,
                                &bindings,
                                DataTypeEnum::F32,
                            );
                            let mut out_coords = row_dims.clone();
                            if free_dim.is_some() {
                                out_coords.push(lane.clone());
                            }
                            let store_index = layout_index(&output_meta, &out_coords);
                            output_storage.store(
                                program,
                                store_index,
                                ValueTile::F32(value).cast_to(output_dtype),
                                Mask::all(),
                            );
                        });
                    }
                }
            });
            Some(())
        },
    )?;

    match combine_kernel {
        Some(combine) => Some(DirectKernel::sequence(
            "row_program_split",
            vec![partials, combine],
        )),
        None => Some(partials),
    }
}

/// Join the span partials a blocked streaming fold wrote.
///
/// One workgroup per row, one lane per free-dimension position. Each lane
/// folds all `splits` records with `combine`, recomputing the scalar slots
/// redundantly — they are a handful of values per span, and duplicating them
/// buys a combine kernel with no cross-lane communication at all.
#[allow(clippy::too_many_arguments)]
fn build_fold_span_combine_kernel(
    operation: &RowProgramOperation,
    carrier: &FoldOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
    scratch: MaybeQData,
    output: MaybeQData,
    row_shape: &[usize],
    rows: u32,
    splits: u32,
    lanes_used: u32,
    output_dtype: DataTypeEnum,
) -> Option<DirectKernel> {
    let block = workgroup_shape.x();
    let max_dispatch_dim = graph.device().limits().max_compute_workgroups_per_dimension;
    let dispatch_spec =
        crate::row_dispatch::RowDispatchSpec::distributed(rows, block, max_dispatch_dim);
    let dispatch_size = dispatch_spec.dispatch_size;
    let key = row_program_cache_key(operation, workgroup_shape, dispatch_size, inputs, 2, splits);
    let carrier = carrier.clone();
    let (slot_offsets, _) = carrier.slot_offsets();
    let output_spec = carrier.outputs.first()?.clone();
    let free_dim = carrier.output_free_dim(&output_spec);
    let row_shape = row_shape.to_vec();
    kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        format!("{}_combine", operation.name()),
        key,
        dispatch_size,
        move |kb| {
            let (scratch_storage, scratch_meta) = declare_value(kb, &scratch, false)?;
            let (output_storage, output_meta) = declare_value(kb, &output, true)?;
            crate::row_dispatch::emit_row_grid(kb.program(), dispatch_spec, |program, ctx| {
                let lane = ctx.lane;
                let row_flat = ctx.row;
                let row_dims = output_dims_from_flat(row_flat.clone(), &row_shape);
                let active = ctx.active & lane.clone().lt(lanes_used);
                let running: Vec<_> = carrier
                    .carrier
                    .iter()
                    .map(|slot| program.private(datatype_element(slot.datatype)))
                    .collect();
                let init: Vec<Tile> = carrier
                    .init
                    .iter()
                    .zip(&carrier.carrier)
                    .map(|(expression, slot)| {
                        let bindings = vec![
                            (ValueTile::F32(tile_u32(0)), DataTypeEnum::F32);
                            carrier.base()
                        ];
                        eval_carrier_expr(program, expression, &[], &[], &bindings, slot.datatype)
                    })
                    .collect();
                for (local, value) in running.iter().zip(init) {
                    program.store_local(local, value);
                }
                program.if_then(active, |program| {
                    program.loop_range(splits, |program, span| {
                        let incoming: Vec<Tile> = carrier
                            .carrier
                            .iter()
                            .enumerate()
                            .map(|(index, slot)| {
                                let offset = slot_offsets[index] as u32;
                                let position = match slot.free_dim {
                                    Some(_) => program.bind(lane.clone() + tile_u32(offset)),
                                    None => tile_u32(offset),
                                };
                                let read = layout_index(
                                    &scratch_meta,
                                    &[row_flat.clone(), span.clone(), position],
                                );
                                raw_tile(scratch_storage.load(program, read, Mask::all()))
                            })
                            .collect();
                        let current: Vec<Tile> = running
                            .iter()
                            .map(|local| {
                                let value = program.load_local(local);
                                program.bind(value)
                            })
                            .collect();
                        let mut bindings = vec![
                            (ValueTile::F32(tile_u32(0)), DataTypeEnum::F32);
                            carrier.base()
                        ];
                        bindings.extend(carrier_bindings(&carrier, &current));
                        bindings.extend(carrier_bindings(&carrier, &incoming));
                        let next: Vec<Tile> = carrier
                            .carrier
                            .iter()
                            .enumerate()
                            .map(|(index, slot)| {
                                let value = eval_carrier_expr(
                                    program,
                                    &carrier.combine[index],
                                    &[],
                                    &[],
                                    &bindings,
                                    slot.datatype,
                                );
                                program.bind(value)
                            })
                            .collect();
                        for (local, value) in running.iter().zip(next) {
                            program.store_local(local, value);
                        }
                    });
                    let values: Vec<Tile> = running
                        .iter()
                        .map(|local| program.load_local(local))
                        .collect();
                    let mut bindings = vec![
                        (ValueTile::F32(tile_u32(0)), DataTypeEnum::F32);
                        carrier.base()
                    ];
                    bindings.extend(carrier_bindings(&carrier, &values));
                    let value = eval_carrier_expr(
                        program,
                        &output_spec.expression,
                        &[],
                        &[],
                        &bindings,
                        DataTypeEnum::F32,
                    );
                    let mut out_coords = row_dims.clone();
                    if free_dim.is_some() {
                        out_coords.push(lane.clone());
                    }
                    let store_index = layout_index(&output_meta, &out_coords);
                    output_storage.store(
                        program,
                        store_index,
                        ValueTile::F32(value).cast_to(output_dtype),
                        Mask::all(),
                    );
                });
            });
            Some(())
        },
    )
}

fn build_row_program_kernel(
    operation: &RowProgramOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    // A dynamic axis means a joint carrier streamed over the axis, which is a
    // fold rather than a phase list — everything below is the chunked-map
    // path and reads `steps`.
    if let Some(dynamic) = &operation.dynamic_axis {
        return build_streaming_fold_kernel(operation, dynamic, graph, workgroup_shape, inputs);
    }
    // Validation spike: re-express every built-in phase as the equivalent
    // general combine, so the general path runs on real reductions and must
    // reproduce the closed-operator results bit for bit.
    let rewritten;
    let operation = if graph.device().config().spike_general_combine
        && let Some(general) = operation.as_general_combines()
    {
        rewritten = general;
        &rewritten
    } else {
        operation
    };
    // Reject before emitting rather than panicking inside it: a general
    // combine is a capability gap in the cross-lane reduction, and the caller
    // already surfaces a missing direct kernel as a lowering error.
    if operation.has_general_combine() && !graph.device().config().spike_general_combine {
        return None;
    }
    let (output, producers) = inputs.split_last()?;
    let output = output.as_tensor()?.clone();
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

    let output_kind = operation.output_step().clone();
    let phase_steps = operation.phase_steps().to_vec();
    // Chunked programs pack several rows per workgroup: lanes split into
    // `block / k_group` groups of `k_group` lanes, each group owning one row,
    // reduced with per-group reductions. Without packing a k=64 reduce runs
    // one row per 256-lane workgroup with 75% of the lanes idle (and a k=1
    // bias-grad sum runs one workgroup per output scalar).
    let subgroups = fixed_subgroups(&graph.device());
    let k_group: u32 = lane_group_width(&graph.device().dispatch_policy(), subgroups, k, block);
    let rows_per_workgroup = block / k_group;
    let dispatch_rows = rows.div_ceil(rows_per_workgroup);

    let max_dispatch_dim = graph.device().limits().max_compute_workgroups_per_dimension;
    let dispatch_spec =
        crate::row_dispatch::RowDispatchSpec::distributed(dispatch_rows, block, max_dispatch_dim);
    let dispatch_size = dispatch_spec.dispatch_size;
    let cache_key = row_program_cache_key(operation, workgroup_shape, dispatch_size, inputs, 1, 1);

    let input_count = operation.inputs.len();
    let stage_budget = graph
        .device()
        .dispatch_policy()
        .work_per_thread(crate::occupancy::RegPressure::ElementwiseFew);
    // Chunked-map programs whose axis fits the per-lane register budget stage
    // each distinct tensor read once per chunk and evaluate every phase (and
    // the output) from the registers instead of re-reading storage per phase.
    let staged_reads = (k.div_ceil(k_group) <= stage_budget)
        .then(|| stage_chunk_reads(&phase_steps, &output_kind, input_count))
        .flatten();
    let output_dtype = output.datatype();
    let output_value = MaybeQData::Tensor(output);

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
            let (output_storage, output_meta) = declare_value(kb, &output_value, true)?;

            let phase_handle = kb.program();
            crate::row_dispatch::emit_row_grid(phase_handle, dispatch_spec, |program, ctx| {
                let wg_flat = ctx.row;
                // With packing, `lane` becomes the position inside the row's
                // `k_group`-wide lane group; every later axis index and the
                // scalar-store lane test use it unchanged.
                let (row_flat, lane) = if rows_per_workgroup > 1 {
                    let (row_local, lane) =
                        split_lane_groups(program, subgroups, ctx.lane, k_group, block);
                    (
                        program.bind(wg_flat * rows_per_workgroup + row_local),
                        lane,
                    )
                } else {
                    (wg_flat, ctx.lane)
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

                let mut slots: Vec<(ValueTile, DataTypeEnum)> = Vec::new();

                // Chunked map program: lanes stride the axis by their lane
                // group's width for each phase's fold. Programs within the
                // register budget stage each distinct tensor read per chunk up
                // front (`staged_reads`) and unroll the stride; longer axes
                // roll the stride into a loop and re-evaluate per phase — the
                // same trade every multi-pass normalization kernel makes.
                let chunks = k.div_ceil(k_group);
                let staged_values = staged_reads.as_ref().map(|staged| {
                    stage_chunk_values(
                        program,
                        &staged.probes,
                        chunks,
                        k,
                        k_group,
                        &lane,
                        &in_bounds,
                        &full_coords,
                        &storages,
                        &metas,
                    )
                });
                for (phase_index, phase) in phase_steps.iter().enumerate() {
                    let RowStep::Reduce(reduce) = phase else {
                        unreachable!("element phases require a dynamic-axis row program")
                    };
                    let function_dtype = reduce.combine.datatype();
                    // Half-precision folds accumulate in f32 and round once
                    // after the group reduction — the same accumulator policy
                    // as the matmul kernels and the composed fused reduce.
                    let phase_dtype = match function_dtype {
                        DataTypeEnum::F16 => DataTypeEnum::F32,
                        other => other,
                    };
                    let identity = || {
                        Tile::literal(tile_literal_for(reduce.combine.initial_value(), phase_dtype))
                    };
                    // Only the built-in branch consults this; both dispatch
                    // sites below match on the combine kind first, so a
                    // general combine never reads it.
                    let reduce_op = reduce
                        .combine
                        .built_in()
                        .map(|function| tile_reduce_op(function.op))
                        .unwrap_or(tile_ir::TileReduceOp::Sum);
                    // Built-ins keep the closed-operator path exactly; a
                    // general combine evaluates its own body. The kernel
                    // builder rejects general combines up front today, so this
                    // is inert for every program that currently lowers.
                    let combine_kind = &reduce.combine;
                    let accumulate = |program: &mut tile_ir::tile::TileBlock<'_>,
                                      acc: Tile,
                                      value: Tile,
                                      slots: &[(ValueTile, DataTypeEnum)]|
                     -> Tile {
                        match combine_kind {
                            RowCombine::BuiltIn(_) => acc.binary(reduce_op.binary(), value),
                            RowCombine::General { step, .. } => {
                                emit_carrier_body(
                                    program, step, &storages, &metas, slots, acc, value,
                                    phase_dtype,
                                )
                            }
                        }
                    };
                    let chunk_value = |program: &mut tile_ir::tile::TileBlock<'_>,
                                       k_index: Tile,
                                       chunk_reads: Option<&Vec<(ValueTile, DataTypeEnum)>>,
                                       slots: &[(ValueTile, DataTypeEnum)]|
                     -> Tile {
                        let active = in_bounds.clone() & k_index.clone().lt(k);
                        let coords = full_coords(k_index);
                        let (value, _) = match (&staged_reads, chunk_reads) {
                            (Some(staged), Some(chunk_reads)) => {
                                let mut extras = chunk_reads.clone();
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
                                slots,
                            ),
                        };
                        let value = raw_tile(value.cast_to(phase_dtype));
                        Tile::select(active, value, identity())
                    };
                    let partial = match &staged_values {
                        Some(staged_values) => {
                            let mut acc = identity();
                            for (chunk, chunk_reads) in staged_values.iter().enumerate() {
                                let k_index = lane.clone() + tile_u32(chunk as u32 * k_group);
                                let masked =
                                    chunk_value(program, k_index, Some(chunk_reads), &slots);
                                acc = accumulate(program, acc, masked, &slots);
                            }
                            acc
                        }
                        None => {
                            let [partial] = program.fold(
                                tile_ir::tile::range(chunks),
                                [identity()],
                                |program, chunk, [acc]| {
                                    let k_index = chunk * k_group + lane.clone();
                                    let masked = chunk_value(program, k_index, None, &slots);
                                    [accumulate(program, acc, masked, &slots)]
                                },
                            );
                            partial
                        }
                    };
                    // The per-group workgroup reduction broadcasts: every lane
                    // in the row's `k_group`-wide lane group reads the combined
                    // value, so later phases can use it directly. With one row
                    // per workgroup `k_group == block` and this is the full
                    // workgroup reduction.
                    let combined =
                        emit_group_reduce_combined(
                            program,
                            subgroups,
                            combine_kind,
                            &storages,
                            &metas,
                            &slots,
                            phase_dtype,
                            reduce_op,
                            k_group,
                            block,
                            partial,
                        );
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
                                   chunk_reads: Option<&Vec<(ValueTile, DataTypeEnum)>>,
                                   slots: &[(ValueTile, DataTypeEnum)]|
                 -> ValueTile {
                    let (value, _) = match (&staged_reads, chunk_reads) {
                        (Some(staged), Some(chunk_reads)) => {
                            let mut extras = chunk_reads.clone();
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
                        let store_chunk = |program: &mut tile_ir::tile::TileBlock<'_>,
                                               k_index: Tile,
                                               chunk_reads: Option<
                            &Vec<(ValueTile, DataTypeEnum)>,
                        >| {
                            let active = in_bounds.clone() & k_index.clone().lt(k);
                            let coords = full_coords(k_index);
                            let value = eval_output(
                                program,
                                output_expr,
                                &coords,
                                active.clone(),
                                chunk_reads,
                                &slots,
                            );
                            let value = value.cast_to(output_dtype);
                            let output_index = layout_index(&output_meta, &coords);
                            output_storage.store(program, output_index, value, active);
                        };
                        match &staged_values {
                            Some(staged_values) => {
                                for (chunk, chunk_reads) in staged_values.iter().enumerate() {
                                    let k_index = lane.clone() + tile_u32(chunk as u32 * k_group);
                                    store_chunk(program, k_index, Some(chunk_reads));
                                }
                            }
                            None => program.loop_range(chunks, |program, chunk| {
                                store_chunk(program, chunk * k_group + lane.clone(), None);
                            }),
                        }
                    }
                    RowOutput::Scalar(output_expr) => {
                        let active = in_bounds & lane.eq(0u32);
                        let coords = full_coords(tile_u32(0));
                        // Only lane zero of the group stores, and chunk zero is
                        // the chunk it staged position zero of the axis in.
                        let chunk_reads = staged_values.as_ref().map(|values| &values[0]);
                        let value = eval_output(
                            program,
                            output_expr,
                            &coords,
                            active.clone(),
                            chunk_reads,
                            &slots,
                        );
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

    Some(partials)
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
    // The shared workgroup width is the widest any segment would pick on its
    // own (`static_axis_block`), so a merge never widens a segment's lane
    // groups relative to its unmerged lowering.
    let policy = device.dispatch_policy();
    let subgroups = fixed_subgroups(&device);
    let block = segments
        .iter()
        .map(|op| static_axis_block(&device, op.shape[op.axis]))
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
        let k_group = lane_group_width(&policy, subgroups, k, block);
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
    let stage_budget = policy.work_per_thread(crate::occupancy::RegPressure::ElementwiseFew);
    // Segments within the per-lane register budget stage each distinct tensor
    // read per chunk (see `stage_chunk_reads`); the rewrite is deterministic
    // from the segment's steps, so it stays out of the cache key.
    let staged_segments: Vec<Option<StagedReads>> = segments
        .iter()
        .zip(&prepared)
        .map(|(op, segment)| {
            (segment.k.div_ceil(segment.k_group) <= stage_budget)
                .then(|| stage_chunk_reads(op.phase_steps(), op.output_step(), op.inputs.len()))
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
                            let (row_local, lane) = split_lane_groups(
                                program,
                                subgroups,
                                full_lane.clone(),
                                segment.k_group,
                                block,
                            );
                            (
                                program.bind(
                                    wg_local.clone() * segment.rows_per_workgroup + row_local,
                                ),
                                lane,
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
                        let k_group = segment.k_group;
                        let chunks = k.div_ceil(k_group);
                        let staged_values = staged_reads.as_ref().map(|staged| {
                            stage_chunk_values(
                                program,
                                &staged.probes,
                                chunks,
                                k,
                                k_group,
                                &lane,
                                &in_bounds,
                                &full_coords,
                                storages,
                                metas,
                            )
                        });
                        let mut slots: Vec<(ValueTile, DataTypeEnum)> = Vec::new();
                        for (phase_index, phase) in op.phase_steps().iter().enumerate() {
                            let RowStep::Reduce(reduce) = phase else {
                                unreachable!("merged row programs are reduce-phase only")
                            };
                            let function_dtype = reduce.combine.datatype();
                            // Same f32 accumulation policy as the standalone
                            // reduce phases above.
                            let phase_dtype = match function_dtype {
                                DataTypeEnum::F16 => DataTypeEnum::F32,
                                other => other,
                            };
                            let identity = || {
                                Tile::literal(tile_literal_for(
                                    reduce.combine.initial_value(),
                                    phase_dtype,
                                ))
                            };
                            // Only the built-in branch consults this; both dispatch
                    // sites below match on the combine kind first, so a
                    // general combine never reads it.
                    let reduce_op = reduce
                        .combine
                        .built_in()
                        .map(|function| tile_reduce_op(function.op))
                        .unwrap_or(tile_ir::TileReduceOp::Sum);
                    // Built-ins keep the closed-operator path exactly; a
                    // general combine evaluates its own body. The kernel
                    // builder rejects general combines up front today, so this
                    // is inert for every program that currently lowers.
                    let combine_kind = &reduce.combine;
                    let accumulate = |program: &mut tile_ir::tile::TileBlock<'_>,
                                      acc: Tile,
                                      value: Tile,
                                      slots: &[(ValueTile, DataTypeEnum)]|
                     -> Tile {
                        match combine_kind {
                            RowCombine::BuiltIn(_) => acc.binary(reduce_op.binary(), value),
                            RowCombine::General { step, .. } => {
                                emit_carrier_body(
                                    program, step, &storages, &metas, slots, acc, value,
                                    phase_dtype,
                                )
                            }
                        }
                    };
                            let chunk_value =
                                |program: &mut tile_ir::tile::TileBlock<'_>,
                                 k_index: Tile,
                                 chunk_reads: Option<&Vec<(ValueTile, DataTypeEnum)>>,
                                 slots: &[(ValueTile, DataTypeEnum)]|
                                 -> Tile {
                                    let active = in_bounds.clone() & k_index.clone().lt(k);
                                    let coords = full_coords(k_index);
                                    let (value, _) = match (&staged_reads, chunk_reads) {
                                        (Some(staged), Some(chunk_reads)) => {
                                            let mut extras = chunk_reads.clone();
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
                                            slots,
                                        ),
                                    };
                                    let value = raw_tile(value.cast_to(phase_dtype));
                                    Tile::select(active, value, identity())
                                };
                            let partial = match &staged_values {
                                Some(staged_values) => {
                                    let mut acc = identity();
                                    for (chunk, chunk_reads) in staged_values.iter().enumerate() {
                                        let k_index =
                                            lane.clone() + tile_u32(chunk as u32 * k_group);
                                        let masked = chunk_value(
                                            program,
                                            k_index,
                                            Some(chunk_reads),
                                            &slots,
                                        );
                                        acc = accumulate(program, acc, masked, &slots);
                                    }
                                    acc
                                }
                                None => {
                                    let [partial] = program.fold(
                                        tile_ir::tile::range(chunks),
                                        [identity()],
                                        |program, chunk, [acc]| {
                                            let k_index = chunk * k_group + lane.clone();
                                            let masked =
                                                chunk_value(program, k_index, None, &slots);
                                            [accumulate(program, acc, masked, &slots)]
                                        },
                                    );
                                    partial
                                }
                            };
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
                                    ValueTile::F32(program.bind(combined)).cast_to(function_dtype),
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
                                           chunk_reads: Option<&Vec<(ValueTile, DataTypeEnum)>>,
                                           slots: &[(ValueTile, DataTypeEnum)]|
                         -> ValueTile {
                            let (value, _) = match (&staged_reads, chunk_reads) {
                                (Some(staged), Some(chunk_reads)) => {
                                    let mut extras = chunk_reads.clone();
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
                                let store_chunk = |program: &mut tile_ir::tile::TileBlock<'_>,
                                                   k_index: Tile,
                                                   chunk_reads: Option<
                                    &Vec<(ValueTile, DataTypeEnum)>,
                                >| {
                                    let active = in_bounds.clone() & k_index.clone().lt(k);
                                    let coords = full_coords(k_index);
                                    let value = eval_output(
                                        program,
                                        output_expr,
                                        &coords,
                                        active.clone(),
                                        chunk_reads,
                                        &slots,
                                    );
                                    let value = value.cast_to(output_dtype);
                                    let output_index = layout_index(output_meta, &coords);
                                    output_storage.store(program, output_index, value, active);
                                };
                                match &staged_values {
                                    Some(staged_values) => {
                                        for (chunk, chunk_reads) in staged_values.iter().enumerate()
                                        {
                                            let k_index =
                                                lane.clone() + tile_u32(chunk as u32 * k_group);
                                            store_chunk(program, k_index, Some(chunk_reads));
                                        }
                                    }
                                    None => program.loop_range(chunks, |program, chunk| {
                                        store_chunk(program, chunk * k_group + lane.clone(), None);
                                    }),
                                }
                            }
                            RowOutput::Scalar(output_expr) => {
                                let active = in_bounds.clone() & lane.clone().eq(0u32);
                                let coords = full_coords(tile_u32(0));
                                // Lane zero of the group stores, and chunk zero
                                // is where it staged axis position zero.
                                let chunk_reads = staged_values.as_ref().map(|values| &values[0]);
                                let value = eval_output(
                                    program,
                                    output_expr,
                                    &coords,
                                    active.clone(),
                                    chunk_reads,
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
    // The inner fold's value: the score `q·k` the scaled-score expression
    // reads. `fold_value` and `slot_expr` name the same binding.
    let slot = |p: usize| slot_expr(input_count, p);
    debug_assert_eq!(slot(0), crate::fold::fold_value(input_count));
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
    let mut input_axis_dims = vec![None, Some(2), Some(2)];
    if mask.is_some() {
        input_axis_dims.push(Some(1));
    }
    let shape: Box<[usize]> = [batch, num_heads, q_seq_len, kv_len].into();
    // The whole program is one carrier: the running score maximum, the
    // softmax normalizer, and the `Σ p·v` accumulator over the head dim,
    // advancing together over the KV axis.
    let carrier = crate::fold::streaming_attention_carrier(
        graph_inputs.clone(),
        crate::fold::ElementFold {
            expression: binary(NaryOp::Mul, q_read, k_read),
            len: head_dim,
            function: sum_fn(f32),
        },
        scaled_score(),
        v_read,
        shape.clone(),
        3,
        head_dim,
        max_fn(f32).initial_value,
        input_dtype,
    );
    debug_assert!(carrier.validate().is_ok(), "{:?}", carrier.validate());
    Some(RowProgramOperation {
        inputs: graph_inputs,
        shape,
        axis: 3,
        steps: Vec::new(),
        output_datatype: input_dtype,
        dynamic_axis: Some(DynamicAxis {
            block,
            input_axis_dims,
            axis_bound_dim: causal.then_some(2),
            carrier,
        }),
    })
}
