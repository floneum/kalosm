use std::hash::Hash;

use crate::{
    DataTypeEnum, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    mir::{
        inputs::MirValue,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_wise::{ElementwiseOperation, NaryExpr, NaryOp, NaryScalar},
    visit_tiled::distribute_workgroups,
};

const BLOCKSIZE: u32 = 256;

/// One stage of a view: a quasi-affine relayout of the space below it.
///
/// `layout` maps the stage's output coordinates to flat indices in the
/// row-major space of `input_shape` (the stage below, or the input node's
/// logical value space). `defined` is a prefix box of `layout.shape()`:
/// coordinates with `coord_i < defined[i]` for every axis read data;
/// anything outside reads `fill`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewStage {
    pub(crate) layout: Layout,
    pub(crate) input_shape: Box<[usize]>,
    pub(crate) defined: Box<[usize]>,
    pub(crate) fill: NaryScalar,
}

impl ViewStage {
    pub(crate) fn fully_defined(layout: Layout, input_shape: impl Into<Box<[usize]>>) -> Self {
        let defined = layout.shape().into();
        Self {
            layout,
            input_shape: input_shape.into(),
            defined,
            fill: NaryScalar::U32(0),
        }
    }

    pub(crate) fn is_fully_defined(&self) -> bool {
        self.defined.as_ref() == self.layout.shape()
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Coordinate expressions of the space below, given this stage's output
    /// coordinate expressions: the divmod-free affine per-dim form when one
    /// exists, otherwise the flat index delinearized over `input_shape`.
    /// Returns the coordinates and whether delinearization was needed.
    /// Coordinates are clamped in-bounds when `clamp` is set (callers set it
    /// for partially-defined stages, where out-of-box coordinates still
    /// evaluate the data branch of the fill select).
    pub(crate) fn coords_below(
        &self,
        out: &[NaryExpr],
        clamp: bool,
    ) -> Option<(Vec<NaryExpr>, bool)> {
        let (coords, delinearized) = match affine_dim_indices(&self.layout, &self.input_shape) {
            Some(affine) => (
                affine.iter().map(|index| index.to_expr(out)).collect(),
                false,
            ),
            None => {
                let flat = flat_index_expression(&self.layout, out)?;
                (row_major_indices_from_flat(flat, &self.input_shape)?, true)
            }
        };
        if !clamp {
            return Some((coords, delinearized));
        }
        let clamped = coords
            .into_iter()
            .zip(&*self.input_shape)
            .map(|(index, &extent)| {
                if extent == 0 {
                    index
                } else {
                    NaryExpr::unary_op(
                        index,
                        "clamp_dim",
                        NaryOp::MinConst(NaryScalar::U32(extent as u32 - 1)),
                        DataTypeEnum::U32,
                        DataTypeEnum::U32,
                    )
                }
            })
            .collect();
        Some((clamped, delinearized))
    }

    /// `Some(condition)` over this stage's output coordinate expressions when
    /// partially defined; `None` for a pure relayout.
    pub(crate) fn bounds_condition(&self, out: &[NaryExpr]) -> Option<NaryExpr> {
        if self.is_fully_defined() {
            return None;
        }
        let mut condition = NaryExpr::scalar(NaryScalar::U32(1));
        for (dim, (&defined, &size)) in self.defined.iter().zip(self.layout.shape()).enumerate() {
            if defined >= size {
                continue;
            }
            let lt_defined = NaryExpr::unary_op(
                out[dim].clone(),
                "lt_defined",
                NaryOp::LessConst(NaryScalar::U32(defined as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            condition = NaryExpr::mul(condition, lt_defined, DataTypeEnum::U32);
        }
        Some(condition)
    }
}

/// A zero-dispatch view of a node's logical value space, as a stack of
/// quasi-affine stages.
///
/// `stages[0]` indexes the input node's logical row-major value space;
/// each later stage indexes the output space of the stage below it. Most
/// relayouts merge into the top stage at construction; a stage boundary
/// remains only where the composition is not expressible as one strided
/// layout (a reshape regrouping elements across non-mergeable strides) or
/// where a fill region intervenes. The compiler chooses how a consumer
/// concretizes the map: a zero-cost buffer view when the stack collapses
/// over the buffer's layout, index expressions folded into the consumer's
/// loads, or a gather dispatch materializing the result.
///
/// `input` is never itself a view node: view-over-view always collapses
/// into one node at construction.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewOperation {
    pub(crate) input: NodeIndex,
    /// Innermost first; never empty.
    pub(crate) stages: Vec<ViewStage>,
    pub(crate) datatype: DataTypeEnum,
}

impl ViewOperation {
    /// A fully-defined single-stage view: pure relayout of the input's
    /// logical space.
    pub(crate) fn fully_defined(
        input: NodeIndex,
        layout: Layout,
        input_shape: impl Into<Box<[usize]>>,
        datatype: DataTypeEnum,
    ) -> Self {
        Self {
            input,
            stages: vec![ViewStage::fully_defined(layout, input_shape)],
            datatype,
        }
    }

    pub(crate) fn is_fully_defined(&self) -> bool {
        self.stages.iter().all(ViewStage::is_fully_defined)
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.stages.last().expect("views have stages").shape()
    }

    /// The single stage, when this view is one relayout deep. Pattern
    /// matchers that need a specific affine layout use this and bail on
    /// staged views.
    pub(crate) fn plain(&self) -> Option<&ViewStage> {
        match self.stages.as_slice() {
            [stage] => Some(stage),
            _ => None,
        }
    }

    /// Collapse the stack into one strided layout over the input's logical
    /// space, when every stage is fully defined and the compositions hold.
    pub(crate) fn composed_layout(&self) -> Option<Layout> {
        if !self.is_fully_defined() {
            return None;
        }
        let mut stages = self.stages.iter();
        let mut layout = stages.next()?.layout.clone();
        for stage in stages {
            layout = compose_layouts(&stage.layout, &layout)?;
        }
        Some(layout)
    }

    /// Resolve as a zero-cost view over the input's concrete buffer, if the
    /// stack composes with the buffer's layout as a single strided layout.
    /// Partially-defined views never qualify — their fill region has no
    /// backing memory.
    pub(crate) fn try_map_tensor(&self, input: &TensorData) -> Option<TensorData> {
        if !self.is_fully_defined() {
            return None;
        }
        let mut composed = input.layout().clone();
        for stage in &self.stages {
            composed = compose_layouts(&stage.layout, &composed)?;
        }
        Some(TensorData::new_from_parts(
            input.device(),
            input.buffer().clone(),
            composed,
            input.datatype(),
        ))
    }

    /// The value of this view at the given output coordinate expressions, as
    /// a load of input `input_slot` walked through every stage (with fill
    /// selects where stages are partially defined). Returns the expression
    /// and whether any stage needed delinearization (divmod address
    /// arithmetic) — the signal consumers use to decide whether folding this
    /// view into their loads beats materializing it.
    pub(crate) fn value_expression(
        &self,
        input_slot: usize,
        out: &[NaryExpr],
    ) -> Option<(NaryExpr, bool)> {
        let mut coords: Vec<NaryExpr> = out.to_vec();
        // (condition, fill) per partially-defined stage, outermost first.
        let mut fills: Vec<(NaryExpr, NaryScalar)> = Vec::new();
        let mut delinearized = false;
        for stage in self.stages.iter().rev() {
            if let Some(condition) = stage.bounds_condition(&coords) {
                fills.push((condition, stage.fill));
            }
            let (below, stage_delinearized) =
                stage.coords_below(&coords, !stage.is_fully_defined())?;
            delinearized |= stage_delinearized;
            coords = below;
        }
        let mut value = NaryExpr::IndexedInput {
            input_idx: input_slot,
            indices: coords,
        };
        // Innermost select wraps first so each stage's fill masks everything
        // below it.
        for (condition, fill) in fills.into_iter().rev() {
            value = NaryExpr::select(
                condition,
                value,
                NaryExpr::scalar(fill),
                DataTypeEnum::U32,
                self.datatype,
            );
        }
        Some((value, delinearized))
    }

    /// The gather expression materializing this view: per output coordinate,
    /// load the input at the mapped coordinates, or fill outside the defined
    /// boxes.
    fn copy_expression(&self) -> Option<NaryExpr> {
        let out: Vec<NaryExpr> = (0..self.shape().len()).map(NaryExpr::DimIndex).collect();
        Some(self.value_expression(0, &out)?.0)
    }
}

/// `offset + Σ coords[d] * strides[d]` as a u32 expression.
fn flat_index_expression(layout: &Layout, coords: &[NaryExpr]) -> Option<NaryExpr> {
    let mut flat = NaryExpr::scalar(NaryScalar::U32(layout.offset().try_into().ok()?));
    for (axis, (&stride, &dim)) in layout.strides().iter().zip(layout.shape()).enumerate() {
        if stride == 0 || dim == 1 {
            continue;
        }
        let stride: u32 = stride.try_into().ok()?;
        let coord = coords[axis].clone();
        let term = if stride == 1 {
            coord
        } else {
            NaryExpr::unary_op(
                coord,
                "mul_const",
                NaryOp::MulConst(NaryScalar::U32(stride)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        };
        flat = NaryExpr::add(flat, term, DataTypeEnum::U32);
    }
    Some(flat)
}

pub(crate) fn zero_scalar(datatype: DataTypeEnum) -> NaryScalar {
    match datatype {
        DataTypeEnum::F32 => NaryScalar::F32(0.0),
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(0.0)),
        DataTypeEnum::U32 => NaryScalar::U32(0),
    }
}

/// Re-express `outer` over `inner`'s index space.
///
/// `outer` maps its output coordinates to flat indices in the row-major space
/// of `inner.shape()`; `inner` maps its own coordinates to some target space
/// (a logical input space, or a concrete buffer). The result maps `outer`'s
/// coordinates directly to `inner`'s target space, or `None` when the
/// composition is not expressible as a single strided layout (e.g. a reshape
/// that regroups elements across non-contiguous strides).
///
/// The check is exact: each outer stride and the outer offset are decomposed
/// as mixed-radix digits over `inner`'s contiguity chunks, and the per-chunk
/// digit spans must never carry for any in-range coordinate.
pub(crate) fn compose_layouts(outer: &Layout, inner: &Layout) -> Option<Layout> {
    // Merge inner dims into contiguity chunks (right to left): a run of dims
    // whose strides chain as stride[i] == stride[i+1] * shape[i+1] acts as one
    // flat axis. Size-1 dims are transparent.
    let mut chunks: Vec<(usize, usize)> = Vec::new(); // (extent, target stride)
    for (&dim, &stride) in inner.shape().iter().zip(inner.strides()).rev() {
        if dim == 1 {
            continue;
        }
        match chunks.last_mut() {
            Some((extent, last_stride)) if stride == *last_stride * *extent => {
                *extent *= dim;
            }
            _ => chunks.push((dim, stride)),
        }
    }
    chunks.reverse();
    if chunks.iter().any(|(extent, _)| *extent == 0) {
        return None;
    }

    // Row-major radix strides over the chunk extents: chunk k covers flat
    // positions in steps of radix[k].
    let mut radix = vec![0usize; chunks.len()];
    let mut acc = 1usize;
    for (k, (extent, _)) in chunks.iter().enumerate().rev() {
        radix[k] = acc;
        acc = acc.checked_mul(*extent)?;
    }

    // Mixed-radix decomposition over the chunk extents. Oversized digits are
    // allowed here; the no-carry span check below is the correctness gate.
    let digits = |mut value: usize| -> Option<Vec<usize>> {
        let mut digits = vec![0usize; chunks.len()];
        for (k, radix) in radix.iter().enumerate() {
            digits[k] = value / radix;
            value %= radix;
        }
        (value == 0).then_some(digits)
    };

    let offset_digits = digits(outer.offset())?;
    let stride_digits = outer
        .strides()
        .iter()
        .zip(outer.shape())
        .map(|(&stride, &dim)| {
            if dim <= 1 {
                // A dim that never steps contributes nothing; its stride is
                // irrelevant (and may be a degenerate placeholder).
                Some(vec![0; chunks.len()])
            } else {
                digits(stride)
            }
        })
        .collect::<Option<Vec<_>>>()?;

    // No-carry check: the offset digit plus every dim's maximum travel along
    // each chunk must stay within the chunk extent.
    for (k, (extent, _)) in chunks.iter().enumerate() {
        let mut max_coord = offset_digits[k];
        for (digits, &dim) in stride_digits.iter().zip(outer.shape()) {
            max_coord = max_coord.checked_add(digits[k].checked_mul(dim.saturating_sub(1))?)?;
        }
        if max_coord >= *extent {
            return None;
        }
    }

    let offset = inner.offset()
        + offset_digits
            .iter()
            .zip(&chunks)
            .map(|(digit, (_, stride))| digit * stride)
            .sum::<usize>();
    let strides: Box<[usize]> = stride_digits
        .iter()
        .map(|digits| {
            digits
                .iter()
                .zip(&chunks)
                .map(|(digit, (_, stride))| digit * stride)
                .sum()
        })
        .collect();
    Some(Layout::from_parts(offset, outer.shape().into(), strides))
}

/// One base-dimension coordinate of an affine view:
/// `base_coord = constant + Σ coefficient * out_coord[dim]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AffineIndex {
    pub(crate) constant: u32,
    pub(crate) terms: Vec<(usize, u32)>,
}

impl AffineIndex {
    /// Render as an index expression, substituting `out[j]` for the view's
    /// output coordinate `j`. Collapses to the bare coordinate when this is
    /// an identity mapping.
    pub(crate) fn to_expr(&self, out: &[NaryExpr]) -> NaryExpr {
        if self.constant == 0
            && let [(dim, 1)] = self.terms.as_slice()
        {
            return out[*dim].clone();
        }
        let mut expr = NaryExpr::scalar(NaryScalar::U32(self.constant));
        for &(dim, coefficient) in &self.terms {
            let term = if coefficient == 1 {
                out[dim].clone()
            } else {
                NaryExpr::unary_op(
                    out[dim].clone(),
                    "mul_const",
                    NaryOp::MulConst(NaryScalar::U32(coefficient)),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                )
            };
            expr = NaryExpr::add(expr, term, DataTypeEnum::U32);
        }
        expr
    }
}

/// Decompose a view layout into per-base-dimension affine coordinate
/// expressions, when every output dimension's stride and the offset split
/// into mixed-radix digits over `base_shape` without carries. This is the
/// divmod-free form of the view: reshape/restride/broadcast/slice over a
/// dense base all qualify.
pub(crate) fn affine_dim_indices(
    layout: &Layout,
    base_shape: &[usize],
) -> Option<Vec<AffineIndex>> {
    if base_shape.contains(&0) {
        return None;
    }
    let radix = Layout::continuous_strides(base_shape);

    let digits = |mut value: usize| -> Option<Vec<u32>> {
        let mut digits = vec![0u32; base_shape.len()];
        for (k, radix) in radix.iter().enumerate() {
            digits[k] = u32::try_from(value / radix).ok()?;
            value %= radix;
        }
        (value == 0).then_some(digits)
    };

    let offset_digits = digits(layout.offset())?;
    let stride_digits = layout
        .strides()
        .iter()
        .zip(layout.shape())
        .map(|(&stride, &dim)| {
            if dim <= 1 {
                Some(vec![0; base_shape.len()])
            } else {
                digits(stride)
            }
        })
        .collect::<Option<Vec<_>>>()?;

    // No-carry check: the maximum coordinate reached along each base dim must
    // stay inside that dim.
    for (k, &extent) in base_shape.iter().enumerate() {
        let mut max_coord = offset_digits[k] as usize;
        for (digits, &dim) in stride_digits.iter().zip(layout.shape()) {
            max_coord =
                max_coord.checked_add((digits[k] as usize).checked_mul(dim.saturating_sub(1))?)?;
        }
        if max_coord >= extent {
            return None;
        }
    }

    Some(
        (0..base_shape.len())
            .map(|k| AffineIndex {
                constant: offset_digits[k],
                terms: stride_digits
                    .iter()
                    .enumerate()
                    .filter(|(_, digits)| digits[k] != 0)
                    .map(|(j, digits)| (j, digits[k]))
                    .collect(),
            })
            .collect(),
    )
}

pub(crate) fn row_major_indices_from_flat(
    flat: NaryExpr,
    shape: &[usize],
) -> Option<Vec<NaryExpr>> {
    // Peel innermost-out with a running quotient instead of dividing by each
    // axis' suffix product — the Apple Metal compiler miscompiles u32
    // div/mod chains with large non-power-of-two constants (see
    // output_dims_from_flat). The outermost non-trivial axis keeps the raw
    // quotient, matching the previous form.
    let mut indices = vec![NaryExpr::scalar(NaryScalar::U32(0)); shape.len()];
    let mut rest = flat;
    for axis in (0..shape.len()).rev() {
        let dim = u32::try_from(shape[axis]).ok()?;
        if dim == 1 {
            continue;
        }
        let outer_nontrivial = shape[..axis].iter().any(|&outer| outer != 1);
        if !outer_nontrivial {
            indices[axis] = rest;
            break;
        }
        indices[axis] = NaryExpr::unary_op(
            rest.clone(),
            "rem_const",
            NaryOp::RemConst(NaryScalar::U32(dim)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        );
        rest = NaryExpr::unary_op(
            rest,
            "div_const",
            NaryOp::DivConst(NaryScalar::U32(dim)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        );
    }
    Some(indices)
}

impl Operation for ViewOperation {
    fn hash_kernel_fields(&self, state: &mut rustc_hash::FxHasher) {
        self.stages.len().hash(state);
        for stage in &self.stages {
            stage.layout.offset().hash(state);
            stage.layout.shape().hash(state);
            stage.layout.strides().hash(state);
            stage.input_shape.hash(state);
            stage.defined.hash(state);
            stage.fill.hash(state);
        }
    }

    fn workgroup_shape_constraints(&self, _: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(BLOCKSIZE));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let output = inputs[1].as_tensor().unwrap();
        let total_workgroups = (output.layout().shape().iter().product::<usize>() as u32)
            .div_ceil(crate::TILE_SIZE * BLOCKSIZE);
        distribute_workgroups(
            total_workgroups,
            output
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        f(&mut self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap().clone();
        let output = TensorData::new_for_shape(input.device(), self.shape(), input.datatype());
        vec![input.into(), output.into()]
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[1].clone()
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let operation = ElementwiseOperation {
            inputs: vec![self.input],
            expression: self.copy_expression()?,
            shape: self.shape().into(),
            output_datatype: self.datatype,
        };
        crate::nary_direct::build_nary_direct_kernel_to_output(
            &operation,
            graph,
            workgroup_shape,
            inputs,
            1,
        )
    }

    fn name(&self) -> String {
        format!(
            "view_{}",
            self.shape()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl Tensor {
    /// The stage stack to extend when layering a new view on this tensor:
    /// the existing view's parts when this tensor is an unresolved view, or
    /// an empty stack over this tensor's own logical space.
    fn view_parts(&self) -> (NodeIndex, Vec<ViewStage>) {
        match self.device().compute_graph().get_view(self.key()) {
            Some(view) => (view.input, view.stages),
            None => (self.key(), Vec::new()),
        }
    }

    fn add_view_op(&self, op: ViewOperation) -> Tensor {
        Tensor::from_parts(self.data.view(op))
    }

    /// Layer `stage` — a relayout of this tensor's current output space —
    /// onto the stack: merged into the top stage when the composition stays
    /// one strided layout over a pure relayout, pushed as a new stage
    /// otherwise. Either way the result is a single view node over the
    /// non-view base.
    fn add_view_stage(&self, stage: ViewStage) -> Tensor {
        let (input, mut stages) = self.view_parts();
        let stage = match stages.last_mut() {
            Some(top) if top.is_fully_defined() => {
                match compose_layouts(&stage.layout, &top.layout) {
                    Some(composed) => {
                        top.layout = composed;
                        top.defined = stage.defined;
                        top.fill = stage.fill;
                        None
                    }
                    None => Some(stage),
                }
            }
            _ => Some(stage),
        };
        if let Some(stage) = stage {
            stages.push(stage);
        }
        self.add_view_op(ViewOperation {
            input,
            stages,
            datatype: self.datatype(),
        })
    }

    pub fn restride(&self, specs: impl Into<Box<[crate::StrideSpec]>>) -> Tensor {
        let specs = specs.into();
        let (input, mut stages) = self.view_parts();
        match stages.last_mut() {
            // Restride is direct stride arithmetic on the top stage — the
            // composition is infallible, no merge check needed.
            Some(top) if top.is_fully_defined() => {
                top.layout = top.layout.restride(&specs);
                top.defined = top.layout.shape().into();
            }
            Some(top) => {
                let below: Box<[usize]> = top.shape().into();
                stages.push(ViewStage::fully_defined(
                    Layout::contiguous(&below).restride(&specs),
                    below,
                ));
            }
            None => {
                stages.push(ViewStage::fully_defined(
                    Layout::contiguous(self.shape()).restride(&specs),
                    self.shape(),
                ));
            }
        }
        self.add_view_op(ViewOperation {
            input,
            stages,
            datatype: self.datatype(),
        })
    }

    /// Replace the tensor's layout with `new_layout` over its logical value
    /// space (the row-major order of its elements). The offset and strides
    /// are in logical elements, independent of how the data is laid out in
    /// any concrete buffer.
    pub fn restride_layout(&self, new_layout: Layout) -> Tensor {
        let numel = self.shape().iter().product::<usize>();
        let max_index = new_layout
            .shape()
            .iter()
            .zip(new_layout.strides())
            .fold(new_layout.offset(), |acc, (dim, stride)| {
                acc + dim.saturating_sub(1) * stride
            });
        assert!(
            numel == 0 || max_index < numel,
            "restride_layout out of bounds: layout reaches element {max_index} \
             but the input has only {numel} elements"
        );
        self.add_view_stage(ViewStage::fully_defined(new_layout, self.shape()))
    }

    pub fn broadcast_as(&self, out_shape: impl AsRef<[usize]>) -> Tensor {
        let out_shape = out_shape.as_ref();
        let shape = self.shape();
        assert!(
            out_shape.len() >= shape.len(),
            "The output rank must be at least the input rank"
        );
        let specs: Vec<crate::StrideSpec> = (0..out_shape.len())
            .map(|out_i| {
                let in_i = out_i as isize - (out_shape.len() as isize - shape.len() as isize);
                if in_i < 0 {
                    crate::StrideSpec::dim_with(0, out_shape[out_i], 0)
                } else {
                    let in_i = in_i as usize;
                    if shape[in_i] == 1 && out_shape[out_i] > 1 {
                        crate::StrideSpec::dim_with(in_i, out_shape[out_i], 0)
                    } else {
                        crate::StrideSpec::dim(in_i, out_shape[out_i])
                    }
                }
            })
            .collect();
        self.restride(specs)
    }

    pub(crate) fn broadcast_together(first: &Tensor, second: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(first.datatype(), second.datatype());
        let first_shape = first.shape();
        let second_shape = second.shape();
        let rank = first_shape.len().max(second_shape.len());
        let shape: Vec<usize> = (0..rank)
            .map(|i| {
                let a = i + first_shape.len();
                let b = i + second_shape.len();
                let a = if a >= rank { first_shape[a - rank] } else { 1 };
                let b = if b >= rank { second_shape[b - rank] } else { 1 };
                assert!(
                    a == b || a == 1 || b == 1,
                    "Cannot broadcast shapes {:?} and {:?}",
                    first_shape,
                    second_shape
                );
                a.max(b)
            })
            .collect();
        (first.broadcast_as(&shape), second.broadcast_as(&shape))
    }

    pub(crate) fn broadcast_then_elementwise_op(
        first: &Tensor,
        second: &Tensor,
        op: impl Fn(Tensor, Tensor) -> Tensor,
    ) -> Tensor {
        let (b1, b2) = Tensor::broadcast_together(first, second);
        assert_eq!(b1.shape(), b2.shape());
        op(b1, b2)
    }

    pub fn reshape(&self, new_shape: impl AsRef<[usize]>) -> Tensor {
        let new_shape = new_shape.as_ref();
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.shape().iter().product::<usize>(),
            "Reshape requires the number of elements to be the same. \
            Current shape: {:?}, target shape: {:?}",
            self.shape(),
            new_shape
        );
        // When the reshape regroups elements across non-mergeable strides
        // the reinterpret stays its own stage; otherwise it merges into the
        // top stage.
        let reinterpret = Layout::contiguous(new_shape);
        self.add_view_stage(ViewStage::fully_defined(reinterpret, self.shape()))
    }

    /// Resize to `new_shape`, clipping or zero-padding each axis: coordinates
    /// inside `min(old, new)` per axis keep their values, anything beyond is
    /// zero.
    pub fn resize(&self, new_shape: impl AsRef<[usize]>) -> Tensor {
        let new_shape = new_shape.as_ref();
        let old_shape = self.shape();
        assert_eq!(
            new_shape.len(),
            old_shape.len(),
            "resize requires matching ranks (got {old_shape:?} -> {new_shape:?}); use reshape \
             to change rank"
        );
        let defined: Box<[usize]> = new_shape
            .iter()
            .zip(old_shape)
            .map(|(new, old)| (*new).min(*old))
            .collect();
        // Within the defined box the old row-major strides address the input
        // exactly; outside it the load is masked to `fill`.
        let resize = Layout::from_parts(0, new_shape.into(), Layout::continuous_strides(old_shape));
        self.add_view_stage(ViewStage {
            layout: resize,
            input_shape: old_shape.into(),
            defined,
            fill: zero_scalar(self.datatype()),
        })
    }

    pub fn flatten_last_n(&self, from_end: usize) -> Tensor {
        assert!(
            from_end < self.rank(),
            "flatten_last_n FROM_END must be less than input rank"
        );
        let out_rank = self.rank() - from_end;
        let new_shape: Vec<usize> = (0..out_rank)
            .map(|i| {
                if i < self.rank() - 1 - from_end {
                    self.shape()[i]
                } else if i == self.rank() - 1 - from_end {
                    self.shape()[i..].iter().product()
                } else {
                    1
                }
            })
            .collect();
        self.reshape(new_shape)
    }

    pub fn flatten_first_n(&self, from_start: usize) -> Tensor {
        assert!(
            from_start < self.rank(),
            "flatten_first_n FROM_START must be less than input rank"
        );
        let out_rank = self.rank() - from_start;
        let new_shape: Vec<usize> = (0..out_rank)
            .map(|i| {
                if i == 0 {
                    self.shape()[..=from_start].iter().product()
                } else {
                    self.shape()[i + from_start]
                }
            })
            .collect();
        self.reshape(new_shape)
    }

    pub fn flatten_all(&self) -> Tensor {
        let size = self.shape().iter().product();
        self.reshape([size])
    }
}

pub use fusor_types::ShapeWithOneHole;

#[cfg(test)]
mod tests {
    use super::*;

    fn strided(offset: usize, shape: &[usize], strides: &[usize]) -> Layout {
        Layout::from_parts(offset, shape.into(), strides.into())
    }

    #[test]
    fn compose_with_contiguous_is_identity() {
        let inner = Layout::contiguous(&[4, 6]);
        let outer = strided(3, &[2, 6], &[12, 1]);
        let composed = compose_layouts(&outer, &inner).unwrap();
        assert_eq!(composed.offset(), 3);
        assert_eq!(composed.shape(), &[2, 6]);
        assert_eq!(composed.strides(), &[12, 1]);
    }

    #[test]
    fn compose_transpose_then_slice() {
        // inner: transpose of a [4, 6] tensor -> shape [6, 4], strides [1, 6]
        let inner = strided(0, &[6, 4], &[1, 6]);
        // outer: narrow rows 2..5 of the transposed view
        let outer = strided(2 * 4, &[3, 4], &[4, 1]);
        let composed = compose_layouts(&outer, &inner).unwrap();
        assert_eq!(composed.shape(), &[3, 4]);
        // flat index f over inner's [6,4] output decomposes as (r, c) with
        // target = r * 1 + c * 6
        assert_eq!(composed.offset(), 2);
        assert_eq!(composed.strides(), &[1, 6]);
    }

    #[test]
    fn compose_reshape_of_transpose_fails() {
        // Flat reinterpret of a transposed (non-contiguous) view regroups
        // elements: not expressible as one strided layout.
        let inner = strided(0, &[6, 4], &[1, 6]);
        let outer = Layout::contiguous(&[8, 3]);
        assert!(compose_layouts(&outer, &inner).is_none());
    }

    #[test]
    fn compose_reshape_merges_contiguous_chunks() {
        // inner: a sliced batch of contiguous rows: [2, 12] with strides
        // [24, 1] inside a larger allocation (row padding -> chunked).
        let inner = strided(0, &[2, 12], &[24, 1]);
        // outer: reshape each row into [3, 4] -> [2, 3, 4] flat over [2, 12]
        let outer = Layout::contiguous(&[2, 3, 4]);
        let composed = compose_layouts(&outer, &inner).unwrap();
        assert_eq!(composed.shape(), &[2, 3, 4]);
        assert_eq!(composed.strides(), &[24, 4, 1]);
    }

    #[test]
    fn compose_broadcast_strides() {
        let inner = Layout::contiguous(&[4, 6]);
        // outer broadcasts a [6] row across 5: shape [5, 6], strides [0, 1]
        let outer = strided(6, &[5, 6], &[0, 1]);
        let composed = compose_layouts(&outer, &inner).unwrap();
        assert_eq!(composed.offset(), 6);
        assert_eq!(composed.strides(), &[0, 1]);
    }

    #[test]
    fn compose_rejects_out_of_bounds_span() {
        let inner = strided(0, &[2, 12], &[24, 1]);
        // Steps of 12 over 4 elements span 36 flat positions, crossing the
        // row chunk boundary (each chunk holds 12).
        let outer = strided(0, &[4], &[12]);
        assert!(compose_layouts(&outer, &inner).is_none());
    }
}
