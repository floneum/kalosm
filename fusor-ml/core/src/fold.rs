//! Folds with an explicit loop-carried tuple and an explicit combine.
//!
//! [`ReduceOperation`] is a fold whose carrier is one value and whose combine
//! is one of four built-in operators. That closure is what forces every
//! interesting blocked reduction in this compiler to be hand-written below the
//! IR: split-K lives in the matmul kernel builder, online softmax lives in a
//! `RowStep` matcher, and Welford does not exist at all. They are three
//! instances of one law.
//!
//! A [`FoldOperation`] names the carrier and the combine, so the law becomes a
//! rewrite:
//!
//! > a fold whose `combine` is associative may be split along its axis into
//! > partial folds joined by `combine`.
//!
//! The element expression stays an ordinary [`NaryExpr`], so producer inlining
//! keeps working on it exactly as it does for a reduce. Only the carrier
//! algebra needs a term of its own, and it is closed and small — the same
//! split this codebase already makes between a fusable `expression` and a
//! restricted `post_element_wise` chain.

use crate::DataTypeEnum;
use crate::compute_graph::NodeIndex;
use crate::nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryScalar, UnaryFunctionChain};
use crate::reduce::{ReduceFunction, ReduceOp, ReduceOperation};

/// One slot of a fold's loop-carried state.
///
/// A slot is a scalar per row unless it declares a `free_dim`, in which case
/// it carries one value per position of a dimension appended to the fold's
/// output shape. Attention's `Σ p·v` is that: a vector over the head dim
/// whose step reads a *different* tensor (`v`) than the scalar slots do, at
/// the axis coordinate extended by the free coordinate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CarrierSlot {
    pub(crate) name: Option<String>,
    pub(crate) datatype: DataTypeEnum,
    /// Extent of the dimension this slot ranges over, appended to the output
    /// shape. `None` for a scalar slot.
    pub(crate) free_dim: Option<usize>,
    /// A per-element value private to this slot, evaluated at the axis
    /// coordinate extended by this slot's free coordinate (`DimIndex(rank)`).
    /// Bound in `step` immediately after the shared element — see
    /// [`slot_element`].
    pub(crate) element: Option<NaryExpr>,
}

impl CarrierSlot {
    pub(crate) fn new(name: &str, datatype: DataTypeEnum) -> Self {
        Self {
            name: Some(name.to_string()),
            datatype,
            free_dim: None,
            element: None,
        }
    }

    /// A slot ranging over `free_dim`, whose step absorbs `element` read at
    /// the axis coordinate extended by the free coordinate.
    pub(crate) fn vector(
        name: &str,
        datatype: DataTypeEnum,
        free_dim: usize,
        element: NaryExpr,
    ) -> Self {
        Self {
            name: Some(name.to_string()),
            datatype,
            free_dim: Some(free_dim),
            element: Some(element),
        }
    }

    /// How many values this slot carries per row.
    pub(crate) fn width(&self) -> usize {
        self.free_dim.unwrap_or(1)
    }
}

/// A private inner axis folded into the element before the element expression
/// runs: attention's `q·k` dot over the head dim.
///
/// `expression` is evaluated `len` times with the fold coordinate appended to
/// the element's coordinates (`DimIndex(rank)`) and combined with `function`.
/// The folded value is what [`fold_value`] binds inside
/// [`FoldOperation::expression`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElementFold {
    pub(crate) expression: NaryExpr,
    pub(crate) len: usize,
    pub(crate) function: ReduceFunction,
}

/// Carrier bodies are ordinary [`NaryExpr`]s.
///
/// A slot read is `IndexedInput { input_idx: base + k, indices: vec![] }` —
/// the same convention `row_program::slot_expr` already uses for cross-step
/// references, so one expression language serves tensors and carriers alike
/// and every existing pass over `NaryExpr` works on fold bodies unchanged.
///
/// Binding layout, where `base` is the number of tensor inputs and `n` the
/// carrier width:
///
/// - `step` sees `[acc_0 .. acc_{n-1}, element]`
/// - `combine` sees `[acc_0 .. acc_{n-1}, rhs_0 .. rhs_{n-1}]`
/// - `init` and `outputs` see `[acc_0 .. acc_{n-1}]` (init reads none of them)
pub(crate) fn slot(base: usize, index: usize) -> NaryExpr {
    NaryExpr::IndexedInput {
        input_idx: base + index,
        indices: Vec::new(),
    }
}

/// Accumulator slot `k`.
pub(crate) fn acc(base: usize, k: usize) -> NaryExpr {
    slot(base, k)
}

/// The element value, valid in `step` only.
pub(crate) fn element(base: usize, width: usize) -> NaryExpr {
    slot(base, width)
}

/// A slot's own per-element value ([`CarrierSlot::element`]), valid in that
/// slot's `step` only.
pub(crate) fn slot_element(base: usize, width: usize) -> NaryExpr {
    slot(base, width + 1)
}

/// The inner fold's value ([`ElementFold`]), valid in
/// [`FoldOperation::expression`] only — it is the element's own private
/// binding space, not the carrier's.
pub(crate) fn fold_value(base: usize) -> NaryExpr {
    slot(base, 0)
}

/// Incoming carrier slot `k`, valid in `combine` only.
pub(crate) fn rhs(base: usize, width: usize, k: usize) -> NaryExpr {
    slot(base, width + k)
}

fn binary(op: NaryOp, lhs: NaryExpr, rhs: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
    NaryExpr::Op {
        children: vec![lhs, rhs],
        function: NaryFunction::binary(None, op, datatype, datatype, datatype),
    }
}

fn unary(op: NaryOp, value: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
    NaryExpr::Op {
        children: vec![value],
        function: NaryFunction::unary(None, op, datatype, datatype),
    }
}

/// Whether `expression` reads any slot at or above `index`.
fn reads_slot_from(expression: &NaryExpr, base: usize, index: usize) -> bool {
    match expression {
        NaryExpr::IndexedInput { input_idx, indices } => {
            indices.is_empty() && *input_idx >= base + index
        }
        NaryExpr::Op { children, .. } => children
            .iter()
            .any(|child| reads_slot_from(child, base, index)),
        _ => false,
    }
}


/// One finalized output of a fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FoldOutput {
    pub(crate) expression: NaryExpr,
    pub(crate) datatype: DataTypeEnum,
}

/// What is known about `combine`, which decides which rewrites may fire.
///
/// Recorded rather than inferred because for floating point associativity is a
/// *policy*: exact reassociation (max) and error-introducing reassociation
/// (sum) are different permissions, and the online-softmax lift is neither —
/// it is exp's shift-invariance, which introduces fresh correction factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoldAlgebra {
    /// Reassociating changes nothing bit-for-bit (min/max absent NaN).
    ExactMonoid,
    /// Associative in the reals; reassociating perturbs rounding.
    ApproximateMonoid,
    /// Associative in the reals, but combining evaluates transcendentals at
    /// new arguments. Online softmax lives here.
    RescalingMonoid,
    /// No associativity claim; the split law must not fire.
    Unspecified,
}

impl FoldAlgebra {
    pub(crate) fn splittable_under(self, policy: NumericsPolicy) -> bool {
        match self {
            FoldAlgebra::ExactMonoid => true,
            FoldAlgebra::ApproximateMonoid => policy >= NumericsPolicy::ReassociationPermitted,
            FoldAlgebra::RescalingMonoid => policy >= NumericsPolicy::RelativeErrorPermitted,
            FoldAlgebra::Unspecified => false,
        }
    }
}

/// How much numerical licence a rewrite may take. Ordered by permissiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum NumericsPolicy {
    Exact,
    ReassociationPermitted,
    RelativeErrorPermitted,
}

/// A fold over one axis with a named carrier and an explicit combine.
///
/// `expression` is evaluated at every coordinate of `shape` (including `axis`)
/// exactly as in [`ReduceOperation`]; `step` absorbs one such element into the
/// carrier; `combine` joins two carriers; `outputs` finalize.
///
/// Unlike a row program's phases — which are sequential *independent*
/// reductions, each seeing the previous one's completed value — a fold's slots
/// update **jointly in one pass**: `step[i]` reads the whole running carrier.
/// That is what online softmax needs and what phases cannot express.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FoldOperation {
    pub(crate) inputs: Vec<NodeIndex>,
    pub(crate) expression: NaryExpr,
    /// A private inner axis folded into the element before `expression` runs;
    /// its value is [`fold_value`].
    pub(crate) element_fold: Option<ElementFold>,
    pub(crate) shape: Box<[usize]>,
    pub(crate) axis: usize,
    pub(crate) carrier: Vec<CarrierSlot>,
    pub(crate) init: Vec<NaryExpr>,
    pub(crate) step: Vec<NaryExpr>,
    pub(crate) combine: Vec<NaryExpr>,
    pub(crate) outputs: Vec<FoldOutput>,
    pub(crate) algebra: FoldAlgebra,
    pub(crate) block: Option<usize>,
    /// This fold's element is itself a carrier: it folds the partial carriers
    /// a blocked fold produced, so its `step` is the original `combine`.
    pub(crate) element_is_carrier: bool,
}

impl FoldOperation {
    /// Slot base: carrier slots are numbered after the tensor inputs.
    pub(crate) fn base(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn width(&self) -> usize {
        self.carrier.len()
    }

    pub(crate) fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for &input in &self.inputs {
            f(input);
        }
    }

    pub(crate) fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        for input in &mut self.inputs {
            f(input);
        }
    }

    /// Shape of the row grid: the input shape with the folded axis removed.
    /// An output that reads a free-dimension slot appends that slot's extent
    /// ([`Self::output_shape`]).
    pub(crate) fn out_shape(&self) -> Vec<usize> {
        let mut shape = self.shape.to_vec();
        shape.remove(self.axis);
        shape
    }

    /// The free dimension an output ranges over: the extent of any
    /// free-dimension slot it reads. Two different ones would make the output
    /// rank ambiguous, so [`Self::validate`] rejects that.
    pub(crate) fn output_free_dim(&self, output: &FoldOutput) -> Option<usize> {
        let base = self.base();
        self.carrier
            .iter()
            .enumerate()
            .find(|(index, slot)| {
                slot.free_dim.is_some() && output.expression.uses_input(base + index)
            })
            .and_then(|(_, slot)| slot.free_dim)
    }

    /// Shape of output `index`, including its free dimension when it has one.
    pub(crate) fn output_shape(&self, output: &FoldOutput) -> Vec<usize> {
        let mut shape = self.out_shape();
        if let Some(free) = self.output_free_dim(output) {
            shape.push(free);
        }
        shape
    }

    /// Whether any slot ranges over a free dimension.
    pub(crate) fn has_free_dim(&self) -> bool {
        self.carrier.iter().any(|slot| slot.free_dim.is_some())
    }

    /// Byte offsets of each slot inside a flattened carrier record — the
    /// scratch layout a blocked fold writes its partials into.
    pub(crate) fn slot_offsets(&self) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(self.carrier.len());
        let mut total = 0;
        for slot in &self.carrier {
            offsets.push(total);
            total += slot.width();
        }
        (offsets, total)
    }

    /// Approximate arithmetic cost of absorbing one element.
    pub(crate) fn step_work(&self) -> u128 {
        fn work(expression: &NaryExpr) -> u128 {
            match expression {
                NaryExpr::Op { children, .. } => 1 + children.iter().map(work).sum::<u128>(),
                NaryExpr::IndexedInput { indices, .. } => {
                    1 + indices.iter().map(work).sum::<u128>()
                }
                _ => 1,
            }
        }
        self.step.iter().map(work).sum()
    }

    /// Structural hash of everything deciding the kernel. A fold has no
    /// `Operation` impl of its own yet, so the interner hashes directly.
    pub(crate) fn hash_carrier_fields(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        self.expression.hash(hasher);
        if let Some(fold) = &self.element_fold {
            fold.expression.hash(hasher);
            fold.len.hash(hasher);
            fold.function.hash(hasher);
        }
        self.shape.hash(hasher);
        self.axis.hash(hasher);
        self.block.hash(hasher);
        self.element_is_carrier.hash(hasher);
        self.carrier.len().hash(hasher);
        for slot in &self.carrier {
            slot.datatype.hash(hasher);
            slot.free_dim.hash(hasher);
            slot.element.hash(hasher);
        }
        for bodies in [&self.init, &self.step, &self.combine] {
            bodies.len().hash(hasher);
            for body in bodies {
                body.hash(hasher);
            }
        }
        self.outputs.len().hash(hasher);
        for output in &self.outputs {
            output.datatype.hash(hasher);
            output.expression.hash(hasher);
        }
    }

    /// Structural well-formedness, expressed as slot-range bounds: `init`
    /// reads no slot, `step` reads at most the carrier plus the element, and
    /// `combine` reads at most two carriers.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let (base, width) = (self.base(), self.width());
        if width == 0 {
            return Err("fold carrier must have at least one slot".into());
        }
        for (name, bodies) in [
            ("init", &self.init),
            ("step", &self.step),
            ("combine", &self.combine),
        ] {
            if bodies.len() != width {
                return Err(format!(
                    "fold {name} has {} expressions for {width} carrier slots",
                    bodies.len()
                ));
            }
        }
        if self.init.iter().any(|e| reads_slot_from(e, base, 0)) {
            return Err("fold init may not read the carrier".into());
        }
        let step_limit = if self.element_is_carrier {
            width * 2
        } else if self.carrier.iter().any(|slot| slot.element.is_some()) {
            // A slot with its own element binds it after the shared one.
            width + 2
        } else {
            width + 1
        };
        if self.step.iter().any(|e| reads_slot_from(e, base, step_limit)) {
            return Err("fold step reads past the accumulator and element".into());
        }
        if self
            .carrier
            .iter()
            .any(|slot| slot.element.is_some() != slot.free_dim.is_some())
        {
            return Err("a slot's private element and free dimension go together".into());
        }
        for output in &self.outputs {
            let mut extents = self
                .carrier
                .iter()
                .enumerate()
                .filter(|(index, slot)| {
                    slot.free_dim.is_some() && output.expression.uses_input(base + index)
                })
                .filter_map(|(_, slot)| slot.free_dim);
            let first = extents.next();
            if extents.any(|extent| Some(extent) != first) {
                return Err("an output may not span two different free dimensions".into());
            }
        }
        if let Some(fold) = &self.element_fold
            && fold.len == 0
        {
            return Err("an element fold needs a non-empty axis".into());
        }
        if self
            .combine
            .iter()
            .any(|e| reads_slot_from(e, base, width * 2))
        {
            return Err("fold combine reads past the two carriers".into());
        }
        if self.axis >= self.shape.len() {
            return Err(format!(
                "fold axis {} out of bounds for rank {}",
                self.axis,
                self.shape.len()
            ));
        }
        Ok(())
    }

    /// Lift a built-in reduction into the general form: a one-slot fold whose
    /// step and combine coincide.
    pub(crate) fn from_reduce(reduce: &ReduceOperation) -> Self {
        let datatype = reduce.function.datatype();
        let base = reduce.inputs.len();
        let op = reduce.function.op;
        let join = |lhs: NaryExpr, rhs_expr: NaryExpr| match op {
            ReduceOp::Sum => binary(NaryOp::Add, lhs, rhs_expr, datatype),
            ReduceOp::Product => binary(NaryOp::Mul, lhs, rhs_expr, datatype),
            ReduceOp::Max => binary(NaryOp::Max, lhs, rhs_expr, datatype),
            ReduceOp::Min => binary(NaryOp::Min, lhs, rhs_expr, datatype),
        };
        let algebra = match op {
            ReduceOp::Max | ReduceOp::Min => FoldAlgebra::ExactMonoid,
            ReduceOp::Sum | ReduceOp::Product => FoldAlgebra::ApproximateMonoid,
        };
        // Finalization is just an output expression, so the post chain wraps
        // the carrier read rather than being a separate stage.
        let finalize = reduce
            .post_element_wise
            .functions
            .iter()
            .fold(acc(base, 0), |value, function| NaryExpr::Op {
                children: vec![value],
                function: function.clone(),
            });
        Self {
            inputs: reduce.inputs.clone(),
            expression: reduce.expression.clone(),
            element_fold: None,
            shape: reduce.shape.clone(),
            axis: reduce.axis,
            carrier: vec![CarrierSlot::new(reduce.function.name(), datatype)],
            init: vec![NaryExpr::Scalar(reduce.function.initial_value)],
            step: vec![join(acc(base, 0), element(base, 1))],
            combine: vec![join(acc(base, 0), rhs(base, 1, 0))],
            outputs: vec![FoldOutput {
                expression: finalize,
                datatype: reduce.out_datatype(),
            }],
            algebra,
            block: None,
            element_is_carrier: false,
        }
    }

    /// The inverse of [`Self::from_reduce`]: recover a built-in reduction when
    /// this fold is one. Lets the resolver hold every reduction in the general
    /// form while lowering keeps its single-slot path.
    pub(crate) fn to_reduce(&self) -> Option<ReduceOperation> {
        if self.width() != 1
            || self.block.is_some()
            || self.element_is_carrier
            || self.element_fold.is_some()
            || self.has_free_dim()
        {
            return None;
        }
        let base = self.base();
        let datatype = self.carrier[0].datatype;
        let NaryExpr::Scalar(initial_value) = self.init[0] else {
            return None;
        };
        let op = [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Max,
            ReduceOp::Min,
        ]
        .into_iter()
        .find(|&candidate| {
            let join = |lhs: NaryExpr, rhs_expr: NaryExpr| match candidate {
                ReduceOp::Sum => binary(NaryOp::Add, lhs, rhs_expr, datatype),
                ReduceOp::Product => binary(NaryOp::Mul, lhs, rhs_expr, datatype),
                ReduceOp::Max => binary(NaryOp::Max, lhs, rhs_expr, datatype),
                ReduceOp::Min => binary(NaryOp::Min, lhs, rhs_expr, datatype),
            };
            self.step[0] == join(acc(base, 0), element(base, 1))
                && self.combine[0] == join(acc(base, 0), rhs(base, 1, 0))
        })?;

        // Peel finalization back into a post chain: a unary spine bottoming
        // out at the carrier read.
        let mut functions = Vec::new();
        let mut cursor = &self.outputs.first()?.expression;
        let carrier_read = acc(base, 0);
        loop {
            if *cursor == carrier_read {
                break;
            }
            match cursor {
                NaryExpr::Op { children, function } if children.len() == 1 => {
                    functions.push(function.clone());
                    cursor = &children[0];
                }
                _ => return None,
            }
        }
        functions.reverse();

        let mut function = ReduceFunction::new(op, initial_value, datatype);
        if let Some(name) = &self.carrier[0].name {
            function = function.with_name(name);
        }
        Some(ReduceOperation {
            inputs: self.inputs.clone(),
            expression: self.expression.clone(),
            shape: self.shape.clone(),
            function,
            post_element_wise: UnaryFunctionChain::new(functions, datatype),
            axis: self.axis,
        })
    }

    /// The split law. Blocking `axis` into runs of `factor` turns one fold
    /// into a partial fold plus a joining fold over the block index.
    ///
    /// The joiner's `step` is the original `combine` verbatim: a joining fold's
    /// element *is* a whole carrier, and both bodies bind
    /// `[acc_0..acc_{n-1}, incoming_0..incoming_{n-1}]`. Mapping the incoming
    /// carrier onto a single element would merge the slots.
    pub(crate) fn split(
        &self,
        factor: usize,
        policy: NumericsPolicy,
    ) -> Result<(FoldOperation, FoldOperation), String> {
        if self.block.is_some() {
            return Err("fold is already blocked".into());
        }
        if factor == 0 {
            return Err("split factor must be non-zero".into());
        }
        if !self.algebra.splittable_under(policy) {
            return Err(format!(
                "combine is {:?}, which {policy:?} does not permit splitting",
                self.algebra
            ));
        }
        let extent = self.shape[self.axis];
        if extent % factor != 0 {
            return Err(format!(
                "split factor {factor} does not divide axis extent {extent}"
            ));
        }
        let base = self.base();

        let mut partial = self.clone();
        partial.block = Some(factor);
        // A partial stops at the carrier; finalization belongs to the joiner.
        partial.outputs = self
            .carrier
            .iter()
            .enumerate()
            .map(|(index, slot)| FoldOutput {
                expression: acc(base, index),
                datatype: slot.datatype,
            })
            .collect();

        let mut joiner = self.clone();
        joiner.block = None;
        joiner.shape = {
            let mut shape = self.shape.to_vec();
            shape[self.axis] = extent / factor;
            shape.into()
        };
        joiner.step = self.combine.clone();
        joiner.element_is_carrier = true;
        joiner.outputs = self.outputs.clone();

        partial.validate()?;
        joiner.validate()?;
        Ok((partial, joiner))
    }
}

/// The online-softmax carrier: running max, normalizer, and unnormalized
/// accumulator.
///
/// ```text
/// (m1,l1,o1) (+) (m2,l2,o2) = (M, e^(m1-M)*l1 + e^(m2-M)*l2,
///                                 e^(m1-M)*o1 + e^(m2-M)*o2)   M = max(m1,m2)
/// ```
///
/// A commutative monoid in the reals, but combining evaluates `exp` at
/// arguments depending on both sides, so it is [`FoldAlgebra::RescalingMonoid`]
/// rather than merely approximate. This is the identity flash attention is
/// built on and the reason a carrier must be a tuple.
pub(crate) fn online_softmax_carrier(
    inputs: Vec<NodeIndex>,
    expression: NaryExpr,
    shape: Box<[usize]>,
    axis: usize,
    datatype: DataTypeEnum,
) -> FoldOperation {
    let base = inputs.len();
    let width = 3;
    let neg_inf = NaryScalar::F32(-3.0e38);
    let zero = match datatype {
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(0.0)),
        DataTypeEnum::U32 => NaryScalar::U32(0),
        DataTypeEnum::F32 => NaryScalar::F32(0.0),
    };

    // step: M = max(m, x); l' = l*e^(m-M) + e^(x-M); o' = o*e^(m-M) + e^(x-M)
    let x = element(base, width);
    let new_max = binary(NaryOp::Max, acc(base, 0), x.clone(), datatype);
    let rescale = unary(
        NaryOp::Exp,
        binary(NaryOp::Sub, acc(base, 0), new_max.clone(), datatype),
        datatype,
    );
    let weight = unary(
        NaryOp::Exp,
        binary(NaryOp::Sub, x, new_max.clone(), datatype),
        datatype,
    );
    // Slot 1 accumulates the weights themselves (the normalizer); slot 2
    // accumulates the weighted element. Giving both the same body would make
    // `acc2 / acc1` identically one, which is a carrier that proves nothing.
    let step_slot = |k: usize, contribution: NaryExpr| {
        binary(
            NaryOp::Add,
            binary(NaryOp::Mul, acc(base, k), rescale.clone(), datatype),
            contribution,
            datatype,
        )
    };

    // combine: the same law with the incoming carrier in place of the element.
    let joined_max = binary(NaryOp::Max, acc(base, 0), rhs(base, width, 0), datatype);
    let alpha = |value: NaryExpr| {
        unary(
            NaryOp::Exp,
            binary(NaryOp::Sub, value, joined_max.clone(), datatype),
            datatype,
        )
    };
    let combine_slot = |k: usize| {
        binary(
            NaryOp::Add,
            binary(
                NaryOp::Mul,
                acc(base, k),
                alpha(acc(base, 0)),
                datatype,
            ),
            binary(
                NaryOp::Mul,
                rhs(base, width, k),
                alpha(rhs(base, width, 0)),
                datatype,
            ),
            datatype,
        )
    };

    FoldOperation {
        inputs,
        expression,
        element_fold: None,
        shape,
        axis,
        carrier: vec![
            CarrierSlot::new("max", datatype),
            CarrierSlot::new("normalizer", datatype),
            CarrierSlot::new("accumulator", datatype),
        ],
        init: vec![
            NaryExpr::Scalar(neg_inf),
            NaryExpr::Scalar(zero),
            NaryExpr::Scalar(zero),
        ],
        step: vec![
            new_max,
            step_slot(1, weight.clone()),
            step_slot(
                2,
                binary(NaryOp::Mul, weight.clone(), element(base, width), datatype),
            ),
        ],
        combine: vec![joined_max.clone(), combine_slot(1), combine_slot(2)],
        outputs: vec![
            // The softmax-weighted mean of the element.
            FoldOutput {
                expression: binary(NaryOp::Div, acc(base, 2), acc(base, 1), datatype),
                datatype,
            },
            // The log-sum-exp, free from the same carrier.
            FoldOutput {
                expression: binary(
                    NaryOp::Add,
                    acc(base, 0),
                    unary(NaryOp::Log, acc(base, 1), datatype),
                    datatype,
                ),
                datatype,
            },
        ],
        algebra: FoldAlgebra::RescalingMonoid,
        block: None,
        element_is_carrier: false,
    }
}

/// The carrier streaming attention folds over the key/value axis: the running
/// score maximum, the softmax normalizer, and the unnormalized `Σ p·v`
/// accumulator — a *vector* over the head dimension.
///
/// This is [`online_softmax_carrier`] with two things it cannot express:
///
/// - the element is itself a fold (the `q·k` dot over the head dim), and
/// - the third slot ranges over a free dimension and absorbs its own tensor
///   read (`v` at the key position and the free coordinate), while the first
///   two stay scalar.
///
/// Stating those in the carrier is what retires the hand-written streaming
/// emitter: the recurrence, the rescale and the tile join all follow from
/// `step` and `combine` rather than from a matcher over a fixed phase list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn streaming_attention_carrier(
    inputs: Vec<NodeIndex>,
    score: ElementFold,
    scaled: NaryExpr,
    value: NaryExpr,
    shape: Box<[usize]>,
    axis: usize,
    free_dim: usize,
    max_identity: NaryScalar,
    output_datatype: DataTypeEnum,
) -> FoldOperation {
    // Softmax statistics accumulate in f32 whatever the tensors' wire type;
    // only the finalized output is rounded back.
    let datatype = DataTypeEnum::F32;
    let base = inputs.len();
    let width = 3;
    let zero = NaryExpr::Scalar(NaryScalar::F32(0.0));
    let x = element(base, width);
    let v = slot_element(base, width);

    let new_max = binary(NaryOp::Max, acc(base, 0), x.clone(), datatype);
    let rescale = unary(
        NaryOp::Exp,
        binary(NaryOp::Sub, acc(base, 0), new_max.clone(), datatype),
        datatype,
    );
    let weight = unary(
        NaryOp::Exp,
        binary(NaryOp::Sub, x, new_max.clone(), datatype),
        datatype,
    );
    let absorb = |k: usize, contribution: NaryExpr| {
        binary(
            NaryOp::Add,
            binary(NaryOp::Mul, acc(base, k), rescale.clone(), datatype),
            contribution,
            datatype,
        )
    };

    let joined_max = binary(NaryOp::Max, acc(base, 0), rhs(base, width, 0), datatype);
    let alpha = |source: NaryExpr| {
        unary(
            NaryOp::Exp,
            binary(NaryOp::Sub, source, joined_max.clone(), datatype),
            datatype,
        )
    };
    let join = |k: usize| {
        binary(
            NaryOp::Add,
            binary(NaryOp::Mul, acc(base, k), alpha(acc(base, 0)), datatype),
            binary(
                NaryOp::Mul,
                rhs(base, width, k),
                alpha(rhs(base, width, 0)),
                datatype,
            ),
            datatype,
        )
    };

    FoldOperation {
        inputs,
        expression: scaled,
        element_fold: Some(score),
        shape,
        axis,
        carrier: vec![
            CarrierSlot::new("max", datatype),
            CarrierSlot::new("normalizer", datatype),
            CarrierSlot::vector("accumulator", datatype, free_dim, value),
        ],
        init: vec![
            NaryExpr::Scalar(max_identity),
            zero.clone(),
            zero,
        ],
        step: vec![
            new_max,
            absorb(1, weight.clone()),
            absorb(2, binary(NaryOp::Mul, weight, v, datatype)),
        ],
        combine: vec![joined_max.clone(), join(1), join(2)],
        outputs: vec![FoldOutput {
            expression: binary(NaryOp::Div, acc(base, 2), acc(base, 1), datatype),
            datatype: output_datatype,
        }],
        algebra: FoldAlgebra::RescalingMonoid,
        block: None,
        element_is_carrier: false,
    }
}

/// Welford's algorithm as a fold: running count, mean, and sum of squared
/// deviations. The generality check — if the carrier abstraction is right this
/// costs nothing attention-specific, and blocked mean/variance for layer norm
/// falls out of the same split law that blocks attention.
pub(crate) fn welford_carrier(
    inputs: Vec<NodeIndex>,
    expression: NaryExpr,
    shape: Box<[usize]>,
    axis: usize,
) -> FoldOperation {
    let datatype = DataTypeEnum::F32;
    let base = inputs.len();
    let width = 3;
    let zero = NaryExpr::Scalar(NaryScalar::F32(0.0));
    let one = NaryExpr::Scalar(NaryScalar::F32(1.0));
    let x = element(base, width);

    let n_next = binary(NaryOp::Add, acc(base, 0), one, datatype);
    let delta = binary(NaryOp::Sub, x.clone(), acc(base, 1), datatype);
    let mean_next = binary(
        NaryOp::Add,
        acc(base, 1),
        binary(NaryOp::Div, delta.clone(), n_next.clone(), datatype),
        datatype,
    );
    let m2_next = binary(
        NaryOp::Add,
        acc(base, 2),
        binary(
            NaryOp::Mul,
            delta,
            binary(NaryOp::Sub, x, mean_next.clone(), datatype),
            datatype,
        ),
        datatype,
    );

    // combine: the pairwise Chan-Golub-LeVeque update.
    let n_total = binary(NaryOp::Add, acc(base, 0), rhs(base, width, 0), datatype);
    // Joining two empty partials leaves `n_total == 0`; clamping the divisor
    // keeps the combine total (the numerators are zero there anyway) so a fold
    // whose blocking leaves idle lanes does not manufacture NaNs.
    let n_divisor = binary(
        NaryOp::Max,
        n_total.clone(),
        NaryExpr::Scalar(NaryScalar::F32(1.0)),
        datatype,
    );
    let mean_delta = binary(NaryOp::Sub, rhs(base, width, 1), acc(base, 1), datatype);
    let mean_joined = binary(
        NaryOp::Add,
        acc(base, 1),
        binary(
            NaryOp::Mul,
            mean_delta.clone(),
            binary(NaryOp::Div, rhs(base, width, 0), n_divisor.clone(), datatype),
            datatype,
        ),
        datatype,
    );
    let m2_joined = binary(
        NaryOp::Add,
        binary(NaryOp::Add, acc(base, 2), rhs(base, width, 2), datatype),
        binary(
            NaryOp::Mul,
            binary(NaryOp::Mul, mean_delta.clone(), mean_delta, datatype),
            binary(
                NaryOp::Div,
                binary(NaryOp::Mul, acc(base, 0), rhs(base, width, 0), datatype),
                n_divisor,
                datatype,
            ),
            datatype,
        ),
        datatype,
    );

    FoldOperation {
        inputs,
        expression,
        element_fold: None,
        shape,
        axis,
        carrier: vec![
            CarrierSlot::new("count", datatype),
            CarrierSlot::new("mean", datatype),
            CarrierSlot::new("m2", datatype),
        ],
        init: vec![zero.clone(), zero.clone(), zero],
        step: vec![n_next, mean_next, m2_next],
        combine: vec![n_total.clone(), mean_joined, m2_joined],
        outputs: vec![
            FoldOutput {
                expression: acc(base, 1),
                datatype,
            },
            FoldOutput {
                expression: binary(NaryOp::Div, acc(base, 2), acc(base, 0), datatype),
                datatype,
            },
        ],
        algebra: FoldAlgebra::ApproximateMonoid,
        block: None,
        element_is_carrier: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nary_wise::UnaryFunctionChain;

    /// Host evaluator over the slot convention, enough to prove the split law
    /// on concrete data. Slots are supplied in binding order.
    fn eval(expression: &NaryExpr, slots: &[f32]) -> f32 {
        match expression {
            NaryExpr::IndexedInput { input_idx, indices } => {
                assert!(indices.is_empty(), "carrier bodies read scalar slots");
                slots[*input_idx]
            }
            NaryExpr::Scalar(NaryScalar::F32(value)) => *value,
            NaryExpr::Scalar(other) => panic!("unexpected scalar {other:?}"),
            NaryExpr::DimIndex(_) => panic!("carrier bodies are coordinate-free"),
            NaryExpr::Op { children, function } => {
                let args: Vec<f32> = children.iter().map(|c| eval(c, slots)).collect();
                match function.op {
                    NaryOp::Add => args[0] + args[1],
                    NaryOp::Sub => args[0] - args[1],
                    NaryOp::Mul => args[0] * args[1],
                    NaryOp::Div => args[0] / args[1],
                    NaryOp::Max => args[0].max(args[1]),
                    NaryOp::Min => args[0].min(args[1]),
                    NaryOp::Exp => args[0].exp(),
                    NaryOp::Log => args[0].ln(),
                    NaryOp::Sqrt => args[0].sqrt(),
                    ref other => panic!("unexpected op {other:?}"),
                }
            }
        }
    }

    fn init_carrier(fold: &FoldOperation) -> Vec<f32> {
        fold.init.iter().map(|e| eval(e, &[])).collect()
    }

    /// One `step`: bindings are `[acc.., element]`.
    fn step_once(fold: &FoldOperation, acc: &[f32], value: f32) -> Vec<f32> {
        let mut slots = acc.to_vec();
        slots.push(value);
        fold.step.iter().map(|e| eval(e, &slots)).collect()
    }

    /// One `combine`: bindings are `[acc.., rhs..]`.
    fn combine_once(fold: &FoldOperation, acc: &[f32], incoming: &[f32]) -> Vec<f32> {
        let mut slots = acc.to_vec();
        slots.extend_from_slice(incoming);
        fold.combine.iter().map(|e| eval(e, &slots)).collect()
    }

    fn finish(fold: &FoldOperation, acc: &[f32]) -> Vec<f32> {
        fold.outputs
            .iter()
            .map(|out| eval(&out.expression, acc))
            .collect()
    }

    fn run_whole(fold: &FoldOperation, data: &[f32]) -> Vec<f32> {
        let mut acc = init_carrier(fold);
        for &value in data {
            acc = step_once(fold, &acc, value);
        }
        finish(fold, &acc)
    }

    /// Partial folds per block, joined by the joiner's step (which is the
    /// original combine), then finalized.
    fn run_split(fold: &FoldOperation, data: &[f32], factor: usize) -> Vec<f32> {
        let (partial, joiner) = fold
            .split(factor, NumericsPolicy::RelativeErrorPermitted)
            .expect("split");
        let partials: Vec<Vec<f32>> = data
            .chunks(factor)
            .map(|block| {
                let mut acc = init_carrier(&partial);
                for &value in block {
                    acc = step_once(&partial, &acc, value);
                }
                acc
            })
            .collect();
        let mut acc = init_carrier(&joiner);
        for incoming in &partials {
            // A carrier-element fold's step binds two whole carriers.
            acc = combine_once(fold, &acc, incoming);
        }
        finish(&joiner, &acc)
    }

    fn parts() -> (Vec<NodeIndex>, NaryExpr, Box<[usize]>) {
        (Vec::new(), NaryExpr::input(0, 2), vec![1usize, 8].into())
    }

    fn reduce_of(function: crate::reduce::ReduceFunction, post: UnaryFunctionChain) -> ReduceOperation {
        let (inputs, expression, shape) = parts();
        ReduceOperation {
            inputs,
            expression,
            shape,
            function,
            post_element_wise: post,
            axis: 1,
        }
    }

    #[test]
    fn split_preserves_sum_and_max() {
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let sum = FoldOperation::from_reduce(&reduce_of(
            crate::reduce::sum_fn(DataTypeEnum::F32),
            UnaryFunctionChain::empty(DataTypeEnum::F32),
        ));
        sum.validate().unwrap();
        assert_eq!(run_whole(&sum, &data), vec![36.0]);
        assert_eq!(run_split(&sum, &data, 4), vec![36.0]);
        assert_eq!(run_split(&sum, &data, 2), vec![36.0]);

        let values = vec![3.0, -1.0, 7.5, 2.0, 0.0, 7.4, -9.0, 1.0];
        let max = FoldOperation::from_reduce(&reduce_of(
            crate::reduce::max_fn(DataTypeEnum::F32),
            UnaryFunctionChain::empty(DataTypeEnum::F32),
        ));
        assert_eq!(run_whole(&max, &values), vec![7.5]);
        assert_eq!(run_split(&max, &values, 4), vec![7.5]);
    }

    #[test]
    fn split_preserves_online_softmax() {
        let (inputs, expression, shape) = parts();
        let fold = online_softmax_carrier(inputs, expression, shape, 1, DataTypeEnum::F32);
        fold.validate().unwrap();
        let data = vec![1.0, 3.0, 2.0, -4.0, 0.5, 8.0, 7.0, -2.0];

        let max = data.iter().cloned().fold(f32::MIN, f32::max);
        let expected_lse = max + data.iter().map(|v| (v - max).exp()).sum::<f32>().ln();

        let whole = run_whole(&fold, &data);
        assert!((whole[1] - expected_lse).abs() < 1e-5, "lse {whole:?}");
        for factor in [2usize, 4] {
            let split = run_split(&fold, &data, factor);
            for (a, b) in whole.iter().zip(&split) {
                assert!((a - b).abs() < 1e-5, "factor {factor}: {whole:?} vs {split:?}");
            }
        }
    }

    /// One `step` for a free-dimension slot: bindings are
    /// `[acc.., element, slot_element]`.
    fn step_once_free(fold: &FoldOperation, acc: &[f32], value: f32, slot_value: f32) -> Vec<f32> {
        let mut slots = acc.to_vec();
        slots.push(value);
        slots.push(slot_value);
        fold.step.iter().map(|e| eval(e, &slots)).collect()
    }

    /// Run a carrier with one free-dimension slot the way the lowering does:
    /// once per free coordinate, with the scalar slots recomputed alongside.
    /// Returns `outputs[0]` per free coordinate.
    fn run_whole_free(fold: &FoldOperation, elements: &[f32], values: &[Vec<f32>]) -> Vec<f32> {
        let free = fold
            .carrier
            .iter()
            .find_map(|slot| slot.free_dim)
            .expect("a free-dimension slot");
        (0..free)
            .map(|d| {
                let mut acc = init_carrier(fold);
                for (j, &value) in elements.iter().enumerate() {
                    acc = step_once_free(fold, &acc, value, values[j][d]);
                }
                eval(&fold.outputs[0].expression, &acc)
            })
            .collect()
    }

    /// The same fold blocked into runs of `factor`, joined by `combine`.
    fn run_split_free(
        fold: &FoldOperation,
        elements: &[f32],
        values: &[Vec<f32>],
        factor: usize,
    ) -> Vec<f32> {
        let (partial, joiner) = fold
            .split(factor, NumericsPolicy::RelativeErrorPermitted)
            .expect("split");
        let free = fold
            .carrier
            .iter()
            .find_map(|slot| slot.free_dim)
            .expect("a free-dimension slot");
        (0..free)
            .map(|d| {
                let partials: Vec<Vec<f32>> = elements
                    .chunks(factor)
                    .enumerate()
                    .map(|(block, chunk)| {
                        let mut acc = init_carrier(&partial);
                        for (offset, &value) in chunk.iter().enumerate() {
                            let j = block * factor + offset;
                            acc = step_once_free(&partial, &acc, value, values[j][d]);
                        }
                        acc
                    })
                    .collect();
                let mut acc = init_carrier(&joiner);
                for incoming in &partials {
                    acc = combine_once(fold, &acc, incoming);
                }
                eval(&joiner.outputs[0].expression, &acc)
            })
            .collect()
    }

    fn reference_attention(elements: &[f32], values: &[Vec<f32>], free: usize) -> Vec<f32> {
        let max = elements.iter().cloned().fold(f32::MIN, f32::max);
        let weights: Vec<f32> = elements.iter().map(|x| (x - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        (0..free)
            .map(|d| {
                weights
                    .iter()
                    .zip(values)
                    .map(|(w, row)| w * row[d])
                    .sum::<f32>()
                    / total
            })
            .collect()
    }

    fn attention_fold(free: usize) -> FoldOperation {
        let (inputs, expression, shape) = parts();
        streaming_attention_carrier(
            inputs,
            ElementFold {
                expression: expression.clone(),
                len: 4,
                function: crate::reduce::sum_fn(DataTypeEnum::F32),
            },
            expression,
            NaryExpr::input(1, 3),
            shape,
            1,
            free,
            NaryScalar::F32(f32::MIN),
            DataTypeEnum::F32,
        )
    }

    #[test]
    fn online_softmax_carrier_weights_the_element() {
        let (inputs, expression, shape) = parts();
        let fold = online_softmax_carrier(inputs, expression, shape, 1, DataTypeEnum::F32);
        let data = vec![1.0, 3.0, 2.0, -4.0, 0.5, 8.0, 7.0, -2.0];
        let max = data.iter().cloned().fold(f32::MIN, f32::max);
        let weights: Vec<f32> = data.iter().map(|v| (v - max).exp()).collect();
        let expected = weights.iter().zip(&data).map(|(w, v)| w * v).sum::<f32>()
            / weights.iter().sum::<f32>();

        let whole = run_whole(&fold, &data);
        // Not identically one: slots 1 and 2 must accumulate different things.
        assert!(
            (whole[0] - expected).abs() < 1e-4,
            "weighted mean {whole:?} vs {expected}"
        );
        assert!((whole[0] - 1.0).abs() > 1e-3, "carrier is degenerate");
    }

    #[test]
    fn streaming_attention_carrier_matches_reference_attention() {
        let free = 3;
        let fold = attention_fold(free);
        fold.validate().unwrap();
        assert!(fold.has_free_dim());
        assert_eq!(fold.slot_offsets(), (vec![0, 1, 2], 2 + free));
        assert_eq!(fold.output_free_dim(&fold.outputs[0]), Some(free));
        assert_eq!(
            fold.output_shape(&fold.outputs[0]),
            vec![1, free],
            "an output reading the accumulator gains its free dimension"
        );

        let elements = vec![1.0, 3.0, 2.0, -4.0, 0.5, 8.0, 7.0, -2.0];
        let values: Vec<Vec<f32>> = (0..elements.len())
            .map(|j| (0..free).map(|d| (j as f32) - 2.0 * (d as f32)).collect())
            .collect();
        let expected = reference_attention(&elements, &values, free);
        let whole = run_whole_free(&fold, &elements, &values);
        for (got, want) in whole.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "{whole:?} vs {expected:?}");
        }
        for factor in [2usize, 4] {
            let split = run_split_free(&fold, &elements, &values, factor);
            for (got, want) in split.iter().zip(&expected) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "factor {factor}: {split:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn a_free_dimension_slot_has_no_reduce_form() {
        assert!(attention_fold(3).to_reduce().is_none());
    }

    #[test]
    fn split_preserves_welford() {
        let (inputs, expression, shape) = parts();
        let fold = welford_carrier(inputs, expression, shape, 1);
        fold.validate().unwrap();
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let whole = run_whole(&fold, &data);
        assert!((whole[0] - 5.0).abs() < 1e-5, "mean {whole:?}");
        assert!((whole[1] - 4.0).abs() < 1e-5, "variance {whole:?}");
        for factor in [2usize, 4] {
            let split = run_split(&fold, &data, factor);
            for (a, b) in whole.iter().zip(&split) {
                assert!((a - b).abs() < 1e-4, "factor {factor}: {whole:?} vs {split:?}");
            }
        }
    }

    #[test]
    fn numerics_policy_gates_the_split_law() {
        let (inputs, expression, shape) = parts();
        let softmax = online_softmax_carrier(inputs, expression, shape, 1, DataTypeEnum::F32);
        // The rescaling lift is not licensed by mere reassociation.
        assert!(softmax.split(4, NumericsPolicy::ReassociationPermitted).is_err());
        assert!(softmax.split(4, NumericsPolicy::RelativeErrorPermitted).is_ok());
        // Max is exact, so it splits even under the strictest policy.
        let max = FoldOperation::from_reduce(&reduce_of(
            crate::reduce::max_fn(DataTypeEnum::F32),
            UnaryFunctionChain::empty(DataTypeEnum::F32),
        ));
        assert!(max.split(4, NumericsPolicy::Exact).is_ok());
    }

    #[test]
    fn built_in_reductions_round_trip_through_the_general_form() {
        let post = UnaryFunctionChain::new(
            vec![NaryFunction::unary(
                None,
                NaryOp::Sqrt,
                DataTypeEnum::F32,
                DataTypeEnum::F32,
            )],
            DataTypeEnum::F32,
        );
        for function in [
            crate::reduce::sum_fn(DataTypeEnum::F32),
            crate::reduce::max_fn(DataTypeEnum::F32),
        ] {
            for chain in [UnaryFunctionChain::empty(DataTypeEnum::F32), post.clone()] {
                let reduce = reduce_of(function.clone(), chain);
                let fold = FoldOperation::from_reduce(&reduce);
                fold.validate().unwrap();
                let recovered = fold.to_reduce().expect("built-in fold has a reduce form");
                assert_eq!(recovered, reduce, "round trip changed the reduction");
            }
        }
    }

    #[test]
    fn general_folds_have_no_reduce_form() {
        let (inputs, expression, shape) = parts();
        assert!(
            online_softmax_carrier(inputs, expression, shape, 1, DataTypeEnum::F32)
                .to_reduce()
                .is_none()
        );
        let (inputs, expression, shape) = parts();
        assert!(welford_carrier(inputs, expression, shape, 1).to_reduce().is_none());

        let sum = FoldOperation::from_reduce(&reduce_of(
            crate::reduce::sum_fn(DataTypeEnum::F32),
            UnaryFunctionChain::empty(DataTypeEnum::F32),
        ));
        let (partial, joiner) = sum.split(4, NumericsPolicy::ReassociationPermitted).unwrap();
        assert!(partial.to_reduce().is_none(), "blocked fold is not a reduce");
        assert!(joiner.to_reduce().is_none(), "carrier-element fold is not a reduce");
    }

    #[test]
    fn split_rejects_indivisible_factors_and_double_blocking() {
        let policy = NumericsPolicy::RelativeErrorPermitted;
        let sum = FoldOperation::from_reduce(&reduce_of(
            crate::reduce::sum_fn(DataTypeEnum::F32),
            UnaryFunctionChain::empty(DataTypeEnum::F32),
        ));
        assert!(sum.split(3, policy).is_err());
        let (partial, _) = sum.split(4, policy).unwrap();
        assert!(partial.split(2, policy).is_err());
    }
}
