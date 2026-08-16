//! The differentiable const-rank tensor.
//!
//! [`Tensor<R, T>`] is a [`RawTensor<R, T>`](crate::tensor::typed::Tensor)
//! plus the [`Graph`] it was built in. There is no separate tape: the e-graph
//! is the tape, so a forward op and its adjoint are nodes in one arena and
//! "save this activation or recompute it" is the extractor's materialization
//! bit.
//!
//! # `with_backwards` and where a user rule runs
//!
//! A rule registered here is a chain-rule boundary, not a node in the reverse
//! walk: [`Tensor::backward_with`] runs the ordinary backward down to each
//! boundary, hands the boundary's gradient to the user closure, and continues
//! from whatever slots the closure names. The closure builds its gradients
//! with the ordinary tensor API and may name parents that are not its
//! operands — a fake-quantized weight is a `constant_from_raw` with no
//! operands whose parent is the master weight.

use std::collections::HashMap;
use std::sync::Arc;

use fusor_ir::egraph::Id;
use parking_lot::Mutex;

use crate::graph::Graph as CoreGraph;
use crate::tensor::Tensor as Dyn;
use crate::tensor::typed::{Element, Tensor as RawTensor};
use crate::{Error, Result};

use crate::tensor::typed::Axis;
pub use fusor_ir::autograd::{GradientSlot, Parent};

/// How many boundary hops one backward may take before it is a cycle.
///
/// A hop is either one user rule or one partial reverse walk; the counter is
/// a cycle guard.
const MAX_BOUNDARY_HOPS: usize = 512;

/// One gradient the user's [`Tensor::with_backwards`] rule produced, and the
/// slot it belongs to.
#[derive(Clone, Debug)]
pub struct BackwardTarget {
    slot: GradientSlot,
    gradient: Dyn,
}

impl BackwardTarget {
    /// Route `gradient` to `slot`. `slot` comes from [`Tensor::slot`]; it is a
    /// bare node id, never a tensor handle, so a rule cannot close a cycle
    /// over the graph by naming its own target.
    pub fn to<const R: usize, T: Element>(slot: GradientSlot, gradient: RawTensor<R, T>) -> Self {
        Self {
            slot,
            gradient: gradient.into_inner(),
        }
    }

    /// The gradient slot this target feeds.
    pub fn slot(&self) -> GradientSlot {
        self.slot
    }
}

/// A user backward, type-erased over the value's rank and dtype.
type Rule = Arc<dyn Fn(Dyn) -> Result<Vec<BackwardTarget>>>;

#[derive(Default)]
struct Boundary {
    parents: Vec<Parent>,
    rule: Option<Rule>,
}

#[derive(Default)]
struct Tape {
    /// Values [`Graph::leaf`] minted, in creation order. These are what a
    /// backward differentiates with respect to.
    leaves: Vec<Id>,
    /// Chain-rule boundaries, keyed by the value they hang off.
    boundaries: HashMap<Id, Boundary>,
}

/// A differentiable program under construction.
///
/// Cloning shares the tape: a model holds one and the loss holds the same one.
#[derive(Clone)]
pub struct Graph {
    core: CoreGraph,
    tape: Arc<Mutex<Tape>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// A fresh tape over the ambient device graph.
    ///
    /// # Panics
    /// If no [`crate::device::Device`] has been created yet — see
    /// [`crate::device`] for why the graph is ambient rather than an argument.
    pub fn new() -> Self {
        Self::over(crate::device::ambient_graph())
    }

    /// A fresh tape over an explicit graph.
    pub fn over(core: CoreGraph) -> Self {
        Self {
            core,
            tape: Arc::new(Mutex::new(Tape::default())),
        }
    }

    /// The underlying node graph.
    pub fn core(&self) -> &CoreGraph {
        &self.core
    }

    /// A trainable value. Gradients come back for exactly these, in creation
    /// order.
    pub fn leaf<const R: usize, T: Element>(&self, value: RawTensor<R, T>) -> Tensor<R, T> {
        self.tape.lock().leaves.push(value.id());
        Tensor {
            value,
            graph: self.clone(),
        }
    }

    /// A value the backward does not descend into.
    pub fn constant<const R: usize, T: Element>(&self, value: RawTensor<R, T>) -> Tensor<R, T> {
        Tensor {
            value,
            graph: self.clone(),
        }
    }

    /// Every leaf this tape minted, in creation order.
    pub fn leaves(&self) -> Vec<Id> {
        self.tape.lock().leaves.clone()
    }

    fn register(&self, value: Id, parents: Vec<Parent>, rule: Rule) {
        self.tape.lock().boundaries.insert(
            value,
            Boundary {
                parents,
                rule: Some(rule),
            },
        );
    }

    /// Boundary ids and leaf ids: the frontier a partial backward stops at.
    fn frontier(&self) -> Vec<Id> {
        let tape = self.tape.lock();
        let mut ids: Vec<Id> = tape.leaves.clone();
        ids.extend(tape.boundaries.keys().copied());
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// A value on the tape.
pub struct Tensor<const R: usize, T: Element = f32> {
    value: RawTensor<R, T>,
    graph: Graph,
}

impl<const R: usize, T: Element> Clone for Tensor<R, T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            graph: self.graph.clone(),
        }
    }
}

impl<const R: usize, T: Element> std::fmt::Debug for Tensor<R, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "autograd::{:?}", self.value)
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Adopt a raw value as a constant of `graph`.
    pub fn constant_from_raw(graph: &Graph, value: RawTensor<R, T>) -> Self {
        graph.constant(value)
    }

    /// The tape this value lives on.
    pub fn graph(&self) -> Graph {
        self.graph.clone()
    }

    /// The value underneath. Reading it does not detach anything: the node is
    /// the same node.
    pub fn raw(&self) -> &RawTensor<R, T> {
        &self.value
    }

    /// [`Tensor::raw`], consuming the handle.
    pub fn into_raw(self) -> RawTensor<R, T> {
        self.value
    }

    /// This value as a backward parent, requiring a gradient.
    pub fn parent(&self) -> Parent {
        Parent {
            value: self.value.id(),
            requires_grad: true,
        }
    }

    /// This value's gradient slot: a bare node id.
    pub fn slot(&self) -> GradientSlot {
        GradientSlot(self.value.id())
    }

    /// The underlying graph node id.
    pub fn id(&self) -> Id {
        self.value.id()
    }

    #[track_caller]
    /// The concrete shape.
    pub fn shape(&self) -> [usize; R] {
        self.value.shape()
    }

    #[track_caller]
    /// The number of elements.
    pub fn elements(&self) -> usize {
        self.value.elements()
    }

    /// The runtime dtype.
    pub fn dtype(&self) -> fusor_ir::dtype::Dtype {
        self.value.dtype()
    }

    fn like<const O: usize, E: Element>(&self, value: RawTensor<O, E>) -> Tensor<O, E> {
        Tensor {
            value,
            graph: self.graph.clone(),
        }
    }

    /// Attach a backward rule and declare the parents it feeds.
    ///
    /// The rule is a closure, not a bare `fn`: it runs outside the reverse
    /// walk (see the module docs), so it may capture, and it builds its
    /// gradients with the ordinary tensor API.
    ///
    /// # Panics
    /// Never here. A rule that misses a `requires_grad` parent is caught by
    /// [`Tensor::backward_with`], which is where it can name the value.
    pub fn with_backwards<const N: usize, F>(self, parents: [Parent; N], rule: F) -> Self
    where
        F: Fn(RawTensor<R, T>) -> Result<Vec<BackwardTarget>> + 'static,
    {
        let erased: Rule = Arc::new(move |grad: Dyn| {
            let typed = RawTensor::<R, T>::try_from_dyn(grad)?;
            rule(typed)
        });
        self.graph
            .register(self.value.id(), parents.to_vec(), erased);
        self
    }

    /// Differentiate with respect to every [`Graph::leaf`], seeding this
    /// value's gradient with `seed`.
    ///
    /// The seed is a value, not a `1.0`: a mixed-precision step scales the
    /// loss before the backward and undoes it before the optimizer, and the
    /// scale has to enter here or the f16 activations underflow first.
    pub fn backward_with(&self, seed: RawTensor<R, T>) -> Result<Gradients> {
        self.backward_from(seed.into_inner())
    }

    /// [`Tensor::backward_with`] seeded with ones.
    pub fn backward(&self) -> Result<Gradients> {
        self.backward_with(self.value.ones_like())
    }

    fn backward_from(&self, seed: Dyn) -> Result<Gradients> {
        let leaves = self.graph.leaves();
        let frontier = self.graph.frontier();
        let mut acc: HashMap<Id, Dyn> = HashMap::new();
        let mut work: Vec<(Id, Dyn)> = vec![(self.value.id(), seed)];
        let mut hops = 0usize;

        while let Some((value, grad)) = work.pop() {
            hops += 1;
            if hops > MAX_BOUNDARY_HOPS {
                return Err(Error::Plan(format!(
                    "backward made more than {MAX_BOUNDARY_HOPS} boundary hops; a \
                     with_backwards rule is feeding itself"
                )));
            }
            // A boundary owns its own chain rule; run it and follow the slots
            // it names.
            let boundary = {
                let tape = self.graph.tape.lock();
                tape.boundaries
                    .get(&value)
                    .map(|b| (b.parents.clone(), b.rule.as_ref().map(Arc::clone)))
            };
            if let Some((parents, rule)) = boundary {
                let Some(rule) = rule else { continue };
                let targets = rule(grad)?;
                validate(&parents, &targets)?;
                for target in targets {
                    work.push((target.slot.0, target.gradient));
                }
                continue;
            }
            // A leaf is where a gradient lands.
            if leaves.contains(&value) {
                accumulate(&mut acc, value, grad)?;
                continue;
            }
            // Anything else: an ordinary reverse walk down to the frontier.
            // Only the part of the frontier this value actually depends on —
            // `backward_seeded` rejects a `wrt` it cannot reach, and most of
            // the frontier is behind some *other* boundary.
            let value_tensor = self.graph.core.handle().tensor(value);
            let seed_tensor = grad;
            let candidates: Vec<Dyn> = frontier
                .iter()
                .filter(|id| **id != value)
                .map(|id| self.graph.core.handle().tensor(*id))
                .collect();
            let wrt = self.graph.core.reachable_from(&value_tensor, &candidates);
            if wrt.is_empty() {
                continue;
            }
            let grads = self
                .graph
                .core
                .backward_seeded(&value_tensor, &seed_tensor, &wrt)?;
            for target in &wrt {
                if let Some(g) = grads.get(target) {
                    work.push((target.id(), g));
                }
            }
        }

        Ok(Gradients { entries: acc })
    }
}

fn accumulate(acc: &mut HashMap<Id, Dyn>, value: Id, grad: Dyn) -> Result<()> {
    match acc.remove(&value) {
        None => {
            acc.insert(value, grad);
        }
        Some(prev) => {
            acc.insert(value, prev.add_(&grad)?);
        }
    }
    Ok(())
}

/// Every requires-grad parent must receive a gradient. A rule that omits one
/// is an error, not a silent zero: the omitted parent's whole subgraph would
/// starve, and the final check would report the symptom rather than the cause.
fn validate(parents: &[Parent], targets: &[BackwardTarget]) -> Result<()> {
    for parent in parents {
        if !parent.requires_grad {
            continue;
        }
        if !targets.iter().any(|t| t.slot.0 == parent.value) {
            return Err(Error::Plan(format!(
                "a with_backwards rule returned no gradient for parent {:?}, which requires one",
                parent.value
            )));
        }
    }
    Ok(())
}

/// The gradients of one backward, keyed by the leaf they belong to.
#[derive(Clone, Default)]
pub struct Gradients {
    entries: HashMap<Id, Dyn>,
}

impl Gradients {
    /// The gradient of `of`, at the same rank and dtype.
    ///
    /// `None` means there is no gradient — `of` was not a [`Graph::leaf`], or
    /// nothing on the tape connects it to the loss.
    ///
    /// # Panics
    /// If a gradient exists but disagrees with `of` on rank or dtype. That is
    /// a bug in the adjoint, not an absent gradient, and reporting it as
    /// `None` would surface at the caller's `expect("missing gradient")` with
    /// the wrong diagnosis.
    #[track_caller]
    pub fn get<const R: usize, T: Element>(&self, of: &Tensor<R, T>) -> Option<RawTensor<R, T>> {
        let raw = self.entries.get(&of.value.id())?.clone();
        Some(crate::device::ok(
            "gradient",
            RawTensor::<R, T>::try_from_dyn(raw),
        ))
    }

    /// The number of available leaf gradients.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no leaf received a gradient.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Rank- and dtype-preserving unaries.
macro_rules! same {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`RawTensor::", stringify!($name), "`], on the tape.")]
            #[track_caller]
            pub fn $name(&self) -> Self {
                self.like(self.value.$name())
            }
        )*}
    };
}

same!(
    exp, log, sqrt, abs, neg, sqr, recip, sign, tanh, relu, sigmoid, silu, gelu, gelu_exact,
    softplus, zeros_like, ones_like,
);

/// Rank-preserving scalar arithmetic.
macro_rules! same_scalar {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`RawTensor::", stringify!($name), "`], on the tape.")]
            #[track_caller]
            pub fn $name(&self, s: impl Into<crate::tensor::Scalar>) -> Self {
                self.like(self.value.$name(s))
            }
        )*}
    };
}

same_scalar!(
    add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, max_scalar, min_scalar,
);

/// Rank-reducing folds; `O` is `R - 1`.
macro_rules! reduce {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`RawTensor::", stringify!($name), "`], output rank `O = R - 1`.")]
            #[track_caller]
            pub fn $name<const O: usize>(&self, axis: impl Axis<R>) -> Tensor<O, T> {
                self.like(self.value.$name::<O>(axis))
            }
        )*}
    };
}

reduce!(sum, product, max, min, mean);

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Same-shape binaries, on the tape.
    #[track_caller]
    pub fn add(&self, rhs: &Self) -> Self {
        self.like(self.value.add(&rhs.value))
    }
    #[track_caller]
    /// Subtract two same-shape values.
    pub fn sub(&self, rhs: &Self) -> Self {
        self.like(self.value.sub(&rhs.value))
    }
    #[track_caller]
    /// Multiply two same-shape values.
    pub fn mul(&self, rhs: &Self) -> Self {
        self.like(self.value.mul(&rhs.value))
    }
    #[track_caller]
    /// Divide two same-shape values.
    pub fn div(&self, rhs: &Self) -> Self {
        self.like(self.value.div(&rhs.value))
    }

    /// Broadcasting `a + b`, output rank `O = max(R, R2)`.
    #[track_caller]
    pub fn add_<const R2: usize, const O: usize>(&self, rhs: &Tensor<R2, T>) -> Tensor<O, T> {
        self.like(self.value.add_::<R2, O, RawTensor<R2, T>>(&rhs.value))
    }

    /// Broadcasting `a * b`.
    #[track_caller]
    pub fn mul_<const R2: usize, const O: usize>(&self, rhs: &Tensor<R2, T>) -> Tensor<O, T> {
        self.like(self.value.mul_::<R2, O, RawTensor<R2, T>>(&rhs.value))
    }

    /// Broadcasting `a / b`.
    #[track_caller]
    pub fn div_<const R2: usize, const O: usize>(&self, rhs: &Tensor<R2, T>) -> Tensor<O, T> {
        self.like(self.value.div_::<R2, O, RawTensor<R2, T>>(&rhs.value))
    }

    /// Convert the dtype. Differentiable, so a gradient taken in f16 still
    /// lands on the f32 master.
    #[track_caller]
    pub fn cast<E: Element>(&self) -> Tensor<R, E> {
        self.like(self.value.cast::<E>())
    }

    #[track_caller]
    /// Reshape the value.
    pub fn reshape<const O: usize>(&self, shape: [usize; O]) -> Tensor<O, T> {
        self.like(self.value.reshape(shape))
    }

    #[track_caller]
    /// Swap two axes.
    pub fn transpose(&self, d0: impl Axis<R>, d1: impl Axis<R>) -> Self {
        self.like(self.value.transpose(d0, d1))
    }

    #[track_caller]
    /// Reorder the axes.
    pub fn permute(&self, order: [usize; R]) -> Self {
        self.like(self.value.permute(order))
    }

    #[track_caller]
    /// Keep `len` elements of one axis starting at `start`.
    pub fn narrow(&self, dim: impl Axis<R>, start: usize, len: usize) -> Self {
        self.like(self.value.narrow(dim, start, len))
    }

    #[track_caller]
    /// Flatten every axis into one.
    pub fn flatten_all(&self) -> Tensor<1, T> {
        self.like(self.value.flatten_all())
    }

    #[track_caller]
    /// Gather indices along one axis.
    pub fn index_select(&self, dim: impl Axis<R>, idx: &RawTensor<1, u32>) -> Self {
        self.like(self.value.index_select(dim, idx))
    }

    #[track_caller]
    /// Clamp every element to the inclusive scalar interval.
    pub fn clamp(
        &self,
        lo: impl Into<crate::tensor::Scalar>,
        hi: impl Into<crate::tensor::Scalar>,
    ) -> Self {
        self.like(self.value.clamp(lo, hi))
    }

    #[track_caller]
    /// Compare every element with `s` using `<=`.
    pub fn lte_scalar(&self, s: impl Into<crate::tensor::Scalar>) -> Self {
        self.like(self.value.lte_scalar(s))
    }

    #[track_caller]
    /// Compare every element with `s` using `>=`.
    pub fn gte_scalar(&self, s: impl Into<crate::tensor::Scalar>) -> Self {
        self.like(self.value.gte_scalar(s))
    }

    #[track_caller]
    /// Matrix multiplication over the trailing two axes.
    pub fn matmul(&self, rhs: &Self) -> Self {
        self.like(self.value.matmul(&rhs.value))
    }

    /// [`RawTensor::conv`], on the tape.
    #[track_caller]
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const WINDOWED: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        stride: [usize; DIFF],
    ) -> Self {
        self.like(self.value.conv::<WEIGHT_RANK, DIFF, WINDOWED>(
            &weight.value,
            bias.map(|b| &b.value),
            padding,
            stride,
        ))
    }

    /// Join values along `dim`.
    #[track_caller]
    pub fn cat<I>(parts: I, dim: usize) -> Self
    where
        I: IntoIterator<Item = Tensor<R, T>>,
    {
        let parts: Vec<Tensor<R, T>> = parts.into_iter().collect();
        let graph = parts
            .first()
            .map(|p| p.graph.clone())
            .unwrap_or_else(Graph::new);
        let value = crate::tensor::typed::cat(parts.into_iter().map(|p| p.value), dim);
        Tensor { value, graph }
    }

    /// Detach from the tape: the same value, no gradient path.
    pub fn detach(&self) -> RawTensor<R, T> {
        self.value.clone()
    }
}

/// Join values along `dim`; the free spelling of [`Tensor::cat`].
#[track_caller]
pub fn cat<const R: usize, T: Element, I>(parts: I, dim: usize) -> Tensor<R, T>
where
    I: IntoIterator<Item = Tensor<R, T>>,
{
    Tensor::cat(parts, dim)
}
