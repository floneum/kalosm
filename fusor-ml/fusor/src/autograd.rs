use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::{
    Device, Dim, Error, Layout, MaskKind, Result, Tensor as RawTensor, ToVec1, ToVec2,
    layers::Embedding,
};
use fusor_types::{SlidingWindow, StrideSpec};

type NodeId = usize;
#[cfg(not(target_arch = "wasm32"))]
type BackwardRule =
    Arc<dyn Fn(Box<dyn AnyTensorValue>) -> Result<Vec<BackwardTarget>> + Send + Sync>;
#[cfg(target_arch = "wasm32")]
type BackwardRule = Arc<dyn Fn(Box<dyn AnyTensorValue>) -> Result<Vec<BackwardTarget>>>;

#[cfg(not(target_arch = "wasm32"))]
trait BackwardClosure: Send + Sync + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T> BackwardClosure for T where T: Send + Sync + 'static {}

#[cfg(target_arch = "wasm32")]
trait BackwardClosure: 'static {}
#[cfg(target_arch = "wasm32")]
impl<T> BackwardClosure for T where T: 'static {}

#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

#[derive(Clone)]
pub struct Tensor<const R: usize> {
    value: RawTensor<R, f32>,
    handle: NodeHandle,
}

pub struct Gradients {
    gradients: HashMap<NodeId, Box<dyn AnyTensorValue>>,
}

pub struct BackwardTarget {
    node: NodeId,
    gradient: Box<dyn AnyTensorValue>,
}

#[derive(Clone)]
pub struct Parent {
    handle: NodeHandle,
}

#[derive(Clone)]
struct NodeHandle {
    graph: Arc<GraphInner>,
    id: NodeId,
}

#[derive(Clone)]
struct Node {
    parents: Vec<NodeId>,
    backward: Option<BackwardRule>,
    requires_grad: bool,
}

struct GraphInner {
    state: Mutex<GraphState>,
}

struct GraphState {
    next_id: NodeId,
    nodes: HashMap<NodeId, Node>,
}

#[cfg(not(target_arch = "wasm32"))]
trait AnyTensorValue: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn AnyTensorValue>;
    fn into_detached(self: Box<Self>) -> Box<dyn AnyTensorValue>;
    fn add_box(&self, other: &dyn AnyTensorValue) -> Result<Box<dyn AnyTensorValue>>;
}

#[cfg(target_arch = "wasm32")]
trait AnyTensorValue {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn AnyTensorValue>;
    fn into_detached(self: Box<Self>) -> Box<dyn AnyTensorValue>;
    fn add_box(&self, other: &dyn AnyTensorValue) -> Result<Box<dyn AnyTensorValue>>;
}

impl Graph {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(GraphInner {
                state: Mutex::new(GraphState {
                    next_id: 0,
                    nodes: HashMap::new(),
                }),
            }),
        }
    }

    pub fn leaf<const R: usize>(&self, value: RawTensor<R, f32>) -> Tensor<R> {
        self.tensor_with_grad(value, true)
    }

    pub fn constant<const R: usize>(&self, value: RawTensor<R, f32>) -> Tensor<R> {
        self.tensor_with_grad(value, false)
    }

    pub fn tensor<const R: usize, T>(&self, device: &Device, data: T) -> Tensor<R>
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        self.leaf(RawTensor::new(device, data))
    }

    pub fn constant_from_data<const R: usize, T>(&self, device: &Device, data: T) -> Tensor<R>
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        self.constant(RawTensor::new(device, data))
    }

    fn tensor_with_grad<const R: usize>(
        &self,
        value: RawTensor<R, f32>,
        requires_grad: bool,
    ) -> Tensor<R> {
        let id = self.inner.add_node(Vec::new(), None, requires_grad);
        Tensor {
            value,
            handle: NodeHandle {
                graph: self.inner.clone(),
                id,
            },
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl<const R: usize> Tensor<R> {
    pub fn from_raw(graph: &Graph, value: RawTensor<R, f32>) -> Self {
        graph.leaf(value)
    }

    pub fn constant_from_raw(graph: &Graph, value: RawTensor<R, f32>) -> Self {
        graph.constant(value)
    }

    pub fn new<T>(graph: &Graph, device: &Device, data: T) -> Self
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        graph.tensor(device, data)
    }

    pub fn from_array<T>(graph: &Graph, device: &Device, data: T) -> Self
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        Self::new(graph, device, data)
    }

    pub fn from_slice(graph: &Graph, device: &Device, shape: [usize; R], data: &[f32]) -> Self {
        graph.leaf(RawTensor::from_slice(device, shape, data))
    }

    pub fn zeros(graph: &Graph, device: &Device, shape: [usize; R]) -> Self {
        graph.leaf(RawTensor::zeros(device, shape))
    }

    pub fn splat(graph: &Graph, device: &Device, value: f32, shape: [usize; R]) -> Self {
        graph.leaf(RawTensor::splat(device, value, shape))
    }

    pub fn full(graph: &Graph, device: &Device, shape: [usize; R], value: f32) -> Self {
        Self::splat(graph, device, value, shape)
    }

    pub fn zeros_like(&self) -> Self {
        Self::zeros(&self.graph(), &self.device(), self.shape())
    }

    pub fn raw(&self) -> &RawTensor<R, f32> {
        &self.value
    }

    pub fn into_raw(self) -> RawTensor<R, f32> {
        self.value
    }

    pub fn shape(&self) -> [usize; R] {
        self.value.shape()
    }

    pub fn device(&self) -> Device {
        self.value.device()
    }

    pub fn graph(&self) -> Graph {
        Graph {
            inner: self.handle.graph.clone(),
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.handle.graph.requires_grad(self.handle.id)
    }

    pub fn parent(&self) -> Parent {
        Parent {
            handle: self.handle.clone(),
        }
    }

    pub fn detach(&self) -> Self {
        let requires_grad = self.requires_grad();
        let id = self.handle.graph.add_node(Vec::new(), None, requires_grad);
        Self {
            value: self.value.to_concrete(),
            handle: NodeHandle {
                graph: self.handle.graph.clone(),
                id,
            },
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_backwards<I, F>(self, parents: I, backwards: F) -> Self
    where
        I: IntoIterator<Item = Parent>,
        F: Fn(RawTensor<R, f32>) -> Result<Vec<BackwardTarget>> + Send + Sync + 'static,
    {
        self.with_backwards_impl(parents, backwards)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn with_backwards<I, F>(self, parents: I, backwards: F) -> Self
    where
        I: IntoIterator<Item = Parent>,
        F: Fn(RawTensor<R, f32>) -> Result<Vec<BackwardTarget>> + 'static,
    {
        self.with_backwards_impl(parents, backwards)
    }

    fn with_backwards_impl<I, F>(self, parents: I, backwards: F) -> Self
    where
        I: IntoIterator<Item = Parent>,
        F: Fn(RawTensor<R, f32>) -> Result<Vec<BackwardTarget>> + BackwardClosure,
    {
        let parent_handles = parents
            .into_iter()
            .map(|parent| parent.handle)
            .collect::<Vec<_>>();
        let requires_grad = parent_handles
            .iter()
            .any(|parent| parent.graph.requires_grad(parent.id));
        let parent_ids = parent_handles
            .iter()
            .map(|parent| parent.id)
            .collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = gradient
                .as_any()
                .downcast_ref::<RawTensor<R, f32>>()
                .ok_or_else(|| Error::msg("gradient rank mismatch in custom backward"))?
                .clone();
            backwards(gradient)
        });
        self.handle.graph.replace_node(
            self.handle.id,
            Node {
                parents: parent_ids,
                backward: Some(backward),
                requires_grad,
            },
        );
        self
    }

    pub fn backward(&self) -> Result<Gradients> {
        let elements = self.shape().iter().product::<usize>();
        if elements != 1 {
            return Err(Error::msg(
                "backward() requires a single-element tensor; use backward_with() for non-scalars",
            ));
        }
        let seed = RawTensor::splat(&self.device(), 1.0, self.shape());
        self.backward_with(seed)
    }

    pub fn backward_with(&self, seed: RawTensor<R, f32>) -> Result<Gradients> {
        self.handle.graph.backward(self.handle.id, Box::new(seed))
    }

    fn from_op<const OUT: usize>(
        &self,
        value: RawTensor<OUT, f32>,
        parents: Vec<NodeHandle>,
        backward: Option<BackwardRule>,
    ) -> Tensor<OUT> {
        for parent in &parents {
            assert!(
                Arc::ptr_eq(&self.handle.graph, &parent.graph),
                "cannot mix autograd tensors from different graphs"
            );
        }
        let requires_grad = parents
            .iter()
            .any(|parent| parent.graph.requires_grad(parent.id));
        let parent_ids = parents.into_iter().map(|parent| parent.id).collect();
        let id = self
            .handle
            .graph
            .add_node(parent_ids, backward, requires_grad);
        Tensor {
            value,
            handle: NodeHandle {
                graph: self.handle.graph.clone(),
                id,
            },
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() + rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), grad.to_concrete()],
        )
    }

    pub fn add_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.add(&rhs)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() - rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), (-grad).to_concrete()],
        )
    }

    pub fn sub_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.sub(&rhs)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() * rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                vec![
                    (grad.clone() * rhs).to_concrete(),
                    (grad * lhs).to_concrete(),
                ]
            },
        )
    }

    pub fn mul_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.mul(&rhs)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() / rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                let lhs_grad = (grad.clone() / rhs.clone()).to_concrete();
                let rhs_grad = (-((grad * lhs) / rhs.sqr().to_concrete())).to_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn div_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.div(&rhs)
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op(rhs, self.value.pow(&rhs.value).to_concrete(), |grad, lhs, rhs| {
            let rhs_minus_one = rhs.sub_scalar(1.0).to_concrete();
            let lhs_power = lhs.pow(&rhs_minus_one).to_concrete();
            let lhs_grad = ((grad.clone() * rhs.clone()).to_concrete() * lhs_power).to_concrete();
            let rhs_grad = ((grad * lhs.pow(&rhs).to_concrete()).to_concrete() * lhs.log().to_concrete())
                .to_concrete();
            vec![lhs_grad, rhs_grad]
        })
    }

    pub fn pow_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.pow(&rhs)
    }

    pub fn pow_elementwise(&self, exponent: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.pow_elementwise(exponent).to_concrete(), move |grad, _| {
            let power = input.pow_elementwise(exponent - 1.0).to_concrete();
            (grad * power).to_concrete().mul_scalar(exponent).to_concrete()
        })
    }

    pub fn pow_scalar(&self, exponent: f32) -> Self {
        self.pow_elementwise(exponent)
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.add_scalar(scalar), move |grad, _| grad)
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.sub_scalar(scalar), move |grad, _| grad)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.mul_scalar(scalar).to_concrete(),
            move |grad, _| grad.mul_scalar(scalar).to_concrete(),
        )
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.div_scalar(scalar).to_concrete(),
            move |grad, _| grad.div_scalar(scalar).to_concrete(),
        )
    }

    pub fn neg(&self) -> Self {
        self.unary_from_value((-self.value.clone()).to_concrete(), move |grad, _| {
            (-grad).to_concrete()
        })
    }

    pub fn sqr(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sqr().to_concrete(), move |grad, _| {
            ((grad * input.clone()).to_concrete().mul_scalar(2.0)).to_concrete()
        })
    }

    pub fn abs(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.abs().to_concrete(), move |grad, _| {
            let positive = input.mt(0.0).to_concrete();
            let negative = input.lt(0.0).to_concrete();
            ((grad.clone() * positive).to_concrete() - (grad * negative).to_concrete()).to_concrete()
        })
    }

    pub fn acos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acos().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete()
            .sqrt()
            .to_concrete();
            (-(grad / denom).to_concrete()).to_concrete()
        })
    }

    pub fn acosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acosh().to_concrete(), move |grad, _| {
            let lower = input.add_scalar(-1.0).to_concrete().sqrt().to_concrete();
            let upper = input.add_scalar(1.0).to_concrete().sqrt().to_concrete();
            (grad / (lower * upper).to_concrete()).to_concrete()
        })
    }

    pub fn approximate_exp(&self) -> Self {
        self.unary_from_value(self.value.approximate_exp().to_concrete(), move |grad, out| {
            (grad * out).to_concrete()
        })
    }

    pub fn asin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asin().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete()
            .sqrt()
            .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn asinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asinh().to_concrete(), move |grad, _| {
            let denom = input
                .sqr()
                .add_scalar(1.0)
                .to_concrete()
                .sqrt()
                .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn atan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atan().to_concrete(), move |grad, _| {
            let denom = input.sqr().add_scalar(1.0).to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn atanh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atanh().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn cos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cos().to_concrete(), move |grad, _| {
            (-(grad * input.sin().to_concrete()).to_concrete()).to_concrete()
        })
    }

    pub fn cosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cosh().to_concrete(), move |grad, _| {
            (grad * input.sinh().to_concrete()).to_concrete()
        })
    }

    pub fn exp2(&self) -> Self {
        self.unary_from_value(self.value.exp2().to_concrete(), move |grad, out| {
            (grad * out).to_concrete().mul_scalar(std::f32::consts::LN_2).to_concrete()
        })
    }

    pub fn less_approximate_exp(&self) -> Self {
        self.unary_from_value(self.value.less_approximate_exp().to_concrete(), move |grad, out| {
            (grad * out).to_concrete()
        })
    }

    pub fn log2(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log2().to_concrete(), move |grad, _| {
            (grad / input.clone()).to_concrete().div_scalar(std::f32::consts::LN_2).to_concrete()
        })
    }

    pub fn sin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sin().to_concrete(), move |grad, _| {
            (grad * input.cos().to_concrete()).to_concrete()
        })
    }

    pub fn sinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sinh().to_concrete(), move |grad, _| {
            (grad * input.cosh().to_concrete()).to_concrete()
        })
    }

    pub fn tan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.tan().to_concrete(), move |grad, _| {
            let cos = input.cos().to_concrete();
            (grad / (cos.clone() * cos).to_concrete()).to_concrete()
        })
    }

    pub fn tanh_exact(&self) -> Self {
        self.unary_from_value(self.value.tanh_exact().to_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), 1.0, out.shape())
                - out.sqr().to_concrete())
            .to_concrete();
            (grad * one_minus_sq).to_concrete()
        })
    }

    pub fn cast<D2>(&self) -> crate::Tensor<R, D2>
    where
        f32: crate::CastTo<D2> + crate::CastTensor<D2>,
        D2: crate::SimdElement + crate::DataType + Default,
    {
        self.value.cast()
    }

    pub fn relu(&self) -> Self {
        let output = self.value.max_scalar(0.0).to_concrete();
        self.unary_from_value(output.clone(), move |grad, out| {
            let zeros = RawTensor::zeros(&out.device(), out.shape());
            let ones = RawTensor::splat(&out.device(), 1.0, out.shape());
            (grad * out.where_cond(&ones, &zeros)).to_concrete()
        })
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.clamp(min, max).to_concrete(), move |grad, _| {
            let lower = input.mt(min).to_concrete();
            let upper = input.lt(max).to_concrete();
            ((grad * lower).to_concrete() * upper).to_concrete()
        })
    }

    pub fn eq(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.eq(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn eq_scalar(&self, rhs: f32) -> Self {
        self.eq(rhs)
    }

    pub fn eq_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.eq_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn gt_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.gt_scalar(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn gt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.gt_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn gte_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.gte_scalar(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn gte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.gte_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn lt(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.lt(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn lt_scalar(&self, rhs: f32) -> Self {
        self.lt(rhs)
    }

    pub fn lt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.lt_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn lte(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.lte(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn lte_scalar(&self, rhs: f32) -> Self {
        self.lte(rhs)
    }

    pub fn lte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.lte_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn max_elementwise(&self, rhs: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.max_elementwise(rhs).to_concrete(), move |grad, _| {
            (grad * input.mt(rhs).to_concrete()).to_concrete()
        })
    }

    pub fn max_scalar(&self, rhs: f32) -> Self {
        self.max_elementwise(rhs)
    }

    pub fn min_elementwise(&self, rhs: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.min_elementwise(rhs).to_concrete(), move |grad, _| {
            (grad * input.lt(rhs).to_concrete()).to_concrete()
        })
    }

    pub fn min_scalar(&self, rhs: f32) -> Self {
        self.min_elementwise(rhs)
    }

    pub fn mt(&self, rhs: f32) -> Self {
        self.gt_scalar(rhs)
    }

    pub fn mte(&self, rhs: f32) -> Self {
        self.gte_scalar(rhs)
    }

    pub fn ne(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.ne(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn ne_scalar(&self, rhs: f32) -> Self {
        self.ne(rhs)
    }

    pub fn ne_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(rhs, self.value.ne_tensor(&rhs.value).to_concrete(), move |_, lhs, rhs| {
            vec![
                RawTensor::zeros(&lhs.device(), lhs.shape()),
                RawTensor::zeros(&rhs.device(), rhs.shape()),
            ]
        })
    }

    pub fn silu(&self) -> Self {
        let denom = self.mul_scalar(-1.0).exp().add_scalar(1.0);
        self.div(&denom)
    }

    pub fn gelu(&self) -> Self {
        let cubic = self.sqr().mul(self);
        let inner = self
            .add(&cubic.mul_scalar(0.044_715))
            .mul_scalar((2.0 / std::f32::consts::PI).sqrt());
        let gate = inner.tanh().add_scalar(1.0);
        self.mul(&gate).mul_scalar(0.5)
    }

    pub fn tanh(&self) -> Self {
        self.unary_from_value(self.value.tanh().to_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), 1.0, out.shape())
                - out.sqr().to_concrete())
            .to_concrete();
            (grad * one_minus_sq).to_concrete()
        })
    }

    pub fn exp(&self) -> Self {
        self.unary_from_value(self.value.exp().to_concrete(), move |grad, out| {
            (grad * out).to_concrete()
        })
    }

    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        assert_same_graph(self, on_true);
        assert_same_graph(self, on_false);

        let value = self.value.where_cond(&on_true.value, &on_false.value).to_concrete();
        let condition_id = self.handle.id;
        let true_id = on_true.handle.id;
        let false_id = on_false.handle.id;
        let condition = self.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "where_cond")?;
            let zeros = RawTensor::zeros(&condition.device(), condition.shape());
            let ones = RawTensor::splat(&condition.device(), 1.0, condition.shape());
            let true_mask = condition.where_cond(&ones, &zeros).to_concrete();
            let false_mask = condition.where_cond(&zeros, &ones).to_concrete();
            Ok(vec![
                BackwardTarget {
                    node: condition_id,
                    gradient: Box::new(zeros),
                },
                BackwardTarget {
                    node: true_id,
                    gradient: Box::new((gradient.clone() * true_mask).to_concrete()),
                },
                BackwardTarget {
                    node: false_id,
                    gradient: Box::new((gradient * false_mask).to_concrete()),
                },
            ])
        });
        self.from_op(
            value,
            vec![
                self.handle.clone(),
                on_true.handle.clone(),
                on_false.handle.clone(),
            ],
            Some(backward),
        )
    }

    pub fn log(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log().to_concrete(), move |grad, _| {
            (grad / input.clone()).to_concrete()
        })
    }

    pub fn sqrt(&self) -> Self {
        self.unary_from_value(self.value.sqrt().to_concrete(), move |grad, out| {
            let denom = out.mul_scalar(2.0).to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn reshape<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.reshape(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "reshape")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.reshape(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let value = self.value.transpose(dim0, dim1).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "transpose")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.transpose(dim0, dim1).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn permute(&self, axes: [usize; R]) -> Self {
        let value = self.value.permute(axes).to_concrete();
        let input_id = self.handle.id;
        let mut inverse = [0usize; R];
        for (index, axis) in axes.iter().copied().enumerate() {
            inverse[axis] = index;
        }
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "permute")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.permute(inverse).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn slice(&self, slices: [Range<usize>; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.slice(slices.clone()).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "slice")?;
            let zeros = RawTensor::zeros(&gradient.device(), input_shape);
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(zeros.slice_assign(slices.clone(), &gradient).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn broadcast_as<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.broadcast_as(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "broadcast_as")?;
            let reduced = reduce_broadcast_gradient(gradient, input_shape)?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: reduced,
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn expand<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        self.broadcast_as(shape)
    }

    pub fn flatten_all(&self) -> Tensor<1> {
        self.reshape([self.shape().iter().product()])
    }

    pub fn flatten_last_n<const FROM_END: usize, const OUT: usize>(&self) -> Tensor<OUT>
    where
        fusor_core::Tensor<R, f32>: fusor_core::SmallerRank<FROM_END, OUT, f32>,
    {
        let shape = self.shape();
        let new_shape: [usize; OUT] = std::array::from_fn(|i| {
            if i < R - 1 - FROM_END {
                shape[i]
            } else if i == R - 1 - FROM_END {
                shape[R - 1 - FROM_END..].iter().product()
            } else {
                1
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_first_n<const FROM_START: usize, const OUT: usize>(&self) -> Tensor<OUT>
    where
        fusor_core::Tensor<R, f32>: fusor_core::SmallerRank<FROM_START, OUT, f32>,
    {
        let shape = self.shape();
        let new_shape: [usize; OUT] = std::array::from_fn(|i| {
            if i == 0 {
                shape[..=FROM_START].iter().product()
            } else {
                shape[i + FROM_START]
            }
        });
        self.reshape(new_shape)
    }

    pub fn narrow(&self, dim: impl Dim<R>, start: usize, length: usize) -> Self {
        let dim = dim.resolve();
        let shape = self.shape();
        let slices: [Range<usize>; R] = std::array::from_fn(|axis| {
            if axis == dim {
                start..start + length
            } else {
                0..shape[axis]
            }
        });
        self.slice(slices)
    }

    pub fn chunk(&self, chunks: usize, dim: impl Dim<R>) -> Vec<Self> {
        let dim = dim.resolve();
        let shape = self.shape();
        let dim_size = shape[dim];
        let chunk_size = dim_size.div_ceil(chunks);

        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;
        while start < dim_size {
            let length = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, length));
            start += length;
        }
        result
    }

    pub fn repeat(&self, repeats: [usize; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.repeat(repeats).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "repeat")?;
            let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
            for_each_index(repeats, |repeat_index| {
                let slices: [Range<usize>; R] = std::array::from_fn(|axis| {
                    let start = repeat_index[axis] * input_shape[axis];
                    start..start + input_shape[axis]
                });
                let patch = gradient.slice(slices).to_concrete();
                input_gradient = (input_gradient.clone() + patch).to_concrete();
            });
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.resize(new_shape).to_concrete();
        let input_id = self.handle.id;
        let copy_shape = std::array::from_fn(|axis| input_shape[axis].min(new_shape[axis]));
        let copy_slices = copy_shape.map(|size| 0..size);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "resize")?;
            let patch = gradient.slice(copy_slices.clone()).to_concrete();
            let zeros = RawTensor::zeros(&gradient.device(), input_shape);
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(zeros.slice_assign(copy_slices.clone(), &patch).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn restride<const OUT: usize>(&self, specs: [StrideSpec; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.restride(specs).to_concrete();
        let input_id = self.handle.id;
        let output_shape: [usize; OUT] = specs.map(|spec| spec.size);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "restride")?;
            let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
            for_each_index(output_shape, |output_index| {
                let input_index: [usize; R] = restride_input_index(specs, output_index);
                let output_slices: [Range<usize>; OUT] =
                    std::array::from_fn(|axis| output_index[axis]..output_index[axis] + 1);
                let patch = gradient.slice(output_slices).reshape([1; R]).to_concrete();
                let target: [Range<usize>; R] =
                    std::array::from_fn(|axis| input_index[axis]..input_index[axis] + 1);
                let current = input_gradient.slice(target.clone()).to_concrete();
                let updated = (current + patch).to_concrete();
                input_gradient = input_gradient.slice_assign(target, &updated).to_concrete();
            });
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn restride_layout<const OUT: usize>(&self, new_layout: Layout) -> Tensor<OUT> {
        assert_eq!(new_layout.rank(), OUT, "restride_layout rank mismatch");
        let input_shape = self.shape();
        let value = self.value.restride_layout(new_layout.clone()).to_concrete();
        let input_id = self.handle.id;
        let output_shape: [usize; OUT] = std::array::from_fn(|axis| new_layout.shape()[axis]);
        let input_strides = Layout::continuous_strides(&input_shape);
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "restride_layout")?;
            let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
            for_each_index(output_shape, |output_index| {
                let linear = new_layout.linear_index(&output_index);
                let input_index: [usize; R] =
                    contiguous_index_from_linear::<R>(linear, &input_strides);
                let output_slices: [Range<usize>; OUT] =
                    std::array::from_fn(|axis| output_index[axis]..output_index[axis] + 1);
                let patch = gradient.slice(output_slices).reshape([1; R]).to_concrete();
                let target: [Range<usize>; R] =
                    std::array::from_fn(|axis| input_index[axis]..input_index[axis] + 1);
                let current = input_gradient.slice(target.clone()).to_concrete();
                let updated = (current + patch).to_concrete();
                input_gradient = input_gradient.slice_assign(target, &updated).to_concrete();
            });
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn squeeze_dims<const DIFF: usize, const OUT: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<OUT>
    where
        fusor_core::Tensor<R, f32>: fusor_core::SmallerRank<DIFF, OUT, f32>,
    {
        let shape = self.shape();
        for &axis in &axes {
            assert_eq!(shape[axis], 1, "Squeeze dimension {} must have size 1", axis);
        }
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut input_axis = 0;
        let mut axis_index = 0;
        let specs: [StrideSpec; OUT] = std::array::from_fn(|_| {
            while axis_index < DIFF && input_axis == sorted_axes[axis_index] {
                input_axis += 1;
                axis_index += 1;
            }
            let spec = StrideSpec::dim(input_axis, shape[input_axis]);
            input_axis += 1;
            spec
        });
        self.restride(specs)
    }

    pub fn unsqueeze_dims<const DIFF: usize, const OUT: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<OUT>
    where
        fusor_core::Tensor<R, f32>: fusor_core::LargerRank<DIFF, OUT, f32>,
    {
        let shape = self.shape();
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut input_axis = 0;
        let mut axis_index = 0;
        let specs: [StrideSpec; OUT] = std::array::from_fn(|output_axis| {
            if axis_index < DIFF && output_axis == sorted_axes[axis_index] {
                axis_index += 1;
                StrideSpec::dim_with(0, 1, 0)
            } else {
                let spec = StrideSpec::dim(input_axis, shape[input_axis]);
                input_axis += 1;
                spec
            }
        });
        self.restride(specs)
    }

    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        assert_same_graph(self, value);

        let output = self.value.slice_assign(slices.clone(), &value.value).to_concrete();
        let input_id = self.handle.id;
        let value_id = value.handle.id;
        let slice_shape = slices
            .clone()
            .map(|range| range.end.saturating_sub(range.start));
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "slice_assign")?;
            let zeros = RawTensor::zeros(&gradient.device(), slice_shape);
            Ok(vec![
                BackwardTarget {
                    node: input_id,
                    gradient: Box::new(gradient.slice_assign(slices.clone(), &zeros).to_concrete()),
                },
                BackwardTarget {
                    node: value_id,
                    gradient: Box::new(gradient.slice(slices.clone()).to_concrete()),
                },
            ])
        });
        self.from_op(
            output,
            vec![self.handle.clone(), value.handle.clone()],
            Some(backward),
        )
    }

    fn pad_axis(&self, axis: usize, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let input_shape = self.shape();
        let mut output_shape = input_shape;
        output_shape[axis] += padding * 2;
        let slices: [Range<usize>; R] = std::array::from_fn(|dim| {
            if dim == axis {
                padding..padding + input_shape[dim]
            } else {
                0..input_shape[dim]
            }
        });
        let value = RawTensor::zeros(&self.device(), output_shape)
            .slice_assign(slices.clone(), &self.value)
            .to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "pad_axis")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.slice(slices.clone()).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn unary_from_value(
        &self,
        value: RawTensor<R, f32>,
        backward: impl Fn(RawTensor<R, f32>, RawTensor<R, f32>) -> RawTensor<R, f32> + BackwardClosure,
    ) -> Self {
        let input_id = self.handle.id;
        let output = value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "unary")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(backward(gradient, output.clone()).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn binary_op(
        &self,
        rhs: &Self,
        value: RawTensor<R, f32>,
        backward: impl Fn(
            RawTensor<R, f32>,
            RawTensor<R, f32>,
            RawTensor<R, f32>,
        ) -> Vec<RawTensor<R, f32>>
        + BackwardClosure,
    ) -> Self {
        assert!(
            Arc::ptr_eq(&self.handle.graph, &rhs.handle.graph),
            "cannot mix autograd tensors from different graphs"
        );
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "binary")?;
            let gradients = backward(gradient, lhs_value.clone(), rhs_value.clone());
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradients[0].clone().to_concrete()),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(gradients[1].clone().to_concrete()),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    fn replay_unary<const OUT: usize>(
        &self,
        context: &'static str,
        value: RawTensor<OUT, f32>,
        replay: impl Fn(Tensor<R>) -> Tensor<OUT> + BackwardClosure,
    ) -> Tensor<OUT> {
        let input_id = self.handle.id;
        let input_value = self.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, context)?;
            let graph = Graph::new();
            let replay_input = Tensor::from_raw(&graph, input_value.clone());
            let replay_output = replay(replay_input.clone());
            let gradients = replay_output.backward_with(gradient)?;
            let input_gradient = gradients
                .get(&replay_input)
                .ok_or_else(|| Error::msg(format!("missing replay gradient in {context}")))?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn replay_binary<const R2: usize, const OUT: usize>(
        &self,
        rhs: &Tensor<R2>,
        context: &'static str,
        value: RawTensor<OUT, f32>,
        replay: impl Fn(Tensor<R>, Tensor<R2>) -> Tensor<OUT> + BackwardClosure,
    ) -> Tensor<OUT> {
        assert_same_graph(self, rhs);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, context)?;
            let graph = Graph::new();
            let replay_lhs = Tensor::from_raw(&graph, lhs_value.clone());
            let replay_rhs = Tensor::from_raw(&graph, rhs_value.clone());
            let replay_output = replay(replay_lhs.clone(), replay_rhs.clone());
            let gradients = replay_output.backward_with(gradient)?;
            let lhs_gradient = gradients
                .get(&replay_lhs)
                .ok_or_else(|| Error::msg(format!("missing lhs replay gradient in {context}")))?;
            let rhs_gradient = gradients
                .get(&replay_rhs)
                .ok_or_else(|| Error::msg(format!("missing rhs replay gradient in {context}")))?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(lhs_gradient),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(rhs_gradient),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    fn replay_ternary<const R2: usize, const R3: usize, const OUT: usize>(
        &self,
        second: &Tensor<R2>,
        third: &Tensor<R3>,
        context: &'static str,
        value: RawTensor<OUT, f32>,
        replay: impl Fn(Tensor<R>, Tensor<R2>, Tensor<R3>) -> Tensor<OUT> + BackwardClosure,
    ) -> Tensor<OUT> {
        assert_same_graph(self, second);
        assert_same_graph(self, third);
        let first_id = self.handle.id;
        let second_id = second.handle.id;
        let third_id = third.handle.id;
        let first_value = self.value.clone();
        let second_value = second.value.clone();
        let third_value = third.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, context)?;
            let graph = Graph::new();
            let replay_first = Tensor::from_raw(&graph, first_value.clone());
            let replay_second = Tensor::from_raw(&graph, second_value.clone());
            let replay_third = Tensor::from_raw(&graph, third_value.clone());
            let replay_output = replay(
                replay_first.clone(),
                replay_second.clone(),
                replay_third.clone(),
            );
            let gradients = replay_output.backward_with(gradient)?;
            let first_gradient = gradients
                .get(&replay_first)
                .ok_or_else(|| Error::msg(format!("missing first replay gradient in {context}")))?;
            let second_gradient = gradients.get(&replay_second).ok_or_else(|| {
                Error::msg(format!("missing second replay gradient in {context}"))
            })?;
            let third_gradient = gradients
                .get(&replay_third)
                .ok_or_else(|| Error::msg(format!("missing third replay gradient in {context}")))?;
            Ok(vec![
                BackwardTarget {
                    node: first_id,
                    gradient: Box::new(first_gradient),
                },
                BackwardTarget {
                    node: second_id,
                    gradient: Box::new(second_gradient),
                },
                BackwardTarget {
                    node: third_id,
                    gradient: Box::new(third_gradient),
                },
            ])
        });
        self.from_op(
            value,
            vec![
                self.handle.clone(),
                second.handle.clone(),
                third.handle.clone(),
            ],
            Some(backward),
        )
    }

    fn mat_mul_internal(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(
                        gradient.clone().mat_mul(&rhs_value.transpose(R - 2, R - 1)),
                    ),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(R - 2, R - 1).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        self.mat_mul_internal(rhs)
    }

    pub fn t(&self) -> Self {
        assert!(R >= 2, "t requires rank >= 2");
        self.transpose(R - 2, R - 1)
    }

    fn sum_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<OUT_RANK>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn max_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let input = self.value.clone();
        let value = input.max_keepdim::<OUT_RANK>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "max_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK>(
                    input.clone(),
                    axis,
                    gradient,
                    true,
                )),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn min_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MinOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let input = self.value.clone();
        let value = input.min_keepdim::<OUT_RANK>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "min_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK>(
                    input.clone(),
                    axis,
                    gradient,
                    false,
                )),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn mean_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.sum_keepdim_any::<OUT_RANK>(axis)
            .div_scalar(self.shape()[axis] as f32)
    }

    fn product_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::ProdOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::EqOp: fusor_cpu::SimdBinaryOp<f32>,
    {
        let input = self.value.clone();
        let input_shape = self.shape();
        let value = input.product_keepdim::<OUT_RANK>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "product_keepdim")?;
            let upstream = gradient.broadcast_as(input_shape).to_concrete();
            let zeros = RawTensor::zeros(&input.device(), input_shape);
            let ones = RawTensor::splat(&input.device(), 1.0, input_shape);
            let zero_mask = input.eq(0.0).to_concrete();
            let safe_input = zero_mask.where_cond(&ones, &input).to_concrete();
            let zero_count = zero_mask.sum_keepdim::<OUT_RANK>(axis).to_concrete();
            let zero_count_broadcast = zero_count.broadcast_as(input_shape).to_concrete();
            let product_non_zero = safe_input
                .product_keepdim::<OUT_RANK>(axis)
                .broadcast_as(input_shape)
                .to_concrete();
            let no_zero_grad = (upstream.clone() * (product_non_zero.clone() / safe_input).to_concrete())
                .to_concrete();
            let single_zero_grad = zero_mask
                .where_cond(&(upstream * product_non_zero).to_concrete(), &zeros)
                .to_concrete();
            let gradient = ((no_zero_grad * zero_count_broadcast.eq(0.0).to_concrete()).to_concrete()
                + (single_zero_grad * zero_count_broadcast.eq(1.0).to_concrete()).to_concrete())
            .to_concrete();
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn var_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let mean = self.mean_keepdim_any::<OUT_RANK>(axis);
        let centered = self.sub(&mean.broadcast_as(self.shape()));
        centered.sqr().mean_keepdim_any::<OUT_RANK>(axis)
    }

    fn softmax_composite<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let input_shape = self.shape();
        let max_values = self.max_keepdim_any::<OUT_RANK>(axis);
        let shifted = self.sub(&max_values.broadcast_as(input_shape));
        let exp_values = shifted.exp();
        let normalization = exp_values
            .sum_keepdim_any::<OUT_RANK>(axis)
            .broadcast_as(input_shape);
        exp_values.div(&normalization)
    }

    fn rms_norm_composite<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let std = self
            .sqr()
            .mean_keepdim_any::<OUT_RANK>(R - 1)
            .add_scalar(eps)
            .sqrt();
        let normalized = self.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    fn layer_norm_composite<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        let centered = {
            let mean = self.mean_keepdim_any::<OUT_RANK>(R - 1);
            self.sub(&mean.broadcast_as(self.shape()))
        };
        let variance = centered.sqr().mean_keepdim_any::<OUT_RANK>(R - 1);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    pub fn pool<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
        with: impl Fn(&Tensor<O>, usize) -> Self + Copy,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LargerRank<R2, DIFF, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: fusor_cpu::LargerRank<R3, 1, f32>,
        fusor_core::Tensor<R2, f32>: fusor_core::LargerRank<1, R3, f32>,
        fusor_core::Tensor<R3, f32>: fusor_core::SmallerRank<DIFF, O, f32>,
    {
        let pools: [crate::composite::pool::PoolSize; DIFF] = pools.map(|pool| pool.into());
        let axis_start = R - DIFF;
        let windows: [SlidingWindow; DIFF] = std::array::from_fn(|i| {
            let pool = pools[i];
            SlidingWindow::new(axis_start + i, pool.size, pool.stride)
        });
        let shape = self.shape();
        let mut sorted_windows = windows;
        sorted_windows.sort_by_key(|window| window.axis);
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            if out_i < R {
                if let Some(window) = sorted_windows.iter().find(|window| window.axis == out_i) {
                    let positions = (shape[out_i] - window.window_size) / window.step + 1;
                    StrideSpec::dim_with(out_i, positions, window.step)
                } else {
                    StrideSpec::dim(out_i, shape[out_i])
                }
            } else {
                let window = &sorted_windows[out_i - R];
                StrideSpec::dim(window.axis, window.window_size)
            }
        });

        let tiled: Tensor<R2> = self.restride(specs);
        let unsqueezed: Tensor<R3> = tiled.unsqueeze_dims::<1, R3>([R2]);
        let flattened: Tensor<O> = unsqueezed.flatten_last_n::<DIFF, O>();
        with(&flattened, O - 1)
    }

    pub fn pool_max<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LargerRank<R2, DIFF, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: fusor_cpu::LargerRank<R3, 1, f32>,
        fusor_core::Tensor<R2, f32>: fusor_core::LargerRank<1, R3, f32>,
        fusor_core::Tensor<R3, f32>: fusor_core::SmallerRank<DIFF, O, f32>,
        crate::ConcreteTensor<f32, O>: fusor_cpu::LastRank<R, f32>,
        fusor_core::Tensor<O, f32>:
            fusor_core::LastRank<R, f32> + fusor_core::SmallerRank<1, R, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.max::<R>(axis))
    }

    pub fn pool_min<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LargerRank<R2, DIFF, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: fusor_cpu::LargerRank<R3, 1, f32>,
        fusor_core::Tensor<R2, f32>: fusor_core::LargerRank<1, R3, f32>,
        fusor_core::Tensor<R3, f32>: fusor_core::SmallerRank<DIFF, O, f32>,
        crate::ConcreteTensor<f32, O>: fusor_cpu::LastRank<R, f32>,
        fusor_core::Tensor<O, f32>:
            fusor_core::LastRank<R, f32> + fusor_core::SmallerRank<1, R, f32>,
        fusor_cpu::MinOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.min::<R>(axis))
    }

    pub fn q_mat_mul(&self, weights: &crate::QMatrix) -> Self {
        assert!(R >= 2, "q_mat_mul requires rank >= 2");
        let value = self.value.q_mat_mul(weights).to_concrete();
        let dequantized: RawTensor<2, f32> = weights.dequantize();
        let n = weights.shape()[0];
        let k = weights.shape()[1];
        let batch_dims = R - 2;
        let weight_shape: [usize; R] = std::array::from_fn(|i| {
            if i < batch_dims {
                1
            } else if i == batch_dims {
                k
            } else {
                n
            }
        });
        let weight = dequantized.transpose(0, 1).reshape(weight_shape).to_concrete();
        self.replay_unary("q_mat_mul", value, move |input| {
            let weight = Tensor::constant_from_raw(&input.graph(), weight.clone());
            input.mat_mul_internal(&weight)
        })
    }

    pub fn stack<const OUT: usize>(tensors: impl IntoIterator<Item = Self>, dim: usize) -> Tensor<OUT>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LargerRank<OUT, 1, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LargerRank<1, OUT, f32>,
    {
        let tensors: Vec<Self> = tensors.into_iter().collect();
        assert!(!tensors.is_empty(), "stack requires at least one tensor");

        let graph = tensors[0].handle.graph.clone();
        let input_shape = tensors[0].shape();
        let raw = tensors
            .iter()
            .map(|tensor| {
                assert!(
                    Arc::ptr_eq(&graph, &tensor.handle.graph),
                    "cannot mix autograd tensors from different graphs"
                );
                assert_eq!(tensor.shape(), input_shape, "stack requires matching shapes");
                tensor.value.unsqueeze_dims::<1, OUT>([dim]).to_concrete()
            })
            .collect::<Vec<_>>();
        let value = RawTensor::cat(raw, dim);
        let parents = tensors
            .iter()
            .map(|tensor| tensor.handle.clone())
            .collect::<Vec<_>>();
        let parent_ids = parents.iter().map(|parent| parent.id).collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "stack")?;
            let mut targets = Vec::with_capacity(parent_ids.len());
            for (index, &parent_id) in parent_ids.iter().enumerate() {
                let slices: [Range<usize>; OUT] = std::array::from_fn(|axis| {
                    if axis == dim {
                        index..index + 1
                    } else {
                        0..gradient.shape()[axis]
                    }
                });
                let grad = gradient.slice(slices).reshape(input_shape).to_concrete();
                targets.push(BackwardTarget {
                    node: parent_id,
                    gradient: Box::new(grad),
                });
            }
            Ok(targets)
        });
        let id = graph.add_node(
            parents.iter().map(|parent| parent.id).collect(),
            Some(backward),
            parents
                .iter()
                .any(|parent| parent.graph.requires_grad(parent.id)),
        );
        Tensor {
            value,
            handle: NodeHandle { graph, id },
        }
    }

    pub fn max_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn max<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>:
            fusor_core::LastRank<OUT_RANK, f32> + fusor_core::SmallerRank<1, OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn min_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MinOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn min<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>:
            fusor_core::LastRank<OUT_RANK, f32> + fusor_core::SmallerRank<1, OUT_RANK, f32>,
        fusor_cpu::MinOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn mean_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn mean<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>:
            fusor_core::LastRank<OUT_RANK, f32> + fusor_core::SmallerRank<1, OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>:
            fusor_core::LastRank<OUT_RANK, f32> + fusor_core::SmallerRank<1, OUT_RANK, f32>,
        fusor_cpu::ProdOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::EqOp: fusor_cpu::SimdBinaryOp<f32>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::ProdOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::EqOp: fusor_cpu::SimdBinaryOp<f32>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn var<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>:
            fusor_core::LastRank<OUT_RANK, f32> + fusor_core::SmallerRank<1, OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn var_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn softmax<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.softmax_composite::<OUT_RANK>(axis)
    }

    pub fn softmax_slow<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.softmax::<OUT_RANK>(axis)
    }

    pub fn softmax_last_dim<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.softmax::<OUT_RANK>(R - 1)
    }

    pub fn softmax_slow_last_dim<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
    {
        self.softmax_last_dim::<OUT_RANK>()
    }

    pub fn softmax_last_dim_fused<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRankInner,
    {
        let value = self
            .value
            .softmax_last_dim_fused::<OUT_RANK>()
            .to_concrete();
        self.replay_unary("softmax_last_dim_fused", value, |input| {
            input.softmax_last_dim::<OUT_RANK>()
        })
    }

    pub fn rms_norm_fused<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        <fusor_core::Tensor<R, f32> as fusor_core::LastRankInner>::LastRank:
            fusor_core::NextRankInner<NextRank = fusor_core::Tensor<R, f32>>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: fusor_cpu::SimdUnaryOp<f32>,
        (fusor_core::Tensor<R, f32>, fusor_core::Tensor<1, f32>): fusor_core::MaxRank<R, f32>,
    {
        let value = self.value.rms_norm_fused::<1, OUT_RANK>(
            &weight.value,
            bias.as_ref().map(|bias| &bias.value),
            eps,
        );
        if let Some(bias) = bias {
            self.replay_ternary(
                weight,
                bias,
                "rms_norm_fused",
                value,
                move |input, weight, bias| {
                    input.rms_norm_composite::<OUT_RANK>(&weight, Some(&bias), eps)
                },
            )
        } else {
            self.replay_binary(weight, "rms_norm_fused", value, move |input, weight| {
                input.rms_norm_composite::<OUT_RANK>(&weight, None, eps)
            })
        }
    }

    pub fn rms_norm_fused_no_bias<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        <fusor_core::Tensor<R, f32> as fusor_core::LastRankInner>::LastRank:
            fusor_core::NextRankInner<NextRank = fusor_core::Tensor<R, f32>>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: fusor_cpu::SimdUnaryOp<f32>,
        (fusor_core::Tensor<R, f32>, fusor_core::Tensor<1, f32>): fusor_core::MaxRank<R, f32>,
    {
        self.rms_norm_fused::<OUT_RANK>(weight, None, eps)
    }

    pub fn layer_norm_last_dim_fused<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
        <fusor_core::Tensor<R, f32> as fusor_core::LastRankInner>::LastRank:
            fusor_core::NextRankInner<NextRank = fusor_core::Tensor<R, f32>>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::SubOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: fusor_cpu::SimdUnaryOp<f32>,
    {
        let value = self.value.layer_norm_last_dim_fused::<OUT_RANK, 1, _, _>(
            &weight.value,
            bias.as_ref().map(|bias| &bias.value),
            eps,
        );
        if let Some(bias) = bias {
            self.replay_ternary(
                weight,
                bias,
                "layer_norm_last_dim_fused",
                value,
                move |input, weight, bias| {
                    input.layer_norm_composite::<OUT_RANK>(&weight, Some(&bias), eps)
                },
            )
        } else {
            self.replay_binary(
                weight,
                "layer_norm_last_dim_fused",
                value,
                move |input, weight| input.layer_norm_composite::<OUT_RANK>(&weight, None, eps),
            )
        }
    }
}

impl Tensor<1> {
    pub fn arange(graph: &Graph, device: &Device, start: f32, end: f32) -> Tensor<1> {
        graph.leaf(crate::arange(device, start, end))
    }

    pub fn arange_step(graph: &Graph, device: &Device, start: f32, end: f32, step: f32) -> Tensor<1> {
        graph.leaf(crate::arange_step(device, start, end, step))
    }

    pub fn sum(&self) -> Tensor<0> {
        let input_shape = self.shape();
        let value = self.value.sum::<0>(0);
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<0>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor<2> {
        let value = self.value.unsqueeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "unsqueeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.squeeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }
}

impl Tensor<2> {
    pub fn mat_mul(&self, rhs: &Tensor<2>) -> Tensor<2> {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradient.clone().mat_mul(&rhs_value.transpose(0, 1))),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(0, 1).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn squeeze(&self, dim: usize) -> Tensor<1> {
        let value = self.value.squeeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "squeeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.unsqueeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor<3> {
        let value = self.value.unsqueeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "unsqueeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.squeeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum(&self, axis: usize) -> Tensor<1> {
        let input_shape = self.shape();
        let value = self.value.sum::<1>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .unsqueeze(axis)
                        .broadcast_as(input_shape)
                        .to_concrete(),
                ),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<2> {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<1>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<2> {
        let centered = {
            let mean = self.sum_keepdim(1).div_scalar(self.shape()[1] as f32);
            self.sub(&mean.broadcast_as(self.shape()))
        };
        let variance = centered
            .sqr()
            .sum_keepdim(1)
            .div_scalar(self.shape()[1] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<2> {
        let variance = self.sqr().sum_keepdim(1).div_scalar(self.shape()[1] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = self.div(&std.broadcast_as(self.shape()));
        normalized.mul(&weight.broadcast_as(self.shape()))
    }

    pub fn index_select(&self, dimension: usize, indices: &RawTensor<1, u32>) -> Tensor<2> {
        let input_shape = self.shape();
        assert!(dimension < 2, "index_select dimension out of bounds");

        let index_values = pollster::block_on(indices.clone().as_slice())
            .unwrap()
            .to_vec1();
        for &index in &index_values {
            assert!(
                (index as usize) < input_shape[dimension],
                "index_select index {index} out of bounds for dimension size {}",
                input_shape[dimension]
            );
        }

        let value = self.value.index_select(dimension, indices).to_concrete();
        let input_id = self.handle.id;
        let device = self.device();
        let output_shape = value.shape();
        let index_values = index_values.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "index_select")?;
            let gradient_values = pollster::block_on(
                gradient
                    .clone()
                    .reshape([output_shape.iter().product()])
                    .as_slice(),
            )?
            .to_vec1();

            let mut input_gradient = vec![0.0f32; input_shape.iter().product()];
            let input_strides = Layout::continuous_strides(&input_shape);
            let output_strides = Layout::continuous_strides(&output_shape);

            for (linear_index, value) in gradient_values.into_iter().enumerate() {
                let mut remainder = linear_index;
                let mut input_linear_index = 0;
                for axis in 0..2 {
                    let coordinate = remainder / output_strides[axis];
                    remainder %= output_strides[axis];
                    let input_coordinate = if axis == dimension {
                        index_values[coordinate] as usize
                    } else {
                        coordinate
                    };
                    input_linear_index += input_coordinate * input_strides[axis];
                }
                input_gradient[input_linear_index] += value;
            }

            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(RawTensor::from_slice(&device, input_shape, &input_gradient)),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn gather_last(&self, indices: &RawTensor<1, u32>) -> Tensor<1> {
        let shape = self.shape();
        assert_eq!(
            shape[0],
            indices.shape()[0],
            "gather_last expects one index per row"
        );
        let width = shape[1];
        let device = self.device();
        let index_values = pollster::block_on(indices.clone().as_slice())
            .unwrap()
            .to_vec1();
        let linear_indices = index_values
            .iter()
            .enumerate()
            .map(|(row, &column)| {
                assert!(
                    (column as usize) < width,
                    "gather_last index {} out of bounds for width {}",
                    column,
                    width
                );
                (row * width + column as usize) as u32
            })
            .collect::<Vec<_>>();
        let linear_indices_tensor = RawTensor::from_slice(&device, [shape[0]], &linear_indices);
        let flat = self.value.reshape([shape[0] * width]).to_concrete();
        let value = flat.index_select(0, &linear_indices_tensor).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "gather_last")?;
            let gradient_values = pollster::block_on(gradient.clone().as_slice())?.to_vec1();
            let mut input_gradient = vec![0.0f32; shape[0] * width];
            for (row, &linear_index) in linear_indices.iter().enumerate() {
                input_gradient[linear_index as usize] += gradient_values[row];
            }
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(RawTensor::from_slice(&device, shape, &input_gradient)),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn embedding(&self, indices: &RawTensor<2, u32>) -> Tensor<3> {
        let value: RawTensor<3, f32> =
            Embedding::new_from_tensor(self.value.clone()).forward(indices);
        let table_id = self.handle.id;
        let table_shape = self.shape();
        let device = self.device();
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "embedding")?;
            let index_values = pollster::block_on(indices.clone().as_slice())?.to_vec2();
            let grad_shape = gradient.shape();
            let grad_flat = gradient.reshape([grad_shape[0] * grad_shape[1], grad_shape[2]]);

            let mut rows_by_token = HashMap::<u32, Vec<u32>>::new();
            for (batch, row) in index_values.iter().enumerate() {
                for (position, &token) in row.iter().enumerate() {
                    let flat_row = (batch * grad_shape[1] + position) as u32;
                    rows_by_token.entry(token).or_default().push(flat_row);
                }
            }

            let mut embedding_gradient = RawTensor::zeros(&device, table_shape);
            for (token, rows) in rows_by_token {
                let row_indices = RawTensor::from_slice(&device, [rows.len()], &rows);
                let token_gradient = grad_flat
                    .index_select(0, &row_indices)
                    .sum::<1>(0)
                    .unsqueeze::<2>(0)
                    .to_concrete();
                embedding_gradient = embedding_gradient.slice_assign(
                    [token as usize..token as usize + 1, 0..table_shape[1]],
                    &token_gradient,
                );
            }

            Ok(vec![BackwardTarget {
                node: table_id,
                gradient: Box::new(embedding_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }
}

impl Tensor<3> {
    pub fn sliding_window_view(&self, window: SlidingWindow) -> Tensor<4> {
        assert_eq!(
            window.axis, 2,
            "autograd sliding_window_view for Tensor<3> currently supports axis=2"
        );
        let input_shape = self.shape();
        let value = self.value.sliding_window_view([window]).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<4>(&*gradient, "sliding_window_view")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(sliding_window_view_backward_3(
                    &gradient,
                    input_shape,
                    window,
                )),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn mat_mul(&self, rhs: &Tensor<3>) -> Tensor<3> {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradient.clone().mat_mul(&rhs_value.transpose(1, 2))),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(1, 2).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn squeeze(&self, dim: usize) -> Tensor<2> {
        let value = self.value.squeeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "squeeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.unsqueeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum(&self, axis: usize) -> Tensor<2> {
        let input_shape = self.shape();
        let value = self.value.sum::<2>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .unsqueeze(axis)
                        .broadcast_as(input_shape)
                        .to_concrete(),
                ),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<3> {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<2>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn cat(tensors: Vec<Tensor<3>>, dim: usize) -> Tensor<3> {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        let graph = tensors[0].handle.graph.clone();
        let raw = tensors
            .iter()
            .map(|tensor| tensor.value.clone())
            .collect::<Vec<_>>();
        let value = RawTensor::cat(raw, dim);
        let parents = tensors
            .iter()
            .map(|tensor| tensor.handle.clone())
            .collect::<Vec<_>>();
        let parent_ids = parents.iter().map(|parent| parent.id).collect::<Vec<_>>();
        let slices = tensors
            .iter()
            .scan(0usize, |offset, tensor| {
                let start = *offset;
                let length = tensor.shape()[dim];
                *offset += length;
                Some(start..start + length)
            })
            .collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "cat")?;
            let mut targets = Vec::with_capacity(parent_ids.len());
            for (&parent_id, slice) in parent_ids.iter().zip(slices.iter()) {
                let grad_slice = match dim {
                    0 => gradient.slice([
                        slice.clone(),
                        0..gradient.shape()[1],
                        0..gradient.shape()[2],
                    ]),
                    1 => gradient.slice([
                        0..gradient.shape()[0],
                        slice.clone(),
                        0..gradient.shape()[2],
                    ]),
                    2 => gradient.slice([
                        0..gradient.shape()[0],
                        0..gradient.shape()[1],
                        slice.clone(),
                    ]),
                    _ => panic!("invalid cat dim"),
                }
                .to_concrete();
                targets.push(BackwardTarget {
                    node: parent_id,
                    gradient: Box::new(grad_slice),
                });
            }
            Ok(targets)
        });
        let id = graph.add_node(
            parents.iter().map(|parent| parent.id).collect(),
            Some(backward),
            parents
                .iter()
                .any(|parent| parent.graph.requires_grad(parent.id)),
        );
        Tensor {
            value,
            handle: NodeHandle { graph, id },
        }
    }

    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<3> {
        let centered = {
            let mean = self.sum_keepdim(2).div_scalar(self.shape()[2] as f32);
            self.sub(&mean.broadcast_as(self.shape()))
        };
        let variance = centered
            .sqr()
            .sum_keepdim(2)
            .div_scalar(self.shape()[2] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<3> {
        let variance = self.sqr().sum_keepdim(2).div_scalar(self.shape()[2] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = self.div(&std.broadcast_as(self.shape()));
        normalized.mul(&weight.broadcast_as(self.shape()))
    }

    pub fn conv(
        &self,
        weight: &Tensor<3>,
        bias: Option<&Tensor<1>>,
        padding: [usize; 1],
        strides: [usize; 1],
    ) -> Tensor<3> {
        assert_same_graph(self, weight);
        if let Some(bias) = bias {
            assert_same_graph(self, bias);
        }

        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        let kernel_size = weight_shape[2];

        let padded = self.pad_axis(2, padding[0]);
        let out_len = (input_shape[2] + 2 * padding[0] - kernel_size) / strides[0] + 1;
        let windows = padded.sliding_window_view(SlidingWindow::new(2, kernel_size, strides[0]));
        let windows_flat = windows
            .permute(conv_window_permutation::<4, 1>())
            .reshape([batch * out_len, in_channels * kernel_size]);
        let weight_t = weight
            .reshape([out_channels, in_channels * kernel_size])
            .transpose(0, 1);
        let output = windows_flat.mat_mul(&weight_t);
        let mut output_final = output
            .reshape([batch, out_len, out_channels])
            .permute(conv_output_permutation::<3, 1>())
            .reshape([batch, out_channels, out_len]);
        if let Some(bias) = bias {
            output_final = output_final.add(&bias.reshape([1, out_channels, 1]).broadcast_as([
                batch,
                out_channels,
                out_len,
            ]));
        }
        output_final
    }
}

impl Tensor<4> {
    fn rotate_half(&self) -> Tensor<4> {
        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let first_half = self.narrow(3, 0, half);
        let second_half = self.narrow(3, half, embed - half).mul_scalar(-1.0);
        let graph = self.graph();
        let device = self.device();
        let zeros = Tensor::zeros(&graph, &device, [batch, heads, sequence_length, embed]);
        let combined = zeros.slice_assign(
            [0..batch, 0..heads, 0..sequence_length, 0..half],
            &second_half,
        );
        combined.slice_assign(
            [0..batch, 0..heads, 0..sequence_length, half..embed],
            &first_half,
        )
    }

    fn rope_interleaved_composite(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, 1, sequence_length, half, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, 1, sequence_length, half, 1]);
        let x = self.reshape([batch, heads, sequence_length, half, 2]);
        let x0 = x.narrow(4, 0, 1);
        let x1 = x.narrow(4, 1, 1);
        let y0 = x0.mul(&cos).sub(&x1.mul(&sin));
        let y1 = x0.mul(&sin).add(&x1.mul(&cos));
        let graph = self.graph();
        let device = self.device();
        let zeros = Tensor::zeros(&graph, &device, [batch, heads, sequence_length, half, 2]);
        let combined = zeros.slice_assign(
            [0..batch, 0..heads, 0..sequence_length, 0..half, 0..1],
            &y0,
        );
        combined
            .slice_assign(
                [0..batch, 0..heads, 0..sequence_length, 0..half, 1..2],
                &y1,
            )
            .flatten_last_n::<1, 4>()
    }

    pub fn sum(&self, axis: usize) -> Tensor<3> {
        let input_shape = self.shape();
        let value = self.value.sum::<3>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .unsqueeze(axis)
                        .broadcast_as(input_shape)
                        .to_concrete(),
                ),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn rope(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let graph = self.graph();
        let device = self.device();
        let cos_base = cos.narrow(0, 0, sequence_length);
        let sin_base = sin.narrow(0, 0, sequence_length);
        let cos = Tensor::zeros(&graph, &device, [sequence_length, embed])
            .slice_assign([0..sequence_length, 0..half], &cos_base)
            .slice_assign([0..sequence_length, half..embed], &cos_base)
            .unsqueeze_dims::<2, 4>([0, 1])
            .broadcast_as([batch, heads, sequence_length, embed]);
        let sin = Tensor::zeros(&graph, &device, [sequence_length, embed])
            .slice_assign([0..sequence_length, 0..half], &sin_base)
            .slice_assign([0..sequence_length, half..embed], &sin_base)
            .unsqueeze_dims::<2, 4>([0, 1])
            .broadcast_as([batch, heads, sequence_length, embed]);
        let rotated = self.rotate_half();
        self.mul(&cos).add(&rotated.mul(&sin))
    }

    pub fn rope_interleaved(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        self.rope_interleaved_composite(cos, sin)
    }

    pub fn flash_attention(
        &self,
        k: &Tensor<4>,
        v: &Tensor<4>,
        scale: f32,
        mask: Option<(&RawTensor<2, f32>, MaskKind)>,
    ) -> Tensor<4> {
        let value = self.value.flash_attention(
            &k.value,
            &v.value,
            scale,
            mask.map(|(mask, kind)| (mask, kind)),
        );
        let mask_value = mask.map(|(mask, kind)| (mask.clone(), kind));
        self.replay_ternary(k, v, "flash_attention", value, move |q, k, v| {
            q.flash_attention_composite(&k, &v, scale, mask_value.as_ref())
        })
    }

    pub fn rope_fused(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self.value.rope_fused(&cos.value, &sin.value).to_concrete();
        self.replay_ternary(cos, sin, "rope_fused", value, |input, cos, sin| {
            input.rope_interleaved_composite(&cos, &sin)
        })
    }

    pub fn rope_normal_fused(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self
            .value
            .rope_normal_fused(&cos.value, &sin.value)
            .to_concrete();
        self.replay_ternary(cos, sin, "rope_normal_fused", value, |input, cos, sin| {
            input.rope(&cos, &sin)
        })
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<4> {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<3>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<4>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<4> {
        let centered = {
            let mean = self.sum_keepdim(3).div_scalar(self.shape()[3] as f32);
            self.sub(&mean.broadcast_as(self.shape()))
        };
        let variance = centered
            .sqr()
            .sum_keepdim(3)
            .div_scalar(self.shape()[3] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<4> {
        let variance = self.sqr().sum_keepdim(3).div_scalar(self.shape()[3] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = self.div(&std.broadcast_as(self.shape()));
        normalized.mul(&weight.broadcast_as(self.shape()))
    }

    pub fn sliding_window_view(&self, windows: [SlidingWindow; 2]) -> Tensor<6> {
        assert_eq!(
            windows[0].axis, 2,
            "autograd sliding_window_view for Tensor<4> currently supports axis=2 for the first window"
        );
        assert_eq!(
            windows[1].axis, 3,
            "autograd sliding_window_view for Tensor<4> currently supports axis=3 for the second window"
        );
        let input_shape = self.shape();
        let value = self.value.sliding_window_view(windows).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<6>(&*gradient, "sliding_window_view")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(sliding_window_view_backward_4(
                    &gradient,
                    input_shape,
                    windows,
                )),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn conv(
        &self,
        weight: &Tensor<4>,
        bias: Option<&Tensor<1>>,
        padding: [usize; 2],
        strides: [usize; 2],
    ) -> Tensor<4> {
        assert_same_graph(self, weight);
        if let Some(bias) = bias {
            assert_same_graph(self, bias);
        }

        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];
        let out_h = (input_shape[2] + 2 * padding[0] - kernel_h) / strides[0] + 1;
        let out_w = (input_shape[3] + 2 * padding[1] - kernel_w) / strides[1] + 1;
        let out_spatial = out_h * out_w;
        let kernel_size = kernel_h * kernel_w;

        let padded = self.pad_axis(2, padding[0]).pad_axis(3, padding[1]);
        let windows = padded.sliding_window_view([
            SlidingWindow::new(2, kernel_h, strides[0]),
            SlidingWindow::new(3, kernel_w, strides[1]),
        ]);
        let windows_flat = windows
            .permute(conv_window_permutation::<6, 2>())
            .reshape([batch * out_spatial, in_channels * kernel_size]);
        let weight_t = weight
            .reshape([out_channels, in_channels * kernel_size])
            .transpose(0, 1);
        let output = windows_flat.mat_mul(&weight_t);
        let mut output_final = output
            .reshape([batch, out_h, out_w, out_channels])
            .permute(conv_output_permutation::<4, 2>())
            .reshape([batch, out_channels, out_h, out_w]);
        if let Some(bias) = bias {
            output_final = output_final.add(&bias.reshape([1, out_channels, 1, 1]).broadcast_as([
                batch,
                out_channels,
                out_h,
                out_w,
            ]));
        }
        output_final
    }

    fn flash_attention_composite(
        &self,
        k: &Tensor<4>,
        v: &Tensor<4>,
        scale: f32,
        mask: Option<&(RawTensor<2, f32>, MaskKind)>,
    ) -> Tensor<4> {
        let q_shape = self.shape();
        let k_shape = k.shape();
        let batch = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let head_dim = q_shape[3];
        let num_kv_heads = k_shape[1];
        let kv_seq_len = k_shape[2];
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({num_heads}) must be divisible by number of K/V heads ({num_kv_heads})"
        );

        let num_key_value_groups = num_heads / num_kv_heads;
        let (k_expanded, v_expanded) = if num_key_value_groups > 1 {
            let k_broadcast = k
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);
            let v_broadcast = v
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);
            (
                k_broadcast.reshape([batch, num_heads, kv_seq_len, head_dim]),
                v_broadcast.reshape([batch, num_heads, kv_seq_len, head_dim]),
            )
        } else {
            (k.clone(), v.clone())
        };

        let scores = self
            .mat_mul_internal(&k_expanded.transpose(2, 3))
            .div_scalar(scale.recip());
        let masked_scores = if let Some((mask, kind)) = mask {
            let mask_tensor = Tensor::constant_from_raw(&self.graph(), mask.clone());
            let mask_4d = match kind {
                MaskKind::QKMask => {
                    assert_eq!(mask_tensor.shape(), [q_seq_len, kv_seq_len]);
                    mask_tensor.reshape([1, 1, q_seq_len, kv_seq_len])
                }
                MaskKind::BatchKeyMask => {
                    assert_eq!(mask_tensor.shape(), [batch, kv_seq_len]);
                    mask_tensor.reshape([batch, 1, 1, kv_seq_len])
                }
            };
            scores.add(&mask_4d.broadcast_as([batch, num_heads, q_seq_len, kv_seq_len]))
        } else {
            scores
        };
        masked_scores
            .softmax_last_dim::<3>()
            .mat_mul_internal(&v_expanded)
    }
}

impl Gradients {
    pub fn get<const R: usize>(&self, tensor: &Tensor<R>) -> Option<RawTensor<R, f32>> {
        self.gradients
            .get(&tensor.handle.id)
            .and_then(|gradient| gradient.as_any().downcast_ref::<RawTensor<R, f32>>())
            .cloned()
    }

    pub fn into_detached(self) -> Self {
        Self {
            gradients: self
                .gradients
                .into_iter()
                .map(|(id, gradient)| (id, gradient.into_detached()))
                .collect(),
        }
    }
}

impl BackwardTarget {
    pub fn wrt<const R: usize>(tensor: &Tensor<R>, gradient: RawTensor<R, f32>) -> Self {
        Self {
            node: tensor.handle.id,
            gradient: Box::new(gradient),
        }
    }
}

impl GraphInner {
    fn add_node(
        &self,
        parents: Vec<NodeId>,
        backward: Option<BackwardRule>,
        requires_grad: bool,
    ) -> NodeId {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.nodes.insert(
            id,
            Node {
                parents,
                backward,
                requires_grad,
            },
        );
        id
    }

    fn replace_node(&self, id: NodeId, node: Node) {
        self.state.lock().unwrap().nodes.insert(id, node);
    }

    fn requires_grad(&self, id: NodeId) -> bool {
        self.state
            .lock()
            .unwrap()
            .nodes
            .get(&id)
            .map(|node| node.requires_grad)
            .unwrap_or(false)
    }

    fn backward(&self, root: NodeId, seed: Box<dyn AnyTensorValue>) -> Result<Gradients> {
        let nodes = self.reachable_nodes(root);
        let mut pending_children = HashMap::<NodeId, usize>::new();
        for (id, node) in &nodes {
            pending_children.entry(*id).or_insert(0);
            for parent in &node.parents {
                *pending_children.entry(*parent).or_insert(0) += 1;
            }
        }

        let mut gradients = HashMap::<NodeId, Box<dyn AnyTensorValue>>::new();
        gradients.insert(root, seed);

        let mut queue = VecDeque::new();
        queue.push_back(root);

        while let Some(node_id) = queue.pop_front() {
            let Some(node) = nodes.get(&node_id) else {
                continue;
            };
            let Some(backward) = node.backward.as_ref() else {
                continue;
            };
            let gradient = gradients
                .get(&node_id)
                .ok_or_else(|| Error::msg(format!("missing gradient for node {node_id}")))?
                .clone_box();

            for target in backward(gradient)? {
                let Some(parent_node) = nodes.get(&target.node) else {
                    continue;
                };
                if !parent_node.requires_grad {
                    continue;
                }
                accumulate_gradient(&mut gradients, target.node, target.gradient)?;
                let remaining = pending_children.get_mut(&target.node).ok_or_else(|| {
                    Error::msg(format!("missing child count for node {}", target.node))
                })?;
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    queue.push_back(target.node);
                }
            }
        }

        Ok(Gradients { gradients })
    }

    fn reachable_nodes(&self, root: NodeId) -> HashMap<NodeId, Node> {
        let snapshot = self.state.lock().unwrap().nodes.clone();
        let mut reachable = HashMap::new();
        let mut stack = vec![root];
        let mut visited = HashSet::new();
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            if let Some(node) = snapshot.get(&node_id) {
                reachable.insert(node_id, node.clone());
                stack.extend(node.parents.iter().copied());
            }
        }
        reachable
    }
}

impl<const R: usize> AnyTensorValue for RawTensor<R, f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnyTensorValue> {
        Box::new(self.clone())
    }

    fn into_detached(self: Box<Self>) -> Box<dyn AnyTensorValue> {
        match *self {
            RawTensor::Cpu(tensor) => Box::new(RawTensor::Cpu(tensor.to_concrete())),
            RawTensor::Gpu(tensor) => Box::new(RawTensor::Gpu(tensor.detach())),
        }
    }

    fn add_box(&self, other: &dyn AnyTensorValue) -> Result<Box<dyn AnyTensorValue>> {
        let other = other
            .as_any()
            .downcast_ref::<RawTensor<R, f32>>()
            .ok_or_else(|| Error::msg("gradient rank mismatch while accumulating"))?;
        Ok(Box::new((self.clone() + other.clone()).to_concrete()))
    }
}

fn accumulate_gradient(
    gradients: &mut HashMap<NodeId, Box<dyn AnyTensorValue>>,
    node: NodeId,
    gradient: Box<dyn AnyTensorValue>,
) -> Result<()> {
    match gradients.get(&node) {
        Some(existing) => {
            let accumulated = existing.add_box(&*gradient)?;
            gradients.insert(node, accumulated);
        }
        None => {
            gradients.insert(node, gradient);
        }
    }
    Ok(())
}

fn downcast_tensor<const R: usize>(
    value: &dyn AnyTensorValue,
    context: &str,
) -> Result<RawTensor<R, f32>> {
    value
        .as_any()
        .downcast_ref::<RawTensor<R, f32>>()
        .cloned()
        .ok_or_else(|| Error::msg(format!("gradient rank mismatch in {context}")))
}

fn assert_same_graph<const R: usize, const R2: usize>(lhs: &Tensor<R>, rhs: &Tensor<R2>) {
    assert!(
        Arc::ptr_eq(&lhs.handle.graph, &rhs.handle.graph),
        "cannot mix autograd tensors from different graphs"
    );
}

fn conv_window_permutation<const R2: usize, const DIFF: usize>() -> [usize; R2] {
    std::array::from_fn(|index| {
        if index == 0 {
            0
        } else if index <= DIFF {
            index + 1
        } else if index == DIFF + 1 {
            1
        } else {
            index
        }
    })
}

fn conv_output_permutation<const R: usize, const DIFF: usize>() -> [usize; R] {
    std::array::from_fn(|index| {
        if index == 0 {
            0
        } else if index == 1 {
            DIFF + 1
        } else {
            index - 1
        }
    })
}

fn for_each_index<const R: usize>(limits: [usize; R], mut visitor: impl FnMut([usize; R])) {
    if limits.contains(&0) {
        return;
    }

    let mut index = [0; R];
    loop {
        visitor(index);

        let mut axis = R;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < limits[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn restride_input_index<const R: usize, const OUT: usize>(
    specs: [StrideSpec; OUT],
    output_index: [usize; OUT],
) -> [usize; R] {
    let mut input_index = [0; R];
    for axis in 0..OUT {
        let spec = specs[axis];
        input_index[spec.input_dim] += spec.offset + output_index[axis] * spec.multiplier;
    }
    input_index
}

fn contiguous_index_from_linear<const R: usize>(
    mut linear: usize,
    strides: &[usize],
) -> [usize; R] {
    let mut input_index = [0; R];
    for axis in 0..R {
        input_index[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
    input_index
}

fn sliding_window_view_backward_3(
    gradient: &RawTensor<4, f32>,
    input_shape: [usize; 3],
    window: SlidingWindow,
) -> RawTensor<3, f32> {
    let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
    let out_len = gradient.shape()[2];
    for out_index in 0..out_len {
        let start = out_index * window.step;
        let patch = gradient
            .slice([
                0..gradient.shape()[0],
                0..gradient.shape()[1],
                out_index..out_index + 1,
                0..window.window_size,
            ])
            .reshape([input_shape[0], input_shape[1], window.window_size])
            .to_concrete();
        let target = [
            0..input_shape[0],
            0..input_shape[1],
            start..start + window.window_size,
        ];
        let current = input_gradient.slice(target.clone()).to_concrete();
        let updated = (current + patch).to_concrete();
        input_gradient = input_gradient.slice_assign(target, &updated).to_concrete();
    }
    input_gradient
}

fn sliding_window_view_backward_4(
    gradient: &RawTensor<6, f32>,
    input_shape: [usize; 4],
    windows: [SlidingWindow; 2],
) -> RawTensor<4, f32> {
    let mut input_gradient = RawTensor::zeros(&gradient.device(), input_shape);
    let out_h = gradient.shape()[2];
    let out_w = gradient.shape()[3];
    for y in 0..out_h {
        for x in 0..out_w {
            let start_y = y * windows[0].step;
            let start_x = x * windows[1].step;
            let patch = gradient
                .slice([
                    0..gradient.shape()[0],
                    0..gradient.shape()[1],
                    y..y + 1,
                    x..x + 1,
                    0..windows[0].window_size,
                    0..windows[1].window_size,
                ])
                .reshape([
                    input_shape[0],
                    input_shape[1],
                    windows[0].window_size,
                    windows[1].window_size,
                ])
                .to_concrete();
            let target = [
                0..input_shape[0],
                0..input_shape[1],
                start_y..start_y + windows[0].window_size,
                start_x..start_x + windows[1].window_size,
            ];
            let current = input_gradient.slice(target.clone()).to_concrete();
            let updated = (current + patch).to_concrete();
            input_gradient = input_gradient.slice_assign(target, &updated).to_concrete();
        }
    }
    input_gradient
}

fn reduce_broadcast_gradient<const IN: usize, const OUT: usize>(
    gradient: RawTensor<OUT, f32>,
    input_shape: [usize; IN],
) -> Result<Box<dyn AnyTensorValue>> {
    let output_shape = gradient.shape();
    let mut aligned_input_shape = [1usize; OUT];
    for axis in 0..IN {
        aligned_input_shape[OUT - IN + axis] = input_shape[axis];
    }

    for axis in 0..OUT {
        let output_dim = output_shape[axis];
        let input_dim = aligned_input_shape[axis];
        if input_dim != 1 && input_dim != output_dim {
            return Err(Error::msg("incompatible broadcast gradient shape"));
        }
    }

    let mut reduced = RawTensor::zeros(&gradient.device(), input_shape);
    for_each_index(output_shape, |output_index| {
        let mut input_index = [0usize; IN];
        for axis in 0..IN {
            let output_axis = OUT - IN + axis;
            input_index[axis] = if input_shape[axis] == 1 {
                0
            } else {
                output_index[output_axis]
            };
        }

        let output_slices: [Range<usize>; OUT] =
            std::array::from_fn(|axis| output_index[axis]..output_index[axis] + 1);
        let input_slices: [Range<usize>; IN] =
            std::array::from_fn(|axis| input_index[axis]..input_index[axis] + 1);
        let patch = gradient.slice(output_slices).reshape([1]).to_concrete();
        let current = reduced.slice(input_slices.clone()).reshape([1]).to_concrete();
        let updated = (current + patch).reshape([1usize; IN]).to_concrete();
        reduced = reduced.slice_assign(input_slices, &updated).to_concrete();
    });

    Ok(Box::new(reduced))
}

fn reduce_same_rank_broadcast_1(
    mut gradient: RawTensor<1, f32>,
    input_shape: [usize; 1],
) -> RawTensor<1, f32> {
    if input_shape[0] == 1 && gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<0>(0).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_same_rank_broadcast_2(
    mut gradient: RawTensor<2, f32>,
    input_shape: [usize; 2],
) -> RawTensor<2, f32> {
    let grad_shape = gradient.shape();
    if input_shape[0] == 1 && grad_shape[0] != 1 {
        gradient = gradient.sum_keepdim::<1>(0).to_concrete();
    }
    if input_shape[1] == 1 && grad_shape[1] != 1 {
        gradient = gradient.sum_keepdim::<1>(1).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_same_rank_broadcast_3(
    mut gradient: RawTensor<3, f32>,
    input_shape: [usize; 3],
) -> RawTensor<3, f32> {
    let grad_shape = gradient.shape();
    if input_shape[0] == 1 && grad_shape[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0).to_concrete();
    }
    if input_shape[1] == 1 && grad_shape[1] != 1 {
        gradient = gradient.sum_keepdim::<2>(1).to_concrete();
    }
    if input_shape[2] == 1 && grad_shape[2] != 1 {
        gradient = gradient.sum_keepdim::<2>(2).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_same_rank_broadcast_4(
    mut gradient: RawTensor<4, f32>,
    input_shape: [usize; 4],
) -> RawTensor<4, f32> {
    let grad_shape = gradient.shape();
    if input_shape[0] == 1 && grad_shape[0] != 1 {
        gradient = gradient.sum_keepdim::<3>(0).to_concrete();
    }
    if input_shape[1] == 1 && grad_shape[1] != 1 {
        gradient = gradient.sum_keepdim::<3>(1).to_concrete();
    }
    if input_shape[2] == 1 && grad_shape[2] != 1 {
        gradient = gradient.sum_keepdim::<3>(2).to_concrete();
    }
    if input_shape[3] == 1 && grad_shape[3] != 1 {
        gradient = gradient.sum_keepdim::<3>(3).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_to_1_from_2(mut gradient: RawTensor<2, f32>, target: usize) -> RawTensor<1, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<1>(0);
    }
    if gradient.shape()[1] != target {
        gradient = gradient.sum_keepdim::<1>(1);
    }
    gradient.reshape([target]).to_concrete()
}

fn reduce_to_1_from_3(mut gradient: RawTensor<3, f32>, target: usize) -> RawTensor<1, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0);
    }
    if gradient.shape()[1] != 1 {
        gradient = gradient.sum_keepdim::<2>(1);
    }
    if gradient.shape()[2] != target {
        gradient = gradient.sum_keepdim::<2>(2);
    }
    gradient.reshape([target]).to_concrete()
}

fn reduce_to_2_from_3(mut gradient: RawTensor<3, f32>, target: [usize; 2]) -> RawTensor<2, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0);
    }
    if gradient.shape()[1] != target[0] {
        gradient = gradient.sum_keepdim::<2>(1);
    }
    if gradient.shape()[2] != target[1] {
        gradient = gradient.sum_keepdim::<2>(2);
    }
    gradient.reshape(target).to_concrete()
}

fn reduction_extrema_keepdim_grad<const R: usize, const OUT_RANK: usize>(
    input: RawTensor<R, f32>,
    axis: usize,
    gradient: RawTensor<R, f32>,
    is_max: bool,
) -> RawTensor<R, f32>
where
    crate::ConcreteTensor<f32, R>: fusor_cpu::LastRank<OUT_RANK, f32>,
    fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
    fusor_cpu::EqOp: fusor_cpu::SimdBinaryOp<f32>,
    fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<f32>,
{
    let input_shape = input.shape();
    let extrema = if is_max {
        input.max_keepdim::<OUT_RANK>(axis)
    } else {
        input.min_keepdim::<OUT_RANK>(axis)
    }
    .to_concrete();
    let extrema_broadcast = extrema.broadcast_as(input_shape).to_concrete();
    let mask = (input - extrema_broadcast)
        .to_concrete()
        .eq(0.0)
        .to_concrete();
    let tie_count = mask
        .sum_keepdim::<OUT_RANK>(axis)
        .broadcast_as(input_shape)
        .to_concrete();
    ((mask * gradient.broadcast_as(input_shape)).to_concrete() / tie_count).to_concrete()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToVec1, ToVec2};

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-3, "expected {right}, got {left}");
    }

    fn assert_slice_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len(), "slice lengths differ");
        for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (*left - *right).abs() < 1e-3,
                "mismatch at index {index}: expected {right}, got {left}",
            );
        }
    }

    async fn flatten<const R: usize>(tensor: RawTensor<R, f32>) -> Vec<f32> {
        let elements = tensor.shape().into_iter().product();
        tensor
            .reshape([elements])
            .as_slice()
            .await
            .unwrap()
            .to_vec1()
    }

    #[tokio::test]
    async fn test_backward_squared_sum_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let loss = x.sqr().sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(dx[0], 2.0);
        assert_close(dx[1], 4.0);
        assert_close(dx[2], 6.0);
    }

    #[tokio::test]
    async fn test_autograd_silu_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

        let values = x.silu().raw().clone().as_slice().await.unwrap().to_vec1();

        let expected = [1.0f32, -2.0, 0.5].map(|v| v / (1.0 + (-v).exp()));
        for (value, expected) in values.iter().zip(expected) {
            assert_close(*value, expected);
        }
    }

    #[tokio::test]
    async fn test_autograd_gelu_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

        let values = x.gelu().raw().clone().as_slice().await.unwrap().to_vec1();

        let expected = [1.0f32, -2.0, 0.5].map(|v| {
            0.5 * v
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3))).tanh())
        });
        for (value, expected) in values.iter().zip(expected) {
            assert_close(*value, expected);
        }
    }

    #[tokio::test]
    async fn test_backward_where_cond_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let condition: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, -2.0]);
        let on_true: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0, 4.0]);
        let on_false: Tensor<1> = Tensor::new(&graph, &device, &[10.0f32, 20.0, 30.0]);

        let output = condition.where_cond(&on_true, &on_false);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.flatten_all().sum().backward().unwrap();

        let dcondition = gradients
            .get(&condition)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let dtrue = gradients
            .get(&on_true)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let dfalse = gradients
            .get(&on_false)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![2.0, 20.0, 4.0]);
        assert_eq!(dcondition, vec![0.0, 0.0, 0.0]);
        assert_eq!(dtrue, vec![1.0, 0.0, 1.0]);
        assert_eq!(dfalse, vec![0.0, 1.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_index_select_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = RawTensor::from_slice(&device, [3], &[2u32, 0, 2]);

        let output = input.index_select(1, &indices);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![3.0, 1.0, 3.0], vec![6.0, 4.0, 6.0]]);
        assert_eq!(dinput, vec![vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 2.0]]);
    }

    #[tokio::test]
    async fn test_backward_slice_assign_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );
        let value: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32, 11.0], [12.0, 13.0]]);

        let output = input.slice_assign([0..2, 1..3], &value);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();

        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();
        let dvalue = gradients
            .get(&value)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(
            output_values,
            vec![vec![1.0, 10.0, 11.0], vec![4.0, 12.0, 13.0], vec![7.0, 8.0, 9.0]]
        );
        assert_eq!(
            dinput,
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]]
        );
        assert_eq!(dvalue, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    }

    #[tokio::test]
    async fn test_backward_expand_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0]]);

        let output = input.expand([2, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0]]);
        assert_eq!(dinput, vec![vec![2.0, 2.0, 2.0]]);
    }

    #[tokio::test]
    async fn test_backward_flatten_all_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.flatten_all();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dinput, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    }

    #[tokio::test]
    async fn test_backward_flatten_last_n_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<3> = Tensor::new(
            &graph,
            &device,
            &[
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
        );

        let output = input.flatten_last_n::<1, 2>();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([2, 6])
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(
            output_values,
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        );
        assert_eq!(
            dinput,
            vec![vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        );
    }

    #[tokio::test]
    async fn test_backward_flatten_first_n_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<3> = Tensor::new(
            &graph,
            &device,
            &[
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
        );

        let output = input.flatten_first_n::<1, 2>();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([4, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(
            output_values,
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
                vec![10.0, 11.0, 12.0]
            ]
        );
        assert_eq!(
            dinput,
            vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0]
            ]
        );
    }

    #[tokio::test]
    async fn test_backward_narrow_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );

        let output = input.narrow(1usize, 1, 2);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![2.0, 3.0], vec![5.0, 6.0], vec![8.0, 9.0]]);
        assert_eq!(
            dinput,
            vec![vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 1.0]]
        );
    }

    #[tokio::test]
    async fn test_backward_repeat_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.repeat([2, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(
            output_values,
            vec![
                vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
                vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
            ]
        );
        assert_eq!(dinput, vec![vec![6.0, 6.0], vec![6.0, 6.0]]);
    }

    #[tokio::test]
    async fn test_backward_resize_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );

        let output = input.resize([2, 2]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
        assert_eq!(
            dinput,
            vec![vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]]
        );
    }

    #[tokio::test]
    async fn test_backward_restride_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0]);

        let output = input.restride([StrideSpec::dim(0, 2), StrideSpec::dim(0, 3)]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        assert_eq!(dinput, vec![1.0, 2.0, 2.0, 1.0]);
    }

    #[tokio::test]
    async fn test_backward_restride_layout_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let layout = Layout::contiguous(&[5]).restride(&[
            StrideSpec::dim(0, 2).with_offset(1),
            StrideSpec::dim(0, 2),
        ]);

        let output = input.restride_layout(layout);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![vec![2.0, 3.0], vec![3.0, 4.0]]);
        assert_eq!(dinput, vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_squeeze_dims_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]],
        );

        let output = input.squeeze_dims::<2, 2>([1, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([2, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    }

    #[tokio::test]
    async fn test_backward_unsqueeze_dims_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.unsqueeze_dims::<2, 4>([0, 2]);
        let output_values = output
            .raw()
            .clone()
            .reshape([2, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec2();
        let gradients = output.sum(3).sum(2).sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output.shape(), [1, 2, 1, 3]);
        assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    }

    #[tokio::test]
    async fn test_backward_max_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 5.0, 5.0], [4.0, 2.0, 0.0]]);

        let output = input.max::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![5.0, 4.0]);
        assert_eq!(dinput, vec![vec![0.0, 0.5, 0.5], vec![1.0, 0.0, 0.0]]);
    }

    #[tokio::test]
    async fn test_backward_min_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0, 5.0], [4.0, 2.0, 0.0]]);

        let output = input.min::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![1.0, 0.0]);
        assert_eq!(dinput, vec![vec![0.5, 0.5, 0.0], vec![0.0, 0.0, 1.0]]);
    }

    #[tokio::test]
    async fn test_backward_mean_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.mean::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![2.0, 5.0]);
        assert_eq!(
            dinput,
            vec![
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            ]
        );
    }

    #[tokio::test]
    async fn test_backward_product_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> =
            Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0], [0.0, 0.0, 9.0]]);

        let output = input.product::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![24.0, 0.0, 0.0]);
        assert_eq!(
            dinput,
            vec![vec![12.0, 8.0, 6.0], vec![0.0, 35.0, 0.0], vec![0.0, 0.0, 0.0]]
        );
    }

    #[tokio::test]
    async fn test_backward_product_keepdim_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0]]);

        let output = input.product_keepdim::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![24.0], vec![0.0]]);
        assert_eq!(dinput, vec![vec![12.0, 8.0, 6.0], vec![0.0, 35.0, 0.0]]);
    }

    #[tokio::test]
    async fn test_backward_var_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.var::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![2.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(
            dinput,
            vec![
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0],
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            ]
        );
    }

    #[tokio::test]
    async fn test_backward_var_keepdim_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.var_keepdim::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_eq!(output_values, vec![vec![2.0 / 3.0], vec![2.0 / 3.0]]);
        assert_eq!(
            dinput,
            vec![
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0],
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            ]
        );
    }

    #[tokio::test]
    async fn test_backward_clamp_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0, 5.0]);

        let output = input.clamp(0.0, 3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 0.0, 2.0, 3.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_eq_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 1.0]);

        let output = input.eq(1.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_eq_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 2.0, 3.0]);

        let output = input.eq_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_eq_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

        let output = lhs.eq_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_gt_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.gt_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_gt_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

        let output = lhs.gt_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_gte_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.gte_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_gte_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0, 2.0]);

        let output = lhs.gte_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lt_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lt(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lt_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lt_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lt_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

        let output = lhs.lt_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lte_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lte(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lte_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lte_scalar(1.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_lte_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 2.0, 1.0]);

        let output = lhs.lte_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_max_elementwise_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0]);

        let output = input.max_elementwise(0.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 0.0, 2.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_backward_max_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.max_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![3.0, 4.0, 3.0]);
        assert_eq!(dinput, vec![0.0, 1.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_min_elementwise_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.min_elementwise(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 3.0, 2.0]);
        assert_eq!(dinput, vec![1.0, 0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_backward_min_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.min_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 2.0, 2.0]);
        assert_eq!(dinput, vec![1.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_mt_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.mt(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_mte_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.mte(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_ne_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.ne(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_ne_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.ne_scalar(4.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_ne_tensor_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

        let output = lhs.ne_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_backward_abs_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-2.0f32, 0.0, 3.0]);

        let output = input.abs();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_eq!(output_values, vec![2.0, 0.0, 3.0]);
        assert_eq!(dinput, vec![-1.0, 0.0, 1.0]);
    }

    #[tokio::test]
    async fn test_backward_acos_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.acos();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.acos());
        assert_close(dinput[0], -1.0f32 / (1.0f32 - 0.25f32).sqrt());
    }

    #[tokio::test]
    async fn test_backward_acosh_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let output = input.acosh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 2.0f32.acosh());
        assert_close(dinput[0], 1.0f32 / ((2.0f32 - 1.0f32).sqrt() * (2.0f32 + 1.0f32).sqrt()));
    }

    #[tokio::test]
    async fn test_backward_approximate_exp_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

        let output = input.approximate_exp();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 1.0f32.exp());
        assert_close(dinput[0], 1.0f32.exp());
    }

    #[tokio::test]
    async fn test_backward_asin_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.asin();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.asin());
        assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32).sqrt());
    }

    #[tokio::test]
    async fn test_backward_asinh_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.5f32]);

        let output = input.asinh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 1.5f32.asinh());
        assert_close(dinput[0], 1.0f32 / (1.5f32 * 1.5f32 + 1.0f32).sqrt());
    }

    #[tokio::test]
    async fn test_backward_atan_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.atan();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.atan());
        assert_close(dinput[0], 1.0f32 / (1.0f32 + 0.25f32));
    }

    #[tokio::test]
    async fn test_backward_atanh_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.atanh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.atanh());
        assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32));
    }

    #[tokio::test]
    async fn test_backward_cos_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.cos();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.cos());
        assert_close(dinput[0], -0.5f32.sin());
    }

    #[tokio::test]
    async fn test_backward_cosh_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.cosh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.cosh());
        assert_close(dinput[0], 0.5f32.sinh());
    }

    #[tokio::test]
    async fn test_backward_exp2_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let output = input.exp2();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 2.0f32.exp2());
        assert_close(dinput[0], std::f32::consts::LN_2 * 2.0f32.exp2());
    }

    #[tokio::test]
    async fn test_backward_less_approximate_exp_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

        let output = input.less_approximate_exp();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 1.0f32.exp());
        assert_close(dinput[0], 1.0f32.exp());
    }

    #[tokio::test]
    async fn test_backward_log2_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

        let output = input.log2();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 4.0f32.log2());
        assert_close(dinput[0], 1.0f32 / (4.0f32 * std::f32::consts::LN_2));
    }

    #[tokio::test]
    async fn test_backward_sin_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.sin();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.sin());
        assert_close(dinput[0], 0.5f32.cos());
    }

    #[tokio::test]
    async fn test_backward_sinh_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.sinh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.sinh());
        assert_close(dinput[0], 0.5f32.cosh());
    }

    #[tokio::test]
    async fn test_backward_tan_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.tan();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.tan());
        assert_close(dinput[0], 1.0f32 / (0.5f32.cos() * 0.5f32.cos()));
    }

    #[tokio::test]
    async fn test_backward_tanh_exact_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.tanh_exact();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 0.5f32.tanh());
        assert_close(dinput[0], 1.0f32 - 0.5f32.tanh().powi(2));
    }

    #[tokio::test]
    async fn test_cast_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.cast::<half::f16>();
        let output_values = output.as_slice().await.unwrap().to_vec1();

        assert_close(f32::from(output_values[0]), 1.0);
        assert_close(f32::from(output_values[1]), 2.0);
        assert_close(f32::from(output_values[2]), 3.0);
    }

    #[tokio::test]
    async fn test_arange_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let output = Tensor::<1>::arange(&graph, &device, 1.0, 5.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();

        assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_arange_step_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let output = Tensor::<1>::arange_step(&graph, &device, 1.0, 6.0, 2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();

        assert_eq!(output_values, vec![1.0, 3.0, 5.0]);
    }

    #[tokio::test]
    async fn test_full_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let output: Tensor<2> = Tensor::full(&graph, &device, [2, 3], 1.5);
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 3]);
        for row in 0..2 {
            for col in 0..3 {
                assert_close(output_values[[row, col]], 1.5);
            }
        }
    }

    #[tokio::test]
    async fn test_zeros_like_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.zeros_like();
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 0.0);
        assert_close(output_values[[0, 1]], 0.0);
        assert_close(output_values[[1, 0]], 0.0);
        assert_close(output_values[[1, 1]], 0.0);
    }

    #[tokio::test]
    async fn test_from_array_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let output: Tensor<2> = Tensor::from_array(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 1.0);
        assert_close(output_values[[0, 1]], 2.0);
        assert_close(output_values[[1, 0]], 3.0);
        assert_close(output_values[[1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_backward_add_broadcast_api_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

        let output: Tensor<2> = lhs.add_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

        assert_close(output_values[0][0], 11.0);
        assert_close(output_values[0][1], 12.0);
        assert_close(output_values[1][0], 21.0);
        assert_close(output_values[1][1], 22.0);
        assert_close(dlhs[0], 2.0);
        assert_close(dlhs[1], 2.0);
        assert_close(drhs[0][0], 2.0);
        assert_close(drhs[1][0], 2.0);
    }

    #[tokio::test]
    async fn test_backward_sub_broadcast_api_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[3.0f32], [4.0]]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);

        let output: Tensor<2> = lhs.sub_::<1, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec2();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec1();

        assert_close(output_values[0][0], 2.0);
        assert_close(output_values[0][1], 1.0);
        assert_close(output_values[1][0], 3.0);
        assert_close(output_values[1][1], 2.0);
        assert_close(dlhs[0][0], 2.0);
        assert_close(dlhs[1][0], 2.0);
        assert_close(drhs[0], -2.0);
        assert_close(drhs[1], -2.0);
    }

    #[tokio::test]
    async fn test_backward_mul_broadcast_api_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

        let output: Tensor<2> = lhs.mul_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

        assert_close(output_values[0][0], 20.0);
        assert_close(output_values[0][1], 30.0);
        assert_close(output_values[1][0], 40.0);
        assert_close(output_values[1][1], 60.0);
        assert_close(dlhs[0], 30.0);
        assert_close(dlhs[1], 30.0);
        assert_close(drhs[0][0], 5.0);
        assert_close(drhs[1][0], 5.0);
    }

    #[tokio::test]
    async fn test_backward_div_broadcast_api_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0]);

        let output: Tensor<2> = lhs.div_::<1, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec2();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec1();

        assert_close(output_values[0][0], 5.0);
        assert_close(output_values[0][1], 2.5);
        assert_close(output_values[1][0], 10.0);
        assert_close(output_values[1][1], 5.0);
        assert_close(dlhs[0][0], 0.75);
        assert_close(dlhs[1][0], 0.75);
        assert_close(drhs[0], -7.5);
        assert_close(drhs[1], -1.875);
    }

    #[tokio::test]
    async fn test_backward_pow_broadcast_api_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32], [1.0]]);

        let output: Tensor<2> = lhs.pow_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

        assert_close(output_values[0][0], 4.0);
        assert_close(output_values[0][1], 9.0);
        assert_close(output_values[1][0], 2.0);
        assert_close(output_values[1][1], 3.0);
        assert_close(dlhs[0], 5.0);
        assert_close(dlhs[1], 7.0);
        assert_close(drhs[0][0], 12.660099);
        assert_close(drhs[1][0], 4.6821313);
    }

    #[tokio::test]
    async fn test_backward_chunk_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        );

        let chunks = input.chunk(2, 1);
        assert_eq!(chunks.len(), 2);
        let first = chunks[0].raw().clone().as_slice().await.unwrap().to_vec2();
        let second = chunks[1].raw().clone().as_slice().await.unwrap().to_vec2();
        let loss = chunks[0]
            .flatten_all()
            .sum()
            .add(&chunks[1].flatten_all().sum().mul_scalar(2.0));
        let gradients = loss.backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec2();

        assert_close(first[0][0], 1.0);
        assert_close(first[0][1], 2.0);
        assert_close(first[1][0], 5.0);
        assert_close(first[1][1], 6.0);
        assert_close(second[0][0], 3.0);
        assert_close(second[0][1], 4.0);
        assert_close(second[1][0], 7.0);
        assert_close(second[1][1], 8.0);
        assert_close(dinput[0][0], 1.0);
        assert_close(dinput[0][1], 1.0);
        assert_close(dinput[0][2], 2.0);
        assert_close(dinput[0][3], 2.0);
        assert_close(dinput[1][0], 1.0);
        assert_close(dinput[1][1], 1.0);
        assert_close(dinput[1][2], 2.0);
        assert_close(dinput[1][3], 2.0);
    }

    #[tokio::test]
    async fn test_backward_softmax_slow_matches_softmax_cpu() {
        let device = Device::cpu();

        let slow_graph = Graph::new();
        let slow_input: Tensor<2> =
            Tensor::new(&slow_graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let slow_weights: Tensor<2> =
            Tensor::new(&slow_graph, &device, &[[0.5f32, 1.5], [2.5, 3.5]]);
        let slow_output = slow_input.softmax_slow::<1>(1);
        let slow_values = slow_output.raw().clone().as_slice().await.unwrap().to_vec2();
        let slow_loss = slow_output.mul(&slow_weights).flatten_all().sum();
        let slow_gradients = slow_loss.backward().unwrap();
        let slow_dinput = slow_gradients
            .get(&slow_input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        let regular_graph = Graph::new();
        let regular_input: Tensor<2> =
            Tensor::new(&regular_graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let regular_weights: Tensor<2> =
            Tensor::new(&regular_graph, &device, &[[0.5f32, 1.5], [2.5, 3.5]]);
        let regular_output = regular_input.softmax::<1>(1);
        let regular_values = regular_output.raw().clone().as_slice().await.unwrap().to_vec2();
        let regular_loss = regular_output.mul(&regular_weights).flatten_all().sum();
        let regular_gradients = regular_loss.backward().unwrap();
        let regular_dinput = regular_gradients
            .get(&regular_input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_close(slow_values[0][0], regular_values[0][0]);
        assert_close(slow_values[0][1], regular_values[0][1]);
        assert_close(slow_values[1][0], regular_values[1][0]);
        assert_close(slow_values[1][1], regular_values[1][1]);
        assert_close(slow_dinput[0][0], regular_dinput[0][0]);
        assert_close(slow_dinput[0][1], regular_dinput[0][1]);
        assert_close(slow_dinput[1][0], regular_dinput[1][0]);
        assert_close(slow_dinput[1][1], regular_dinput[1][1]);
    }

    #[tokio::test]
    async fn test_backward_softmax_slow_last_dim_matches_softmax_last_dim_cpu() {
        let device = Device::cpu();

        let slow_graph = Graph::new();
        let slow_input: Tensor<3> = Tensor::new(
            &slow_graph,
            &device,
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        );
        let slow_weights: Tensor<3> = Tensor::new(
            &slow_graph,
            &device,
            &[[[0.5f32, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        );
        let slow_output = slow_input.softmax_slow_last_dim::<2>();
        let slow_values = slow_output.raw().clone().as_slice().await.unwrap();
        let slow_loss = slow_output.mul(&slow_weights).flatten_all().sum();
        let slow_gradients = slow_loss.backward().unwrap();
        let slow_dinput = slow_gradients.get(&slow_input).unwrap().as_slice().await.unwrap();

        let regular_graph = Graph::new();
        let regular_input: Tensor<3> = Tensor::new(
            &regular_graph,
            &device,
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        );
        let regular_weights: Tensor<3> = Tensor::new(
            &regular_graph,
            &device,
            &[[[0.5f32, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        );
        let regular_output = regular_input.softmax_last_dim::<2>();
        let regular_values = regular_output.raw().clone().as_slice().await.unwrap();
        let regular_loss = regular_output.mul(&regular_weights).flatten_all().sum();
        let regular_gradients = regular_loss.backward().unwrap();
        let regular_dinput = regular_gradients
            .get(&regular_input)
            .unwrap()
            .as_slice()
            .await
            .unwrap();

        for batch in 0..2 {
            for row in 0..2 {
                for col in 0..2 {
                    assert_close(slow_values[[batch, row, col]], regular_values[[batch, row, col]]);
                    assert_close(
                        slow_dinput[[batch, row, col]],
                        regular_dinput[[batch, row, col]],
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_backward_matmul_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[5.0f32, 6.0], [7.0, 8.0]]);

        let output = lhs.matmul(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap();
        let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 19.0);
        assert_close(output_values[[0, 1]], 22.0);
        assert_close(output_values[[1, 0]], 43.0);
        assert_close(output_values[[1, 1]], 50.0);

        assert_close(dlhs[[0, 0]], 11.0);
        assert_close(dlhs[[0, 1]], 15.0);
        assert_close(dlhs[[1, 0]], 11.0);
        assert_close(dlhs[[1, 1]], 15.0);

        assert_close(drhs[[0, 0]], 4.0);
        assert_close(drhs[[0, 1]], 4.0);
        assert_close(drhs[[1, 0]], 6.0);
        assert_close(drhs[[1, 1]], 6.0);
    }

    #[tokio::test]
    async fn test_backward_t_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.t();
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 1.0);
        assert_close(output_values[[0, 1]], 3.0);
        assert_close(output_values[[1, 0]], 2.0);
        assert_close(output_values[[1, 1]], 4.0);

        assert_close(dinput[[0, 0]], 1.0);
        assert_close(dinput[[0, 1]], 1.0);
        assert_close(dinput[[1, 0]], 1.0);
        assert_close(dinput[[1, 1]], 1.0);
    }

    #[tokio::test]
    async fn test_backward_pool_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 2.0, 3.0, 4.0]]]);

        let output = input.pool::<1, 4, 5, 4>([(2, 1)], |windowed, axis| windowed.mean::<3>(axis));
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 3]);
        assert_close(output_values[[0, 0, 0]], 1.5);
        assert_close(output_values[[0, 0, 1]], 2.5);
        assert_close(output_values[[0, 0, 2]], 3.5);

        assert_close(dinput[[0, 0, 0]], 0.5);
        assert_close(dinput[[0, 0, 1]], 1.0);
        assert_close(dinput[[0, 0, 2]], 1.0);
        assert_close(dinput[[0, 0, 3]], 0.5);
    }

    #[tokio::test]
    async fn test_backward_pool_max_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 4.0, 2.0, 3.0]]]);

        let output = input.pool_max::<1, 4, 5, 4>([(2, 1)]);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 3]);
        assert_close(output_values[[0, 0, 0]], 4.0);
        assert_close(output_values[[0, 0, 1]], 4.0);
        assert_close(output_values[[0, 0, 2]], 3.0);

        assert_close(dinput[[0, 0, 0]], 0.0);
        assert_close(dinput[[0, 0, 1]], 2.0);
        assert_close(dinput[[0, 0, 2]], 0.0);
        assert_close(dinput[[0, 0, 3]], 1.0);
    }

    #[tokio::test]
    async fn test_backward_pool_min_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 4.0, 2.0, 3.0]]]);

        let output = input.pool_min::<1, 4, 5, 4>([(2, 1)]);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 3]);
        assert_close(output_values[[0, 0, 0]], 1.0);
        assert_close(output_values[[0, 0, 1]], 2.0);
        assert_close(output_values[[0, 0, 2]], 2.0);

        assert_close(dinput[[0, 0, 0]], 1.0);
        assert_close(dinput[[0, 0, 1]], 0.0);
        assert_close(dinput[[0, 0, 2]], 2.0);
        assert_close(dinput[[0, 0, 3]], 0.0);
    }

    #[tokio::test]
    async fn test_backward_q_mat_mul_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0, 1.0, 1.0]]);
        let weight_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let weights = crate::QMatrix::from_raw_bytes(
            &device,
            [2, 4],
            &weight_bytes,
            fusor_gguf::GgmlType::F32,
        )
        .unwrap();

        let output = input.q_mat_mul(&weights);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 2]);
        assert_close(output_values[[0, 0]], 10.0);
        assert_close(output_values[[0, 1]], 26.0);

        assert_close(dinput[[0, 0]], 6.0);
        assert_close(dinput[[0, 1]], 8.0);
        assert_close(dinput[[0, 2]], 10.0);
        assert_close(dinput[[0, 3]], 12.0);
    }

    #[tokio::test]
    async fn test_backward_stack_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let first: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
        let second: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 4.0]);

        let output = Tensor::stack::<2>(vec![first.clone(), second.clone()], 0);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dfirst = gradients.get(&first).unwrap().as_slice().await.unwrap().to_vec1();
        let dsecond = gradients.get(&second).unwrap().as_slice().await.unwrap().to_vec1();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 1.0);
        assert_close(output_values[[0, 1]], 2.0);
        assert_close(output_values[[1, 0]], 3.0);
        assert_close(output_values[[1, 1]], 4.0);

        assert_close(dfirst[0], 1.0);
        assert_close(dfirst[1], 1.0);
        assert_close(dsecond[0], 1.0);
        assert_close(dsecond[1], 1.0);
    }

    #[tokio::test]
    async fn test_backward_rope_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<4> =
            Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
        let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
        let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

        let output = input.rope(&cos, &sin);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
        let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
        let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
        assert_close(output_values[[0, 0, 0, 0]], 1.0);
        assert_close(output_values[[0, 0, 0, 1]], 2.0);
        assert_close(output_values[[0, 0, 0, 2]], 3.0);
        assert_close(output_values[[0, 0, 0, 3]], 4.0);
        assert_close(output_values[[0, 0, 1, 0]], 5.0);
        assert_close(output_values[[0, 0, 1, 1]], 6.0);
        assert_close(output_values[[0, 0, 1, 2]], 7.0);
        assert_close(output_values[[0, 0, 1, 3]], 8.0);

        for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
            assert_close(dinput[index], 1.0);
        }

        assert_close(dcos[[0, 0]], 4.0);
        assert_close(dcos[[0, 1]], 6.0);
        assert_close(dcos[[1, 0]], 12.0);
        assert_close(dcos[[1, 1]], 14.0);

        assert_close(dsin[[0, 0]], -2.0);
        assert_close(dsin[[0, 1]], -2.0);
        assert_close(dsin[[1, 0]], -2.0);
        assert_close(dsin[[1, 1]], -2.0);
    }

    #[tokio::test]
    async fn test_backward_rope_fused_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<4> =
            Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
        let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
        let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

        let output = input.rope_fused(&cos, &sin);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
        let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
        let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
        assert_close(output_values[[0, 0, 0, 0]], 1.0);
        assert_close(output_values[[0, 0, 0, 1]], 2.0);
        assert_close(output_values[[0, 0, 0, 2]], 3.0);
        assert_close(output_values[[0, 0, 0, 3]], 4.0);
        assert_close(output_values[[0, 0, 1, 0]], 5.0);
        assert_close(output_values[[0, 0, 1, 1]], 6.0);
        assert_close(output_values[[0, 0, 1, 2]], 7.0);
        assert_close(output_values[[0, 0, 1, 3]], 8.0);

        for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
            assert_close(dinput[index], 1.0);
        }

        assert_close(dcos[[0, 0]], 3.0);
        assert_close(dcos[[0, 1]], 7.0);
        assert_close(dcos[[1, 0]], 11.0);
        assert_close(dcos[[1, 1]], 15.0);

        assert_close(dsin[[0, 0]], -1.0);
        assert_close(dsin[[0, 1]], -1.0);
        assert_close(dsin[[1, 0]], -1.0);
        assert_close(dsin[[1, 1]], -1.0);
    }

    #[tokio::test]
    async fn test_backward_rope_interleaved_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<4> =
            Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
        let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
        let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

        let output = input.rope_interleaved(&cos, &sin);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
        let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
        let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
        assert_close(output_values[[0, 0, 0, 0]], 1.0);
        assert_close(output_values[[0, 0, 0, 1]], 2.0);
        assert_close(output_values[[0, 0, 0, 2]], 3.0);
        assert_close(output_values[[0, 0, 0, 3]], 4.0);
        assert_close(output_values[[0, 0, 1, 0]], 5.0);
        assert_close(output_values[[0, 0, 1, 1]], 6.0);
        assert_close(output_values[[0, 0, 1, 2]], 7.0);
        assert_close(output_values[[0, 0, 1, 3]], 8.0);

        for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
            assert_close(dinput[index], 1.0);
        }

        assert_close(dcos[[0, 0]], 3.0);
        assert_close(dcos[[0, 1]], 7.0);
        assert_close(dcos[[1, 0]], 11.0);
        assert_close(dcos[[1, 1]], 15.0);

        assert_close(dsin[[0, 0]], -1.0);
        assert_close(dsin[[0, 1]], -1.0);
        assert_close(dsin[[1, 0]], -1.0);
        assert_close(dsin[[1, 1]], -1.0);
    }

    #[tokio::test]
    async fn test_backward_rope_normal_fused_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<4> =
            Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
        let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
        let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

        let output = input.rope_normal_fused(&cos, &sin);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
        let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
        let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
        assert_close(output_values[[0, 0, 0, 0]], 1.0);
        assert_close(output_values[[0, 0, 0, 1]], 2.0);
        assert_close(output_values[[0, 0, 0, 2]], 3.0);
        assert_close(output_values[[0, 0, 0, 3]], 4.0);
        assert_close(output_values[[0, 0, 1, 0]], 5.0);
        assert_close(output_values[[0, 0, 1, 1]], 6.0);
        assert_close(output_values[[0, 0, 1, 2]], 7.0);
        assert_close(output_values[[0, 0, 1, 3]], 8.0);

        for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
            assert_close(dinput[index], 1.0);
        }

        assert_close(dcos[[0, 0]], 4.0);
        assert_close(dcos[[0, 1]], 6.0);
        assert_close(dcos[[1, 0]], 12.0);
        assert_close(dcos[[1, 1]], 14.0);

        assert_close(dsin[[0, 0]], -2.0);
        assert_close(dsin[[0, 1]], -2.0);
        assert_close(dsin[[1, 0]], -2.0);
        assert_close(dsin[[1, 1]], -2.0);
    }

    #[tokio::test]
    async fn test_backward_pow_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

        let output = lhs.pow(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 8.0);
        assert_close(dlhs[0], 12.0);
        assert_close(drhs[0], 8.0 * 2.0f32.ln());
    }

    #[tokio::test]
    async fn test_backward_pow_elementwise_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

        let output = input.pow_elementwise(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 9.0);
        assert_close(dinput[0], 6.0);
    }

    #[tokio::test]
    async fn test_backward_pow_scalar_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

        let output = input.pow_scalar(0.5);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(output_values[0], 2.0);
        assert_close(dinput[0], 0.25);
    }

    #[tokio::test]
    async fn test_autograd_rms_norm_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let weight: Tensor<1> = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [3], &[1.0f32, 1.0, 1.0]),
        );

        let output = input
            .rms_norm(&weight, 1e-5)
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        let expected = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].map(|row| {
            let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
            let scale = 1.0 / (mean_sq + 1e-5).sqrt();
            row.map(|value| value * scale)
        });

        for (actual_row, expected_row) in output.iter().zip(expected.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_close(*actual, *expected);
            }
        }
    }

    #[tokio::test]
    async fn test_backward_matmul_with_broadcast_bias_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32], [1.0], [1.5]]);
        let b: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let y = x.mat_mul(&w).add(&b.broadcast_as([2, 1]));
        let loss = y.sum(1).sum();

        let gradients = loss.backward().unwrap();
        let dw = gradients
            .get(&w)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();
        let db = gradients
            .get(&b)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(dw[0][0], 5.0);
        assert_close(dw[1][0], 7.0);
        assert_close(dw[2][0], 9.0);
        assert_close(db[0], 2.0);
    }

    #[tokio::test]
    async fn test_backward_embedding_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let table: Tensor<2> =
            Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let indices: RawTensor<2, u32> = RawTensor::new(&device, &[[0u32, 2u32]]);
        let embedded = table.embedding(&indices);
        let loss = embedded.sum(2).sum(1).sum();

        let gradients = loss.backward().unwrap();
        let dtable = gradients
            .get(&table)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_close(dtable[0][0], 1.0);
        assert_close(dtable[0][1], 1.0);
        assert_close(dtable[1][0], 0.0);
        assert_close(dtable[1][1], 0.0);
        assert_close(dtable[2][0], 1.0);
        assert_close(dtable[2][1], 1.0);
    }

    #[tokio::test]
    async fn test_backward_gather_last_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let values: Tensor<2> =
            Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices: RawTensor<1, u32> = RawTensor::new(&device, &[2u32, 0u32]);
        let gathered = values.gather_last(&indices);
        let loss = gathered.sum();

        let gradients = loss.backward().unwrap();
        let dvalues = gradients
            .get(&values)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_close(dvalues[0][0], 0.0);
        assert_close(dvalues[0][1], 0.0);
        assert_close(dvalues[0][2], 1.0);
        assert_close(dvalues[1][0], 1.0);
        assert_close(dvalues[1][1], 0.0);
        assert_close(dvalues[1][2], 0.0);
    }

    #[tokio::test]
    async fn test_backward_conv_1d_weights_and_bias_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let input = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [1, 1, 4], &[1.0f32, 2.0, 3.0, 4.0]),
        );
        let weight: Tensor<3> = Tensor::new(&graph, &device, &[[[0.5f32, -1.0]]]);
        let bias: Tensor<1> = Tensor::new(&graph, &device, &[0.25f32]);

        let loss = input
            .conv(&weight, Some(&bias), [0], [1])
            .sum(2)
            .sum(1)
            .sum();
        let gradients = loss.backward().unwrap();

        let dweight = pollster::block_on(gradients.get(&weight).unwrap().reshape([2]).as_slice())
            .unwrap()
            .to_vec1();
        let dbias = gradients
            .get(&bias)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(dweight[0], 6.0);
        assert_close(dweight[1], 9.0);
        assert_close(dbias[0], 3.0);
    }

    #[tokio::test]
    async fn test_backward_conv_1d_input_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 2.0, 3.0, 4.0]]]);
        let weight = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [1, 1, 3], &[1.0f32, 1.0, 1.0]),
        );

        let loss = input.conv(&weight, None, [1], [1]).sum(2).sum(1).sum();
        let gradients = loss.backward().unwrap();
        let dinput = pollster::block_on(gradients.get(&input).unwrap().reshape([4]).as_slice())
            .unwrap()
            .to_vec1();

        assert_close(dinput[0], 2.0);
        assert_close(dinput[1], 3.0);
        assert_close(dinput[2], 3.0);
        assert_close(dinput[3], 2.0);
    }

    #[tokio::test]
    async fn test_backward_conv_2d_weights_bias_and_input_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        );
        let weight: Tensor<4> = Tensor::new(&graph, &device, &[[[[1.0f32, 1.0], [1.0, 1.0]]]]);
        let bias: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let loss = input
            .conv(&weight, Some(&bias), [0, 0], [1, 1])
            .reshape([4])
            .sum();
        let gradients = loss.backward().unwrap();

        let dinput: RawTensor<4, f32> = gradients.get(&input).unwrap();
        let dinput = dinput.reshape([3, 3]).as_slice().await.unwrap().to_vec2();
        let dweight: RawTensor<4, f32> = gradients.get(&weight).unwrap();
        let dweight = dweight.reshape([2, 2]).as_slice().await.unwrap().to_vec2();
        let dbias: RawTensor<1, f32> = gradients.get(&bias).unwrap();
        let dbias = dbias.as_slice().await.unwrap().to_vec1();

        assert_close(dinput[0][0], 1.0);
        assert_close(dinput[0][1], 2.0);
        assert_close(dinput[0][2], 1.0);
        assert_close(dinput[1][0], 2.0);
        assert_close(dinput[1][1], 4.0);
        assert_close(dinput[1][2], 2.0);
        assert_close(dinput[2][0], 1.0);
        assert_close(dinput[2][1], 2.0);
        assert_close(dinput[2][2], 1.0);

        assert_close(dweight[0][0], 12.0);
        assert_close(dweight[0][1], 16.0);
        assert_close(dweight[1][0], 24.0);
        assert_close(dweight[1][1], 28.0);
        assert_close(dbias[0], 4.0);
    }

    #[tokio::test]
    async fn test_backward_softmax_last_dim_fused_matches_composite_cpu() {
        let device = Device::cpu();
        let input_data = &[
            [[0.2f32, -0.4, 1.1], [0.5, 0.3, -0.7]],
            [[-1.0, 0.8, 0.6], [0.9, -0.2, 0.1]],
        ];

        let fused_graph = Graph::new();
        let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
        let fused_output = fused_input.softmax_last_dim_fused::<2>();
        let fused_loss = fused_output.sqr().reshape([12]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
        let composite_output = composite_input.softmax_last_dim::<2>();
        let composite_loss = composite_output.sqr().reshape([12]).sum();
        let composite_gradients = composite_loss.backward().unwrap();

        let fused_output = flatten(fused_output.raw().clone()).await;
        let composite_output = flatten(composite_output.raw().clone()).await;
        let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
        let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;

        assert_slice_close(&fused_output, &composite_output);
        assert_slice_close(&fused_dx, &composite_dx);
    }

    #[tokio::test]
    async fn test_backward_rms_norm_fused_matches_composite_cpu() {
        let device = Device::cpu();
        let input_data = &[
            [[0.3f32, -1.2, 0.7], [1.5, 0.1, -0.8]],
            [[-0.4, 0.9, 1.3], [0.2, -0.6, 0.5]],
        ];
        let weight_data = &[1.0f32, 0.75, 1.25];
        let eps = 1e-5;

        let fused_graph = Graph::new();
        let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
        let fused_weight: Tensor<1> = Tensor::new(&fused_graph, &device, weight_data);
        let fused_output = fused_input.rms_norm_fused_no_bias::<2>(&fused_weight, eps);
        let fused_loss = fused_output.sqr().reshape([12]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
        let composite_weight: Tensor<1> = Tensor::new(&composite_graph, &device, weight_data);
        let composite_output = composite_input.rms_norm(&composite_weight, eps);
        let composite_loss = composite_output.sqr().reshape([12]).sum();
        let composite_gradients = composite_loss.backward().unwrap();

        let fused_output = flatten(fused_output.raw().clone()).await;
        let composite_output = flatten(composite_output.raw().clone()).await;
        let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
        let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;
        let fused_dw = flatten(fused_gradients.get(&fused_weight).unwrap()).await;
        let composite_dw = flatten(composite_gradients.get(&composite_weight).unwrap()).await;

        assert_slice_close(&fused_output, &composite_output);
        assert_slice_close(&fused_dx, &composite_dx);
        assert_slice_close(&fused_dw, &composite_dw);
    }

    #[tokio::test]
    async fn test_backward_layer_norm_last_dim_fused_matches_composite_cpu() {
        let device = Device::cpu();
        let input_data = &[
            [[0.25f32, -0.5, 1.0], [1.25, -1.5, 0.75]],
            [[-0.8, 0.4, 1.2], [0.6, -0.1, -0.9]],
        ];
        let weight_data = &[1.0f32, 0.9, 1.1];
        let bias_data = &[0.1f32, -0.2, 0.05];
        let eps = 1e-5;

        let fused_graph = Graph::new();
        let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
        let fused_weight: Tensor<1> = Tensor::new(&fused_graph, &device, weight_data);
        let fused_bias: Tensor<1> = Tensor::new(&fused_graph, &device, bias_data);
        let fused_output =
            fused_input.layer_norm_last_dim_fused::<2>(&fused_weight, Some(&fused_bias), eps);
        let fused_loss = fused_output.sqr().reshape([12]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
        let composite_weight: Tensor<1> = Tensor::new(&composite_graph, &device, weight_data);
        let composite_bias: Tensor<1> = Tensor::new(&composite_graph, &device, bias_data);
        let composite_output =
            composite_input.layer_norm(&composite_weight, Some(&composite_bias), eps);
        let composite_loss = composite_output.sqr().reshape([12]).sum();
        let composite_gradients = composite_loss.backward().unwrap();

        let fused_output = flatten(fused_output.raw().clone()).await;
        let composite_output = flatten(composite_output.raw().clone()).await;
        let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
        let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;
        let fused_dw = flatten(fused_gradients.get(&fused_weight).unwrap()).await;
        let composite_dw = flatten(composite_gradients.get(&composite_weight).unwrap()).await;
        let fused_db = flatten(fused_gradients.get(&fused_bias).unwrap()).await;
        let composite_db = flatten(composite_gradients.get(&composite_bias).unwrap()).await;

        assert_slice_close(&fused_output, &composite_output);
        assert_slice_close(&fused_dx, &composite_dx);
        assert_slice_close(&fused_dw, &composite_dw);
        assert_slice_close(&fused_db, &composite_db);
    }

    #[tokio::test]
    async fn test_backward_flash_attention_matches_composite_cpu() {
        let device = Device::cpu();
        let q_data = &[[[[0.2f32, 0.6], [1.0, -0.3]]]];
        let k_data = &[[[[0.4f32, -0.7], [0.9, 0.1]]]];
        let v_data = &[[[[1.1f32, -0.5], [0.3, 0.8]]]];
        let scale = (2.0f32).sqrt();

        let fused_graph = Graph::new();
        let fused_q: Tensor<4> = Tensor::new(&fused_graph, &device, q_data);
        let fused_k: Tensor<4> = Tensor::new(&fused_graph, &device, k_data);
        let fused_v: Tensor<4> = Tensor::new(&fused_graph, &device, v_data);
        let fused_output = fused_q.flash_attention(&fused_k, &fused_v, scale, None);
        let fused_loss = fused_output.sqr().reshape([4]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_q: Tensor<4> = Tensor::new(&composite_graph, &device, q_data);
        let composite_k: Tensor<4> = Tensor::new(&composite_graph, &device, k_data);
        let composite_v: Tensor<4> = Tensor::new(&composite_graph, &device, v_data);
        let composite_output =
            composite_q.flash_attention_composite(&composite_k, &composite_v, scale, None);
        let composite_loss = composite_output.sqr().reshape([4]).sum();
        let composite_gradients = composite_loss.backward().unwrap();

        let fused_output = flatten(fused_output.raw().clone()).await;
        let composite_output = flatten(composite_output.raw().clone()).await;
        let fused_dq = flatten(fused_gradients.get(&fused_q).unwrap()).await;
        let composite_dq = flatten(composite_gradients.get(&composite_q).unwrap()).await;
        let fused_dk = flatten(fused_gradients.get(&fused_k).unwrap()).await;
        let composite_dk = flatten(composite_gradients.get(&composite_k).unwrap()).await;
        let fused_dv = flatten(fused_gradients.get(&fused_v).unwrap()).await;
        let composite_dv = flatten(composite_gradients.get(&composite_v).unwrap()).await;

        assert_slice_close(&fused_output, &composite_output);
        assert_slice_close(&fused_dq, &composite_dq);
        assert_slice_close(&fused_dk, &composite_dk);
        assert_slice_close(&fused_dv, &composite_dv);
    }

    #[tokio::test]
    async fn test_cpu_graph_drops_after_backward() {
        let graph = Graph::new();
        let weak = Arc::downgrade(&graph.inner);
        let device = Device::cpu();

        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32, -1.0], [1.5, 2.0]]);
        let loss = x.mat_mul(&w).sum(1).sum();
        let gradients = loss.backward().unwrap();
        assert!(gradients.get(&x).is_some());
        assert!(gradients.get(&w).is_some());

        drop(gradients);
        drop(loss);
        drop(x);
        drop(w);
        drop(graph);

        assert!(
            weak.upgrade().is_none(),
            "autograd graph stayed alive after all tensors were dropped",
        );
    }

    #[test]
    fn test_gpu_gradients_can_detach() {
        let Ok(device) = Device::gpu_blocking() else {
            eprintln!("skipping GPU gradient detach regression test: GPU unavailable");
            return;
        };

        let graph = Graph::new();
        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32, -1.0], [1.5, 2.0]]);
        let gradients = x
            .mat_mul(&w)
            .sum(1)
            .sum()
            .backward()
            .unwrap()
            .into_detached();
        let dx = gradients.get(&x).expect("missing x gradient");
        let dw = gradients.get(&w).expect("missing w gradient");

        assert_eq!(
            dx.as_gpu()
                .expect("expected GPU x gradient")
                .count_kernels_to_resolve(),
            0,
            "detached x gradient should not retain backward compute graph",
        );
        assert_eq!(
            dw.as_gpu()
                .expect("expected GPU w gradient")
                .count_kernels_to_resolve(),
            0,
            "detached w gradient should not retain backward compute graph",
        );
    }
}
