use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
};

use crate::{Device, Error, Result, Tensor as RawTensor};

mod composite;
mod elementwise;
mod indexing;
pub mod layers;
mod matmul;
mod reduce;
mod view;

#[cfg(test)]
mod tests;

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

    pub fn ones(graph: &Graph, device: &Device, shape: [usize; R]) -> Self {
        Self::splat(graph, device, 1.0, shape)
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

    pub fn ones_like(&self) -> Self {
        Self::ones(&self.graph(), &self.device(), self.shape())
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
            let targets = backwards(gradient)?;
            // The scheduler only unlocks a parent once every child sends it a
            // gradient, so a missing target would silently starve that
            // parent's whole subgraph.
            for parent in &parent_handles {
                if parent.graph.requires_grad(parent.id)
                    && !targets.iter().any(|target| target.node == parent.id)
                {
                    return Err(Error::msg(
                        "custom backward omitted a gradient for a parent that requires grad",
                    ));
                }
            }
            Ok(targets)
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

    fn emit_op<const OUT: usize>(
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
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
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
        self.emit_op(
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
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
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
        self.emit_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    fn replay_quaternary<const R2: usize, const R3: usize, const R4: usize, const OUT: usize>(
        &self,
        second: &Tensor<R2>,
        third: &Tensor<R3>,
        fourth: &Tensor<R4>,
        context: &'static str,
        value: RawTensor<OUT, f32>,
        replay: impl Fn(Tensor<R>, Tensor<R2>, Tensor<R3>, Tensor<R4>) -> Tensor<OUT> + BackwardClosure,
    ) -> Tensor<OUT> {
        assert_same_graph(self, second);
        assert_same_graph(self, third);
        assert_same_graph(self, fourth);
        let ids = [
            self.handle.id,
            second.handle.id,
            third.handle.id,
            fourth.handle.id,
        ];
        let first_value = self.value.clone();
        let second_value = second.value.clone();
        let third_value = third.value.clone();
        let fourth_value = fourth.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, context)?;
            let graph = Graph::new();
            let replay_first = Tensor::from_raw(&graph, first_value.clone());
            let replay_second = Tensor::from_raw(&graph, second_value.clone());
            let replay_third = Tensor::from_raw(&graph, third_value.clone());
            let replay_fourth = Tensor::from_raw(&graph, fourth_value.clone());
            let replay_output = replay(
                replay_first.clone(),
                replay_second.clone(),
                replay_third.clone(),
                replay_fourth.clone(),
            );
            let gradients = replay_output.backward_with(gradient)?;
            let missing =
                || Error::msg(format!("missing replay gradient in {context}"));
            Ok(vec![
                BackwardTarget {
                    node: ids[0],
                    gradient: Box::new(gradients.get(&replay_first).ok_or_else(missing)?),
                },
                BackwardTarget {
                    node: ids[1],
                    gradient: Box::new(gradients.get(&replay_second).ok_or_else(missing)?),
                },
                BackwardTarget {
                    node: ids[2],
                    gradient: Box::new(gradients.get(&replay_third).ok_or_else(missing)?),
                },
                BackwardTarget {
                    node: ids[3],
                    gradient: Box::new(gradients.get(&replay_fourth).ok_or_else(missing)?),
                },
            ])
        });
        self.emit_op(
            value,
            vec![
                self.handle.clone(),
                second.handle.clone(),
                third.handle.clone(),
                fourth.handle.clone(),
            ],
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
        self.emit_op(
            value,
            vec![
                self.handle.clone(),
                second.handle.clone(),
                third.handle.clone(),
            ],
            Some(backward),
        )
    }
}

impl Tensor<1> {
    pub fn arange(graph: &Graph, device: &Device, start: f32, end: f32) -> Tensor<1> {
        graph.leaf(crate::arange(device, start, end))
    }

    pub fn arange_step(
        graph: &Graph,
        device: &Device,
        start: f32,
        end: f32,
        step: f32,
    ) -> Tensor<1> {
        graph.leaf(crate::arange_step(device, start, end, step))
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
