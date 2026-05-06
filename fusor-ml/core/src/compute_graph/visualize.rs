use rustc_hash::FxHashMap;

use super::queue::ComputeQueue;
use super::{ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex, layout_pass};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

#[derive(Default)]
struct GraphVisPass {
    queued: ComputeQueue,
    layout_pass: layout_pass::LayoutPass,
    identities: FxHashMap<NodeIndex, Identity>,
    statements: Vec<Stmt>,
}

impl GraphVisPass {
    fn visit_nary(&mut self, key: NodeIndex, operation: &crate::nary_wise::NaryOperation) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("nary ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        for input in &operation.inputs {
            let input_id = self.identities.get(input).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(input_id.clone(), None).arrow_to_node(id.clone(), None),
            ));
        }
        self.identities.insert(key, id.clone());
    }

    fn visit_mat_mul(&mut self, key: NodeIndex, operation: &crate::MatMulOperation) {
        let first = self.identities.get(&operation.first).unwrap();
        let second = self.identities.get(&operation.second).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("matmul ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(first.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(second.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_q_mat_mul(
        &mut self,
        key: NodeIndex,
        operation: &crate::quantized::matmul::QMatMulOperation,
    ) {
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("qmatmul ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_q_mat_mul_swiglu(
        &mut self,
        key: NodeIndex,
        operation: &crate::quantized::matmul::QMatMulSwiGluOperation,
    ) {
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("qmatmul_swiglu ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_reduce(&mut self, key: NodeIndex, operation: &crate::ReduceOperation) {
        let input = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!(
            "{} ({}) #{:?}",
            operation.function.name(),
            output_layout,
            key
        ));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_rms_norm(&mut self, key: NodeIndex, operation: &crate::RmsNormOperation) {
        let input = self.identities.get(&operation.input).unwrap();
        let weight = self.identities.get(&operation.weight).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("rms_norm ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(weight.clone(), None).arrow_to_node(id.clone(), None),
        ));
        if let Some(bias) = operation.bias {
            let bias = self.identities.get(&bias).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(bias.clone(), None).arrow_to_node(id.clone(), None),
            ));
        }
        self.identities.insert(key, id.clone());
    }

    fn visit_flash_attention(
        &mut self,
        key: NodeIndex,
        operation: &crate::FlashAttentionOperation,
    ) {
        let q = self.identities.get(&operation.q).unwrap();
        let k = self.identities.get(&operation.k).unwrap();
        let v = self.identities.get(&operation.v).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("flash_attention ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(q.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(k.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(v.clone(), None).arrow_to_node(id.clone(), None),
        ));
        if let Some(mask) = operation.mask {
            let mask = self.identities.get(&mask).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(mask.clone(), None).arrow_to_node(id.clone(), None),
            ));
        }
        self.identities.insert(key, id.clone());
    }

    fn visit_map_layout(
        &mut self,
        key: NodeIndex,
        operation: &crate::map_layout::MapLayoutOperation,
    ) {
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("map_layout ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_resize(&mut self, key: NodeIndex, operation: &crate::resize::ResizeOperation) {
        let input = self.identities.get(&operation.input).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("resize ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_slice_assign(
        &mut self,
        key: NodeIndex,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) {
        let input = self.identities.get(&operation.input).unwrap();
        let value = self.identities.get(&operation.value).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("slice_assign ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(input.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.statements.push(Stmt::Edge(
            Edge::head_node(value.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_dequantize(
        &mut self,
        key: NodeIndex,
        _operation: &crate::dequantize::DequantizeOperation,
    ) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("dequantize ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key, id.clone());
    }

    fn visit_q_embedding(
        &mut self,
        key: NodeIndex,
        operation: &crate::quantized::embedding::QEmbeddingOperation,
    ) {
        let indexes = self.identities.get(&operation.indexes).unwrap();
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("q_embedding ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.statements.push(Stmt::Edge(
            Edge::head_node(indexes.clone(), None).arrow_to_node(id.clone(), None),
        ));
        self.identities.insert(key, id.clone());
    }

    fn visit_tensor(&mut self, key: NodeIndex, _operation: &crate::tensor::TensorData) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("tensor ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        self.identities.insert(key, id.clone());
    }
}

impl ComputeGraphInner {
    pub(crate) fn graphvis(&self, root: NodeIndex) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut graph_vis_pass = GraphVisPass {
            layout_pass,
            ..Default::default()
        };
        graph_vis_pass.queued.push_back(root);
        while let Some(node) = graph_vis_pass.queued.pop_front() {
            if graph_vis_pass.identities.contains_key(&node) {
                continue;
            }
            let node_data = self.nodes.nodes.node_weight(node);
            if let Some(data) = node_data.and_then(|n| n.cached.as_ref()) {
                let id = Identity::quoted(format!("cached ({}) #{:?}", data.info(), node));
                graph_vis_pass.statements.push(Stmt::Node {
                    id: id.clone(),
                    port: None,
                    attr: None,
                });
                graph_vis_pass.identities.insert(node, id.clone());
                continue;
            }

            let mut dependencies = Vec::new();
            self.visit_dependencies(node, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            dependencies.retain(|dependency| !graph_vis_pass.identities.contains_key(dependency));
            if !dependencies.is_empty() {
                // If there are dependencies that are not resolved, push them to the queue then
                // revisit this node
                for dependency in dependencies {
                    graph_vis_pass.queued.push_back(dependency);
                }
                graph_vis_pass.queued.push_back(node);
                continue;
            }

            let node_data = self.nodes.nodes.node_weight(node).expect("Node not found");
            match &node_data.variant {
                ComputeGraphNodeVariant::Nary(op) => graph_vis_pass.visit_nary(node, op),
                ComputeGraphNodeVariant::MatMul(op) => graph_vis_pass.visit_mat_mul(node, op),
                ComputeGraphNodeVariant::QMatMul(op) => graph_vis_pass.visit_q_mat_mul(node, op),
                ComputeGraphNodeVariant::QMatMulSwiGlu(op) => {
                    graph_vis_pass.visit_q_mat_mul_swiglu(node, op)
                }
                ComputeGraphNodeVariant::QEmbedding(op) => {
                    graph_vis_pass.visit_q_embedding(node, op)
                }
                ComputeGraphNodeVariant::Reduce(op) => graph_vis_pass.visit_reduce(node, op),
                ComputeGraphNodeVariant::RmsNorm(op) => graph_vis_pass.visit_rms_norm(node, op),
                ComputeGraphNodeVariant::FlashAttention(op) => {
                    graph_vis_pass.visit_flash_attention(node, op)
                }
                ComputeGraphNodeVariant::MapLayout(op) => graph_vis_pass.visit_map_layout(node, op),
                ComputeGraphNodeVariant::Resize(op) => graph_vis_pass.visit_resize(node, op),
                ComputeGraphNodeVariant::SliceAssign(op) => {
                    graph_vis_pass.visit_slice_assign(node, op)
                }
                ComputeGraphNodeVariant::Tensor(op) => graph_vis_pass.visit_tensor(node, op),
                ComputeGraphNodeVariant::Dequantize(op) => {
                    graph_vis_pass.visit_dequantize(node, op)
                }
            }
        }

        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::quoted("ComputeGraph"))
            .stmts(StmtList::new().extend(graph_vis_pass.statements))
            .build()
            .unwrap()
    }
}
