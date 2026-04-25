use rustc_hash::FxHashMap;

use super::queue::ComputeQueue;
use super::{ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex, layout_pass};
use crate::mir::operation::Operation;
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
    fn visit_tensor_expr(&mut self, key: NodeIndex, operation: &crate::TensorExprOperation) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("tensor_expr ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        operation.visit_dependencies(&mut |input| {
            let input_id = self.identities.get(&input).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(input_id.clone(), None).arrow_to_node(id.clone(), None),
            ));
        });
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

    fn visit_custom(
        &mut self,
        key: NodeIndex,
        operation: &std::sync::Arc<dyn crate::mir::operation::Operation + Send + Sync>,
    ) {
        let output_layout = self.layout_pass.output_layout.get(&key).unwrap();
        let id = Identity::quoted(format!("custom ({}) #{:?}", output_layout, key));
        self.statements.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        operation.visit_dependencies(&mut |dep| {
            let dep_id = self.identities.get(&dep).unwrap();
            self.statements.push(Stmt::Edge(
                Edge::head_node(dep_id.clone(), None).arrow_to_node(id.clone(), None),
            ));
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
                ComputeGraphNodeVariant::TensorExpr(op) => {
                    graph_vis_pass.visit_tensor_expr(node, op)
                }
                ComputeGraphNodeVariant::MapLayout(op) => graph_vis_pass.visit_map_layout(node, op),
                ComputeGraphNodeVariant::Tensor(op) => graph_vis_pass.visit_tensor(node, op),
                ComputeGraphNodeVariant::Custom(op) => graph_vis_pass.visit_custom(node, op),
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
