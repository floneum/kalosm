//! Allocation-independent fusion-plan sharing.
//!
//! The value e-graph cannot equate two transformer layers: their activation
//! and weight buffers are different values. Fusion planning can still share
//! allocation-independent structural templates. A compact structural
//! interner canonicalizes those templates without creating a second e-graph;
//! the first occurrence records a rewrite that later occurrences instantiate
//! by rebinding dependency roles and the concrete QMatrix.

use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};

use super::super::ExecutionVariant;
use super::EGraphDriver;
use super::extract::ExtractState;
use super::interner::{SpecId, rebind_variant_dependencies, variant_dependencies};
use super::lang::Prov;
use super::rules_fuse::FusionView;
use crate::DataTypeEnum;
use crate::compute_graph::NodeIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PlanAtomId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct StructuralId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructuralNode {
    atom: PlanAtomId,
    children: Box<[StructuralId]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LayoutSpec {
    datatype: DataTypeEnum,
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PlanNodeKind {
    Operator(SpecId),
    Tensor,
    Boundary,
    Missing,
    Frontier,
}

/// Exact facts a generator is permitted to observe for one role in its
/// local window. Including role numbers preserves aliasing: `[x, x]` and
/// `[x, y]` never share a plan even when x and y have identical layouts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlanAtom {
    role: u32,
    kind: PlanNodeKind,
    /// Canonical local identity of an embedded QMatrix allocation. This
    /// preserves the distinction between "two views of the same qmatmul"
    /// and "two same-shaped qmatmuls with different weights" while still
    /// allowing corresponding allocations in different layers to share.
    matrix_alias: Option<u32>,
    layout: Option<LayoutSpec>,
    reads: u32,
    cached: bool,
    externally_live: bool,
    is_target: bool,
}

pub(super) struct PlanInstance {
    pub(super) root: StructuralId,
    nodes: Vec<NodeIndex>,
    role_of: FxHashMap<NodeIndex, u32>,
}

#[derive(Clone)]
struct VariantTemplate {
    variant: ExecutionVariant,
    dependency_roles: Vec<u32>,
    matrix_role: Option<u32>,
    spec: Option<SpecId>,
}

#[derive(Clone)]
enum PlanDecision {
    NoRewrite,
    Rewrite(VariantTemplate),
}

pub(super) enum PlanLookup {
    Miss,
    Hit(Option<ExecutionVariant>),
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PlanSharingStats {
    pub(super) windows: u64,
    pub(super) unique_windows: u64,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) templates: u64,
    pub(super) negative_templates: u64,
}

/// Per-resolve memo. It is intentionally not global: device capabilities,
/// graph liveness and policy are part of a planning decision, while repeated
/// layers within this resolve are the redundancy we want to remove.
pub(super) struct FusionPlanMemo {
    atoms: Vec<PlanAtom>,
    atom_ids: FxHashMap<PlanAtom, PlanAtomId>,
    nodes: Vec<StructuralNode>,
    node_ids: FxHashMap<StructuralNode, StructuralId>,
    decisions: FxHashMap<StructuralId, PlanDecision>,
    seen_windows: FxHashSet<StructuralId>,
    stats: PlanSharingStats,
}

impl Default for FusionPlanMemo {
    fn default() -> Self {
        Self {
            atoms: Vec::new(),
            atom_ids: FxHashMap::default(),
            nodes: Vec::new(),
            node_ids: FxHashMap::default(),
            decisions: FxHashMap::default(),
            seen_windows: FxHashSet::default(),
            stats: PlanSharingStats::default(),
        }
    }
}

impl FusionPlanMemo {
    pub(super) fn capture(
        &mut self,
        driver: &EGraphDriver,
        state: &ExtractState,
        view: &FusionView<'_>,
        prov: Prov,
    ) -> PlanInstance {
        self.stats.windows += 1;
        let inner = driver.egraph.analysis.facts_of(prov).inner;
        let mut builder = WindowBuilder {
            memo: self,
            driver,
            state,
            view,
            nodes: Vec::new(),
            role_of: FxHashMap::default(),
            local_ids: FxHashMap::default(),
            matrix_aliases: FxHashMap::default(),
        };
        let root = builder.add(inner, 0);
        builder.memo.seen_windows.insert(root);
        builder.memo.stats.unique_windows = builder.memo.seen_windows.len() as u64;
        PlanInstance {
            root,
            nodes: builder.nodes,
            role_of: builder.role_of,
        }
    }

    pub(super) fn lookup(&mut self, instance: &PlanInstance, view: &FusionView<'_>) -> PlanLookup {
        let Some(decision) = self.decisions.get(&instance.root) else {
            self.stats.misses += 1;
            return PlanLookup::Miss;
        };
        self.stats.hits += 1;
        match decision {
            PlanDecision::NoRewrite => PlanLookup::Hit(None),
            PlanDecision::Rewrite(template) => {
                PlanLookup::Hit(Some(template.instantiate(instance, view)))
            }
        }
    }

    pub(super) fn record(
        &mut self,
        instance: &PlanInstance,
        view: &FusionView<'_>,
        result: Option<&ExecutionVariant>,
    ) {
        if self.decisions.contains_key(&instance.root) {
            return;
        }
        let decision = match result {
            None => {
                self.stats.negative_templates += 1;
                PlanDecision::NoRewrite
            }
            Some(variant) => {
                let template = VariantTemplate::capture(variant, instance, view);
                self.stats.templates += 1;
                PlanDecision::Rewrite(template)
            }
        };
        self.decisions.insert(instance.root, decision);
    }

    pub(super) fn stats(&self) -> PlanSharingStats {
        self.stats
    }

    pub(super) fn known_spec(&self, instance: &PlanInstance) -> Option<SpecId> {
        let PlanDecision::Rewrite(template) = self.decisions.get(&instance.root)? else {
            return None;
        };
        template.spec
    }

    pub(super) fn record_spec(&mut self, root: StructuralId, spec: SpecId) {
        let Some(PlanDecision::Rewrite(template)) = self.decisions.get_mut(&root) else {
            return;
        };
        match template.spec {
            Some(existing) => debug_assert_eq!(existing, spec),
            None => template.spec = Some(spec),
        }
    }

    fn intern_atom(&mut self, atom: PlanAtom) -> PlanAtomId {
        if let Some(&id) = self.atom_ids.get(&atom) {
            return id;
        }
        let id = PlanAtomId(self.atoms.len() as u32);
        self.atoms.push(atom.clone());
        self.atom_ids.insert(atom, id);
        id
    }

    fn intern_node(&mut self, node: StructuralNode) -> StructuralId {
        if let Some(&id) = self.node_ids.get(&node) {
            return id;
        }
        let id = StructuralId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.node_ids.insert(node, id);
        id
    }
}

struct WindowBuilder<'a> {
    memo: &'a mut FusionPlanMemo,
    driver: &'a EGraphDriver,
    state: &'a ExtractState,
    view: &'a FusionView<'a>,
    nodes: Vec<NodeIndex>,
    role_of: FxHashMap<NodeIndex, u32>,
    local_ids: FxHashMap<NodeIndex, StructuralId>,
    matrix_aliases: FxHashMap<usize, u32>,
}

/// Fusion generators make one single-step decision per visit: they observe
/// the candidate node, its direct inputs' selected variants and facts, and
/// (through the variants they emit) the identities and layouts of the
/// inputs' inputs. Deeper structure is invisible to a generation step, and
/// every hit re-validates kills and switch cost against live state — so the
/// structural window cuts at that horizon. Cutting is what makes repeated
/// layers share: an unbounded walk would drag each window's whole upstream
/// cone in, making every layer's window unique and the walk quadratic.
const WINDOW_STUB_DEPTH: u32 = 2;

impl WindowBuilder<'_> {
    fn add(&mut self, inner: NodeIndex, depth: u32) -> StructuralId {
        if let Some(&id) = self.local_ids.get(&inner) {
            return id;
        }
        let role = if let Some(&role) = self.role_of.get(&inner) {
            role
        } else {
            let role = self.nodes.len() as u32;
            self.nodes.push(inner);
            self.role_of.insert(inner, role);
            role
        };

        let prov = self.driver.prov_of.get(&inner).copied();
        let facts = prov.map(|prov| self.driver.egraph.analysis.facts_of(prov));
        let variant = self.view.variant_of(inner);
        // Rules only look through these five pure operator families. Other
        // operations are materialization frontiers: their output layout and
        // liveness matter, but their private implementation/allocation does
        // not. Embeddings and effectful/multi-output nodes stay frontiers;
        // row programs have a structural equality and rebinding contract.
        let opaque = variant.is_some_and(|variant| {
            matches!(
                variant,
                ExecutionVariant::QMatrix(_)
                    | ExecutionVariant::Assign(_)
                    | ExecutionVariant::Region(_)
                    | ExecutionVariant::QEmbedding(_)
            )
        });
        let matrix_alias = (!opaque)
            .then_some(variant)
            .flatten()
            .and_then(matrix_of)
            .map(|matrix| {
                let allocation = std::sync::Arc::as_ptr(matrix.buffer()) as usize;
                let next = self.matrix_aliases.len() as u32;
                *self.matrix_aliases.entry(allocation).or_insert(next)
            });
        let stub = depth >= WINDOW_STUB_DEPTH;
        let (kind, dependencies) = if opaque || stub {
            let kind = if opaque {
                PlanNodeKind::Frontier
            } else if let Some(variant) = variant {
                match variant {
                    ExecutionVariant::Tensor(_) => PlanNodeKind::Tensor,
                    _ => {
                        let prov = prov.expect("selected execution variant has provenance");
                        let payload = self
                            .state
                            .selected_enode(self.driver, prov)
                            .payload()
                            .expect("non-tensor execution variant has a payload");
                        PlanNodeKind::Operator(
                            self.driver.egraph.analysis.payloads.spec_of(payload),
                        )
                    }
                }
            } else if facts.is_some_and(|facts| facts.exec.is_none()) {
                PlanNodeKind::Boundary
            } else {
                PlanNodeKind::Missing
            };
            (kind, Vec::new())
        } else if let Some(variant) = variant {
            let kind = match variant {
                ExecutionVariant::Tensor(_) => PlanNodeKind::Tensor,
                _ => {
                    let prov = prov.expect("selected execution variant has provenance");
                    let payload = self
                        .state
                        .selected_enode(self.driver, prov)
                        .payload()
                        .expect("non-tensor execution variant has a payload");
                    PlanNodeKind::Operator(self.driver.egraph.analysis.payloads.spec_of(payload))
                }
            };
            (kind, variant_dependencies(variant))
        } else if facts.is_some_and(|facts| facts.exec.is_none()) {
            (PlanNodeKind::Boundary, Vec::new())
        } else {
            (PlanNodeKind::Missing, Vec::new())
        };

        let layout = self.view.layout_of(inner).map(|info| LayoutSpec {
            datatype: info.datatype(),
            offset: info.layout().offset(),
            shape: info.layout().shape().into(),
            strides: info.layout().strides().into(),
        });
        let atom = PlanAtom {
            role,
            kind,
            matrix_alias,
            layout,
            reads: prov
                .map(|prov| self.state.reads[prov.0 as usize])
                .unwrap_or(0),
            cached: facts.is_some_and(|facts| facts.exec.is_none()),
            externally_live: facts.is_some_and(|facts| facts.externally_live),
            is_target: facts.is_some_and(|facts| facts.is_target),
        };
        let atom = self.memo.intern_atom(atom);
        let children: Vec<StructuralId> = dependencies
            .into_iter()
            .map(|dependency| self.add(dependency, depth + 1))
            .collect();
        let id = self.memo.intern_node(StructuralNode {
            atom,
            children: children.into_boxed_slice(),
        });
        self.local_ids.insert(inner, id);
        id
    }
}

impl VariantTemplate {
    fn capture(variant: &ExecutionVariant, instance: &PlanInstance, view: &FusionView<'_>) -> Self {
        assert!(
            !matches!(
                variant,
                ExecutionVariant::Tensor(_)
                    | ExecutionVariant::QMatrix(_)
                    | ExecutionVariant::Assign(_)
                    | ExecutionVariant::Region(_)
            ),
            "fusion generators must produce a structurally rebindable variant"
        );
        let dependency_roles = variant_dependencies(variant)
            .into_iter()
            .map(|dependency| instance.role_of[&dependency])
            .collect();
        let matrix_role = matrix_of(variant).and_then(|matrix| {
            instance
                .nodes
                .iter()
                .position(|&inner| {
                    view.variant_of(inner)
                        .and_then(matrix_of)
                        .is_some_and(|candidate| same_matrix_allocation(matrix, candidate))
                })
                .map(|role| role as u32)
        });
        assert!(
            matrix_of(variant).is_none() || matrix_role.is_some(),
            "quantized fusion template matrix must be part of its structural window"
        );
        Self {
            variant: variant.clone(),
            dependency_roles,
            matrix_role,
            spec: None,
        }
    }

    fn instantiate(&self, instance: &PlanInstance, view: &FusionView<'_>) -> ExecutionVariant {
        let dependencies = self
            .dependency_roles
            .iter()
            .map(|&role| instance.nodes[role as usize])
            .collect::<Vec<_>>();
        let mut variant = self.variant.clone();
        rebind_variant_dependencies(&mut variant, &dependencies);
        if let Some(role) = self.matrix_role {
            let inner = instance.nodes[role as usize];
            let matrix = matrix_of(
                view.variant_of(inner)
                    .expect("template matrix role must have a selected variant"),
            )
            .expect("template matrix role must remain quantized")
            .clone();
            match &mut variant {
                ExecutionVariant::QMatMul(operation) => operation.matrix = matrix,
                ExecutionVariant::QEmbedding(operation) => operation.matrix = matrix,
                _ => unreachable!("only quantized variants carry a matrix role"),
            }
        }
        variant
    }
}

fn matrix_of(variant: &ExecutionVariant) -> Option<&crate::quantized::QMatrix> {
    match variant {
        ExecutionVariant::QMatrix(operation) => Some(&operation.matrix),
        ExecutionVariant::QMatMul(operation) => Some(&operation.matrix),
        ExecutionVariant::QEmbedding(operation) => Some(&operation.matrix),
        _ => None,
    }
}

fn same_matrix_allocation(a: &crate::quantized::QMatrix, b: &crate::quantized::QMatrix) -> bool {
    std::sync::Arc::ptr_eq(a.buffer(), b.buffer())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::resolve::Resolver;
    use crate::compute_graph::resolve::egraph::rules_fuse::FusionCtx;
    use crate::{Device, QMatrix, Tensor};

    #[test]
    fn repeated_windows_share_a_plan_but_rebind_distinct_values() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let left = Tensor::new::<f32, 1, _>(&device, &[1.0, 2.0, 3.0, 4.0]);
            let right = Tensor::new::<f32, 1, _>(&device, &[5.0, 6.0, 7.0, 8.0]);
            let left_out = (&left + 1.0) * 2.0;
            let right_out = (&right + 1.0) * 2.0;
            let targets = [left_out.data().key, right_out.data().key];

            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.to_vec());
                for &target in &targets {
                    resolver.build_execution_graph(graph, target);
                }
                let driver = EGraphDriver::ingest(&resolver, graph);
                let state = ExtractState::new(&driver);
                let ctx = FusionCtx {
                    graph,
                    layouts: std::cell::RefCell::new(Default::default()),
                };
                let view = FusionView::new(&driver, &state, &ctx);
                let left_prov = driver.prov_of[&targets[0]];
                let right_prov = driver.prov_of[&targets[1]];

                // Equal plans are not equal values: allocation identity keeps
                // the executable e-classes separate.
                assert_ne!(driver.class_for(targets[0]), driver.class_for(targets[1]));

                let mut memo = FusionPlanMemo::default();
                let left_instance = memo.capture(&driver, &state, &view, left_prov);
                assert!(matches!(
                    memo.lookup(&left_instance, &view),
                    PlanLookup::Miss
                ));
                let left_variant = view
                    .generate_candidates(left_prov)
                    .into_iter()
                    .next()
                    .expect("left chain fuses");
                memo.record(&left_instance, &view, Some(&left_variant));

                let right_instance = memo.capture(&driver, &state, &view, right_prov);
                assert_eq!(left_instance.root, right_instance.root);
                let PlanLookup::Hit(Some(rebound)) = memo.lookup(&right_instance, &view) else {
                    panic!("second isomorphic window should instantiate the first plan");
                };
                let fresh = view
                    .generate_candidates(right_prov)
                    .into_iter()
                    .next()
                    .expect("right chain fuses");
                match (rebound, fresh) {
                    (
                        ExecutionVariant::Elementwise(rebound),
                        ExecutionVariant::Elementwise(fresh),
                    ) => {
                        assert_eq!(rebound, fresh);
                        assert!(rebound.inputs.iter().all(|input| {
                            right_instance.role_of.contains_key(input)
                                && !left_instance.role_of.contains_key(input)
                        }));
                    }
                    variants => panic!("unexpected generated variants: {variants:?}"),
                }
                let stats = memo.stats();
                assert_eq!(stats.templates, 1);
                assert_eq!(stats.hits, 1);
            });
        });
    }

    #[test]
    fn transformer_sized_repetition_plans_once() {
        pollster::block_on(async {
            const LAYERS: usize = 32;
            let Ok(device) = Device::new().await else {
                return;
            };
            let inputs: Vec<Tensor> = (0..LAYERS)
                .map(|layer| {
                    Tensor::new::<f32, 1, _>(
                        &device,
                        &[
                            layer as f32,
                            layer as f32 + 1.0,
                            layer as f32 + 2.0,
                            layer as f32 + 3.0,
                        ],
                    )
                })
                .collect();
            let outputs: Vec<Tensor> = inputs.iter().map(|input| (input + 1.0) * 2.0).collect();
            let targets: Vec<NodeIndex> = outputs.iter().map(|output| output.data().key).collect();

            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.clone());
                for &target in &targets {
                    resolver.build_execution_graph(graph, target);
                }
                let driver = EGraphDriver::ingest(&resolver, graph);
                let state = ExtractState::new(&driver);
                let ctx = FusionCtx {
                    graph,
                    layouts: std::cell::RefCell::new(Default::default()),
                };
                let view = FusionView::new(&driver, &state, &ctx);
                let mut memo = FusionPlanMemo::default();

                for (layer, &target) in targets.iter().enumerate() {
                    let prov = driver.prov_of[&target];
                    let instance = memo.capture(&driver, &state, &view, prov);
                    match memo.lookup(&instance, &view) {
                        PlanLookup::Miss if layer == 0 => {
                            let variant = view
                                .generate_candidates(prov)
                                .into_iter()
                                .next()
                                .expect("repeated layer chain fuses");
                            memo.record(&instance, &view, Some(&variant));
                        }
                        PlanLookup::Hit(Some(_)) if layer > 0 => {}
                        _ => panic!("unexpected planning result for repeated layer {layer}"),
                    }
                }

                let stats = memo.stats();
                assert_eq!(stats.windows, LAYERS as u64);
                assert_eq!(stats.unique_windows, 1);
                assert_eq!(stats.templates, 1);
                assert_eq!(stats.misses, 1);
                assert_eq!(stats.hits, (LAYERS - 1) as u64);
            });
        });
    }

    fn dense_qmatrix(device: &Device, value: f32) -> QMatrix {
        const N: usize = 4;
        const K: usize = 8;
        let bytes = std::iter::repeat_n(value, N * K)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        QMatrix::from_parts(device, &bytes, Box::new([N, K]), fusor_gguf::GgmlType::F32).unwrap()
    }

    #[test]
    fn qmatmul_plan_rebinds_the_layer_weight_allocation() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let activations: Vec<Tensor> = (0..4)
                .map(|offset| {
                    Tensor::new::<f32, 2, _>(
                        &device,
                        &[vec![
                            1.0 + offset as f32,
                            -2.0,
                            3.0,
                            -4.0,
                            5.0,
                            -6.0,
                            7.0,
                            -8.0,
                        ]],
                    )
                })
                .collect();
            let left_weight = dense_qmatrix(&device, 0.25);
            let right_weight = dense_qmatrix(&device, 0.5);
            let other_weight = dense_qmatrix(&device, 0.75);
            let left = activations[0].q_mat_mul(&left_weight);
            let right = activations[1].q_mat_mul(&right_weight);
            let same_matrix_peer = activations[2].q_mat_mul(&left_weight);
            let other_matrix_peer = activations[3].q_mat_mul(&other_weight);
            // Separate operations over the exact same activation and weight
            // become several observations of one semantic e-class.
            let duplicate_a = activations[0].q_mat_mul(&left_weight);
            let duplicate_b = activations[0].q_mat_mul(&left_weight);
            let same_matrix_sum = &left + &same_matrix_peer;
            let different_matrix_sum = &right + &other_matrix_peer;
            let duplicate_sum = &duplicate_a + &duplicate_b;
            let qmatmuls = [
                left.data().key,
                right.data().key,
                same_matrix_peer.data().key,
                other_matrix_peer.data().key,
                duplicate_a.data().key,
                duplicate_b.data().key,
            ];
            let targets = [
                same_matrix_sum.data().key,
                different_matrix_sum.data().key,
                duplicate_sum.data().key,
            ];

            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.to_vec());
                for &target in &targets {
                    resolver.build_execution_graph(graph, target);
                }
                let mut driver =
                    EGraphDriver::ingest_for_recognition(&resolver, graph).run_recognition();
                let extraction = driver.extract();
                resolver.apply_egraph_deltas(graph, &driver, &extraction);
                resolver.recognize_attention(graph);
                assert_eq!(
                    resolver
                        .execution_graph
                        .node_indices()
                        .filter(|&node| matches!(
                            resolver.execution_graph[node].variant,
                            ExecutionVariant::QMatMul(_)
                        ))
                        .count(),
                    qmatmuls.len(),
                    "every observation of a shared e-class must be specialized"
                );
                let state = ExtractState::from_execution(&mut driver, &resolver);
                let ctx = FusionCtx {
                    graph,
                    layouts: std::cell::RefCell::new(Default::default()),
                };
                let view = FusionView::new(&driver, &state, &ctx);
                assert_eq!(
                    variant_dependencies(view.variant_of(qmatmuls[0]).unwrap()),
                    vec![activations[0].data().key],
                    "a synchronized alternative must keep the first occurrence's concrete dependencies"
                );
                assert_eq!(
                    variant_dependencies(view.variant_of(qmatmuls[2]).unwrap()),
                    vec![activations[2].data().key],
                    "a synchronized alternative must keep this occurrence's concrete dependencies"
                );
                // Use the unique left-weight observation here; qmatmuls[0]
                // deliberately shares a value e-class with the duplicate
                // pair below and therefore has an aggregated read count.
                let left_prov = driver.prov_of[&qmatmuls[2]];
                let right_prov = driver.prov_of[&qmatmuls[1]];
                assert_ne!(
                    driver.class_for(qmatmuls[2]),
                    driver.class_for(qmatmuls[1]),
                    "different weight buffers remain different values"
                );

                let mut memo = FusionPlanMemo::default();
                let left_instance = memo.capture(&driver, &state, &view, left_prov);
                let left_variant = view.variant_of(qmatmuls[2]).unwrap().clone();
                memo.record(&left_instance, &view, Some(&left_variant));

                let right_instance = memo.capture(&driver, &state, &view, right_prov);
                assert_eq!(left_instance.root, right_instance.root);
                let PlanLookup::Hit(Some(ExecutionVariant::QMatMul(rebound))) =
                    memo.lookup(&right_instance, &view)
                else {
                    panic!("qmatmul planning template should be shared");
                };
                assert!(std::sync::Arc::ptr_eq(
                    rebound.matrix.buffer(),
                    right_weight.buffer()
                ));
                assert!(!std::sync::Arc::ptr_eq(
                    rebound.matrix.buffer(),
                    left_weight.buffer()
                ));

                // Local matrix-alias identity is part of the plan key. Two
                // distinct qmatmul nodes sharing one weight may take the
                // same-base accumulator rewrite; same-shaped but different
                // weights must not reuse that decision.
                let same_sum = memo.capture(&driver, &state, &view, driver.prov_of[&targets[0]]);
                let different_sum =
                    memo.capture(&driver, &state, &view, driver.prov_of[&targets[1]]);
                assert_ne!(same_sum.root, different_sum.root);
            });
        });
    }
}
