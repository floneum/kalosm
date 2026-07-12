//! Allocation-independent fusion-plan sharing.
//!
//! The value e-graph cannot equate two transformer layers: their activation
//! and weight buffers are different values.  Fusion planning, however, only
//! needs a bounded local window of operator shapes, layouts and liveness
//! facts.  This module builds those windows in a second, private egg e-graph.
//! Egg's hash-consing gives every isomorphic window one e-class, and the
//! first occurrence records a rewrite template that later occurrences can
//! instantiate by rebinding dependency roles and the concrete QMatrix.

use std::hash::Hash;

use egg::{EGraph, Id, Language};
use rustc_hash::{FxHashMap, FxHashSet};

use super::super::ExecutionVariant;
use super::EGraphDriver;
use super::extract::ExtractState;
use super::interner::{SpecId, rebind_variant_dependencies, variant_dependencies};
use super::lang::Prov;
use super::rules_fuse::FusionView;
use crate::DataTypeEnum;
use crate::compute_graph::NodeIndex;

/// The deepest operation a single fusion generator can inspect. Views are
/// canonicalized at construction (view-over-view is collapsed), so four
/// expanded levels cover nary/reduce fusion and both sides of the qmatmul
/// pre/post epilogue rules. The frontier still records exact value layout and
/// liveness, but deliberately abstracts its producer history; that is what
/// lets layer-local plans share without equating layer values.
const PLANNING_WINDOW_DEPTH: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PlanAtomId(u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PlanLang {
    Node(PlanAtomId, Box<[Id]>),
}

impl Language for PlanLang {
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Node(a, _), Self::Node(b, _)) => a == b,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Node(_, children) => children,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Node(_, children) => children,
        }
    }
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
    pub(super) root: Id,
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
    pub(super) fallback_rebuilds: u64,
}

/// Per-resolve memo. It is intentionally not global: device capabilities,
/// graph liveness and policy are part of a planning decision, while repeated
/// layers within this resolve are the redundancy we want to remove.
pub(super) struct FusionPlanMemo {
    egraph: EGraph<PlanLang, ()>,
    atoms: Vec<PlanAtom>,
    atom_ids: FxHashMap<PlanAtom, PlanAtomId>,
    decisions: FxHashMap<Id, PlanDecision>,
    seen_windows: FxHashSet<Id>,
    stats: PlanSharingStats,
}

impl Default for FusionPlanMemo {
    fn default() -> Self {
        Self {
            egraph: EGraph::default(),
            atoms: Vec::new(),
            atom_ids: FxHashMap::default(),
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
        let root = builder.memo.egraph.find(root);
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
            PlanDecision::Rewrite(template) => match template.instantiate(instance, view) {
                Some(variant) => PlanLookup::Hit(Some(variant)),
                None => {
                    // A missing role means the window invariant changed in a
                    // way its key did not capture. Recompute conservatively;
                    // never apply a partially rebound template.
                    self.stats.fallback_rebuilds += 1;
                    PlanLookup::Miss
                }
            },
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
                let Some(template) = VariantTemplate::capture(variant, instance, view) else {
                    // Opaque/effectful variants do not have a safe generic
                    // rebinding contract. They simply keep the per-instance
                    // path instead of weakening correctness.
                    return;
                };
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

    pub(super) fn record_spec(&mut self, root: Id, spec: SpecId) {
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
}

struct WindowBuilder<'a> {
    memo: &'a mut FusionPlanMemo,
    driver: &'a EGraphDriver,
    state: &'a ExtractState,
    view: &'a FusionView<'a>,
    nodes: Vec<NodeIndex>,
    role_of: FxHashMap<NodeIndex, u32>,
    local_ids: FxHashMap<NodeIndex, Id>,
    matrix_aliases: FxHashMap<usize, u32>,
}

impl WindowBuilder<'_> {
    fn add(&mut self, inner: NodeIndex, depth: usize) -> Id {
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
        let expanded = depth < PLANNING_WINDOW_DEPTH;
        let variant = expanded.then(|| self.view.variant_of(inner)).flatten();
        // Rules only look through these five pure operator families. Other
        // operations are materialization frontiers: their output layout and
        // liveness matter, but their private implementation/allocation does
        // not. Treating row programs and embeddings this way is particularly
        // important for repeated transformer blocks.
        let opaque = variant.is_some_and(|variant| {
            matches!(
                variant,
                ExecutionVariant::QMatrix(_)
                    | ExecutionVariant::Assign(_)
                    | ExecutionVariant::Region(_)
                    | ExecutionVariant::QEmbedding(_)
                    | ExecutionVariant::GraphOp(_)
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
        let (kind, dependencies) = if !expanded || opaque {
            (PlanNodeKind::Frontier, Vec::new())
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
        let children: Vec<Id> = dependencies
            .into_iter()
            .map(|dependency| self.add(dependency, depth + 1))
            .collect();
        let id = self
            .memo
            .egraph
            .add(PlanLang::Node(atom, children.into_boxed_slice()));
        self.local_ids.insert(inner, id);
        id
    }
}

impl VariantTemplate {
    fn capture(
        variant: &ExecutionVariant,
        instance: &PlanInstance,
        view: &FusionView<'_>,
    ) -> Option<Self> {
        if matches!(
            variant,
            ExecutionVariant::Tensor(_)
                | ExecutionVariant::QMatrix(_)
                | ExecutionVariant::Assign(_)
                | ExecutionVariant::Region(_)
                | ExecutionVariant::GraphOp(_)
        ) {
            return None;
        }
        let dependency_roles = variant_dependencies(variant)
            .into_iter()
            .map(|dependency| instance.role_of.get(&dependency).copied())
            .collect::<Option<Vec<_>>>()?;
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
        if matrix_of(variant).is_some() && matrix_role.is_none() {
            return None;
        }
        Some(Self {
            variant: variant.clone(),
            dependency_roles,
            matrix_role,
            spec: None,
        })
    }

    fn instantiate(
        &self,
        instance: &PlanInstance,
        view: &FusionView<'_>,
    ) -> Option<ExecutionVariant> {
        let dependencies = self
            .dependency_roles
            .iter()
            .map(|&role| instance.nodes.get(role as usize).copied())
            .collect::<Option<Vec<_>>>()?;
        let mut variant = self.variant.clone();
        rebind_variant_dependencies(&mut variant, &dependencies);
        if let Some(role) = self.matrix_role {
            let inner = *instance.nodes.get(role as usize)?;
            let matrix = matrix_of(view.variant_of(inner)?)?.clone();
            match &mut variant {
                ExecutionVariant::QMatMul(operation) => operation.matrix = matrix,
                ExecutionVariant::QEmbedding(operation) => operation.matrix = matrix,
                _ => return None,
            }
        }
        Some(variant)
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
    use crate::compute_graph::resolve::egraph::rules_fuse::Stage2Ctx;
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
                let ctx = Stage2Ctx {
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
                let ctx = Stage2Ctx {
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
                resolver.recognize_operations(graph);
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
                let driver = EGraphDriver::ingest(&resolver, graph);
                let state = ExtractState::new(&driver);
                let ctx = Stage2Ctx {
                    graph,
                    layouts: std::cell::RefCell::new(Default::default()),
                };
                let view = FusionView::new(&driver, &state, &ctx);
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
