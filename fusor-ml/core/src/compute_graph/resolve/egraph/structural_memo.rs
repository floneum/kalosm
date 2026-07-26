//! Allocation-independent fusion-plan sharing.
//!
//! The value e-graph cannot equate two transformer layers: their activation
//! and weight buffers are different values. Fusion planning can still share
//! allocation-independent structural templates. A compact structural
//! interner canonicalizes those templates without creating a second e-graph;
//! the first occurrence records a rewrite that later occurrences instantiate
//! by rebinding dependency roles and the concrete QMatrix.

use std::hash::Hash;

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use super::super::ExecutionVariant;
use super::EGraphDriver;
use super::extract::ExtractState;
use super::interner::{
    PayloadKey, SpecId, TwoLane, local_hash, rebind_variant_dependencies, variant_dependencies,
};
use super::lang::Prov;
use super::rules_fuse::FusionView;
use crate::compute_graph::NodeIndex;
use crate::quantized::QMatrix;
use crate::quantized::embedding::QEmbeddingOperation;
use crate::quantized::matmul::{ElementwiseEpilogue, QMatMulOperation};
use crate::{DataTypeEnum, FusorConfig};

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
pub(super) struct VariantTemplate {
    variant: ExecutionVariant,
    dependency_roles: Vec<u32>,
    matrix_role: Option<u32>,
    spec: Option<SpecId>,
}

#[derive(Clone)]
pub(super) enum PlanDecision {
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
    /// Rewrites reading past the window horizon, which plan fresh every
    /// visit ([`VariantTemplate::capture`]).
    pub(super) unshareable: u64,
    /// Per-resolve misses answered by the device-scoped [`FusionPlanStore`].
    pub(super) store_hits: u64,
    pub(super) store_misses: u64,
}

/// Resolve-independent identity of one planning window: a two-lane
/// structural hash over the window's atoms and topology. Like
/// `FlushPlanKey`, the key is trusted without exact verification;
/// `FUSOR_VERIFY_PLAN_SHARING` regenerates and compares on every hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct WindowKey([u64; 2]);

/// A stored template body with quantized-matrix identity erased. The store
/// outlives resolves on its device, so entries must hold no [`QMatrix`]: its
/// buffer Arc would pin dropped weights and its `Device` handle would cycle
/// the store back to `DeviceInner`. The template's `matrix_role` re-supplies
/// the concrete matrix at instantiation ([`VariantTemplate::capture`]
/// asserts one exists for every matrix-carrying variant).
#[derive(Clone)]
enum StoredBody {
    /// Variant kinds that hold no buffer or device handles.
    Plain(ExecutionVariant),
    QMatMul(StoredQMatMul),
    QEmbedding(StoredQEmbedding),
}

#[derive(Clone)]
struct StoredQMatMul {
    input_datatype: DataTypeEnum,
    input: NodeIndex,
    in_shape: Box<[usize]>,
    out_shape: Box<[usize]>,
    pre_element_wise_expr: Option<ElementwiseEpilogue>,
    post_element_wise_expr: Option<ElementwiseEpilogue>,
    post_accumulator_offsets: Box<[u32]>,
}

#[derive(Clone)]
struct StoredQEmbedding {
    indexes: NodeIndex,
    out_shape: Box<[usize]>,
    datatype: DataTypeEnum,
}

impl StoredBody {
    fn capture(variant: &ExecutionVariant) -> Self {
        match variant {
            ExecutionVariant::QMatMul(op) => {
                let QMatMulOperation {
                    input_datatype,
                    input,
                    matrix: _,
                    in_shape,
                    out_shape,
                    pre_element_wise_expr,
                    post_element_wise_expr,
                    post_accumulator_offsets,
                } = op.as_ref().clone();
                StoredBody::QMatMul(StoredQMatMul {
                    input_datatype,
                    input,
                    in_shape,
                    out_shape,
                    pre_element_wise_expr,
                    post_element_wise_expr,
                    post_accumulator_offsets,
                })
            }
            ExecutionVariant::QEmbedding(op) => {
                let QEmbeddingOperation {
                    indexes,
                    matrix: _,
                    out_shape,
                    datatype,
                } = op.clone();
                StoredBody::QEmbedding(StoredQEmbedding {
                    indexes,
                    out_shape,
                    datatype,
                })
            }
            ExecutionVariant::Elementwise(_)
            | ExecutionVariant::Reduce(_)
            | ExecutionVariant::Fold(_)
            | ExecutionVariant::View(_)
            | ExecutionVariant::MatMul(_)
            | ExecutionVariant::RowProgram(_)
            | ExecutionVariant::Attention(_) => StoredBody::Plain(variant.clone()),
            ExecutionVariant::Tensor(_)
            | ExecutionVariant::QMatrix(_)
            | ExecutionVariant::Assign(_)
            | ExecutionVariant::Region(_) => {
                unreachable!("fusion templates never store this variant kind")
            }
        }
    }

    fn rebuild(&self, matrix: Option<QMatrix>) -> ExecutionVariant {
        match self {
            StoredBody::Plain(variant) => {
                debug_assert!(matrix.is_none());
                variant.clone()
            }
            StoredBody::QMatMul(stored) => {
                let StoredQMatMul {
                    input_datatype,
                    input,
                    in_shape,
                    out_shape,
                    pre_element_wise_expr,
                    post_element_wise_expr,
                    post_accumulator_offsets,
                } = stored.clone();
                ExecutionVariant::QMatMul(Box::new(QMatMulOperation {
                    input_datatype,
                    input,
                    matrix: matrix.expect("stored qmatmul template requires a matrix role"),
                    in_shape,
                    out_shape,
                    pre_element_wise_expr,
                    post_element_wise_expr,
                    post_accumulator_offsets,
                }))
            }
            StoredBody::QEmbedding(stored) => {
                let StoredQEmbedding {
                    indexes,
                    out_shape,
                    datatype,
                } = stored.clone();
                ExecutionVariant::QEmbedding(QEmbeddingOperation {
                    indexes,
                    matrix: matrix.expect("stored qembedding template requires a matrix role"),
                    out_shape,
                    datatype,
                })
            }
        }
    }
}

#[derive(Clone)]
struct StoredTemplate {
    body: StoredBody,
    dependency_roles: Vec<u32>,
    matrix_role: Option<u32>,
}

impl StoredTemplate {
    fn from_template(template: &VariantTemplate) -> Self {
        Self {
            body: StoredBody::capture(&template.variant),
            dependency_roles: template.dependency_roles.clone(),
            matrix_role: template.matrix_role,
        }
    }

    fn instantiate(&self, instance: &PlanInstance, view: &FusionView<'_>) -> ExecutionVariant {
        let matrix = self.matrix_role.map(|role| {
            let inner = instance.nodes[role as usize];
            matrix_of(
                view.variant_of(inner)
                    .expect("stored template matrix role must have a selected variant"),
            )
            .expect("stored template matrix role must remain quantized")
            .clone()
        });
        let mut variant = self.body.rebuild(matrix);
        let dependencies = self
            .dependency_roles
            .iter()
            .map(|&role| instance.nodes[role as usize])
            .collect::<Vec<_>>();
        rebind_variant_dependencies(&mut variant, &dependencies);
        variant
    }
}

#[derive(Clone)]
enum StoredDecision {
    NoRewrite,
    Rewrite(StoredTemplate),
}

/// Entries are individually cheap to regenerate; the cap only guards
/// runaway unique structure, so eviction is a wholesale reset.
const FUSION_PLAN_STORE_CAP: usize = 4096;

/// Device-scoped fusion-plan decisions keyed by [`WindowKey`], shared across
/// resolves: the first resolve to plan a window pays generation, every later
/// isomorphic window on the device — next training step, next decode token —
/// instantiates the stored template. Templates are matrix-free (see
/// [`StoredBody`]) and every hit still re-validates kills and switch cost
/// against live state, exactly like intra-resolve sharing.
#[derive(Default)]
pub(crate) struct FusionPlanStore {
    decisions: Mutex<FxHashMap<WindowKey, StoredDecision>>,
}

impl FusionPlanStore {
    /// `None`: no stored decision. `Some(None)`: stored no-rewrite.
    /// `Some(Some(variant))`: stored template instantiated for `instance`.
    pub(super) fn instantiate(
        &self,
        key: WindowKey,
        instance: &PlanInstance,
        view: &FusionView<'_>,
    ) -> Option<Option<ExecutionVariant>> {
        let decisions = self.decisions.lock();
        Some(match decisions.get(&key)? {
            StoredDecision::NoRewrite => None,
            StoredDecision::Rewrite(template) => Some(template.instantiate(instance, view)),
        })
    }

    pub(super) fn record(&self, key: WindowKey, decision: &PlanDecision) {
        let stored = match decision {
            PlanDecision::NoRewrite => StoredDecision::NoRewrite,
            PlanDecision::Rewrite(template) => {
                StoredDecision::Rewrite(StoredTemplate::from_template(template))
            }
        };
        let mut decisions = self.decisions.lock();
        if decisions.len() >= FUSION_PLAN_STORE_CAP && !decisions.contains_key(&key) {
            tracing::debug!(
                "fusion plan store reached {FUSION_PLAN_STORE_CAP} unique windows; resetting"
            );
            decisions.clear();
        }
        decisions.entry(key).or_insert(stored);
    }
}

/// Per-resolve memo. Liveness facts and read counts are part of every window
/// key, so repeated layers within this resolve share plans while windows
/// with different liveness never conflate. Per-resolve misses fall through
/// to the device-scoped [`FusionPlanStore`].
pub(super) struct FusionPlanMemo {
    atoms: Vec<PlanAtom>,
    atom_ids: FxHashMap<PlanAtom, PlanAtomId>,
    nodes: Vec<StructuralNode>,
    node_ids: FxHashMap<StructuralNode, StructuralId>,
    decisions: FxHashMap<StructuralId, PlanDecision>,
    seen_windows: FxHashSet<StructuralId>,
    stats: PlanSharingStats,
    stub_depth: u32,
    /// Spike ledger: total time spent capturing windows, accumulated only
    /// when `FUSOR_SPIKE_HOISTING` asked for it.
    capture_time: Option<std::time::Duration>,
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
            stub_depth: WINDOW_STUB_DEPTH,
            capture_time: None,
        }
    }
}

impl FusionPlanMemo {
    pub(super) fn for_config(config: &FusorConfig) -> Self {
        Self {
            stub_depth: config.spike_window_depth.unwrap_or(WINDOW_STUB_DEPTH),
            capture_time: config.spike_hoisting.then(Default::default),
            ..Default::default()
        }
    }

    pub(super) fn capture(
        &mut self,
        driver: &EGraphDriver,
        state: &ExtractState,
        view: &FusionView<'_>,
        prov: Prov,
    ) -> PlanInstance {
        let start = self.capture_time.map(|_| std::time::Instant::now());
        self.stats.windows += 1;
        let inner = driver.egraph.analysis.facts_of(prov).inner;
        let stub_depth = self.stub_depth;
        let mut builder = WindowBuilder {
            memo: self,
            driver,
            state,
            view,
            stub_depth,
            nodes: Vec::new(),
            role_of: FxHashMap::default(),
            local_ids: FxHashMap::default(),
            matrix_aliases: FxHashMap::default(),
        };
        let root = builder.add(inner, 0);
        builder.memo.seen_windows.insert(root);
        builder.memo.stats.unique_windows = builder.memo.seen_windows.len() as u64;
        let instance = PlanInstance {
            root,
            nodes: builder.nodes,
            role_of: builder.role_of,
        };
        if let (Some(total), Some(start)) = (self.capture_time.as_mut(), start) {
            *total += start.elapsed();
        }
        instance
    }

    pub(super) fn capture_time(&self) -> std::time::Duration {
        self.capture_time.unwrap_or_default()
    }

    pub(super) fn stub_depth(&self) -> u32 {
        self.stub_depth
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
    ) -> Option<&PlanDecision> {
        if !self.decisions.contains_key(&instance.root) {
            let decision = match result {
                None => {
                    self.stats.negative_templates += 1;
                    PlanDecision::NoRewrite
                }
                Some(variant) => {
                    let Some(template) = VariantTemplate::capture(variant, instance, view) else {
                        self.stats.unshareable += 1;
                        return None;
                    };
                    self.stats.templates += 1;
                    PlanDecision::Rewrite(template)
                }
            };
            self.decisions.insert(instance.root, decision);
        }
        Some(&self.decisions[&instance.root])
    }

    pub(super) fn note_store_hit(&mut self) {
        self.stats.store_hits += 1;
    }

    pub(super) fn note_store_miss(&mut self) {
        self.stats.store_misses += 1;
    }

    /// Resolve-independent identity of one planning window. Per-resolve
    /// `SpecId`s are replaced by their stable structural spec keys; roles,
    /// facts, layouts and topology hash in exactly the interned content, so
    /// equal keys reproduce equal role numbering on both sides.
    pub(super) fn window_key(&self, root: StructuralId, driver: &EGraphDriver) -> WindowKey {
        let mut lanes = TwoLane::new();
        let mut order: FxHashMap<StructuralId, u64> = FxHashMap::default();
        self.hash_window(root, driver, &mut lanes, &mut order);
        WindowKey(lanes.finish().0)
    }

    fn hash_window(
        &self,
        id: StructuralId,
        driver: &EGraphDriver,
        lanes: &mut TwoLane,
        order: &mut FxHashMap<StructuralId, u64>,
    ) {
        if let Some(&back) = order.get(&id) {
            // Shared subterm: a back-reference, disjoint from the kind tags.
            lanes.write_u64(u64::MAX);
            lanes.write_u64(back);
            return;
        }
        order.insert(id, order.len() as u64);
        let node = &self.nodes[id.0 as usize];
        let atom = &self.atoms[node.atom.0 as usize];
        lanes.write_u64(atom.role as u64);
        match atom.kind {
            PlanNodeKind::Operator(spec) => {
                lanes.write_u64(0);
                let PayloadKey(words) = driver.egraph.analysis.payloads.spec_key(spec);
                lanes.write_u64(words[0]);
                lanes.write_u64(words[1]);
            }
            PlanNodeKind::Tensor => lanes.write_u64(1),
            PlanNodeKind::Boundary => lanes.write_u64(2),
            PlanNodeKind::Missing => lanes.write_u64(3),
            PlanNodeKind::Frontier => lanes.write_u64(4),
        }
        lanes.write_u64(local_hash(|hasher| {
            atom.matrix_alias.hash(hasher);
            atom.layout.hash(hasher);
            atom.reads.hash(hasher);
            atom.cached.hash(hasher);
            atom.externally_live.hash(hasher);
            atom.is_target.hash(hasher);
        }));
        lanes.write_u64(node.children.len() as u64);
        for &child in node.children.iter() {
            self.hash_window(child, driver, lanes, order);
        }
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
    stub_depth: u32,
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
/// `FUSOR_SPIKE_WINDOW_DEPTH` widens the horizon for measurement only —
/// widening is always sound (windows only get more specific), it just costs
/// capture time and sharing.
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
        let stub = depth >= self.stub_depth;
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
    /// `None` when the rewrite reads a node the window cannot name. Epilogue
    /// generators fold a whole producer chain in one step and so reach the
    /// chain's own operands, which sit past the horizon; those roots plan
    /// fresh every visit rather than share a template that cannot say which
    /// node a later instance should rebind to.
    fn capture(
        variant: &ExecutionVariant,
        instance: &PlanInstance,
        view: &FusionView<'_>,
    ) -> Option<Self> {
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
        assert!(
            matrix_of(variant).is_none() || matrix_role.is_some(),
            "quantized fusion template matrix must be part of its structural window"
        );
        Some(Self {
            variant: variant.clone(),
            dependency_roles,
            matrix_role,
            spec: None,
        })
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
    fn plan_store_shares_templates_across_resolves() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            for step in 0..2u32 {
                let input = Tensor::new::<f32, 1, _>(&device, &[step as f32, 2.0, 3.0, 4.0]);
                let out = (&input + 1.0) * 2.0;
                let targets = [out.data().key];
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
                    let prov = driver.prov_of[&targets[0]];
                    let mut memo = FusionPlanMemo::default();
                    let instance = memo.capture(&driver, &state, &view, prov);
                    let key = memo.window_key(instance.root, &driver);
                    let store = device.fusion_plan_store();
                    let fresh = view
                        .generate_candidates(prov)
                        .into_iter()
                        .next()
                        .expect("chain fuses");
                    match store.instantiate(key, &instance, &view) {
                        None => {
                            assert_eq!(step, 0, "second resolve must hit the store");
                            let decision = memo
                                .record(&instance, &view, Some(&fresh))
                                .expect("chain rewrite stays inside its window");
                            store.record(key, decision);
                        }
                        Some(Some(rebound)) => {
                            assert_eq!(step, 1, "first resolve cannot hit an empty store");
                            let (
                                ExecutionVariant::Elementwise(rebound),
                                ExecutionVariant::Elementwise(fresh),
                            ) = (rebound, fresh)
                            else {
                                panic!("unexpected variant kinds");
                            };
                            assert_eq!(rebound, fresh);
                            assert!(
                                rebound
                                    .inputs
                                    .iter()
                                    .all(|input| { instance.role_of.contains_key(input) })
                            );
                        }
                        Some(None) => panic!("no-rewrite stored for a fusible window"),
                    }
                });
            }
        });
    }

    #[test]
    fn plan_store_rebinds_weights_across_resolves() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let weights = [dense_qmatrix(&device, 0.25), dense_qmatrix(&device, 0.5)];
            for step in 0..2usize {
                let activation = Tensor::new::<f32, 2, _>(
                    &device,
                    &[vec![
                        1.0 + step as f32,
                        -2.0,
                        3.0,
                        -4.0,
                        5.0,
                        -6.0,
                        7.0,
                        -8.0,
                    ]],
                );
                let out = activation.q_mat_mul(&weights[step]);
                let targets = [out.data().key];
                device.compute_graph().with_mut(|graph| {
                    let mut resolver = Resolver::new_batch(graph, targets.to_vec());
                    for &target in &targets {
                        resolver.build_execution_graph(graph, target);
                    }
                    resolver.recognize_contractions(graph);
                    let driver = EGraphDriver::ingest(&resolver, graph);
                    let state = ExtractState::new(&driver);
                    let ctx = FusionCtx {
                        graph,
                        layouts: std::cell::RefCell::new(Default::default()),
                    };
                    let view = FusionView::new(&driver, &state, &ctx);
                    let prov = driver.prov_of[&targets[0]];
                    let mut memo = FusionPlanMemo::default();
                    let instance = memo.capture(&driver, &state, &view, prov);
                    let key = memo.window_key(instance.root, &driver);
                    let store = device.fusion_plan_store();
                    match store.instantiate(key, &instance, &view) {
                        None => {
                            assert_eq!(step, 0, "second resolve must hit the store");
                            let variant = view.variant_of(targets[0]).unwrap().clone();
                            let decision = memo
                                .record(&instance, &view, Some(&variant))
                                .expect("qmatmul rewrite stays inside its window");
                            store.record(key, decision);
                        }
                        Some(Some(ExecutionVariant::QMatMul(rebound))) => {
                            assert_eq!(step, 1, "first resolve cannot hit an empty store");
                            assert!(std::sync::Arc::ptr_eq(
                                rebound.matrix.buffer(),
                                weights[1].buffer()
                            ));
                            assert!(!std::sync::Arc::ptr_eq(
                                rebound.matrix.buffer(),
                                weights[0].buffer()
                            ));
                            assert_eq!(rebound.input, activation.data().key);
                        }
                        other => panic!(
                            "unexpected stored decision (step {step}, some={})",
                            other.is_some()
                        ),
                    }
                });
            }
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
                resolver.recognize_contractions(graph);
                resolver.recognize_embeddings(graph);
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
                let driver = EGraphDriver::ingest(&resolver, graph);
                let state = ExtractState::new(&driver);
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
