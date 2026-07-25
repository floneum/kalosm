//! Deterministic ingestion: execution graph → e-graph.
//!
//! Traversal mirrors `build_execution_graph` (and the flush-replay
//! fingerprint): targets in order, dependencies in `visit_dependencies`
//! order, provenance assigned at first visit (pre-order). Payloads are
//! interned byte-faithful — no dedup, no normalization — because recognition
//! matchers depend on the exact API-emitted forms.
//!
//! Inputs outside the execution graph (cached before this resolve, the
//! `resolved_set`) become opaque [`FusorLang::Boundary`] leaves: the
//! structural equivalent of every `check_cached` guard — fusion rules only
//! match operation e-nodes, and a cached producer simply is not one.

use egg::{EGraph, Id};
use rustc_hash::FxHashSet;

use super::super::{ExecutionVariant, Resolver};
use super::EGraphDriver;
use super::analysis::{FusorAnalysis, NodeFacts};
use super::interner::{SpecId, variant_dependencies};
use super::lang::{AllocationId, FusorLang, Prov};
use crate::compute_graph::{ComputeGraphInner, NodeIndex};

/// Build the e-node for `variant` with the given children (in
/// `visit_dependencies` order). Shared by ingestion and rule appliers so
/// the payload/children lockstep has one owner.
pub(super) fn enode_for(
    analysis: &mut FusorAnalysis,
    variant: &ExecutionVariant,
    prov: Prov,
    children: Vec<Id>,
    dedup: bool,
    known_spec: Option<SpecId>,
) -> FusorLang {
    let intern = |analysis: &mut FusorAnalysis, variant: &ExecutionVariant| {
        if let Some(spec) = known_spec {
            analysis
                .payloads
                .push_unique_with_spec(variant.clone(), spec)
        } else if dedup {
            analysis.payloads.intern(variant.clone())
        } else {
            analysis.payloads.push_unique(variant.clone())
        }
    };
    match variant {
        ExecutionVariant::Tensor(data) => {
            debug_assert!(children.is_empty());
            FusorLang::TensorLeaf(AllocationId(std::sync::Arc::as_ptr(data.buffer()) as usize))
        }
        ExecutionVariant::QMatrix(op) => {
            debug_assert!(children.is_empty());
            FusorLang::QMatrixLeaf(
                AllocationId(std::sync::Arc::as_ptr(op.matrix.buffer()) as usize),
                intern(analysis, variant),
            )
        }
        ExecutionVariant::Elementwise(_) => {
            FusorLang::Elementwise(intern(analysis, variant), children.into_boxed_slice())
        }
        ExecutionVariant::Reduce(_) => {
            FusorLang::Reduce(intern(analysis, variant), children.into_boxed_slice())
        }
        ExecutionVariant::View(_) => FusorLang::View(intern(analysis, variant), [children[0]]),
        ExecutionVariant::Assign(_) => {
            FusorLang::Assign(prov, intern(analysis, variant), [children[0], children[1]])
        }
        ExecutionVariant::Region(_) => {
            FusorLang::Region(prov, intern(analysis, variant), children.into_boxed_slice())
        }
        ExecutionVariant::MatMul(_) => {
            FusorLang::MatMul(intern(analysis, variant), [children[0], children[1]])
        }
        ExecutionVariant::QMatMul(_) => {
            FusorLang::QMatMul(intern(analysis, variant), children.into_boxed_slice())
        }
        ExecutionVariant::QEmbedding(_) => {
            FusorLang::QEmbedding(intern(analysis, variant), [children[0]])
        }
        ExecutionVariant::RowProgram(_) | ExecutionVariant::Attention(_) => {
            FusorLang::RowProgram(intern(analysis, variant), children.into_boxed_slice())
        }
    }
}

impl EGraphDriver {
    /// Ingest the resolver's execution graph reachable from its targets.
    pub(super) fn ingest(resolver: &Resolver, graph: &ComputeGraphInner) -> Self {
        let mut driver = EGraphDriver {
            egraph: EGraph::new(FusorAnalysis {
                facts: Vec::new(),
                payloads: Default::default(),
                class_of_inner: Default::default(),
            }),
            class_of: Vec::new(),
            identity_enodes: Vec::new(),
            identity_variants: Vec::new(),
            prov_of: Default::default(),
            provs_of_class: Default::default(),
        };
        let target_set: FxHashSet<NodeIndex> = resolver.targets.iter().copied().collect();

        enum Frame {
            Enter(NodeIndex),
            Exit { inner: NodeIndex, prov: Prov },
        }
        let mut stack = Vec::new();
        // Nodes whose Exit frame has not run yet. Re-entering one means the
        // execution graph has a dependency cycle, which the placeholder
        // class slots downstream would turn into an opaque unionfind panic.
        #[cfg(debug_assertions)]
        let mut open: FxHashSet<NodeIndex> = FxHashSet::default();
        for &target in resolver.targets.iter().rev() {
            stack.push(Frame::Enter(target));
        }
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(inner) => {
                    if driver.prov_of.contains_key(&inner) {
                        #[cfg(debug_assertions)]
                        if open.contains(&inner) {
                            let describe = |node: NodeIndex| {
                                resolver
                                    .node_mapping
                                    .get(&node)
                                    .map(|&exec| {
                                        format!("{:?}", resolver.execution_graph[exec].variant)
                                    })
                                    .unwrap_or_else(|| "<boundary>".into())
                            };
                            let variant = describe(inner);
                            let deps = resolver
                                .node_mapping
                                .get(&inner)
                                .map(|&exec| {
                                    variant_dependencies(&resolver.execution_graph[exec].variant)
                                        .into_iter()
                                        .map(|dep| format!("{dep:?}: {}", describe(dep)))
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            let spine: Vec<String> = stack
                                .iter()
                                .filter_map(|frame| match frame {
                                    Frame::Exit { inner, .. } => {
                                        let mut text = describe(*inner);
                                        text.truncate(120);
                                        Some(format!("{inner:?}: {text}"))
                                    }
                                    _ => None,
                                })
                                .collect();
                            panic!(
                                "dependency cycle through inner node {inner:?} ({variant}) reached \
                                 egraph ingest; dependencies: {deps:#?}; open spine: {spine:#?}"
                            );
                        }
                        continue;
                    }
                    let exec = resolver.node_mapping.get(&inner).copied();
                    let Some(exec_idx) = exec else {
                        // Cached before this resolve (or otherwise excluded
                        // from the execution graph): opaque boundary leaf.
                        let prov = driver.alloc_prov(
                            inner,
                            NodeFacts {
                                inner,
                                exec: None,
                                externally_live: graph.has_live_reference(inner),
                                is_target: target_set.contains(&inner),
                            },
                        );
                        let allocation = graph
                            .get_cached_result(inner)
                            .map(
                                |data| AllocationId(std::sync::Arc::as_ptr(data.buffer()) as usize),
                            )
                            .unwrap_or(AllocationId(inner.index()));
                        let boundary = FusorLang::Boundary(allocation);
                        let id = driver.egraph.add(boundary.clone());
                        driver.egraph.analysis.class_of_inner.insert(inner, id);
                        driver.class_of.push(id);
                        driver.identity_enodes.push(boundary);
                        driver.identity_variants.push(None);
                        debug_assert_eq!(driver.class_of.len(), prov.0 as usize + 1);
                        continue;
                    };
                    let prov = driver.alloc_prov(
                        inner,
                        NodeFacts {
                            inner,
                            exec: Some(exec_idx),
                            externally_live: graph.has_live_reference(inner),
                            is_target: target_set.contains(&inner),
                        },
                    );
                    // Reserve the class slot now (pre-order prov => index
                    // into class_of); filled on exit. Safe placeholder: a
                    // consumer only reads a dependency's slot after that
                    // dependency's Exit frame ran (DAG + stack ordering),
                    // and the validation pass below re-checks every slot.
                    driver.class_of.push(Id::from(0usize));
                    driver
                        .identity_enodes
                        .push(FusorLang::Boundary(AllocationId(inner.index())));
                    driver.identity_variants.push(None);
                    #[cfg(debug_assertions)]
                    open.insert(inner);
                    stack.push(Frame::Exit { inner, prov });
                    let deps = variant_dependencies(&resolver.execution_graph[exec_idx].variant);
                    for &dep in deps.iter().rev() {
                        stack.push(Frame::Enter(dep));
                    }
                }
                Frame::Exit { inner, prov } => {
                    #[cfg(debug_assertions)]
                    open.remove(&inner);
                    let exec_idx = resolver.node_mapping[&inner];
                    let variant = resolver.execution_graph[exec_idx].variant.clone();
                    let children: Vec<Id> = variant_dependencies(&variant)
                        .into_iter()
                        .map(|dep| driver.class_of[driver.prov_of[&dep].0 as usize])
                        .collect();
                    let enode = enode_for(
                        &mut driver.egraph.analysis,
                        &variant,
                        prov,
                        children,
                        true,
                        None,
                    );
                    driver.identity_enodes[prov.0 as usize] = enode.clone();
                    driver.identity_variants[prov.0 as usize] = Some(variant);
                    let id = driver.egraph.add(enode);
                    driver.class_of[prov.0 as usize] = id;
                    driver.egraph.analysis.class_of_inner.insert(inner, id);
                }
            }
        }
        driver.egraph.rebuild();
        driver.refresh_prov_classes();
        debug_assert_eq!(
            driver.class_of.len(),
            driver.egraph.analysis.facts.len(),
            "one class per provenance"
        );
        #[cfg(debug_assertions)]
        for (index, &id) in driver.class_of.iter().enumerate() {
            let class = driver.egraph.find(id);
            assert!(
                driver.provs_of_class[&class].contains(&Prov(index as u32)),
                "class slot must contain its own provenance"
            );
        }
        driver
    }

    fn alloc_prov(&mut self, inner: NodeIndex, facts: NodeFacts) -> Prov {
        let prov = Prov(self.egraph.analysis.facts.len() as u32);
        self.egraph.analysis.facts.push(facts);
        self.prov_of.insert(inner, prov);
        prov
    }

    /// The (canonical) e-class of an ingested inner node.
    pub(super) fn class_for(&self, inner: NodeIndex) -> Option<Id> {
        self.prov_of
            .get(&inner)
            .map(|prov| self.egraph.find(self.class_of[prov.0 as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// Build the execution graph for `targets` exactly like resolve pass 1,
    /// ingest it, and hand everything to the assertion closure.
    fn with_ingested<R>(
        device: &Device,
        targets: &[&Tensor],
        f: impl FnOnce(&Resolver, &ComputeGraphInner, EGraphDriver) -> R,
    ) -> R {
        let keys: Vec<NodeIndex> = targets.iter().map(|t| t.data().key).collect();
        device.compute_graph().with_mut(|inner| {
            let mut resolver = Resolver::new_batch(inner, keys.clone());
            for &key in &keys {
                resolver.build_execution_graph(inner, key);
            }
            let driver = EGraphDriver::ingest(&resolver, inner);
            f(&resolver, inner, driver)
        })
    }

    fn interesting_graph(device: &Device) -> (Tensor, Tensor) {
        let rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 8];
        let input = Tensor::new::<f32, 2, _>(device, &rows);
        let weight_rows = vec![vec![0.25f32, 0.5, 0.75, 1.0]; 4];
        let weight = Tensor::new::<f32, 2, _>(device, &weight_rows);
        // Elementwise -> composed matmul (views + multiply + sum) -> reduce:
        // covers Tensor leaves, View, Elementwise, and Reduce variants.
        let x = (&input * 2.0) + 1.0;
        let m = x.mat_mul(&weight);
        let s = m.sum(1);
        (s, m)
    }

    fn dump(driver: &EGraphDriver) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (index, &id) in driver.class_of.iter().enumerate() {
            let facts = &driver.egraph.analysis.facts[index];
            let class = &driver.egraph[driver.egraph.find(id)];
            let mut nodes: Vec<String> =
                class.nodes.iter().map(|node| format!("{node:?}")).collect();
            nodes.sort();
            writeln!(
                out,
                "prov={index} inner={} exec={:?} live={} target={} nodes={nodes:?}",
                facts.inner.index(),
                facts.exec.map(|e| e.index()),
                facts.externally_live,
                facts.is_target,
            )
            .unwrap();
        }
        out
    }

    #[test]
    fn ingestion_covers_execution_graph() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let (target, _mid) = interesting_graph(&device);
            with_ingested(&device, &[&target], |resolver, _inner, driver| {
                let non_boundary = driver
                    .egraph
                    .analysis
                    .facts
                    .iter()
                    .filter(|facts| facts.exec.is_some())
                    .count();
                assert_eq!(
                    non_boundary,
                    resolver.execution_graph.node_count(),
                    "every execution node ingests exactly once"
                );
                // Identity-only ingestion: one e-node per provenance.
                assert_eq!(
                    driver.egraph.total_number_of_nodes(),
                    driver.egraph.analysis.facts.len(),
                );
                assert!(driver.class_for(resolver.targets[0]).is_some());
                let target_prov = driver.prov_of[&resolver.targets[0]];
                assert!(driver.egraph.analysis.facts_of(target_prov).is_target);
            });
        });
    }

    #[test]
    fn ingestion_is_deterministic() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let (target, _mid) = interesting_graph(&device);
            let first = with_ingested(&device, &[&target], |_, _, driver| dump(&driver));
            let second = with_ingested(&device, &[&target], |_, _, driver| dump(&driver));
            assert_eq!(
                first, second,
                "ingestion must be a pure function of the graph"
            );
        });
    }

    #[test]
    fn cached_inputs_become_boundaries() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 8];
            let input = Tensor::new::<f32, 2, _>(&device, &rows);
            let x = (&input * 2.0) + 1.0;
            // Materialize `x` so it is cached before the next resolve starts.
            let _ = x.data().materialize();
            let y = x.sin();
            with_ingested(&device, &[&y], |_resolver, _inner, driver| {
                let x_prov = driver.prov_of[&x.data().key];
                let facts = driver.egraph.analysis.facts_of(x_prov);
                assert!(
                    facts.exec.is_none(),
                    "cached producer must ingest as an opaque boundary leaf"
                );
                let class = &driver.egraph[driver.egraph.find(driver.class_of[x_prov.0 as usize])];
                assert!(matches!(class.nodes.as_slice(), [FusorLang::Boundary(..)]));
            });
        });
    }

    #[test]
    fn equivalent_pure_nodes_share_an_eclass() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = &input * 2.0;
            let right = &input * 2.0;
            with_ingested(&device, &[&left, &right], |_, _, driver| {
                let left_class = driver.class_for(left.data().key).unwrap();
                let right_class = driver.class_for(right.data().key).unwrap();
                assert_eq!(left_class, right_class);
                assert_eq!(driver.provs_of_class[&left_class].len(), 2);
            });
        });
    }

    #[test]
    fn class_resolution_selects_a_needed_equivalent_observation() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = &input * 2.0;
            let right = &input * 2.0;
            with_ingested(&device, &[&left, &right], |_, _, driver| {
                let class = driver.class_for(left.data().key).unwrap();
                let provenances = &driver.provs_of_class[&class];
                assert_eq!(provenances.len(), 2);
                let mut needed = vec![true; driver.egraph.analysis.facts.len()];
                needed[provenances[0].0 as usize] = false;

                let selected = driver.prov_of_class(class, &needed);
                assert_eq!(selected, provenances[1]);
            });
        });
    }

    #[test]
    fn allocation_identity_distinguishes_equal_contents() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let a = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let b = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = &a * 2.0;
            let right = &b * 2.0;
            with_ingested(&device, &[&left, &right], |_, _, driver| {
                assert_ne!(
                    driver.class_for(left.data().key),
                    driver.class_for(right.data().key)
                );
            });
        });
    }

    #[test]
    fn congruence_shares_equivalent_nested_subgraphs() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = (&input * 2.0) + 1.0;
            let right = (&input * 2.0) + 1.0;
            with_ingested(&device, &[&left, &right], |_, _, driver| {
                assert_eq!(
                    driver.class_for(left.data().key),
                    driver.class_for(right.data().key)
                );
            });
        });
    }

    #[test]
    fn shared_eclass_coalesces_to_one_execution_node() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = &input * 2.0;
            let right = &input * 2.0;
            let targets = vec![left.data().key, right.data().key];
            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.clone());
                for &target in &targets {
                    resolver.build_execution_graph(graph, target);
                }
                let driver = EGraphDriver::ingest(&resolver, graph);
                resolver.coalesce_equivalent_eclasses(graph, &driver);
                assert_eq!(resolver.execution_graph.node_count(), 2);
                assert_eq!(
                    resolver
                        .shared_outputs
                        .values()
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>(),
                    vec![targets[1]]
                );
            });
        });
    }

    #[test]
    fn nested_shared_eclasses_coalesce_without_stale_execution_nodes() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let mut left = &input * 2.0;
            let mut right = &input * 2.0;
            const DEPTH: usize = 2;
            for _ in 0..DEPTH {
                left = &left + 1.0;
                right = &right + 1.0;
            }
            let targets = vec![left.data().key, right.data().key];
            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.clone());
                for &target in &targets {
                    resolver.build_execution_graph(graph, target);
                }

                let driver = EGraphDriver::ingest(&resolver, graph);
                resolver.coalesce_equivalent_eclasses(graph, &driver);

                assert_eq!(resolver.execution_graph.node_count(), DEPTH + 2);
                assert!(
                    resolver
                        .node_mapping
                        .values()
                        .all(|&execution| { resolver.execution_graph.contains_node(execution) })
                );
                assert_eq!(
                    resolver
                        .shared_outputs
                        .values()
                        .map(Vec::len)
                        .sum::<usize>(),
                    DEPTH + 1
                );
            });
        });
    }
}
