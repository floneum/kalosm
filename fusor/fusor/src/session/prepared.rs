//! Small, immutable plan indexes. Weak ownership prevents stale pointer reuse
//! without pinning a plan, graph, tensor buffer, or command record.
use std::sync::{Arc, Weak};

use fusor_ir::{egraph::Id, extract::Plan};
use rustc_hash::FxHashSet;

pub(crate) struct PreparedRun {
    pub(crate) owner: Weak<Plan>,
    pub(crate) external: Vec<Id>,
}

impl PreparedRun {
    pub(crate) fn matches(&self, plan: &Arc<Plan>) -> bool {
        // The weak reference keeps this allocation's address from being
        // reused. Arc::make_mut detaches weak references before mutation.
        std::ptr::eq(self.owner.as_ptr(), Arc::as_ptr(plan))
    }

    pub(crate) fn capture(plan: &Arc<Plan>, external: impl Fn(Id) -> bool) -> Option<Self> {
        // Bound the work and temporary set by binding visits, including empty
        // dispatches. Many small launches need no larger retained leaf index.
        plan.launches.iter().try_fold(plan.launches.len(), |n, l| {
            n.checked_add(l.bindings.len()).filter(|n| *n <= 4096)
        })?;
        let mut seen = FxHashSet::default();
        let mut leaves = Vec::new();
        for launch in &plan.launches {
            for binding in &launch.bindings {
                let id = binding.value;
                if seen.insert(id) && external(id) {
                    leaves.push(id);
                }
            }
        }
        Some(Self {
            owner: Arc::downgrade(plan),
            external: leaves,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::extract::{BindKind, BindingPlan, Dispatch};

    #[test]
    fn indexes_follow_the_immutable_plan_and_do_not_pin_it() {
        let launch = |root, ids: &[u32]| Dispatch {
            root: Id(root),
            members: smallvec::smallvec![Id(root)],
            bindings: ids
                .iter()
                .enumerate()
                .map(|(i, id)| BindingPlan {
                    binding: i as u32,
                    value: Id(*id),
                    kind: BindKind::Read,
                    arena: false,
                })
                .collect(),
            grid: [1, 1, 1],
            block: 32,
        };
        let mut plan = Arc::new(Plan {
            extraction: Default::default(),
            launches: vec![launch(3, &[1, 2, 1]), launch(4, &[1, 3])],
            buffers: Vec::new(),
            arena_bytes: 0,
            symbols: Vec::new(),
            scalar_symbols: Vec::new(),
            hash: Default::default(),
            cost: fusor_ir::cost::Picoseconds(0),
        });
        let cached = PreparedRun::capture(&plan, |id| id.0 < 3).unwrap();
        assert_eq!(cached.external, [Id(1), Id(2)]);
        assert!(cached.matches(&plan));
        assert_eq!(Arc::strong_count(&plan), 1);
        // Same plan hash, but a changed binding must never reuse the indexes.
        Arc::make_mut(&mut plan).launches[1].bindings[0].value = Id(2);
        assert!(!cached.matches(&plan));
        let updated = PreparedRun::capture(&plan, |id| id.0 < 3).unwrap();
        assert!(updated.matches(&plan));
        drop(plan);
        assert!(updated.owner.upgrade().is_none());
    }

    #[test]
    fn sparse_plans_use_the_same_bounded_leaf_index() {
        use fusor_ir::extract::{BindKind, BindingPlan, Dispatch};
        let mut plan = Arc::new(Plan {
            extraction: Default::default(),
            launches: vec![
                Dispatch {
                    root: Id(3),
                    members: smallvec::smallvec![Id(3)],
                    bindings: vec![BindingPlan {
                        binding: 1,
                        value: Id(1),
                        kind: BindKind::Read,
                        arena: false,
                    }],
                    grid: [1, 1, 1],
                    block: 32,
                };
                1024
            ],
            buffers: Vec::new(),
            arena_bytes: 0,
            symbols: Vec::new(),
            scalar_symbols: Vec::new(),
            hash: Default::default(),
            cost: fusor_ir::cost::Picoseconds(0),
        });
        let cached = PreparedRun::capture(&plan, |_| true).unwrap();
        assert_eq!(cached.external, [Id(1)]);
        assert_eq!(Arc::strong_count(&plan), 1);
        let launch = plan.launches[0].clone();
        Arc::make_mut(&mut plan).launches.resize(4097, launch);
        assert!(PreparedRun::capture(&plan, |_| true).is_none());
    }
}
