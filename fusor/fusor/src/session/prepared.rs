//! Small, immutable plan indexes. Weak ownership prevents stale pointer reuse
//! without pinning a plan, graph, tensor buffer, or command record.
use std::sync::{Arc, Weak};

use fusor_ir::{egraph::Id, extract::Plan};
use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) struct PreparedRun {
    owner: Weak<Plan>,
    pub(crate) external: Vec<Id>,
    pub(crate) last_uses: FxHashMap<Id, usize>,
    pub(crate) outputs: FxHashMap<Id, usize>,
}

impl PreparedRun {
    pub(crate) fn matches(&self, plan: &Arc<Plan>) -> bool {
        // The weak reference keeps this allocation's address from being
        // reused. Arc::make_mut detaches weak references before mutation.
        std::ptr::eq(self.owner.as_ptr(), Arc::as_ptr(plan))
    }

    pub(crate) fn capture(plan: &Arc<Plan>, external: impl Fn(Id) -> bool) -> Option<Self> {
        if plan.launches.len() > 512
            || plan.buffers.len() > 1024
            || plan
                .launches
                .iter()
                .map(|l| l.bindings.len())
                .sum::<usize>()
                > 4096
        {
            return None;
        }
        let mut seen = FxHashSet::default();
        let mut leaves = Vec::new();
        let mut last_uses = FxHashMap::default();
        for (index, launch) in plan.launches.iter().enumerate() {
            for binding in &launch.bindings {
                let id = binding.value;
                if seen.insert(id) && external(id) {
                    leaves.push(id);
                }
                last_uses.insert(id, index);
            }
        }
        Some(Self {
            owner: Arc::downgrade(plan),
            external: leaves,
            last_uses,
            outputs: plan
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (b.value, i))
                .collect(),
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
        assert_eq!(cached.last_uses[&Id(1)], 1);
        assert_eq!(cached.last_uses[&Id(2)], 0);
        assert!(cached.matches(&plan));
        assert_eq!(Arc::strong_count(&plan), 1);
        // Same plan hash, but a changed binding must never reuse the indexes.
        Arc::make_mut(&mut plan).launches[1].bindings[0].value = Id(2);
        assert!(!cached.matches(&plan));
        let updated = PreparedRun::capture(&plan, |id| id.0 < 3).unwrap();
        assert_eq!(updated.last_uses[&Id(2)], 1);
        drop(plan);
        assert!(updated.owner.upgrade().is_none());
    }
}
