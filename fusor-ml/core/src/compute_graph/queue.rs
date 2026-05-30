use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use super::NodeIndex;

#[derive(Default, Debug)]
pub(crate) struct ComputeQueue {
    nodes: VecDeque<NodeIndex>,
    set: FxHashSet<NodeIndex>,
}

impl ComputeQueue {
    pub(crate) fn push_back(&mut self, key: NodeIndex) {
        if self.set.insert(key) {
            self.nodes.push_back(key);
        }
    }

    pub(crate) fn pop_front(&mut self) -> Option<NodeIndex> {
        self.nodes.pop_front().inspect(|key| {
            self.set.remove(key);
        })
    }
}
