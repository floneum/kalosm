//! Order independent expression evaluation by estimated temporary pressure.
//! Operand positions and arithmetic stay unchanged. Collectives and cooperative
//! matrix operations fence scheduling, including their descendant trees.
use fusor_ir::ir::kernel::{ElementType, TileExpr, TileExprKind as K};
use rustc_hash::FxHashMap;

#[derive(Default)]
pub(super) struct Pressure(FxHashMap<TileExpr, Option<u32>>);

pub(super) fn result_words(e: &TileExpr) -> u32 {
    if matches!(e.kind(), K::Literal(_)) {
        return 0;
    }
    match e.element() {
        ElementType::Scalar(_) => 1,
        ElementType::Vector { lanes, .. } => lanes,
        ElementType::CoopMatrix { .. } => 0,
    }
}

impl Pressure {
    pub(super) fn get(&mut self, e: &TileExpr) -> Option<u32> {
        if let Some(value) = self.0.get(e) {
            return *value;
        }
        let result = if matches!(
            e.kind(),
            K::Reduce { .. } | K::CoopLoad { .. } | K::CoopMma { .. } | K::CoopZero { .. }
        ) {
            None
        } else {
            let mut children = Vec::new();
            e.kind()
                .visit_children(&mut |child| children.push((self.get(child), result_words(child))));
            let children: Option<Vec<_>> = children
                .into_iter()
                .map(|(peak, words)| peak.map(|p| (p, words)))
                .collect();
            children.map(|mut children| {
                children
                    .sort_by_key(|&(peak, words)| std::cmp::Reverse(peak.saturating_sub(words)));
                let mut held = 0u32;
                let mut peak = result_words(e);
                for (child_peak, words) in children {
                    peak = peak.max(held.saturating_add(child_peak));
                    held = held.saturating_add(words);
                }
                peak.max(held)
            })
        };
        self.0.insert(e.clone(), result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::ir::kernel::{Builtin, ReduceKind, TileReduceOp};
    use fusor_tile::build::Kernel;

    #[test]
    fn collectives_fence_the_entire_operand_tree() {
        let b = Kernel::new();
        let value = b.cast(
            b.builtin(Builtin::Lane),
            fusor_ir::ir::kernel::ScalarElement::F32.element(),
        );
        let sum = b.reduce(TileReduceOp::Sum, ReduceKind::Subgroup, value);
        let nested = b.add(b.f32(1.), b.mul(b.f32(2.), sum));
        assert_eq!(Pressure::default().get(&nested), None);
    }
}
