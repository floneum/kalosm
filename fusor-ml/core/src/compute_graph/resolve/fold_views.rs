use crate::view::ViewOperation;

use super::*;

impl Resolver {
    /// Rewrite every access to input `target_idx` through `view`'s
    /// coordinate map: the original index expressions (the view's output
    /// coordinates) walk down the stage stack to base coordinates, with
    /// fill selects and in-bounds clamps where stages are partially defined
    /// (both select branches evaluate).
    pub(super) fn rewrite_view_input(
        expr: &NaryExpr,
        target_idx: usize,
        view: &ViewOperation,
    ) -> Option<NaryExpr> {
        Some(match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::rewrite_view_input(child, target_idx, view))
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let indices: Vec<NaryExpr> = indices
                    .iter()
                    .map(|index| Self::rewrite_view_input(index, target_idx, view))
                    .collect::<Option<Vec<_>>>()?;
                if *input_idx != target_idx {
                    NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices,
                    }
                } else {
                    view.value_expression(*input_idx, &indices)?.0
                }
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        })
    }
}

/// The worst re-read factor across this slot's loads: the product of
/// index-space dims a load's coordinates never reference — each such dim
/// re-reads the same element once per step.
pub(super) fn input_reread_factor(expr: &NaryExpr, shape: &[usize], slot: usize) -> usize {
    fn collect_dims(expr: &NaryExpr, referenced: &mut [bool]) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    collect_dims(child, referenced);
                }
            }
            NaryExpr::IndexedInput { indices, .. } => {
                for index in indices {
                    collect_dims(index, referenced);
                }
            }
            NaryExpr::DimIndex(dim) => referenced[*dim] = true,
            NaryExpr::Scalar(_) => {}
        }
    }
    fn visit_loads(expr: &NaryExpr, shape: &[usize], slot: usize, worst: &mut usize) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    visit_loads(child, shape, slot, worst);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                for index in indices {
                    visit_loads(index, shape, slot, worst);
                }
                if *input_idx == slot {
                    let mut referenced = vec![false; shape.len()];
                    for index in indices {
                        collect_dims(index, &mut referenced);
                    }
                    let factor: usize = shape
                        .iter()
                        .zip(&referenced)
                        .filter(|(_, referenced)| !**referenced)
                        .map(|(size, _)| *size)
                        .product();
                    *worst = (*worst).max(factor);
                }
            }
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => {}
        }
    }
    let mut worst = 1;
    visit_loads(expr, shape, slot, &mut worst);
    worst
}
