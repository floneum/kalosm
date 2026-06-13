//! Access-pattern analysis over fused n-ary expressions.
//!
//! Every tiled lowering asks the same two questions about an expression's
//! inputs: *which index-space dims does each input read* (the index lists),
//! and *which of those reads actually vary* (stride-aware — a broadcast view
//! reads a dim through a zero stride, which is reuse, not dependence). The
//! answers drive tile-shape selection for reductions (which inputs stage
//! through workgroup memory or registers) and for elementwise kernels (which
//! loads hoist out of a thread's output run).

use crate::{nary_direct::TensorMeta, nary_wise::NaryExpr};

/// Per-input access structure for an expression whose inputs are all read at
/// consistent, pure `DimIndex` coordinates.
pub(crate) struct InputAccesses {
    /// Per input: the index-space dim read by each input dimension.
    pub(crate) dims: Vec<Vec<usize>>,
    /// Per input: the index-space dims the input *effectively* depends on. A
    /// stride-0 or size-1 input dim is not a dependence — `layout_index`
    /// drops it from addressing too — so broadcast views analyze the same as
    /// index lists that never mention the dim.
    pub(crate) effective: Vec<Vec<usize>>,
}

impl InputAccesses {
    /// Collect access structure, or `None` for any access a tiled kernel
    /// cannot hoist or stage: non-`DimIndex` coordinates (gathers),
    /// inconsistent index lists for one input, or `DimIndex` leaves in value
    /// position (the value depends on the coordinate itself).
    pub(crate) fn collect(
        expr: &NaryExpr,
        input_count: usize,
        metas: &[TensorMeta],
    ) -> Option<Self> {
        let mut accesses = vec![None; input_count];
        collect_accesses(expr, &mut accesses)?;
        let dims = accesses.into_iter().collect::<Option<Vec<Vec<usize>>>>()?;
        let effective = dims
            .iter()
            .zip(metas)
            .map(|(dims, meta)| {
                dims.iter()
                    .enumerate()
                    .filter_map(|(j, &d)| {
                        let stride = meta.strides.get(j).copied().unwrap_or(0);
                        let size = meta.shape.get(j).copied().unwrap_or(1);
                        (stride != 0 && size > 1).then_some(d)
                    })
                    .collect()
            })
            .collect();
        Some(Self { dims, effective })
    }

    /// Whether input `i` effectively depends on index-space dim `d`.
    pub(crate) fn depends_on(&self, i: usize, d: usize) -> bool {
        self.effective[i].contains(&d)
    }
}

fn collect_accesses(expr: &NaryExpr, accesses: &mut Vec<Option<Vec<usize>>>) -> Option<()> {
    match expr {
        NaryExpr::Op { children, .. } => {
            for child in children {
                collect_accesses(child, accesses)?;
            }
            Some(())
        }
        NaryExpr::IndexedInput { input_idx, indices } => {
            let dims = indices
                .iter()
                .map(|index| match index {
                    NaryExpr::DimIndex(dim) => Some(*dim),
                    _ => None,
                })
                .collect::<Option<Vec<usize>>>()?;
            match &accesses[*input_idx] {
                Some(existing) if *existing != dims => None,
                Some(_) => Some(()),
                None => {
                    accesses[*input_idx] = Some(dims);
                    Some(())
                }
            }
        }
        NaryExpr::DimIndex(_) => None,
        NaryExpr::Scalar(_) => Some(()),
    }
}
