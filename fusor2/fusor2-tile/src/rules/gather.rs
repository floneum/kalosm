//! The three gather lowerings. `index_select`, `embedding`, `gather_last`
//! and `i()` are all one `L0::Gather`, so they share these three
//! alternatives.
//!
//! Owned by W4.

use fusor2_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor2_ir::ir::level0::L0;
use fusor2_ir::ir::level1::{GatherMode, IndexSpace, L1, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::shape::Dim;

use crate::domains::{DomainCtx, default_planner, map_domain};
use crate::rules::contract::alias;

rule!(
    GATHER_ROW_PER_GROUP,
    level = Level::L0,
    head = OpTag::Gather,
    tag = RuleTag::StrictlyLowering,
    apply = gather_row_per_group,
);

rule!(
    GATHER_VECTORIZED,
    level = Level::L0,
    head = OpTag::Gather,
    tag = RuleTag::StrictlyLowering,
    apply = gather_vectorized,
);

/// A vectorized row load moves whole 16-byte quads.
const VECTOR_BYTES: u64 = 16;

fn parts(node: &Node) -> Option<(u32, Id, Id)> {
    match &node.op {
        Op::L0(L0::Gather { axis, x, idx }) => Some((*axis, *x, *idx)),
        _ => None,
    }
}

/// Bytes in one gathered row: everything inside the gathered axis.
fn row_bytes(f: &Facts<'_>, axis: u32) -> Option<u64> {
    let x = f.operand(0)?;
    let mut elems: u64 = 1;
    for d in x.shape.iter().skip(axis as usize + 1) {
        elems = elems.checked_mul(d.as_const()?)?;
    }
    elems.checked_mul(x.dtype.byte_size())
}

fn mint(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>, mode: GatherMode) -> Option<Id> {
    let (axis, x_id, idx_id) = parts(node)?;
    let x = f.operand(0)?;
    let idx = f.operand(1)?;
    let out: Vec<Dim> = f.own().shape.iter().copied().collect();
    let x_op = alias(x_id, x);
    let idx_op = alias(idx_id, idx);
    let cx = DomainCtx::new(f.caps(), default_planner());
    let accesses = [x_op.access.clone(), idx_op.access.clone()];
    let op = L1::KGather {
        space: IndexSpace::new(out.iter().copied()),
        axis,
        mode,
        ops: vec![x_op, idx_op],
        sched: ScheduleDomain::Map(map_domain(&out, &accesses, &cx)),
    };
    let new = b.add_l1(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// One workgroup per gathered row. The universal form: no divisibility, no
/// capability, no layout requirement.
pub fn gather_row_per_group(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    parts(node)?;
    if f.operand(0)?.dtype.is_quantized() {
        return None;
    }
    mint(b, id, node, f, GatherMode::RowPerGroup)
}

/// Legal when a gathered row is a whole number of 16-byte quads and its
/// elements are unit-stride — a vector load cannot straddle a row edge or
/// a stride.
pub fn gather_vectorized(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let (axis, _, _) = parts(node)?;
    if f.operand(0)?.dtype.is_quantized() {
        return None;
    }
    let bytes = row_bytes(f, axis)?;
    if bytes == 0 || !bytes.is_multiple_of(VECTOR_BYTES) {
        return None;
    }
    mint(b, id, node, f, GatherMode::Vectorized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::apple_caps;
    use crate::rules::TILE_RULES;
    use crate::rules::testing::{Fixture, l1_of};
    use fusor2_ir::dtype::Dtype;

    fn modes(fx: &Fixture, id: Id) -> Vec<GatherMode> {
        fx.chain(id)
            .into_iter()
            .filter_map(|m| match l1_of(fx, m) {
                Some(L1::KGather { mode, .. }) => Some(mode),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn three_modes_for_dense_unit_stride_row() {
        let mut fx = Fixture::new(apple_caps());
        // 1024 rows of 8 f32 = 32 bytes per row, two whole quads.
        let table = fx.buffer(Dtype::F32, &[1024, 8]);
        let idx = fx.buffer(Dtype::U32, &[128]);
        let g = fx.gather(0, table, idx);
        fx.apply_all(TILE_RULES, g);

        let modes = modes(&fx, g);
        assert!(modes.contains(&GatherMode::RowPerGroup));
        assert!(modes.contains(&GatherMode::Vectorized));
        // The L0 gather plus its two lowerings.
        assert_eq!(fx.chain(g).len(), 3);
    }

    #[test]
    fn a_short_row_declines_the_vector_load() {
        let mut fx = Fixture::new(apple_caps());
        // 3 f32 = 12 bytes: not a whole quad.
        let table = fx.buffer(Dtype::F32, &[1024, 3]);
        let idx = fx.buffer(Dtype::U32, &[128]);
        let g = fx.gather(0, table, idx);
        fx.apply_all(TILE_RULES, g);
        let modes = modes(&fx, g);
        assert_eq!(modes, vec![GatherMode::RowPerGroup]);
    }
}
