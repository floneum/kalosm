//! Cross-lane reductions on CPU: `Reduce{Subgroup}` becomes a horizontal
//! reduce of one register by `log2(W)` shuffle-reduce steps, and
//! `Reduce{Workgroup}` becomes a tree over the thread-local scratch tile,
//! with the segment split supplying the barrier semantics.

use fusor2_ir::ir::kernel::TileReduceOp;

use super::expr::Reg;

/// The identity of a reduction operator in f32.
#[inline(always)]
pub(crate) const fn identity_f32(op: TileReduceOp) -> f32 {
    match op {
        TileReduceOp::Sum => 0.0,
        TileReduceOp::Product => 1.0,
        TileReduceOp::Max => f32::NEG_INFINITY,
        TileReduceOp::Min => f32::INFINITY,
    }
}

#[inline(always)]
pub(crate) fn combine_f32(op: TileReduceOp, a: f32, b: f32) -> f32 {
    match op {
        TileReduceOp::Sum => a + b,
        TileReduceOp::Product => a * b,
        TileReduceOp::Max => {
            if b > a {
                b
            } else {
                a
            }
        }
        TileReduceOp::Min => {
            if b < a {
                b
            } else {
                a
            }
        }
    }
}

/// Elementwise combine of two registers.
#[inline(always)]
pub(crate) fn combine_reg<const W: usize>(op: TileReduceOp, a: Reg<W>, b: Reg<W>) -> Reg<W> {
    a.zipf(b, |x, y| combine_f32(op, x, y))
}

/// Horizontal reduce of one register by `log2(W)` shuffle-reduce steps, with
/// the result broadcast back across every lane.
#[inline(always)]
pub(crate) fn horizontal<const W: usize>(op: TileReduceOp, v: Reg<W>) -> Reg<W> {
    let mut cur = v.to_f();
    let mut half = W / 2;
    while half >= 1 {
        let mut next = [0f32; W];
        for k in 0..W {
            let partner = k ^ half;
            next[k] = combine_f32(op, cur[k], cur[partner]);
        }
        cur = next;
        if half == 1 {
            break;
        }
        half /= 2;
    }
    Reg::from_f(cur)
}

/// Horizontal reduce honouring an active-lane mask: inactive lanes contribute
/// the identity.
#[inline(always)]
pub(crate) fn horizontal_masked<const W: usize>(op: TileReduceOp, v: Reg<W>, mask: Reg<W>) -> Reg<W> {
    let id = Reg::<W>::splat_f32(identity_f32(op));
    horizontal(op, Reg::select(mask, v, id))
}

/// Tree-reduce `values[0..len]` in groups of `group`, writing each group's
/// result back over the whole group. This is the `Reduce{Workgroup}` body; the
/// preceding segment split is what makes the staged writes visible.
pub(crate) fn tree_in_place(op: TileReduceOp, values: &mut [f32], len: usize, group: usize) {
    let group = group.max(1);
    let mut base = 0;
    while base < len {
        let hi = (base + group).min(len);
        let mut acc = identity_f32(op);
        for v in &values[base..hi] {
            acc = combine_f32(op, acc, *v);
        }
        for v in &mut values[base..hi] {
            *v = acc;
        }
        base = hi;
    }
}
