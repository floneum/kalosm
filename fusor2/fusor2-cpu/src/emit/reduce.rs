//! Cross-lane reductions on CPU: `Reduce{Subgroup}` becomes a horizontal
//! reduce of one register by `log2(W)` shuffle-reduce steps, and
//! `Reduce{Workgroup}` becomes a tree over the thread-local scratch tile, with
//! the segment split supplying the barrier semantics.

use fusor2_ir::ir::level2::TileReduceOp;

use super::expr::Reg;

/// The identity of a reduction operator in f32.
#[inline(always)]
pub const fn identity_f32(op: TileReduceOp) -> f32 {
    match op {
        TileReduceOp::Sum => 0.0,
        TileReduceOp::Product => 1.0,
        TileReduceOp::Max => f32::NEG_INFINITY,
        TileReduceOp::Min => f32::INFINITY,
    }
}

#[inline(always)]
pub fn combine_f32(op: TileReduceOp, a: f32, b: f32) -> f32 {
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
pub fn combine_reg<const W: usize>(op: TileReduceOp, a: Reg<W>, b: Reg<W>) -> Reg<W> {
    a.zipf(b, |x, y| combine_f32(op, x, y))
}

/// Horizontal reduce of one register by `log2(W)` butterfly steps, with the
/// result broadcast back across every lane.
#[inline(always)]
pub fn horizontal<const W: usize>(op: TileReduceOp, v: Reg<W>) -> Reg<W> {
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
pub fn horizontal_masked<const W: usize>(op: TileReduceOp, v: Reg<W>, mask: Reg<W>) -> Reg<W> {
    let id = Reg::<W>::splat_f32(identity_f32(op));
    horizontal(op, Reg::select(mask, v, id))
}

/// Tree-reduce `values[0..len]` in groups of `group`, writing each group's
/// result back over the whole group. The `Reduce{Workgroup}` body; the
/// preceding segment split makes the staged writes visible.
pub fn tree_in_place(op: TileReduceOp, values: &mut [f32], len: usize, group: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_reduce_broadcasts_the_result() {
        let v: Reg<8> = Reg::from_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let s = horizontal(TileReduceOp::Sum, v);
        for k in 0..8 {
            assert_eq!(s.f(k), 36.0, "lane {k}");
        }
        let m = horizontal(TileReduceOp::Max, v);
        assert_eq!(m.f(0), 8.0);
        let n = horizontal(TileReduceOp::Min, v);
        assert_eq!(n.f(3), 1.0);
        let p: Reg<4> = Reg::from_f([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(horizontal(TileReduceOp::Product, p).f(0), 24.0);
    }

    #[test]
    fn masked_horizontal_ignores_inactive_lanes() {
        let v: Reg<4> = Reg::from_f([1.0, 2.0, 3.0, 4.0]);
        let m: Reg<4> = Reg([u32::MAX, u32::MAX, 0, 0]);
        assert_eq!(horizontal_masked(TileReduceOp::Sum, v, m).f(0), 3.0);
        assert_eq!(horizontal_masked(TileReduceOp::Max, v, m).f(0), 2.0);
    }

    #[test]
    fn tree_broadcasts_each_group() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        tree_in_place(TileReduceOp::Sum, &mut v, 6, 3);
        assert_eq!(v, vec![6.0, 6.0, 6.0, 15.0, 15.0, 15.0]);
    }
}
