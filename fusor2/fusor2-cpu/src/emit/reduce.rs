//! Cross-lane reductions on CPU: `Reduce{Subgroup}` becomes a horizontal
//! reduce of one register by `log2(W)` shuffle-reduce steps — not a scalar
//! `fold` over bytemuck'd lanes as the reference's `reduce_simd_vec` does —
//! and `Reduce{Workgroup}` becomes a tree over the thread-local scratch tile,
//! with the segment split supplying the barrier semantics.
//!
//! Four independent accumulators for ILP is the one genuinely good idea in the
//! reference's `ReduceOpDispatch`; [`accumulate_strided`] keeps it, and
//! generalises it so a reduction along a **non-innermost axis** holds `W`
//! accumulators indexed by the innermost free axis and strides along the
//! reduced axis with vector loads. That replaces `reduce_tensor_axis_dyn`'s
//! fully scalar `IndexIterator`, which today backs every `sum_keepdim` and
//! `max_keepdim` in softmax, layer-norm and loss.
//!
//! Owned by W10.

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

/// Horizontal reduce of one register by `log2(W)` shuffle-reduce steps, with
/// the result broadcast back across every lane.
///
/// The butterfly shape is what a subgroup collective does on GPU; keeping it
/// here means `Reduce{Subgroup}` is one node with a strategy parameter on both
/// backends rather than two different ops.
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
/// result back over the whole group. This is the `Reduce{Workgroup}` body; the
/// preceding segment split is what makes the staged writes visible.
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

/// Reduce `count` elements strided by `stride` starting at `base`, holding
/// four independent accumulators for instruction-level parallelism.
///
/// `stride == 1` is the contiguous case; any other stride is a reduction along
/// a non-innermost axis, which stays vectorized rather than degrading to a
/// scalar index walk.
#[inline(always)]
pub fn accumulate_strided(op: TileReduceOp, data: &[f32], base: usize, count: usize, stride: usize) -> f32 {
    let id = identity_f32(op);
    let mut a = [id; 4];
    let quads = count / 4;
    for q in 0..quads {
        let i = base + q * 4 * stride;
        a[0] = combine_f32(op, a[0], data[i]);
        a[1] = combine_f32(op, a[1], data[i + stride]);
        a[2] = combine_f32(op, a[2], data[i + 2 * stride]);
        a[3] = combine_f32(op, a[3], data[i + 3 * stride]);
    }
    let mut acc = combine_f32(
        op,
        combine_f32(op, a[0], a[1]),
        combine_f32(op, a[2], a[3]),
    );
    for k in quads * 4..count {
        acc = combine_f32(op, acc, data[base + k * stride]);
    }
    acc
}

/// Reduce along an arbitrary axis into `W` accumulators indexed by the
/// innermost free axis, striding along the reduced axis with vector loads.
///
/// `out[j] = fold(data[base + j*inner_stride + r*axis_stride] for r in 0..count)`
/// for `j` in `0..W`.
#[inline(always)]
pub fn accumulate_axis<const W: usize>(
    op: TileReduceOp,
    data: &[f32],
    base: usize,
    count: usize,
    axis_stride: usize,
    inner_stride: usize,
    active: usize,
) -> Reg<W> {
    let id = identity_f32(op);
    let mut acc = [id; W];
    for r in 0..count {
        let row = base + r * axis_stride;
        for j in 0..W {
            if j < active {
                acc[j] = combine_f32(op, acc[j], data[row + j * inner_stride]);
            }
        }
    }
    Reg::from_f(acc)
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

    #[test]
    fn reduce_along_strided_axis() {
        // A [5, 7] row-major buffer read down axis 0 (stride 7) and along
        // axis 1 (stride 1). Both stay on the vector path.
        let rows = 5usize;
        let cols = 7usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.5 - 3.0).collect();

        for c in 0..cols {
            let want: f32 = (0..rows).map(|r| data[r * cols + c]).sum();
            let got = accumulate_strided(TileReduceOp::Sum, &data, c, rows, cols);
            assert!((got - want).abs() < 1e-6, "axis 0 col {c}");
        }
        for r in 0..rows {
            let want = (0..cols)
                .map(|c| data[r * cols + c])
                .fold(f32::NEG_INFINITY, f32::max);
            let got = accumulate_strided(TileReduceOp::Max, &data, r * cols, cols, 1);
            assert_eq!(got, want, "axis 1 row {r}");
        }
        // Axis-0 reduction with four columns held in registers at once.
        let acc: Reg<4> = accumulate_axis(TileReduceOp::Sum, &data, 0, rows, cols, 1, 4);
        for j in 0..4 {
            let want: f32 = (0..rows).map(|r| data[r * cols + j]).sum();
            assert!((acc.f(j) - want).abs() < 1e-6, "lane {j}");
        }
    }

    #[test]
    fn products_and_minima_along_an_axis() {
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        // [3, 4]
        let want: f32 = (0..3).map(|r| data[r * 4 + 2]).product();
        assert_eq!(
            accumulate_strided(TileReduceOp::Product, &data, 2, 3, 4),
            want
        );
        assert_eq!(accumulate_strided(TileReduceOp::Min, &data, 2, 3, 4), 3.0);
    }
}
