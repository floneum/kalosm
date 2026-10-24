//! Assign reduction-index bits to lanes or to an unrolled register window.
//! Selection uses ordinary scalar expressions, never storage-format metadata.
use crate::build::Kernel;
use fusor_ir::Result;
use fusor_ir::ir::kernel::{TileExpr, TileExprKind as K, simplify_index};
use rustc_hash::FxHashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    lane_bits: Vec<u32>,
    value_bits: Vec<u32>,
    contiguous_bits: u32,
}

impl Window {
    /// The existing split-run layout, when it is a permutation of index bits.
    pub fn new(width: u32, vector: u32, parts: u32, gap: u32) -> Option<Self> {
        if !width.is_power_of_two()
            || !vector.is_power_of_two()
            || !parts.is_power_of_two()
            || vector > 64
            || parts > vector
            || width.checked_mul(vector)? > 4096
        {
            return None;
        }
        let run = vector / parts;
        let low = run.trailing_zeros();
        let span = (width * vector).trailing_zeros();
        let mut value_bits: Vec<_> = (0..low).collect();
        if parts > 1 {
            if !gap.is_power_of_two() || gap <= run || !(width * run).is_multiple_of(gap) {
                return None;
            }
            value_bits.extend(gap.trailing_zeros()..gap.trailing_zeros() + parts.trailing_zeros());
        }
        let lane_bits = (0..span).filter(|bit| !value_bits.contains(bit)).collect();
        Some(Self {
            lane_bits,
            value_bits,
            contiguous_bits: low,
        })
    }

    fn offset(&self, v: u32) -> u32 {
        self.value_bits
            .iter()
            .enumerate()
            .fold(0, |n, (i, bit)| n | (((v >> i) & 1) << bit))
    }

    /// Preserve adjacent elements within a run. Only exchange a register bit
    /// above the run with a lane bit; each candidate remains a bijection.
    fn alternatives(&self) -> Vec<Self> {
        let mut choices = Vec::new();
        for v in self.contiguous_bits as usize..self.value_bits.len() {
            for l in 0..self.lane_bits.len() {
                if choices.len() == 16 {
                    return choices;
                }
                let mut next = self.clone();
                std::mem::swap(&mut next.value_bits[v], &mut next.lane_bits[l]);
                choices.push(next);
            }
        }
        choices
    }

    pub fn index(&self, b: &Kernel, lane: TileExpr, v: u32) -> TileExpr {
        let mut value = b.u32(self.offset(v));
        let mut start = 0;
        while start < self.lane_bits.len() {
            let mut end = start + 1;
            while end < self.lane_bits.len() && self.lane_bits[end] == self.lane_bits[end - 1] + 1 {
                end += 1;
            }
            let bits = b.rem(
                b.div(lane.clone(), b.u32(1 << start)),
                b.u32(1 << (end - start)),
            );
            value = b.add(value, b.mul(bits, b.u32(1 << self.lane_bits[start])));
            start = end;
        }
        simplify_index(&value)
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Cost {
    loads: usize,
    selects: usize,
    work: usize,
}

fn cost(values: impl IntoIterator<Item = TileExpr>) -> Cost {
    fn visit(e: &TileExpr, seen: &mut FxHashSet<TileExpr>) {
        if seen.insert(e.clone()) {
            e.kind().visit_children(&mut |child| visit(child, seen));
        }
    }
    let mut seen = FxHashSet::default();
    for value in values {
        visit(&simplify_index(&value), &mut seen);
    }
    let mut cost = Cost::default();
    for e in seen {
        match e.kind() {
            K::Load { .. } | K::LoadTile { .. } => cost.loads += 1,
            K::Select { .. } => {
                cost.selects += 1;
                cost.work += 1;
            }
            K::Binary { .. } | K::Unary { .. } | K::Compare { .. } | K::Cast { .. } => {
                cost.work += 1
            }
            _ => {}
        }
    }
    cost
}

/// Bounded compile-time search: at most 16 single-element probes and four
/// complete windows. Accept only fewer dynamic selections and fewer operations
/// with no additional unique loads. No GPU races or persistent search cache.
pub fn choose(
    original: Window,
    mut values: impl FnMut(&Window, u32) -> Result<Vec<TileExpr>>,
) -> Result<Option<Window>> {
    let baseline = cost(values(&original, 0)?);
    if baseline.selects == 0 {
        return Ok(None);
    }
    let mut probes = Vec::new();
    for window in original.alternatives() {
        let c = cost(values(&window, 0)?);
        if c.selects < baseline.selects {
            probes.push((c, window));
        }
    }
    if probes.is_empty() {
        return Ok(None);
    }
    probes.sort_by_key(|(c, _)| (c.selects, c.loads, c.work));
    let mut full_cost = |window: &Window| -> Result<Cost> {
        let mut all = Vec::new();
        for v in 0..(1 << window.value_bits.len()) {
            all.extend(values(window, v)?);
        }
        Ok(cost(all))
    };
    let baseline = full_cost(&original)?;
    let mut best = baseline;
    let mut selected = None;
    for (_, window) in probes.into_iter().take(4) {
        let c = full_cost(&window)?;
        if c.loads <= baseline.loads && c.selects < baseline.selects && c.work < best.work {
            best = c;
            selected = Some(window);
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::ir::kernel::{Builtin, TileLiteral};

    fn evaluate(e: &TileExpr, lane: u32) -> u32 {
        use fusor_ir::ir::kernel::TileBinaryOp as Op;
        match e.kind() {
            K::Builtin(Builtin::SubgroupLane) => lane,
            K::Literal(TileLiteral::U32(v)) => *v,
            K::Binary {
                op, left, right, ..
            } => {
                let (a, b) = (evaluate(left, lane), evaluate(right, lane));
                match op {
                    Op::Add => a.wrapping_add(b),
                    Op::Mul => a.wrapping_mul(b),
                    Op::Div => a / b,
                    Op::Rem => a % b,
                    _ => panic!("unexpected index operation"),
                }
            }
            _ => panic!("unexpected index expression"),
        }
    }

    #[test]
    fn every_lane_exchange_covers_each_element_exactly_once() {
        for width in [4, 8, 16, 32, 64] {
            for vector in [1, 2, 4, 8, 16, 32, 64] {
                for parts in [1, 2, 4] {
                    for gap in [4, 8, 16, 32, 64, 128] {
                        let Some(w) = Window::new(width, vector, parts, gap) else {
                            continue;
                        };
                        for w in std::iter::once(w.clone()).chain(w.alternatives()) {
                            let b = Kernel::new();
                            let indices: Vec<_> = (0..vector)
                                .map(|v| w.index(&b, b.builtin(Builtin::SubgroupLane), v))
                                .collect();
                            let mut visited = vec![false; (width * vector) as usize];
                            for lane in 0..width {
                                for e in &indices {
                                    let index = evaluate(e, lane) as usize;
                                    assert!(!visited[index]);
                                    visited[index] = true;
                                }
                            }
                            assert!(visited.into_iter().all(|v| v));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn index_decisions_move_into_the_unrolled_window() {
        let b = Kernel::new();
        let lane = b.builtin(Builtin::SubgroupLane);
        let original = Window::new(32, 32, 4, 32).unwrap();
        let selected = choose(original.clone(), |w, v| {
            let k = w.index(&b, lane.clone(), v);
            Ok(vec![b.select(
                b.lt(b.rem(k, b.u32(256)), b.u32(128)),
                b.f32(2.),
                b.f32(3.),
            )])
        })
        .unwrap()
        .expect("a constant decision needs less work");
        assert_ne!(selected, original);
        assert_eq!(
            cost([b.select(
                b.lt(b.rem(selected.index(&b, lane, 0), b.u32(256)), b.u32(128)),
                b.f32(2.),
                b.f32(3.)
            )])
            .selects,
            0
        );
    }

    #[test]
    fn removing_a_decision_does_not_justify_extra_loads() {
        use fusor_ir::ir::kernel::ScalarElement;
        let b = Kernel::new();
        let lane = b.builtin(Builtin::SubgroupLane);
        let source = b.tile("lookup", ScalarElement::F32.element(), &[8]);
        let original = Window::new(32, 32, 4, 32).unwrap();
        let selected = choose(original, |w, v| {
            let k = w.index(&b, lane.clone(), v);
            Ok(vec![
                b.select(
                    b.lt(b.rem(k.clone(), b.u32(256)), b.u32(128)),
                    b.f32(2.),
                    b.f32(3.),
                ),
                b.load_tile(source.clone(), b.div(k, b.u32(128))),
            ])
        })
        .unwrap();
        assert!(selected.is_none());
    }
}
