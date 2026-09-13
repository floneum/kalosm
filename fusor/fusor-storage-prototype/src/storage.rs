//! Pure storage assignment. Lifetimes include both ends: an input and its
//! consumer's output interfere until a barrier has ordered the stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    Dedicated,
    Colored,
    BestFit,
}
#[derive(Clone, Debug)]
pub struct Slot {
    pub id: usize,
    pub len: usize,
    pub first: usize,
    pub last: usize,
}
impl Slot {
    pub fn interferes(&self, b: &Self) -> bool {
        self.first <= b.last && b.first <= self.last
    }
}
#[derive(Clone, Debug, Default)]
pub struct Allocation {
    pub slots: Vec<Slot>,
    pub offsets: Vec<usize>,
    pub len: usize,
    pub lower_bound: usize,
    pub conflicts: usize,
}
impl Allocation {
    pub fn offset(&self, id: usize) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.id == id)
            .map(|i| self.offsets[i])
    }
    pub fn verify(&self) {
        for (i, a) in self.slots.iter().enumerate() {
            assert!(self.offsets[i] + a.len <= self.len);
            for (j, b) in self.slots.iter().enumerate().take(i) {
                if a.interferes(b) {
                    assert!(
                        self.offsets[i] + a.len <= self.offsets[j]
                            || self.offsets[j] + b.len <= self.offsets[i],
                        "live allocations overlap"
                    );
                }
            }
        }
    }
}
pub fn allocate(slots: Vec<Slot>, mode: Packing) -> Allocation {
    let mut out = Allocation {
        offsets: vec![0; slots.len()],
        slots,
        ..Allocation::default()
    };
    let mut order: Vec<usize> = (0..out.slots.len()).collect();
    order.sort_by_key(|i| {
        (
            std::cmp::Reverse(out.slots[*i].len),
            out.slots[*i].first,
            *i,
        )
    });
    out.lower_bound = (0..=out.slots.iter().map(|s| s.last).max().unwrap_or(0))
        .map(|t| {
            out.slots
                .iter()
                .filter(|s| s.first <= t && t <= s.last)
                .map(|s| s.len)
                .sum()
        })
        .max()
        .unwrap_or(0);
    out.conflicts = out
        .slots
        .iter()
        .enumerate()
        .map(|(i, a)| out.slots[..i].iter().filter(|b| a.interferes(b)).count())
        .sum();
    match mode {
        Packing::Dedicated => {
            for i in order {
                out.offsets[i] = out.len;
                out.len += out.slots[i].len;
            }
        }
        Packing::Colored => {
            // Weighted greedy coloring: a color is one reusable slot sized
            // to its largest value. Largest values first, exact edge checks.
            let mut colors: Vec<Vec<usize>> = vec![];
            for i in order {
                if let Some(c) = colors
                    .iter_mut()
                    .find(|c| c.iter().all(|j| !out.slots[i].interferes(&out.slots[*j])))
                {
                    c.push(i);
                } else {
                    colors.push(vec![i]);
                }
            }
            for color in colors {
                for i in &color {
                    out.offsets[*i] = out.len;
                }
                out.len += color.iter().map(|i| out.slots[*i].len).max().unwrap();
            }
        }
        Packing::BestFit => {
            // Variable-sized interval packing over the interference graph.
            // Holes may be shared partially; no fixed color sizes are needed.
            let mut placed: Vec<usize> = vec![];
            for i in order {
                let mut candidates = vec![0];
                candidates.extend(placed.iter().map(|j| out.offsets[*j] + out.slots[*j].len));
                candidates.sort_unstable();
                candidates.dedup();
                let off = candidates
                    .into_iter()
                    .filter(|off| {
                        placed.iter().all(|j| {
                            !out.slots[i].interferes(&out.slots[*j])
                                || off + out.slots[i].len <= out.offsets[*j]
                                || out.offsets[*j] + out.slots[*j].len <= *off
                        })
                    })
                    .min_by_key(|off| (out.len.max(off + out.slots[i].len), *off))
                    .unwrap();
                out.offsets[i] = off;
                out.len = out.len.max(off + out.slots[i].len);
                placed.push(i);
            }
        }
    };
    out.verify();
    out
}
