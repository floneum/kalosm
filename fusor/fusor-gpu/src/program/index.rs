//! Nonnegative integer index expressions. One representation for composition,
//! evaluation, bounds, ownership proofs and WGSL addressing.
use std::collections::BTreeMap;
pub(super) const GROUP: usize = usize::MAX - 1;
pub(super) const LOCAL: usize = usize::MAX;
pub(super) type Bounds = BTreeMap<usize, u64>; // inclusive upper bounds; lower bound zero

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Expr {
    Const(u64),
    Var(usize),
    Sum(Vec<(Expr, u64)>),
    Div(Box<Expr>, u64),
    Mod(Box<Expr>, u64),
}
impl Expr {
    pub(super) fn var(id: usize) -> Self {
        Self::Var(id)
    }
    pub(super) fn sum(xs: impl IntoIterator<Item = Self>) -> Self {
        Self::weighted(xs.into_iter().map(|x| (x, 1)))
    }
    fn weighted(xs: impl IntoIterator<Item = (Self, u64)>) -> Self {
        let mut terms = BTreeMap::<Self, u64>::new();
        let mut constant = 0;
        for (x, factor) in xs {
            if factor == 0 {
                continue;
            }
            match x {
                Self::Const(c) => constant += c * factor,
                Self::Sum(inner) => {
                    for (term, c) in inner {
                        *terms.entry(term).or_default() += c * factor;
                    }
                }
                x => *terms.entry(x).or_default() += factor,
            }
        }
        // Mixed-radix recomposition: x%a + a*((x/a)%b) = x%(a*b). A group
        // index split into several coordinates reassembles digit by digit.
        loop {
            let pair = terms.iter().find_map(|(term, c)| {
                let Self::Mod(x, a) = term else { return None };
                terms.iter().find_map(|(other, d)| {
                    let Self::Mod(inner, b) = other else { return None };
                    let Self::Div(y, a2) = inner.as_ref() else { return None };
                    (y == x && a2 == a && *d == c * a).then(|| {
                        (term.clone(), other.clone(), Self::Mod(x.clone(), a * b), *c)
                    })
                })
            });
            let Some((low, high, merged, c)) = pair else {
                break;
            };
            terms.remove(&low);
            terms.remove(&high);
            *terms.entry(merged).or_default() += c;
        }
        // Euclidean recomposition: n*(x/n) + x%n = x. This cancels
        // reshape/transpose round trips without inspecting any tensor element.
        loop {
            let pair = terms.iter().find_map(|(term, c)| {
                if let Self::Mod(x, n) = term {
                    let div = Self::Div(x.clone(), *n);
                    if let Some(d) = terms.get(&div) {
                        let count = (*c).min(d / n);
                        if count > 0 {
                            return Some((term.clone(), div, (**x).clone(), *n, count));
                        }
                    }
                }
                None
            });
            let Some((modulo, div, x, n, count)) = pair else {
                break;
            };
            *terms.get_mut(&modulo).unwrap() -= count;
            *terms.get_mut(&div).unwrap() -= n * count;
            match x {
                Self::Const(c) => constant += c * count,
                Self::Sum(inner) => {
                    for (term, c) in inner {
                        *terms.entry(term).or_default() += c * count;
                    }
                }
                x => *terms.entry(x).or_default() += count,
            }
            terms.retain(|_, c| *c != 0);
        }
        // Nested sums can contain constants; normalize those too.
        terms.retain(|term, factor| {
            if let Self::Const(c) = term {
                constant += c * *factor;
                false
            } else {
                *factor != 0
            }
        });
        if constant != 0 {
            terms.insert(Self::Const(constant), 1);
        }
        if terms.is_empty() {
            Self::Const(0)
        } else if terms.len() == 1 && *terms.first_key_value().unwrap().1 == 1 {
            terms.into_iter().next().unwrap().0
        } else {
            Self::Sum(terms.into_iter().collect())
        }
    }
    pub(super) fn scale(self, n: usize) -> Self {
        Self::weighted([(self, n as u64)])
    }
    pub(super) fn coordinate(self, shape: &[u32], axis: usize) -> Self {
        self.div(shape[axis + 1..].iter().product::<u32>() as usize)
            .modulo(shape[axis] as usize)
    }
    pub(super) fn restride(
        self,
        shape: &[u32],
        source: &[u32],
        specs: &[fusor_ir::shape::StrideSpec],
    ) -> Self {
        Self::sum(
            specs
                .iter()
                .enumerate()
                .filter(|(_, s)| s.multiplier != 0)
                .map(|(axis, s)| {
                    Self::sum([
                        self.clone()
                            .coordinate(shape, axis)
                            .scale(s.multiplier as usize),
                        Self::Const(s.offset.as_const().unwrap()),
                    ])
                    .scale(source[s.input_dim as usize + 1..].iter().product::<u32>() as usize)
                }),
        )
    }
    pub(super) fn div(self, n: usize) -> Self {
        assert!(n > 0);
        let n = n as u64;
        if n == 1 {
            return self;
        }
        match self {
            Self::Const(c) => Self::Const(c / n),
            Self::Div(x, d) => x.div((d * n) as usize),
            Self::Sum(xs) => {
                let mut quotient = vec![];
                let mut remainder = vec![];
                for (term, c) in xs {
                    if let Self::Const(v) = term {
                        quotient.push((Self::Const(v * c / n), 1));
                        remainder.push((Self::Const(v * c % n), 1));
                    } else {
                        quotient.push((term.clone(), c / n));
                        remainder.push((term, c % n));
                    }
                }
                let residual = Self::weighted(remainder);
                if residual != Self::Const(0) {
                    quotient.push((Self::Div(Box::new(residual), n), 1));
                }
                Self::weighted(quotient)
            }
            x => Self::Div(Box::new(x), n),
        }
    }
    pub(super) fn modulo(self, n: usize) -> Self {
        assert!(n > 0);
        let n = n as u64;
        if n == 1 {
            return Self::Const(0);
        }
        match self {
            Self::Const(c) => Self::Const(c % n),
            Self::Mod(x, d) if d % n == 0 => x.modulo(n as usize),
            Self::Sum(xs) => {
                let residual = Self::weighted(xs.into_iter().map(|(term, c)| match term {
                    Self::Const(v) => (Self::Const(v * c % n), 1),
                    x => (x, c % n),
                }));
                if let Self::Const(c) = residual {
                    Self::Const(c % n)
                } else {
                    Self::Mod(Box::new(residual), n)
                }
            }
            x => Self::Mod(Box::new(x), n),
        }
    }
    pub(super) fn upper(&self, bounds: &Bounds) -> u64 {
        match self {
            Self::Const(c) => *c,
            Self::Var(v) => bounds[v],
            Self::Sum(xs) => xs.iter().map(|(x, c)| x.upper(bounds) * c).sum(),
            Self::Div(x, n) => x.upper(bounds) / n,
            Self::Mod(x, n) => x.upper(bounds).min(n - 1),
        }
    }
    pub(super) fn simplify(&self, bounds: &Bounds) -> Self {
        let out = match self {
            Self::Var(v) if bounds[v] == 0 => Self::Const(0),
            Self::Sum(xs) => Self::weighted(xs.iter().map(|(x, c)| (x.simplify(bounds), *c))),
            Self::Div(x, n) => {
                let x = x.simplify(bounds);
                if x.upper(bounds) < *n {
                    Self::Const(0)
                } else if let Some((g, major, _)) = x.split_below(*n, bounds) {
                    major.div((*n / g) as usize)
                } else {
                    x.div(*n as usize)
                }
            }
            Self::Mod(x, n) => {
                let x = x.simplify(bounds);
                if x.upper(bounds) < *n {
                    x
                } else if let Some((g, major, minor)) = x.split_below(*n, bounds) {
                    Self::weighted([(major.modulo((*n / g) as usize), g), (minor, 1)])
                } else {
                    x.modulo(*n as usize)
                }
            }
            x => x.clone(),
        };
        // Distribution can expose another bounded quotient or remainder.
        if out != *self {
            out.simplify(bounds)
        } else {
            out
        }
    }
    /// `self = g*major + minor` with `g | n`, `g > 1` and `minor <= g - 1`,
    /// so `self / n = major / (n/g)` and `self % n = g*(major % (n/g)) + minor`.
    /// The largest such `g` among the term coefficients is chosen.
    fn split_below(&self, n: u64, bounds: &Bounds) -> Option<(u64, Self, Self)> {
        let Self::Sum(xs) = self else { return None };
        let mut candidates: Vec<u64> = xs
            .iter()
            .map(|(x, c)| match x {
                Self::Const(v) => v * c,
                _ => *c,
            })
            .filter(|g| *g > 1 && n % g == 0)
            .collect();
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        candidates.dedup();
        for g in candidates {
            let (mut major, mut minor) = (vec![], vec![]);
            for (x, c) in xs {
                match x {
                    Self::Const(v) => {
                        let v = v * c;
                        major.push((Self::Const(v / g), 1));
                        minor.push((Self::Const(v % g), 1));
                    }
                    x if c % g == 0 => major.push((x.clone(), c / g)),
                    x => minor.push((x.clone(), *c)),
                }
            }
            let minor = Self::weighted(minor);
            if minor.upper(bounds) < g {
                return Some((g, Self::weighted(major), minor));
            }
        }
        None
    }
    #[cfg(test)]
    pub(super) fn eval(&self, vars: &impl Fn(usize) -> usize) -> usize {
        match self {
            Self::Const(c) => *c as usize,
            Self::Var(v) => vars(*v),
            Self::Sum(xs) => xs.iter().map(|(x, c)| x.eval(vars) * (*c as usize)).sum(),
            Self::Div(x, n) => x.eval(vars) / (*n as usize),
            Self::Mod(x, n) => x.eval(vars) % (*n as usize),
        }
    }
}

pub(super) fn reduction_index(shape: &[usize], axis: usize, output: Expr, reduction: Expr) -> Expr {
    let inner: usize = shape[axis + 1..].iter().product();
    Expr::sum([
        output.clone().div(inner).scale(shape[axis] * inner),
        reduction.scale(inner),
        output.modulo(inner),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row statistic broadcast back over its row: the `[b, h, i]` value an
    /// element `(b, h, i, j)` reads is owned by the element's own group
    /// whenever a group owns whole `(b, h)` slices.
    #[test]
    fn broadcast_row_statistic_is_group_owned() {
        for (batch, heads, rows, cols) in [(2, 3, 4, 5), (4, 4, 8, 8), (3, 2, 5, 7)] {
            let groups = batch * heads;
            let share = rows * cols;
            let bounds = Bounds::from([(GROUP, groups as u64 - 1), (LOCAL, share as u64 - 1)]);
            let at = Expr::sum([Expr::var(GROUP).scale(share), Expr::var(LOCAL)]);
            let shape = [batch as u32, heads as u32, rows as u32, cols as u32];
            let stat = Expr::sum([
                at.clone().coordinate(&shape, 0).scale(heads * rows),
                at.clone().coordinate(&shape, 1).scale(rows),
                at.clone().coordinate(&shape, 2),
            ]);
            let owner = stat.clone().div(rows);
            let simplified = owner.simplify(&bounds);
            assert_eq!(simplified, Expr::var(GROUP), "{batch}x{heads}x{rows}x{cols}");
            for group in 0..groups {
                for local in 0..share {
                    let vars = |v| if v == GROUP { group } else { local };
                    assert_eq!(owner.eval(&vars), simplified.eval(&vars));
                }
            }
        }
    }

    #[test]
    fn mixed_radix_digits_recombine() {
        let bounds = Bounds::from([(GROUP, 255u64)]);
        let g = || Expr::var(GROUP);
        let owner = Expr::sum([
            g().div(8).scale(8),
            g().modulo(2),
            g().div(2).modulo(4).scale(2),
        ]);
        assert_eq!(owner.simplify(&bounds), g());
        for group in 0..256 {
            assert_eq!(owner.eval(&|_| group), group);
        }
    }

    #[test]
    fn split_division_and_remainder_agree_with_evaluation() {
        let bounds = Bounds::from([(0, 9), (1, 5), (2, 3)]);
        let exprs = [
            Expr::sum([Expr::var(0).scale(24), Expr::var(1).scale(4), Expr::var(2)]),
            Expr::sum([Expr::var(0).scale(8), Expr::var(2), Expr::Const(4)]),
            Expr::sum([Expr::var(1).scale(6), Expr::var(2).scale(2)]),
        ];
        for e in &exprs {
            for n in [2, 3, 4, 6, 8, 12, 24, 48] {
                for (op, simplified) in [
                    (Expr::Div(Box::new(e.clone()), n), Expr::Div(Box::new(e.clone()), n).simplify(&bounds)),
                    (Expr::Mod(Box::new(e.clone()), n), Expr::Mod(Box::new(e.clone()), n).simplify(&bounds)),
                ] {
                    for a in 0..=9 {
                        for b in 0..=5 {
                            for c in 0..=3 {
                                let vars = |v| [a, b, c][v];
                                assert_eq!(op.eval(&vars), simplified.eval(&vars), "{op:?} -> {simplified:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ownership_simplification_agrees_with_exhaustive_indices() {
        for groups in [1, 2, 3, 7] {
            for rows in [1, 2, 5] {
                for width in [1, 3, 8, 17] {
                    let bounds =
                        Bounds::from([(GROUP, groups - 1), (LOCAL, rows - 1), (0, width - 1)]);
                    let output =
                        Expr::sum([Expr::var(GROUP).scale(rows as usize), Expr::var(LOCAL)]);
                    let index = reduction_index(
                        &[(groups * rows) as usize, width as usize],
                        1,
                        output,
                        Expr::var(0),
                    );
                    let owner = index.div((rows * width) as usize);
                    let simplified = owner.simplify(&bounds);
                    for group in 0..groups {
                        for local in 0..rows {
                            for k in 0..width {
                                let vars = |v| match v {
                                    GROUP => group as usize,
                                    LOCAL => local as usize,
                                    _ => k as usize,
                                };
                                assert_eq!(owner.eval(&vars), simplified.eval(&vars));
                                assert_eq!(simplified.eval(&vars), group as usize);
                            }
                        }
                    }
                }
            }
        }
    }
}
