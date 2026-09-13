//! Nonnegative integer index expressions. One representation for composition,
//! evaluation, bounds, ownership proofs and WGSL addressing.
use std::collections::BTreeMap;
pub const OUTPUT: usize = 0;
pub const GROUP: usize = usize::MAX - 1;
pub const LOCAL: usize = usize::MAX;
pub type Bounds = BTreeMap<usize, u64>; // inclusive upper bounds; lower bound zero

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    Const(u64),
    Var(usize),
    Sum(Vec<(Expr, u64)>),
    Div(Box<Expr>, u64),
    Mod(Box<Expr>, u64),
}
impl Expr {
    pub fn var(id: usize) -> Self {
        Self::Var(id)
    }
    pub fn constant(x: usize) -> Self {
        Self::Const(x as u64)
    }
    pub fn sum(xs: impl IntoIterator<Item = Self>) -> Self {
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
    pub fn scale(self, n: usize) -> Self {
        Self::weighted([(self, n as u64)])
    }
    pub fn div(self, n: usize) -> Self {
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
    pub fn modulo(self, n: usize) -> Self {
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
    pub fn upper(&self, bounds: &Bounds) -> u64 {
        match self {
            Self::Const(c) => *c,
            Self::Var(v) => bounds[v],
            Self::Sum(xs) => xs.iter().map(|(x, c)| x.upper(bounds) * c).sum(),
            Self::Div(x, n) => x.upper(bounds) / n,
            Self::Mod(x, n) => x.upper(bounds).min(n - 1),
        }
    }
    pub fn simplify(&self, bounds: &Bounds) -> Self {
        let out = match self {
            Self::Var(v) if bounds[v] == 0 => Self::Const(0),
            Self::Sum(xs) => Self::weighted(xs.iter().map(|(x, c)| (x.simplify(bounds), *c))),
            Self::Div(x, n) => {
                let x = x.simplify(bounds);
                if x.upper(bounds) < *n {
                    Self::Const(0)
                } else {
                    x.div(*n as usize)
                }
            }
            Self::Mod(x, n) => {
                let x = x.simplify(bounds);
                if x.upper(bounds) < *n {
                    x
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
    pub fn substitute(&self, variable: usize, value: &Self) -> Self {
        match self {
            Self::Var(v) if *v == variable => value.clone(),
            Self::Sum(xs) => {
                Self::weighted(xs.iter().map(|(x, c)| (x.substitute(variable, value), *c)))
            }
            Self::Div(x, n) => x.substitute(variable, value).div(*n as usize),
            Self::Mod(x, n) => x.substitute(variable, value).modulo(*n as usize),
            x => x.clone(),
        }
    }
    pub fn eval(&self, vars: &impl Fn(usize) -> usize) -> usize {
        match self {
            Self::Const(c) => *c as usize,
            Self::Var(v) => vars(*v),
            Self::Sum(xs) => xs.iter().map(|(x, c)| x.eval(vars) * (*c as usize)).sum(),
            Self::Div(x, n) => x.eval(vars) / (*n as usize),
            Self::Mod(x, n) => x.eval(vars) % (*n as usize),
        }
    }
    pub fn wgsl(&self, names: &impl Fn(usize) -> String) -> String {
        match self {
            Self::Const(c) => format!("{c}u"),
            Self::Var(v) => names(*v),
            Self::Sum(xs) => format!(
                "({})",
                xs.iter()
                    .map(|(x, c)| if *c == 1 {
                        x.wgsl(names)
                    } else {
                        format!("({} * {c}u)", x.wgsl(names))
                    })
                    .collect::<Vec<_>>()
                    .join(" + ")
            ),
            Self::Div(x, n) => format!("({} / {n}u)", x.wgsl(names)),
            Self::Mod(x, n) => format!("({} % {n}u)", x.wgsl(names)),
        }
    }
}

pub fn reduction_index(shape: &[usize], axis: usize, output: Expr, reduction: Expr) -> Expr {
    let inner: usize = shape[axis + 1..].iter().product();
    Expr::sum([
        output.clone().div(inner).scale(shape[axis] * inner),
        reduction.scale(inner),
        output.modulo(inner),
    ])
}
