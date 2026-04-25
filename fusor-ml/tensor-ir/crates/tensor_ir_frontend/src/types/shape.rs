//! Tensor shapes, dimensions, and strides.

use std::fmt;

/// Runtime values used to evaluate symbolic shape dimensions.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeParams(pub Vec<u32>);

impl ShapeParams {
    #[must_use]
    pub fn new(values: impl Into<Vec<u32>>) -> Self {
        Self(values.into())
    }

    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn storage_words(&self) -> Vec<u32> {
        if self.0.is_empty() {
            // Keep the runtime binding layout uniform even for literal-only
            // programs. WGSL storage arrays cannot be zero-length.
            vec![0]
        } else {
            self.0.clone()
        }
    }

    #[must_use]
    pub fn get(&self, index: u32) -> Option<u32> {
        self.0.get(index as usize).copied()
    }
}

impl From<Vec<u32>> for ShapeParams {
    fn from(values: Vec<u32>) -> Self {
        Self(values)
    }
}

impl<const N: usize> From<[u32; N]> for ShapeParams {
    fn from(values: [u32; N]) -> Self {
        Self(values.into())
    }
}

/// A single dimension extent or shape-derived unsigned integer expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dim {
    Const(u32),
    /// Runtime dimension at slot `index` in [`ShapeParams`].
    Symbol(u32),
    Add(Box<Dim>, Box<Dim>),
    Sub(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    Div(Box<Dim>, Box<Dim>),
    Mod(Box<Dim>, Box<Dim>),
    CeilDiv(Box<Dim>, Box<Dim>),
    Min(Box<Dim>, Box<Dim>),
    Max(Box<Dim>, Box<Dim>),
}

impl Dim {
    fn collect_add_terms(self, terms: &mut Vec<Self>) {
        match self {
            Self::Add(lhs, rhs) => {
                lhs.collect_add_terms(terms);
                rhs.collect_add_terms(terms);
            }
            value => terms.push(value),
        }
    }

    fn collect_mul_terms(self, terms: &mut Vec<Self>) {
        match self {
            Self::Mul(lhs, rhs) => {
                lhs.collect_mul_terms(terms);
                rhs.collect_mul_terms(terms);
            }
            value => terms.push(value),
        }
    }

    fn collect_min_terms(self, terms: &mut Vec<Self>) {
        match self {
            Self::Min(lhs, rhs) => {
                lhs.collect_min_terms(terms);
                rhs.collect_min_terms(terms);
            }
            value => terms.push(value),
        }
    }

    fn collect_max_terms(self, terms: &mut Vec<Self>) {
        match self {
            Self::Max(lhs, rhs) => {
                lhs.collect_max_terms(terms);
                rhs.collect_max_terms(terms);
            }
            value => terms.push(value),
        }
    }

    fn fold_sorted_terms(
        mut terms: Vec<Self>,
        identity: Self,
        raw: impl Fn(Self, Self) -> Self,
    ) -> Self {
        if terms.is_empty() {
            return identity;
        }

        terms.sort();
        let mut terms = terms.into_iter();
        let first = terms.next().unwrap_or(identity);
        terms.fold(first, raw)
    }

    #[must_use]
    pub const fn as_const(&self) -> Option<u32> {
        match self {
            Self::Const(v) => Some(*v),
            Self::Symbol(_)
            | Self::Add(_, _)
            | Self::Sub(_, _)
            | Self::Mul(_, _)
            | Self::Div(_, _)
            | Self::Mod(_, _)
            | Self::CeilDiv(_, _)
            | Self::Min(_, _)
            | Self::Max(_, _) => None,
        }
    }

    #[must_use]
    pub fn eval_u32(&self, params: &ShapeParams) -> Option<u32> {
        match self {
            Self::Const(v) => Some(*v),
            Self::Symbol(index) => params.get(*index),
            Self::Add(a, b) => a.eval_u32(params)?.checked_add(b.eval_u32(params)?),
            Self::Sub(a, b) => a.eval_u32(params)?.checked_sub(b.eval_u32(params)?),
            Self::Mul(a, b) => a.eval_u32(params)?.checked_mul(b.eval_u32(params)?),
            Self::Div(a, b) => {
                let rhs = b.eval_u32(params)?;
                if rhs == 0 {
                    None
                } else {
                    Some(a.eval_u32(params)? / rhs)
                }
            }
            Self::Mod(a, b) => {
                let rhs = b.eval_u32(params)?;
                if rhs == 0 {
                    None
                } else {
                    Some(a.eval_u32(params)? % rhs)
                }
            }
            Self::CeilDiv(a, b) => {
                let rhs = b.eval_u32(params)?;
                if rhs == 0 {
                    None
                } else {
                    Some(a.eval_u32(params)?.div_ceil(rhs))
                }
            }
            Self::Min(a, b) => Some(a.eval_u32(params)?.min(b.eval_u32(params)?)),
            Self::Max(a, b) => Some(a.eval_u32(params)?.max(b.eval_u32(params)?)),
        }
    }

    #[must_use]
    pub fn representative_u32(&self) -> u32 {
        self.as_const().unwrap_or(1024)
    }

    #[must_use]
    pub fn is_multiple_of(&self, rhs: u32) -> bool {
        rhs != 0
            && self
                .as_const()
                .is_some_and(|value| value.is_multiple_of(rhs))
    }

    #[must_use]
    pub fn add(lhs: Self, rhs: Self) -> Self {
        let mut raw_terms = Vec::new();
        lhs.collect_add_terms(&mut raw_terms);
        rhs.collect_add_terms(&mut raw_terms);

        let mut terms = Vec::new();
        let mut const_sum = 0_u32;
        for term in raw_terms {
            match term {
                Self::Const(0) => {}
                Self::Const(value) => match const_sum.checked_add(value) {
                    Some(next) => const_sum = next,
                    None => {
                        if const_sum != 0 {
                            terms.push(Self::Const(const_sum));
                        }
                        const_sum = value;
                    }
                },
                value => terms.push(value),
            }
        }
        if const_sum != 0 {
            terms.push(Self::Const(const_sum));
        }
        Self::fold_sorted_terms(terms, Self::Const(0), |lhs, rhs| {
            Self::Add(Box::new(lhs), Box::new(rhs))
        })
    }

    #[must_use]
    pub fn sub(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (value, Self::Const(0)) => value,
            (Self::Const(a), Self::Const(b)) if b <= a => Self::Const(a - b),
            (lhs, rhs) if lhs == rhs => Self::Const(0),
            (lhs, rhs) => Self::Sub(Box::new(lhs), Box::new(rhs)),
        }
    }

    #[must_use]
    pub fn mul(lhs: Self, rhs: Self) -> Self {
        let mut raw_terms = Vec::new();
        lhs.collect_mul_terms(&mut raw_terms);
        rhs.collect_mul_terms(&mut raw_terms);

        let mut terms = Vec::new();
        let mut const_product = 1_u32;
        for term in raw_terms {
            match term {
                Self::Const(0) => return Self::Const(0),
                Self::Const(1) => {}
                Self::Const(value) => match const_product.checked_mul(value) {
                    Some(next) => const_product = next,
                    None => {
                        if const_product != 1 {
                            terms.push(Self::Const(const_product));
                        }
                        const_product = value;
                    }
                },
                value => terms.push(value),
            }
        }
        if const_product != 1 {
            terms.push(Self::Const(const_product));
        }
        Self::fold_sorted_terms(terms, Self::Const(1), |lhs, rhs| {
            Self::Mul(Box::new(lhs), Box::new(rhs))
        })
    }

    #[must_use]
    pub fn div(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Const(0), _) => Self::Const(0),
            (value, Self::Const(1)) => value,
            (Self::Const(a), Self::Const(b)) if b != 0 => Self::Const(a / b),
            (lhs, rhs) => Self::Div(Box::new(lhs), Box::new(rhs)),
        }
    }

    #[must_use]
    pub fn modulo(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Const(0), _) | (_, Self::Const(1)) => Self::Const(0),
            (Self::Const(a), Self::Const(b)) if b != 0 => Self::Const(a % b),
            (lhs, rhs) => Self::Mod(Box::new(lhs), Box::new(rhs)),
        }
    }

    #[must_use]
    pub fn ceil_div(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Const(0), _) => Self::Const(0),
            (value, Self::Const(1)) => value,
            (Self::Const(a), Self::Const(b)) if b != 0 => Self::Const(a.div_ceil(b)),
            (lhs, rhs) => Self::CeilDiv(Box::new(lhs), Box::new(rhs)),
        }
    }

    #[must_use]
    pub fn min(lhs: Self, rhs: Self) -> Self {
        let mut raw_terms = Vec::new();
        lhs.collect_min_terms(&mut raw_terms);
        rhs.collect_min_terms(&mut raw_terms);

        let mut terms = Vec::new();
        let mut const_min = None::<u32>;
        for term in raw_terms {
            match term {
                Self::Const(value) => {
                    const_min = Some(const_min.map_or(value, |current| current.min(value)));
                }
                value => terms.push(value),
            }
        }
        if let Some(value) = const_min {
            terms.push(Self::Const(value));
        }
        terms.sort();
        terms.dedup();
        Self::fold_sorted_terms(terms, Self::Const(u32::MAX), |lhs, rhs| {
            Self::Min(Box::new(lhs), Box::new(rhs))
        })
    }

    #[must_use]
    pub fn max(lhs: Self, rhs: Self) -> Self {
        let mut raw_terms = Vec::new();
        lhs.collect_max_terms(&mut raw_terms);
        rhs.collect_max_terms(&mut raw_terms);

        let mut terms = Vec::new();
        let mut const_max = None::<u32>;
        for term in raw_terms {
            match term {
                Self::Const(value) => {
                    const_max = Some(const_max.map_or(value, |current| current.max(value)));
                }
                value => terms.push(value),
            }
        }
        if let Some(value) = const_max {
            terms.push(Self::Const(value));
        }
        terms.sort();
        terms.dedup();
        Self::fold_sorted_terms(terms, Self::Const(0), |lhs, rhs| {
            Self::Max(Box::new(lhs), Box::new(rhs))
        })
    }
}

impl From<u32> for Dim {
    fn from(value: u32) -> Self {
        Self::Const(value)
    }
}

impl PartialEq<u32> for Dim {
    fn eq(&self, other: &u32) -> bool {
        self.as_const() == Some(*other)
    }
}

impl PartialEq<Dim> for u32 {
    fn eq(&self, other: &Dim) -> bool {
        other == self
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => write!(f, "{v}"),
            Self::Symbol(idx) => write!(f, "${idx}"),
            Self::Add(a, b) => write!(f, "({a}+{b})"),
            Self::Sub(a, b) => write!(f, "({a}-{b})"),
            Self::Mul(a, b) => write!(f, "({a}*{b})"),
            Self::Div(a, b) => write!(f, "({a}/{b})"),
            Self::Mod(a, b) => write!(f, "({a}%{b})"),
            Self::CeilDiv(a, b) => write!(f, "ceildiv({a},{b})"),
            Self::Min(a, b) => write!(f, "min({a},{b})"),
            Self::Max(a, b) => write!(f, "max({a},{b})"),
        }
    }
}

/// A tensor shape: a list of dimension extents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn numel(&self) -> Dim {
        self.0.iter().cloned().fold(Dim::Const(1), Dim::mul)
    }

    #[must_use]
    pub fn static_numel(&self) -> Option<u32> {
        self.numel().as_const()
    }

    #[must_use]
    pub fn remove_axis(&self, axis: usize) -> Self {
        let mut dims = self.0.clone();
        dims.remove(axis);
        Self(dims)
    }

    /// Try to interpret as a 3D shape with all constant dimensions.
    #[must_use]
    pub fn as_3d_const(&self) -> Option<(u32, u32, u32)> {
        if self.0.len() != 3 {
            return None;
        }
        Some((
            self.0[0].as_const()?,
            self.0[1].as_const()?,
            self.0[2].as_const()?,
        ))
    }

    #[must_use]
    pub fn as_3d_lit(&self) -> Option<(u32, u32, u32)> {
        self.as_3d_const()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Physical strides for a Restride node.
/// A stride of 0 means broadcast along that axis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Strides(pub Vec<Dim>);

impl Strides {
    /// Row-major strides for any algebraic shape.
    #[must_use]
    pub fn row_major_for_shape(shape: &Shape) -> Self {
        let mut strides = vec![Dim::Const(1); shape.rank()];
        for i in (0..shape.rank().saturating_sub(1)).rev() {
            strides[i] = Dim::mul(strides[i + 1].clone(), shape.0[i + 1].clone());
        }
        Self(strides)
    }

    /// Drop an axis (used when propagating strides through a `Reduce`).
    #[must_use]
    pub fn remove_axis(&self, axis: usize) -> Self {
        let mut s = self.0.clone();
        s.remove(axis);
        Self(s)
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, s) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{s}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim, Shape, ShapeParams, Strides};

    #[test]
    fn dim_constructors_canonicalize_and_fold_constants() {
        let m = Dim::Symbol(0);
        let n = Dim::Symbol(1);

        assert_eq!(Dim::add(Dim::Const(0), m.clone()), m);
        assert_eq!(
            Dim::add(Dim::add(n.clone(), Dim::Const(2)), Dim::Const(3)),
            Dim::Add(Box::new(Dim::Const(5)), Box::new(n.clone()))
        );
        assert_eq!(
            Dim::mul(Dim::mul(m.clone(), Dim::Const(2)), Dim::Const(3)),
            Dim::Mul(Box::new(Dim::Const(6)), Box::new(m))
        );
        assert_eq!(Dim::min(Dim::min(n.clone(), n.clone()), n), Dim::Symbol(1));
        assert_eq!(Dim::sub(Dim::Const(3), Dim::Const(7)).as_const(), None);
    }

    #[test]
    fn dim_as_const_display_and_eval() {
        let expr = Dim::ceil_div(
            Dim::mul(Dim::Symbol(0), Dim::add(Dim::Symbol(1), Dim::Const(3))),
            Dim::Const(8),
        );
        let params = ShapeParams::from([4, 5]);

        assert_eq!(Dim::Const(9).as_const(), Some(9));
        assert_eq!(expr.as_const(), None);
        assert_eq!(expr.eval_u32(&params), Some(4));
        assert_eq!(expr.to_string(), "ceildiv(($0*(3+$1)),8)");
    }

    #[test]
    fn symbolic_row_major_strides_are_algebraic() {
        let shape = Shape(vec![Dim::Symbol(0), Dim::Symbol(1), Dim::Symbol(2)]);
        let strides = Strides::row_major_for_shape(&shape);
        let params = ShapeParams::from([2, 3, 4]);

        assert_eq!(shape.numel().eval_u32(&params), Some(24));
        assert_eq!(
            strides
                .0
                .iter()
                .map(|stride| stride.eval_u32(&params).unwrap())
                .collect::<Vec<_>>(),
            vec![12, 4, 1]
        );
    }
}
