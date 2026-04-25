//! Tensor shapes, dimensions, and strides.

use std::fmt;
use std::str::FromStr;

/// A single dimension extent: either a compile-time literal or a typed
/// positional reference to a runtime dim parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dim {
    Lit(u32),
    /// Runtime dim at slot `index` (passed alongside the dispatch).
    Sym(u32),
}

impl Dim {
    #[must_use]
    pub const fn as_lit(&self) -> Option<u32> {
        match self {
            Self::Lit(v) => Some(*v),
            Self::Sym(_) => None,
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lit(v) => write!(f, "{v}"),
            Self::Sym(idx) => write!(f, "${idx}"),
        }
    }
}

impl FromStr for Dim {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if let Some(rest) = s.strip_prefix('$') {
            let idx: u32 = rest
                .parse()
                .map_err(|e| format!("bad dim sym index: {e}"))?;
            Ok(Self::Sym(idx))
        } else {
            s.parse::<u32>()
                .map(Self::Lit)
                .map_err(|e| format!("bad dim: {e}"))
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
    pub fn static_numel(&self) -> Option<u32> {
        self.0
            .iter()
            .try_fold(1u32, |acc, d| d.as_lit().map(|v| acc * v))
    }

    #[must_use]
    pub fn remove_axis(&self, axis: usize) -> Self {
        let mut dims = self.0.clone();
        dims.remove(axis);
        Self(dims)
    }

    /// Try to interpret as a 3D shape with all literal dimensions.
    #[must_use]
    pub fn as_3d_lit(&self) -> Option<(u32, u32, u32)> {
        if self.0.len() != 3 {
            return None;
        }
        match (&self.0[0], &self.0[1], &self.0[2]) {
            (Dim::Lit(a), Dim::Lit(b), Dim::Lit(c)) => Some((*a, *b, *c)),
            _ => None,
        }
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

impl FromStr for Shape {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| format!("shape must be [dims]: {s}"))?;
        if s.is_empty() {
            return Ok(Self(vec![]));
        }
        let dims: Result<Vec<Dim>, _> = s.split(',').map(|d| d.trim().parse()).collect();
        Ok(Self(dims?))
    }
}

/// Physical strides for a Restride node.
/// A stride of 0 means broadcast along that axis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Strides(pub Vec<i64>);

impl Strides {
    /// Row-major strides for a shape whose dimensions are all literal. Returns
    /// `None` if any dimension is symbolic.
    #[must_use]
    pub fn row_major_for_shape(shape: &Shape) -> Option<Self> {
        let mut strides = vec![1i64; shape.rank()];
        for i in (0..shape.rank().saturating_sub(1)).rev() {
            let next_dim = match &shape.0[i + 1] {
                Dim::Lit(v) => i64::from(*v),
                Dim::Sym(_) => return None,
            };
            strides[i] = strides[i + 1] * next_dim;
        }
        Some(Self(strides))
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

impl FromStr for Strides {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| format!("strides must be [vals]: {s}"))?;
        if s.is_empty() {
            return Ok(Self(vec![]));
        }
        let vals: Result<Vec<i64>, _> = s
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<i64>()
                    .map_err(|e| format!("bad stride: {e}"))
            })
            .collect();
        Ok(Self(vals?))
    }
}
