//! Dependence sets — track which thread-hierarchy levels and which named
//! `Var` references a value depends on.

use std::fmt;

use super::binder::BinderKind;
use super::memory::IndexLevel;
use super::var::VarRef;

/// Static bound on the value of a scalar (typically address) expression.
///
/// Populated bottom-up by `TensorAnalysis` for the cases it can prove tight:
/// `Const`, `BinOp(Add, ..)`, `BinOp(Mul, ..)`. The bound is an inclusive
/// upper bound on what the expression can evaluate to at runtime; consumers
/// (skeleton tg-buffer sizing, codegen) use it instead of walking the
/// expression subtree to recompute the same fact.
///
/// `None` for any expression containing a `Var` reference (the bound depends
/// on the enclosing `Theta.count`, which Analysis cannot inspect locally).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AddressProfile {
    /// Inclusive upper bound on the expression's runtime value, or `None`
    /// if the expression's range is unbounded with respect to local facts.
    pub max_value: Option<u32>,
}

impl AddressProfile {
    /// Profile for a literal `u32` constant.
    #[must_use]
    pub const fn from_const(v: u32) -> Self {
        Self { max_value: Some(v) }
    }

    /// Profile for `lhs + rhs`, summing the bounds.
    #[must_use]
    pub fn saturating_sum(lhs: Self, rhs: Self) -> Self {
        let max_value = lhs
            .max_value
            .zip(rhs.max_value)
            .map(|(a, b)| a.saturating_add(b));
        Self { max_value }
    }

    /// Profile for `lhs * rhs`, multiplying the bounds.
    #[must_use]
    pub fn saturating_product(lhs: Self, rhs: Self) -> Self {
        let max_value = lhs
            .max_value
            .zip(rhs.max_value)
            .map(|(a, b)| a.saturating_mul(b));
        Self { max_value }
    }
}

/// Dependence set: tracks which GPU hierarchy levels a value depends on.
/// Represented as bitflags for fast union via bitwise OR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DepSet(pub u8);

impl DepSet {
    pub const EMPTY: Self = Self(0);
    pub const LANE: Self = Self(1);
    pub const SIMDGROUP: Self = Self(2);
    pub const WORKGROUP: Self = Self(4);

    #[must_use]
    pub const fn contains_lane(self) -> bool {
        self.0 & Self::LANE.0 != 0
    }

    #[must_use]
    pub const fn contains_simdgroup(self) -> bool {
        self.0 & Self::SIMDGROUP.0 != 0
    }

    #[must_use]
    pub const fn contains_workgroup(self) -> bool {
        self.0 & Self::WORKGROUP.0 != 0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn from_index_level(level: IndexLevel) -> Self {
        match level {
            IndexLevel::Lane => Self::LANE,
            IndexLevel::Simdgroup => Self::SIMDGROUP,
            IndexLevel::Workgroup => Self::WORKGROUP,
        }
    }
}

impl fmt::Display for DepSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = if self.contains_lane() {
            write!(f, "lane")?;
            false
        } else {
            true
        };
        if self.contains_simdgroup() {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "simdgroup")?;
            first = false;
        }
        if self.contains_workgroup() {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "workgroup")?;
        }
        write!(f, "}}")
    }
}

/// Tracks which `Var` references an expression depends on.
///
/// Holds the set of free Theta-scoped `Bound { kind: Theta, slot, depth }`
/// refs that escape the subtree. Dispatch-scoped refs are kernel-scope
/// and do not participate in Theta-depth tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct VarDepSet {
    pub bound: Vec<(u8, u32)>,
}

impl VarDepSet {
    #[must_use]
    pub const fn empty() -> Self {
        Self { bound: vec![] }
    }

    #[must_use]
    pub fn singleton_theta_bound(slot: u8, depth: u32) -> Self {
        Self {
            bound: vec![(slot, depth)],
        }
    }

    /// Build from a `VarRef`, choosing the right constructor.
    #[must_use]
    pub fn singleton(var: VarRef) -> Self {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot,
                depth,
            } => Self::singleton_theta_bound(slot, depth),
            // Dispatch-scoped refs are kernel-scope; they don't contribute
            // to the Theta-depth dep tracking this set models.
            VarRef::Bound {
                kind: BinderKind::Dispatch,
                ..
            } => Self::empty(),
        }
    }

    #[must_use]
    pub fn contains_theta_bound(&self, slot: u8, depth: u32) -> bool {
        self.bound.contains(&(slot, depth))
    }

    /// True iff the set contains the given `VarRef`.
    #[must_use]
    pub fn contains(&self, var: &VarRef) -> bool {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot,
                depth,
            } => self.contains_theta_bound(*slot, *depth),
            VarRef::Bound {
                kind: BinderKind::Dispatch,
                ..
            } => false,
        }
    }

    /// True iff any free Theta-bound ref of the given slot is present.
    #[must_use]
    pub fn contains_any_theta_slot(&self, slot: u8) -> bool {
        self.bound.iter().any(|(s, _)| *s == slot)
    }

    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut bound = self.bound.clone();
        for v in &other.bound {
            if !bound.contains(v) {
                bound.push(*v);
            }
        }
        bound.sort();
        Self { bound }
    }

    /// On Theta ascent: drop refs to the innermost binder (depth 0) and
    /// shift the remaining bound refs down by one.
    #[must_use]
    pub fn ascend_theta(&self) -> Self {
        let mut bound: Vec<_> = self
            .bound
            .iter()
            .filter_map(|(s, d)| d.checked_sub(1).map(|d1| (*s, d1)))
            .collect();
        bound.sort();
        Self { bound }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bound.is_empty()
    }
}
