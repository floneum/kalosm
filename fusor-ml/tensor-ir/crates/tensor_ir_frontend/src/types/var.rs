//! Typed identifiers for binders and parameters.
//!
//! Every variable reference in the IR is fully typed — `VarRef`
//! uses positional, kind-aware identifiers. `VarRef::Bound` names a slot of
//! some enclosing binder identified by `BinderKind`, with a De Bruijn `depth`
//! counted independently per kind (0 = innermost of that kind).

use std::fmt;
use std::str::FromStr;

use super::binder::{BinderKind, slots};
use super::memory::IndexLevel;

/// Reference to a variable.
///
/// `Bound` references are positional and per-kind: `depth` is a De Bruijn
/// level counting the enclosing binders of the given `kind` outward (0 =
/// innermost). A Theta substitution never disturbs a Dispatch-bound ref and
/// vice versa, because shift/subst filter on `kind`. Two structurally
/// identical terms are alpha-equivalent iff their `Bound` refs match
/// exactly, so the e-graph can union them on its own.
///
/// Register-blocked accumulators no longer get a separate variant — the
/// output's identity comes from its position in the dispatch's
/// `children_list` (one `(value, addr)` pair per output), and every
/// accumulator is the canonical `Bound { Theta, Acc, 0 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarRef {
    Bound {
        kind: BinderKind,
        slot: u8,
        depth: u32,
    },
}

impl VarRef {
    /// Construct a `Bound { kind: Theta, slot: Iter, depth }` ref.
    #[must_use]
    pub const fn iter(depth: u32) -> Self {
        Self::Bound {
            kind: BinderKind::Theta,
            slot: slots::THETA_ITER,
            depth,
        }
    }

    /// Construct a `Bound { kind: Theta, slot: Acc, depth }` ref.
    #[must_use]
    pub const fn acc(depth: u32) -> Self {
        Self::Bound {
            kind: BinderKind::Theta,
            slot: slots::THETA_ACC,
            depth,
        }
    }

    /// Construct a thread-index ref (Lane, Simdgroup, Workgroup) bound by
    /// the enclosing `Dispatch`. Depth is 0 because Dispatch cannot nest.
    #[must_use]
    pub const fn thread(level: IndexLevel) -> Self {
        Self::Bound {
            kind: BinderKind::Dispatch,
            slot: index_level_slot(level),
            depth: 0,
        }
    }
}

const fn index_level_slot(level: IndexLevel) -> u8 {
    match level {
        IndexLevel::Lane => slots::DISPATCH_LANE,
        IndexLevel::Simdgroup => slots::DISPATCH_SIMDGROUP,
        IndexLevel::Workgroup => slots::DISPATCH_WORKGROUP,
    }
}

/// Recover the `IndexLevel` for a Dispatch-bound ref's slot, if the slot
/// names a known thread-index level.
#[must_use]
pub const fn index_level_from_slot(slot: u8) -> Option<IndexLevel> {
    match slot {
        slots::DISPATCH_LANE => Some(IndexLevel::Lane),
        slots::DISPATCH_SIMDGROUP => Some(IndexLevel::Simdgroup),
        slots::DISPATCH_WORKGROUP => Some(IndexLevel::Workgroup),
        _ => None,
    }
}

fn theta_slot_name(slot: u8) -> Option<&'static str> {
    match slot {
        slots::THETA_ITER => Some("iter"),
        slots::THETA_ACC => Some("acc"),
        _ => None,
    }
}

fn parse_theta_slot(s: &str) -> Result<u8, String> {
    match s {
        "iter" => Ok(slots::THETA_ITER),
        "acc" => Ok(slots::THETA_ACC),
        _ => Err(format!("unknown theta slot: {s}")),
    }
}

fn dispatch_slot_name(slot: u8) -> Option<&'static str> {
    match slot {
        slots::DISPATCH_LANE => Some("lane"),
        slots::DISPATCH_SIMDGROUP => Some("simdgroup"),
        slots::DISPATCH_WORKGROUP => Some("workgroup"),
        _ => None,
    }
}

fn parse_dispatch_slot(s: &str) -> Result<u8, String> {
    match s {
        "lane" => Ok(slots::DISPATCH_LANE),
        "simdgroup" => Ok(slots::DISPATCH_SIMDGROUP),
        "workgroup" => Ok(slots::DISPATCH_WORKGROUP),
        _ => Err(format!("unknown dispatch slot: {s}")),
    }
}

impl fmt::Display for VarRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bound { kind, slot, depth } => {
                let slot_name = match kind {
                    BinderKind::Theta => theta_slot_name(*slot),
                    BinderKind::Dispatch => dispatch_slot_name(*slot),
                };
                match slot_name {
                    Some(name) => write!(f, "^{kind}:{name}:{depth}"),
                    None => write!(f, "^{kind}:s{slot}:{depth}"),
                }
            }
        }
    }
}

impl FromStr for VarRef {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let rest = s
            .strip_prefix('^')
            .ok_or_else(|| format!("var ref must start with ^: {s}"))?;
        let parts: Vec<&str> = rest.split(':').collect();
        match parts.as_slice() {
            [kind_s, slot_s, depth_s] => {
                let kind: BinderKind = kind_s.parse()?;
                let slot = match kind {
                    BinderKind::Theta => parse_theta_slot(slot_s)?,
                    BinderKind::Dispatch => parse_dispatch_slot(slot_s)?,
                };
                let depth: u32 = depth_s
                    .parse()
                    .map_err(|e| format!("bad bound depth: {e}"))?;
                Ok(Self::Bound { kind, slot, depth })
            }
            _ => Err(format!("bad var ref: {s}")),
        }
    }
}
