//! Pre-codegen verification — proves the chosen extraction is well-formed.
//!
//! `verify` walks each dispatch's chosen extraction and checks structural
//! invariants that codegen would otherwise have to assert at runtime:
//!
//! - Every `Var(Bound { depth, .. })` reference is satisfied by an enclosing
//!   `Theta` (at least `depth + 1` `Theta`s are in scope when the `Var` is
//!   evaluated).
//! - The chosen extraction is acyclic.
//!
//! On success, returns a [`Verified`] handle wrapping the program by
//! reference. `lower_dispatch_program` accepts only `Verified`, so the
//! corresponding codegen panics (`unbound variable`, `recursive lowering
//! cycle`) become structurally unreachable.

use std::collections::HashSet;

use egg::{Id, Language};

use crate::language::{DispatchNode, SimdNode, TensorIr, extract_list};
use crate::skeleton::DispatchProgram;
use crate::types::{BinderKind, HasBinder, VarRef};

/// Witness that a `DispatchProgram` has passed structural verification.
///
/// Constructed only by [`verify`]. Codegen entry points consume this to
/// trade runtime asserts for type-level guarantees.
pub struct Verified<'a>(&'a DispatchProgram);

impl<'a> Verified<'a> {
    /// Borrow the underlying program.
    #[must_use]
    pub fn program(&self) -> &'a DispatchProgram {
        self.0
    }
}

/// Reasons a `DispatchProgram` may fail verification.
#[derive(Debug, Clone)]
pub enum VerifyError {
    /// A `Var(Bound { depth, .. })` reference was reached with fewer than
    /// `depth + 1` enclosing `Theta` binders in scope.
    UnboundVariable { var: VarRef, binders_in_scope: u32 },
    /// The chosen extraction has a cycle reachable from a dispatch root.
    CycleInExtraction { canonical: Id },
    /// An `Extract { index, tuple }` where `index` is at least the arity
    /// of the corresponding `Pack` reachable from `tuple`.
    TupleExtractOutOfBounds { index: u32, pack_arity: u32 },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnboundVariable {
                var,
                binders_in_scope,
            } => write!(
                f,
                "unbound variable {var:?}: {binders_in_scope} binder(s) in scope"
            ),
            Self::CycleInExtraction { canonical } => {
                write!(f, "cyclic chosen extraction at canonical {canonical:?}")
            }
            Self::TupleExtractOutOfBounds { index, pack_arity } => write!(
                f,
                "tuple extract index {index} out of bounds for pack of arity {pack_arity}"
            ),
        }
    }
}

impl std::error::Error for VerifyError {}

/// Verify that every `Var` in `program`'s chosen extraction is properly
/// bound and that the extraction is acyclic.
///
/// # Errors
///
/// Returns the first violation encountered.
pub fn verify(program: &DispatchProgram) -> Result<Verified<'_>, VerifyError> {
    let mut ctx = WalkCtx {
        program,
        visited: HashSet::new(),
        in_progress: HashSet::new(),
    };
    for dispatch in &program.dispatches {
        // Every dispatch starts with one Dispatch binder in scope; Theta
        // depth starts at 0. Track per-kind so shift/subst invariants hold.
        let scope = ScopeDepth {
            theta: 0,
            dispatch: 1,
        };
        for output in &dispatch.outputs {
            ctx.walk(output.value_id, scope)?;
            ctx.walk(output.addr_id, scope)?;
        }
    }
    Ok(Verified(program))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ScopeDepth {
    theta: u32,
    dispatch: u32,
}

impl ScopeDepth {
    fn bump(self, kind: BinderKind) -> Self {
        match kind {
            BinderKind::Theta => Self {
                theta: self.theta + 1,
                ..self
            },
            BinderKind::Dispatch => Self {
                dispatch: self.dispatch + 1,
                ..self
            },
        }
    }

    fn count(self, kind: BinderKind) -> u32 {
        match kind {
            BinderKind::Theta => self.theta,
            BinderKind::Dispatch => self.dispatch,
        }
    }
}

struct WalkCtx<'a> {
    program: &'a DispatchProgram,
    /// Memoization keyed on (canonical, scope) — different scope depths
    /// re-visit the same node legitimately.
    visited: HashSet<(Id, ScopeDepth)>,
    /// Currently-walking set with the same key. Re-entry signals a cycle.
    in_progress: HashSet<(Id, ScopeDepth)>,
}

impl WalkCtx<'_> {
    fn walk(&mut self, id: Id, scope: ScopeDepth) -> Result<(), VerifyError> {
        let canonical = self.program.egraph.find(id);
        let key = (canonical, scope);
        if self.visited.contains(&key) {
            return Ok(());
        }
        if !self.in_progress.insert(key) {
            return Err(VerifyError::CycleInExtraction { canonical });
        }

        let node = self
            .program
            .chosen_nodes
            .get(&canonical)
            .cloned()
            .or_else(|| self.program.egraph[canonical].iter().next().cloned())
            .expect("e-class has at least one e-node");

        match &node {
            TensorIr::Simd(SimdNode::Var(var)) => {
                if let VarRef::Bound { kind, depth, .. } = var
                    && *depth >= scope.count(*kind)
                {
                    self.in_progress.remove(&key);
                    return Err(VerifyError::UnboundVariable {
                        var: *var,
                        binders_in_scope: scope.count(*kind),
                    });
                }
                // BlockedAcc is kernel-scope; always in scope.
            }
            TensorIr::Dispatch(DispatchNode::Extract { index, tuple }) => {
                if let Some(arity) = self.pack_arity(*tuple)
                    && *index >= arity
                {
                    self.in_progress.remove(&key);
                    return Err(VerifyError::TupleExtractOutOfBounds {
                        index: *index,
                        pack_arity: arity,
                    });
                }
                self.walk(*tuple, scope)?;
            }
            other => {
                let info = other.binder_info();
                let children = other.children();
                for (i, child) in children.iter().enumerate() {
                    let child_scope = if let Some(info) = info
                        && (info.body_mask >> i) & 1 == 1
                    {
                        scope.bump(info.kind)
                    } else {
                        scope
                    };
                    self.walk(*child, child_scope)?;
                }
            }
        }

        self.in_progress.remove(&key);
        self.visited.insert(key);
        Ok(())
    }

    /// Statically resolve the arity of the `Pack` reachable from `tuple_id`,
    /// looking through `Theta` binders (the loop's accumulator carries the
    /// `init` pack's shape). Returns `None` for tuples whose source is a
    /// `Var` reference (kernel-scope resolution required at runtime).
    ///
    /// Non-Pack non-Var leaves (e.g., a single `Const` or scalar BinOp used
    /// as a Theta init) imply scalar (arity-1) tuples — extracting any
    /// index other than 0 is invalid.
    fn pack_arity(&self, tuple_id: Id) -> Option<u32> {
        let mut current = tuple_id;
        let mut guard = 0;
        loop {
            // Bound the chase to avoid pathological cycles (cycle detection
            // happens in `walk`; this is a safety net for `pack_arity` only).
            guard += 1;
            if guard > 32 {
                return None;
            }
            let canonical = self.program.egraph.find(current);
            let node = self
                .program
                .chosen_nodes
                .get(&canonical)
                .cloned()
                .or_else(|| self.program.egraph[canonical].iter().next().cloned())?;
            match node {
                TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
                    let len = extract_list(&self.program.egraph, children_list).len();
                    return u32::try_from(len).ok();
                }
                TensorIr::Simd(SimdNode::Theta {
                    children: [init, ..],
                    ..
                }) => {
                    current = init;
                }
                // A `Var` resolves at codegen time (kernel-scope binders);
                // we can't statically determine its arity here.
                TensorIr::Simd(SimdNode::Var(_)) => return None,
                // Any other leaf (Const, BinOp, Load, ...) is scalar.
                _ => return Some(1),
            }
        }
    }
}
