//! Typed wrapper around `egg::Applier` that hides the four parameters every
//! rule in this crate ignores (`_subst`, `_searcher_ast`, `_rule_name`).
//!
//! egg's `Applier` trait signature is fixed by upstream:
//! ```ignore
//! fn apply_one(
//!     &self,
//!     egraph: &mut EGraph<L, A>,
//!     eclass: Id,
//!     subst: &Subst,
//!     searcher_ast: Option<&PatternAst<L>>,
//!     rule_name: Symbol,
//! ) -> Vec<Id>
//! ```
//! Every searcher in `tensor_ir` returns `vec![Subst::default()]` and every
//! applier ignores `_searcher_ast` and `_rule_name`. Rather than repeat
//! `_subst: &Subst, _searcher_ast: Option<&egg::PatternAst<TensorIr>>,
//! _rule_name: egg::Symbol` in 18 places (and re-import `egg::Symbol` for the
//! sole purpose of writing it down), implement `TypedApplier` on the rule
//! struct and wrap with `AdaptedApplier(rule)` at the `Rewrite::new` site.

use egg::{EGraph, Id, SearchMatches, Searcher, Subst, Var};

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;

/// Drop-in replacement for `egg::Applier` with the ignored parameters elided.
pub trait TypedApplier: Send + Sync + 'static {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id>;
}

/// Newtype that lifts a `TypedApplier` into an `egg::Applier`.
pub struct AdaptedApplier<A: TypedApplier>(pub A);

impl<A: TypedApplier> egg::Applier<TensorIr, TensorAnalysis> for AdaptedApplier<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<TensorIr, TensorAnalysis>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&egg::PatternAst<TensorIr>>,
        _rule_name: egg::Symbol,
    ) -> Vec<Id> {
        self.0.apply(egraph, eclass)
    }
}

/// Generic eclass-level searcher: every rule in this crate either matches
/// against a predicate over the whole e-class or doesn't match. This replaces
/// 16 near-identical `struct XxxSearcher` + `impl Searcher` blocks with one
/// closure-driven type. The predicate is stored inline and invoked per e-class;
/// matching eclasses emit a single empty substitution (no pattern vars are
/// bound because every applier in this crate re-derives its state from the
/// e-class itself).
pub struct SimpleEclassSearcher<F>(pub F);

impl<F> SimpleEclassSearcher<F>
where
    F: Fn(&EGraph<TensorIr, TensorAnalysis>, Id) -> bool + Send + Sync + 'static,
{
    pub fn new(predicate: F) -> Self {
        Self(predicate)
    }
}

impl<F> Searcher<TensorIr, TensorAnalysis> for SimpleEclassSearcher<F>
where
    F: Fn(&EGraph<TensorIr, TensorAnalysis>, Id) -> bool + Send + Sync + 'static,
{
    fn search_eclass(
        &self,
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        eclass: Id,
    ) -> Option<SearchMatches<'_, TensorIr>> {
        if (self.0)(egraph, eclass) {
            Some(SearchMatches {
                eclass,
                substs: vec![Subst::default()],
                ast: None,
            })
        } else {
            None
        }
    }

    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, TensorIr>> {
        self.search_eclass(egraph, eclass)
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}
