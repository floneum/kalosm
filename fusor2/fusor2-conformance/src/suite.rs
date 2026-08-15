//! The case registry, by area.
//!
//! Every area file returns [`Cases`]; every case runs on **every** session in
//! [`crate::harness::sessions`], so nothing in the suite mentions a concrete
//! backend. The shared case shapes live in [`support`] below rather than in a
//! new file, because a new file means a new `pub mod` line in a module list
//! other agents may also be editing.

pub mod attention_rope;
pub mod backward;
pub mod conv_pool;
pub mod dtypes;
pub mod elementwise;
pub mod indexing_scatter;
pub mod layers;
pub mod matmul;
pub mod multi_slot;
pub mod normalization;
pub mod quantized;
pub mod reductions;
pub mod sampling;
pub mod smoke;
pub mod views;

use crate::harness::Cases;

/// Every case, in a fixed area order.
///
/// A function rather than a `static`: a [`crate::harness::Case`] owns a boxed
/// closure, which cannot be built in a `const`.
pub fn registry() -> Cases {
    let mut all = Cases::new();
    // First: nothing below is interpretable while these fail.
    all.extend(smoke::cases());
    all.extend(elementwise::cases());
    all.extend(reductions::cases());
    all.extend(multi_slot::cases());
    all.extend(views::cases());
    all.extend(matmul::cases());
    all.extend(conv_pool::cases());
    all.extend(normalization::cases());
    all.extend(attention_rope::cases());
    all.extend(indexing_scatter::cases());
    all.extend(quantized::cases());
    all.extend(layers::cases());
    all.extend(backward::cases());
    all.extend(dtypes::cases());
    all.extend(sampling::cases());
    all
}

/// The registry as a function pointer, which is what `lib.rs` re-exports.
pub const REGISTRY: fn() -> Cases = registry;

// ---------------------------------------------------------------------------

/// Shared case shapes. Every area file is a table over these.
pub mod support {
    use fusor2::graph::GraphRef;
    use fusor2::{Dim, Dtype, Graph, Session, };
use fusor2::tensor::Dyn as Tensor;

    use crate::compare::{
        self, assert_gradient_matches_finite_difference, finite_difference_gradient,
    };
    use crate::harness::{
        Case, CaseError, CaseResult, dense_len, dims, fill, fill_range, from_f32,
    };

    /// A unary op, as the case table names it.
    pub type UnaryOp = fn(&Tensor) -> fusor2::Result<Tensor>;
    /// A binary op over two same-shape operands.
    pub type BinaryOp = fn(&Tensor, &Tensor) -> fusor2::Result<Tensor>;

    /// The input domain a case draws from. `Wide` is `[-0.5, 0.5)`; the others
    /// exist because `log` needs `x > 0`, `acosh` needs `x >= 1` and `atanh`
    /// needs `|x| < 1`.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum Domain {
        Wide,
        Positive,
        Unit,
        AboveOne,
        Custom(f32, f32),
    }

    impl Domain {
        pub fn sample(self, seed: u32, len: usize) -> Vec<f32> {
            match self {
                Domain::Wide => fill(seed, len),
                Domain::Positive => fill_range(seed, len, 0.25, 2.0),
                Domain::Unit => fill_range(seed, len, -0.8, 0.8),
                Domain::AboveOne => fill_range(seed, len, 1.25, 3.0),
                Domain::Custom(lo, hi) => fill_range(seed, len, lo, hi),
            }
        }
    }

    /// Read a tensor back as f32. One of exactly three host syncs.
    pub fn read(t: &Tensor) -> Result<Vec<f32>, CaseError> {
        t.to_vec_f32()
            .map_err(|e| -> CaseError { e.to_string().into() })
    }

    /// Read a rank-0 (or one-element) tensor.
    pub fn read_scalar(t: &Tensor) -> Result<f32, CaseError> {
        read(t)?
            .first()
            .copied()
            .ok_or_else(|| -> CaseError { "expected a scalar, got an empty tensor".into() })
    }

    /// A fresh graph on this session.
    pub fn graph_of(session: &Session) -> Graph {
        Graph::new(session)
    }

    /// Upload `data` into `graph` as an f32 buffer of `shape`.
    pub fn upload(graph: &GraphRef, shape: &[Dim], data: &[f32]) -> Result<Tensor, CaseError> {
        from_f32(graph, shape, data).map_err(|e| -> CaseError { e.to_string().into() })
    }

    /// The `sum_all` of a tensor, as the scalar loss every backward case seeds.
    pub fn loss_of(y: &Tensor) -> Result<Tensor, CaseError> {
        y.sum_all()
            .map_err(|e| -> CaseError { format!("sum_all: {e}").into() })
    }

    /// Compare against a host-side reference at `dtype`'s tolerance.
    pub fn expect_values(
        session: &Session,
        shape: &[u64],
        dtype: Dtype,
        actual: &[f32],
        expected: &[f32],
    ) -> CaseResult {
        let backend = if crate::harness::is_gpu(session) {
            "gpu"
        } else {
            "cpu"
        };
        let shape: Vec<usize> = shape.iter().map(|n| *n as usize).collect();
        compare::compare_for(dtype)(backend, &shape, expected, actual)?;
        Ok(())
    }

    /// Compare an f32 output against a host reference.
    pub fn expect_shaped(
        session: &Session,
        out_shape: &[u64],
        actual: &[f32],
        expected: &[f32],
    ) -> CaseResult {
        expect_values(session, out_shape, Dtype::F32, actual, expected)
    }

    /// `d(sum(y))/d(wrt)`, as a flat host vector.
    pub fn gradient_of(graph: &Graph, y: &Tensor, wrt: &Tensor) -> Result<Vec<f32>, CaseError> {
        let loss = loss_of(y)?;
        let grads = graph
            .backward_with(&loss, std::slice::from_ref(wrt))
            .map_err(|e| -> CaseError { format!("backward: {e}").into() })?;
        let g = grads.get(wrt).ok_or_else(|| -> CaseError {
            "no gradient reached the input; every requires-grad parent must receive one".into()
        })?;
        read(&g)
    }

    /// Rebuild the graph at `probe` and return `sum(build(x))`. The loss
    /// closure finite differences drive.
    fn probe_loss(
        session: &Session,
        shape: &[Dim],
        probe: &[f32],
        build: &dyn Fn(&Tensor) -> fusor2::Result<Tensor>,
    ) -> Result<f32, CaseError> {
        let graph = graph_of(session);
        let x = upload(graph.handle(), shape, probe)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    }

    /// Forward against a host reference, then backward against central
    /// differences. The shape every elementwise case takes.
    pub fn check_unary(
        session: &Session,
        shape: &[u64],
        data: &[f32],
        build: &dyn Fn(&Tensor) -> fusor2::Result<Tensor>,
        reference: &dyn Fn(f32) -> f32,
    ) -> CaseResult {
        let dimv = dims(shape);
        let graph = graph_of(session);
        let x = upload(graph.handle(), &dimv, data)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

        let actual = read(&y)?;
        let expected: Vec<f32> = data.iter().copied().map(reference).collect();
        expect_values(session, shape, Dtype::F32, &actual, &expected)?;

        let analytic = gradient_of(&graph, &y, &x)?;
        let usize_shape: Vec<usize> = shape.iter().map(|n| *n as usize).collect();
        let numeric = finite_difference_gradient(&usize_shape, data, &mut |probe| {
            probe_loss(session, &dimv, probe, build)
        })?;
        assert_gradient_matches_finite_difference(&analytic, &numeric)?;
        Ok(())
    }

    /// One unary elementwise case: forward parity plus a finite-difference
    /// backward.
    pub fn unary_case(
        area: &'static str,
        name: &'static str,
        shape: &'static [u64],
        seed: u32,
        domain: Domain,
        build: UnaryOp,
        reference: fn(f32) -> f32,
    ) -> Case {
        Case::new(area, name, move |session| {
            let data = domain.sample(seed, dense_len(&dims(shape)));
            check_unary(session, shape, &data, &build, &reference)
        })
    }

    /// One binary elementwise case over two same-shape operands. **Both**
    /// gradients are checked: a rule that forgets `d_rhs` still passes a
    /// forward-only comparison.
    pub fn binary_case(
        area: &'static str,
        name: &'static str,
        shape: &'static [u64],
        domain: Domain,
        build: BinaryOp,
        reference: fn(f32, f32) -> f32,
    ) -> Case {
        Case::new(area, name, move |session| {
            let len = dense_len(&dims(shape));
            let lhs = domain.sample(11, len);
            let rhs = domain.sample(23, len);
            let dimv = dims(shape);
            let usize_shape: Vec<usize> = shape.iter().map(|n| *n as usize).collect();

            let graph = graph_of(session);
            let a = upload(graph.handle(), &dimv, &lhs)?;
            let b = upload(graph.handle(), &dimv, &rhs)?;
            let y = build(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;

            let actual = read(&y)?;
            let expected: Vec<f32> = lhs
                .iter()
                .zip(&rhs)
                .map(|(x, y)| reference(*x, *y))
                .collect();
            expect_values(session, shape, Dtype::F32, &actual, &expected)?;

            let d_lhs = gradient_of(&graph, &y, &a)?;
            let numeric_lhs = finite_difference_gradient(&usize_shape, &lhs, &mut |probe| {
                let g = graph_of(session);
                let a = upload(g.handle(), &dimv, probe)?;
                let b = upload(g.handle(), &dimv, &rhs)?;
                let y = build(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
                read_scalar(&loss_of(&y)?)
            })?;
            assert_gradient_matches_finite_difference(&d_lhs, &numeric_lhs)?;

            let d_rhs = gradient_of(&graph, &y, &b)?;
            let numeric_rhs = finite_difference_gradient(&usize_shape, &rhs, &mut |probe| {
                let g = graph_of(session);
                let a = upload(g.handle(), &dimv, &lhs)?;
                let b = upload(g.handle(), &dimv, probe)?;
                let y = build(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
                read_scalar(&loss_of(&y)?)
            })?;
            assert_gradient_matches_finite_difference(&d_rhs, &numeric_rhs)?;
            Ok(())
        })
    }

    /// One comparison case.
    ///
    /// The forward is checked against a 1.0/0.0 reference in the operand's own
    /// dtype, and the backward must produce a gradient that is **all zeros** —
    /// not an absent rule. The tape validates that every requires-grad parent
    /// receives a gradient, so "no rule" and "a zero rule" are different
    /// outcomes and only one of them is correct.
    pub fn comparison_case(
        area: &'static str,
        name: &'static str,
        build: UnaryOp,
        reference: fn(f32) -> f32,
    ) -> Case {
        const SHAPE: &[u64] = &[4, 6];
        Case::new(area, name, move |session| {
            let data = Domain::Wide.sample(31, dense_len(&dims(SHAPE)));
            let dimv = dims(SHAPE);
            let graph = graph_of(session);
            let x = upload(graph.handle(), &dimv, &data)?;
            let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

            let actual = read(&y)?;
            let expected: Vec<f32> = data.iter().copied().map(reference).collect();
            expect_values(session, SHAPE, Dtype::F32, &actual, &expected)?;
            if let Some((i, v)) = actual
                .iter()
                .enumerate()
                .find(|(_, v)| **v != 0.0 && **v != 1.0)
            {
                return Err(format!(
                    "{name}: comparison {i} produced {v}; booleans are 1.0/0.0 in the \
                     operand's own dtype — there is no bool"
                )
                .into());
            }

            let grad = gradient_of(&graph, &y, &x)?;
            compare::assert_all_zero(name, &grad)?;
            Ok(())
        })
    }

}

// ---------------------------------------------------------------------------

/// The structural half of a generality case: saturate the graph the **real
/// frontend** emits, extract a plan from it, and read both.
///
/// Every generality assert in the acceptance list is one of four questions
/// about that pair, and all four are answered here rather than in each area
/// file:
///
/// * **did the law fire** — [`Probe::fired`], on `report.fired`. A rule that
///   silently stops matching the frontend's chain is how flash attention was
///   unreachable on both backends for a week while every numeric case passed;
/// * **did the law decline** — the same reading, negated. The `STRICT` half of
///   the acceptance list is a *decline* assert, and asserting only that a law
///   fires somewhere else does not cover it;
/// * **what did extraction materialize** — [`Probe::materializes_elements`],
///   over the plan's own buffer list. The whole memory win of a fused
///   reduction lives in that bit: if the extractor materializes the
///   intermediate, every numeric test still passes and the kernel is a memory
///   hog;
/// * **which schedule point did it resolve to** — [`Probe::theta`]. A node can
///   be admissible on paper and unselectable in fact, so "a fold was chosen"
///   is not the same claim as "a schedule was chosen for it".
///
/// A probe is never built from a hand-written graph: the caller passes the
/// tensors a frontend call returned.
pub mod probe {
    use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;
    use fusor2_ir::egraph::{Id, Saturate, SaturationBudget, SaturationReport};
    use fusor2_ir::extract::{ExtractBudget, Extractor, Plan};
    use fusor2_ir::ir::level1::SchedPoint;
    use fusor2_ir::saturate::Driver;

    use crate::harness::CaseError;

    /// One saturation and one extraction of one frontend-built graph.
    pub struct Probe {
        pub report: SaturationReport,
        pub plan: Plan,
    }

    /// Saturate and extract the graph `outs` were built in.
    ///
    /// The pipeline is exactly `Session::resolve`'s: the session's own rule
    /// table, the session's own caps, the shipped `Driver` and the shipped
    /// `LocalSearch`. Anything else would prove a claim about a pipeline
    /// nothing runs.
    pub fn probe(session: &Session, outs: &[Tensor]) -> Result<Probe, CaseError> {
        let first = outs
            .first()
            .ok_or_else(|| -> CaseError { "a probe needs at least one root".into() })?;
        let graph = first.graph().clone();
        let caps = session.caps();
        let cost = fusor2_cost::Roofline::new(session.device().target().facts().clone());
        let extractor = fusor2_cost::LocalSearch::new(fusor2_tile::Planner::shared(), caps.clone())
            .with_registry(session.registry().clone());
        let ids: Vec<Id> = outs.iter().map(|t| t.id()).collect();
        graph
            .with_egraph(|eg| {
                for id in &ids {
                    eg.add_root(*id);
                }
                let report = Driver::new().saturate(
                    eg,
                    &caps,
                    session.rules(),
                    SaturationBudget::default(),
                )?;
                let roots: Vec<Id> = eg.roots().to_vec();
                let plan = extractor.extract(eg, &roots, &cost, ExtractBudget::default())?;
                Ok(Probe { report, plan })
            })
            .map_err(|e| -> CaseError { format!("saturate+extract: {e}").into() })
    }

    impl Probe {
        /// How many times a named rule fired.
        pub fn fired(&self, rule: &str) -> u32 {
            self.report
                .fired
                .iter()
                .find(|(n, _)| *n == rule)
                .map_or(0, |(_, n)| *n)
        }

        /// Every rule that fired at least once, sorted — the message a missing
        /// firing assert prints, so a reader can tell "the law is unlanded"
        /// from "the law stopped matching".
        pub fn fired_names(&self) -> Vec<&'static str> {
            let mut v: Vec<&'static str> = self
                .report
                .fired
                .iter()
                .filter(|(_, n)| *n > 0)
                .map(|(n, _)| *n)
                .collect();
            v.sort_unstable();
            v
        }

        /// The saturation report itself must be readable before anything below
        /// it means anything.
        pub fn require_saturated(&self, what: &str) -> Result<(), CaseError> {
            if !self.report.saturated {
                return Err(format!(
                    "{what}: did not saturate in {} rounds ({} applications, {} nodes). \
                     Every structural claim below this is unreadable while it is false.",
                    self.report.rounds, self.report.applications, self.report.final_nodes
                )
                .into());
            }
            if !self.report.truncated.is_empty() {
                return Err(format!(
                    "{what}: truncated {} class(es) at {} nodes. Truncation is never silent.",
                    self.report.truncated.len(),
                    self.report.final_nodes
                )
                .into());
            }
            Ok(())
        }

        /// A law must have fired on this graph.
        pub fn require_fired(&self, rule: &str, why: &str) -> Result<(), CaseError> {
            if self.fired(rule) > 0 {
                return Ok(());
            }
            Err(format!(
                "`{rule}` never fired while saturating the graph the frontend emits ({why}). \
                 {} applications over {} rounds fired {:?}. A rule that only derives its \
                 motivating example is a recognizer with extra steps.",
                self.report.applications,
                self.report.rounds,
                self.fired_names()
            )
            .into())
        }

        /// A law must **not** have fired on this graph. The `STRICT` half of
        /// the acceptance list is exactly this assert, and it is not implied
        /// by any number of firing asserts elsewhere.
        pub fn require_declined(&self, rule: &str, why: &str) -> Result<(), CaseError> {
            let n = self.fired(rule);
            if n == 0 {
                return Ok(());
            }
            Err(format!(
                "`{rule}` fired {n} time(s) on a value that forbids it ({why}). \
                 The acceptance test for this path is a byte-identical export, so an \
                 inexact law firing here is a wrong answer, not a slow one."
            )
            .into())
        }

        /// Total bytes the plan allocates.
        pub fn buffer_bytes(&self) -> u64 {
            self.plan
                .buffers
                .iter()
                .map(|b| b.elements.as_const().unwrap_or(0) * b.dtype.byte_size())
                .sum()
        }

        /// Every element count the plan allocates a buffer for, ascending.
        pub fn buffer_elements(&self) -> Vec<u64> {
            let mut v: Vec<u64> = self
                .plan
                .buffers
                .iter()
                .filter_map(|b| b.elements.as_const())
                .collect();
            v.sort_unstable();
            v
        }

        /// Dispatches in the extracted plan.
        pub fn launches(&self) -> usize {
            self.plan.launches.len()
        }

        /// Every schedule point the extraction resolved, in node order. A
        /// named-shape assert reads this, because "a fold was selected" and "a
        /// schedule was resolved for it" are different claims and only the
        /// second one is schedulable.
        pub fn thetas(&self) -> Vec<SchedPoint> {
            let mut ids: Vec<Id> = self.plan.extraction.theta.keys().copied().collect();
            ids.sort_unstable();
            ids.iter()
                .filter_map(|i| self.plan.extraction.theta.get(i).copied())
                .collect()
        }

        /// The resolved schedule point of every selected node in the plan's
        /// launches, paired with the launch it belongs to.
        pub fn theta(&self, index: usize) -> Option<SchedPoint> {
            self.thetas().get(index).copied()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_case_name_is_unique() {
        let registry = registry();
        let mut names = registry.names();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(
            before,
            names.len(),
            "a case name is registered twice; `run_case` would run both"
        );
    }

    #[test]
    fn every_case_is_area_qualified() {
        for case in registry().iter() {
            assert!(
                case.name.starts_with(case.area),
                "{} is filed under {}",
                case.name,
                case.area
            );
            assert!(case.name.contains("::"), "{} is not qualified", case.name);
        }
    }

    #[test]
    fn every_area_contributes_cases() {
        let registry = registry();
        for area in [
            "elementwise",
            "reductions",
            "views",
            "matmul",
            "conv_pool",
            "normalization",
            "attention_rope",
            "indexing_scatter",
            "quantized",
            "layers",
            "backward",
            "dtypes",
            "sampling",
        ] {
            assert!(
                registry.iter().any(|c| c.area == area),
                "{area} registered no cases"
            );
        }
    }

    #[test]
    fn the_registry_covers_the_reference_autograd_matrix() {
        // The acceptance bar is the ~181 tests in `fusor/src/autograd/tests.rs`
        // reproduced forward and backward. Case count is a coarse proxy, but a
        // registry that silently shrank is worth catching.
        let registry = registry();
        assert!(
            registry.len() >= 181,
            "the registry has {} cases; the reference autograd matrix alone is 181",
            registry.len()
        );
    }

    #[test]
    fn the_registry_function_pointer_is_the_registry() {
        assert_eq!(REGISTRY().len(), registry().len());
    }
}
