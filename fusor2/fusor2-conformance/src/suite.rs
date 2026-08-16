//! The case registry, by area.
//!
//! Every area file returns [`Cases`]; every case runs on every session in
//! [`crate::harness::sessions`], so nothing in the suite mentions a concrete
//! backend.

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

/// Shared case shapes. Every area file is a table over these.
pub mod support {
    use fusor2::graph::GraphRef;
    use fusor2::{Dim, Dtype, Graph, Session, };
use fusor2::tensor::Dyn as Tensor;

    use crate::compare::{
        self, assert_gradient_matches_finite_difference, finite_difference_gradient,
    };
    use crate::harness::{
        Case, CaseError, CaseResult, FuzzDim, dense_len, dims, fill, fill_range, from_f32,
        fuzz_case,
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

    /// The shape a plain elementwise case fuzzes over: rank 2, both extents
    /// re-sampled per run. Extents stay small because every case also runs a
    /// finite-difference backward, which rebuilds the graph once per element.
    pub const ELEMENTWISE_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 16)];

    /// One unary elementwise case: forward parity plus a finite-difference
    /// backward, at a fresh shape per run.
    pub fn unary_case(
        area: &'static str,
        name: &'static str,
        spec: &'static [FuzzDim],
        domain: Domain,
        build: UnaryOp,
        reference: fn(f32) -> f32,
    ) -> Case {
        fuzz_case(area, name, spec, move |session, shape, seed| {
            let data = domain.sample(seed, dense_len(&dims(shape)));
            check_unary(session, shape, &data, &build, &reference)
        })
    }

    /// One binary elementwise case over two same-shape operands. Both
    /// gradients are checked: a rule that forgets `d_rhs` still passes a
    /// forward-only comparison.
    pub fn binary_case(
        area: &'static str,
        name: &'static str,
        spec: &'static [FuzzDim],
        domain: Domain,
        build: BinaryOp,
        reference: fn(f32, f32) -> f32,
    ) -> Case {
        fuzz_case(area, name, spec, move |session, shape, seed| {
            let len = dense_len(&dims(shape));
            // Offset the rhs seed so the two operand streams are unrelated.
            let lhs = domain.sample(seed, len);
            let rhs = domain.sample(seed ^ 0x9e37_79b9, len);
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
    /// dtype, and the backward must produce an all-zero gradient — not an
    /// absent rule.
    pub fn comparison_case(
        area: &'static str,
        name: &'static str,
        build: UnaryOp,
        reference: fn(f32) -> f32,
    ) -> Case {
        fuzz_case(area, name, ELEMENTWISE_SPEC, move |session, shape, seed| {
            let data = Domain::Wide.sample(seed, dense_len(&dims(shape)));
            let dimv = dims(shape);
            let graph = graph_of(session);
            let x = upload(graph.handle(), &dimv, &data)?;
            let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

            let actual = read(&y)?;
            let expected: Vec<f32> = data.iter().copied().map(reference).collect();
            expect_values(session, shape, Dtype::F32, &actual, &expected)?;
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
        // reproduced forward and backward; catches a registry that silently shrank.
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
