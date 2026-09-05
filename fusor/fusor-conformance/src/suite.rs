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
    use fusor::graph::GraphRef;
    use fusor::tensor::Dyn as Tensor;
    use fusor::{Dim, Dtype, Graph, Session};

    use crate::compare::{
        self, assert_gradient_matches_finite_difference, finite_difference_gradient,
    };
    use crate::harness::{
        Case, CaseError, CaseResult, FuzzDim, dense_len, dims, fill, fill_range, from_f32,
        fuzz_case,
    };

    /// A unary op, as the case table names it.
    pub type UnaryOp = fn(&Tensor) -> fusor::Result<Tensor>;
    /// A binary op over two same-shape operands.
    pub type BinaryOp = fn(&Tensor, &Tensor) -> fusor::Result<Tensor>;

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

    /// Re-evaluate a scalar loss after replacing one f32 input leaf.
    ///
    /// Finite differences visit many values at one fixed shape. Reusing the
    /// graph preserves every perturbation and dispatch while avoiding a fresh
    /// graph build, saturation and extraction for each point.
    pub fn read_probe_loss(input: &Tensor, loss: &Tensor, probe: &[f32]) -> Result<f32, CaseError> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(probe));
        for value in probe {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        input
            .set_bytes(bytes)
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        // `resolve` deliberately returns early for an already-materialized
        // root. The input update invalidates its leaf buffer; invalidate the
        // requested loss as well so this perturbation executes the plan.
        loss.clear_device_buf();
        read_scalar(loss)
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

    /// Forward against a host reference, then backward against central
    /// differences. The shape every elementwise case takes.
    /// `gpu_forward_tol` replaces the F32 `(absolute, relative)` bound for
    /// the forward comparison on the GPU only, for an op whose driver
    /// implementation is a documented approximation; the adjoint check keeps
    /// its own bound.
    pub fn check_unary(
        session: &Session,
        shape: &[u64],
        data: &[f32],
        build: &dyn Fn(&Tensor) -> fusor::Result<Tensor>,
        reference: &dyn Fn(f32) -> f32,
        gpu_forward_tol: Option<(f32, f32)>,
    ) -> CaseResult {
        let dimv = dims(shape);
        let graph = graph_of(session);
        let x = upload(graph.handle(), &dimv, data)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

        let actual = read(&y)?;
        let expected: Vec<f32> = data.iter().copied().map(reference).collect();
        match gpu_forward_tol.filter(|_| crate::harness::is_gpu(session)) {
            Some((abs, rel)) => {
                let usize_shape: Vec<usize> = shape.iter().map(|n| *n as usize).collect();
                compare::approx_or_relative_compare(abs, rel)(
                    "gpu",
                    &usize_shape,
                    &expected,
                    &actual,
                )?;
            }
            None => expect_values(session, shape, Dtype::F32, &actual, &expected)?,
        }

        let analytic = gradient_of(&graph, &y, &x)?;
        let usize_shape: Vec<usize> = shape.iter().map(|n| *n as usize).collect();
        let probe_graph = graph_of(session);
        let probe_x = upload(probe_graph.handle(), &dimv, data)?;
        let probe_y = build(&probe_x).map_err(|e| -> CaseError { e.to_string().into() })?;
        let probe_loss = loss_of(&probe_y)?;
        let numeric = finite_difference_gradient(&usize_shape, data, &mut |probe| {
            read_probe_loss(&probe_x, &probe_loss, probe)
        })?;
        assert_gradient_matches_finite_difference(&analytic, &numeric)?;
        Ok(())
    }

    /// The shape a plain elementwise case fuzzes over: rank 2, both extents
    /// re-sampled per run. Extents stay small because every case also runs a
    /// finite-difference backward at every element.
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
        gpu_forward_tol: Option<(f32, f32)>,
    ) -> Case {
        fuzz_case(area, name, spec, move |session, shape, seed| {
            let data = domain.sample(seed, dense_len(&dims(shape)));
            check_unary(session, shape, &data, &build, &reference, gpu_forward_tol)
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
            let lhs_graph = graph_of(session);
            let lhs_a = upload(lhs_graph.handle(), &dimv, &lhs)?;
            let lhs_b = upload(lhs_graph.handle(), &dimv, &rhs)?;
            let lhs_y = build(&lhs_a, &lhs_b).map_err(|e| -> CaseError { e.to_string().into() })?;
            let lhs_loss = loss_of(&lhs_y)?;
            let numeric_lhs = finite_difference_gradient(&usize_shape, &lhs, &mut |probe| {
                read_probe_loss(&lhs_a, &lhs_loss, probe)
            })?;
            assert_gradient_matches_finite_difference(&d_lhs, &numeric_lhs)?;

            let d_rhs = gradient_of(&graph, &y, &b)?;
            let rhs_graph = graph_of(session);
            let rhs_a = upload(rhs_graph.handle(), &dimv, &lhs)?;
            let rhs_b = upload(rhs_graph.handle(), &dimv, &rhs)?;
            let rhs_y = build(&rhs_a, &rhs_b).map_err(|e| -> CaseError { e.to_string().into() })?;
            let rhs_loss = loss_of(&rhs_y)?;
            let numeric_rhs = finite_difference_gradient(&usize_shape, &rhs, &mut |probe| {
                read_probe_loss(&rhs_b, &rhs_loss, probe)
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
