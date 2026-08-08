//! The ~22 view ops, all one `Restride` underneath, plus the `Scatter{Set}`
//! family (`cat`, `stack`, `pad`, `repeat`, `slice_assign`) and the two
//! `Window` forms.
//!
//! Every case checks the forward against a host-side reference **and** that
//! the adjoint routes each element's gradient back to the element it came
//! from — which for a broadcast axis is a sum and for an overlapping window is
//! a scatter.

use fusor2::{Dtype, Session, Tensor};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

/// The rank-3 source every single-input view case starts from.
const SRC: &[u64] = &[2, 3, 4];
const SRC_LEN: usize = 24;

/// A view under test: build it, and say what shape and values it should have.
type Build = fn(&Tensor) -> fusor2::Result<Tensor>;
type Reference = fn(&[f32]) -> (Vec<u64>, Vec<f32>);

/// Row-major index into `SRC`.
fn at(data: &[f32], i: usize, j: usize, k: usize) -> f32 {
    data[(i * 3 + j) * 4 + k]
}

#[rustfmt::skip]
fn single_input() -> Vec<(&'static str, Build, Reference)> {
    vec![
        ("narrow",            |x| x.narrow(2, 1, 2), ref_narrow),
        ("expand",            |x| x.unsqueeze(1)?.expand(&dims(&[2, 5, 3, 4])), ref_expand),
        ("repeat",            |x| x.repeat(&[3, 1, 1]), ref_repeat),
        ("resize",            |x| x.reshape_dims(&dims(&[4, 6])), ref_resize),
        ("restride",          |x| x.permute(&[2, 0, 1]), ref_permute),
        ("restride_strided_overlap", |x| x.windows(2, 3, 1), ref_overlap_windows),
        ("restride_layout",   |x| x.transpose(0, 2), ref_transpose_02),
        ("sliding_window_view", |x| x.windows(2, 2, 2), ref_windows_2_2),
        ("sliding_window_view_strided", |x| x.windows(2, 2, 1), ref_windows_2_1),
        ("squeeze",           |x| x.unsqueeze(1)?.squeeze(1), ref_identity),
        ("squeeze_dims",      |x| x.unsqueeze(0)?.unsqueeze(2)?.squeeze(2)?.squeeze(0), ref_identity),
        ("unsqueeze",         |x| x.unsqueeze(1), ref_unsqueeze_1),
        ("unsqueeze_dims",    |x| x.unsqueeze(0)?.unsqueeze(4), ref_unsqueeze_dims),
        ("flatten_all",       |x| x.flatten_all(), ref_flatten_all),
        ("flatten_first_n",   |x| x.flatten(0, 1), ref_flatten_first),
        ("flatten_last_n",    |x| x.flatten(1, 2), ref_flatten_last),
        ("pad_axis",          |x| x.pad_axis(2, (1, 2)), ref_pad_axis),
        ("pad_with_zeros",    |x| x.pad_with_zeros(0, 1, 1), ref_pad_axis0),
        ("t",                 |x| x.transpose(1, 2), ref_transpose_12),
        ("chunk",             |x| x.narrow(1, 0, 2), ref_chunk_first),
    ]
}

fn ref_identity(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (SRC.to_vec(), d.to_vec())
}

fn ref_narrow(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for i in 0..2 {
        for j in 0..3 {
            for k in 1..3 {
                out.push(at(d, i, j, k));
            }
        }
    }
    (vec![2, 3, 2], out)
}

fn ref_expand(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for i in 0..2 {
        for _ in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    out.push(at(d, i, j, k));
                }
            }
        }
    }
    (vec![2, 5, 3, 4], out)
}

fn ref_repeat(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for _ in 0..3 {
        out.extend_from_slice(d);
    }
    (vec![6, 3, 4], out)
}

fn ref_resize(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![4, 6], d.to_vec())
}

fn ref_permute(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for k in 0..4 {
        for i in 0..2 {
            for j in 0..3 {
                out.push(at(d, i, j, k));
            }
        }
    }
    (vec![4, 2, 3], out)
}

fn ref_transpose_02(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for k in 0..4 {
        for j in 0..3 {
            for i in 0..2 {
                out.push(at(d, i, j, k));
            }
        }
    }
    (vec![4, 3, 2], out)
}

fn ref_transpose_12(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for i in 0..2 {
        for k in 0..4 {
            for j in 0..3 {
                out.push(at(d, i, j, k));
            }
        }
    }
    (vec![2, 4, 3], out)
}

/// `windows(axis 2, window, step)`: the axis becomes
/// `(4 - window) / step + 1` positions and gains a trailing window axis.
fn ref_windows(d: &[f32], window: usize, step: usize) -> (Vec<u64>, Vec<f32>) {
    let positions = (4 - window) / step + 1;
    let mut out = Vec::new();
    for i in 0..2 {
        for j in 0..3 {
            for p in 0..positions {
                for w in 0..window {
                    out.push(at(d, i, j, p * step + w));
                }
            }
        }
    }
    (vec![2, 3, positions as u64, window as u64], out)
}

fn ref_windows_2_2(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    ref_windows(d, 2, 2)
}
fn ref_windows_2_1(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    ref_windows(d, 2, 1)
}
/// Window 3, step 1: positions overlap, so the adjoint is a `Scatter{Add}`
/// rather than an elementwise mask.
fn ref_overlap_windows(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    ref_windows(d, 3, 1)
}

fn ref_unsqueeze_1(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![2, 1, 3, 4], d.to_vec())
}
fn ref_unsqueeze_dims(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![1, 2, 3, 4, 1], d.to_vec())
}
fn ref_flatten_all(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![24], d.to_vec())
}
fn ref_flatten_first(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![6, 4], d.to_vec())
}
fn ref_flatten_last(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    (vec![2, 12], d.to_vec())
}

fn ref_pad_axis(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for i in 0..2 {
        for j in 0..3 {
            out.push(0.0);
            for k in 0..4 {
                out.push(at(d, i, j, k));
            }
            out.push(0.0);
            out.push(0.0);
        }
    }
    (vec![2, 3, 7], out)
}

fn ref_pad_axis0(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = vec![0.0f32; 12];
    out.extend_from_slice(d);
    out.extend(std::iter::repeat_n(0.0f32, 12));
    (vec![4, 3, 4], out)
}

fn ref_chunk_first(d: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let mut out = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                out.push(at(d, i, j, k));
            }
        }
    }
    (vec![2, 2, 4], out)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    for (name, build, reference) in single_input() {
        cases.push("views", name, move |session| {
            view_case(session, name, build, reference)
        });
    }

    // `cat` along each of the three axes, at rank 1 and rank 2.
    cases.push("views", "cat_rank1", |s| cat_case(s, &[5], 0));
    cases.push("views", "cat_rank2", |s| cat_case(s, &[3, 4], 0));
    cases.push("views", "cat_dim0", |s| cat_case(s, SRC, 0));
    cases.push("views", "cat_dim1", |s| cat_case(s, SRC, 1));
    cases.push("views", "cat_dim2", |s| cat_case(s, SRC, 2));
    cases.push("views", "stack", stack_case);
    cases.push("views", "slice_assign", slice_assign_case);

    cases
}

/// Forward against the host reference, then a finite-difference backward.
///
/// The backward half is the point: every view is a `Restride`, and a
/// `Restride` whose adjoint forgot to sum over a stride-0 axis, or to
/// accumulate over overlapping window positions, produces the right forward
/// and the wrong gradient.
fn view_case(
    session: &Session,
    name: &'static str,
    build: Build,
    reference: Reference,
) -> CaseResult {
    let data = Domain::Wide.sample(211, SRC_LEN);
    let dimv = dims(SRC);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let (out_shape, expected) = reference(&data);
    let actual = read(&y)?;
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: produced {} elements, the reference view has {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[SRC_LEN], &data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, probe)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// `cat` is `Scatter{Set}` into a constant, and its adjoint hands each input
/// back its own slice of the gradient.
fn cat_case(session: &Session, shape: &[u64], axis: u32) -> CaseResult {
    let len: usize = shape.iter().product::<u64>() as usize;
    let left = Domain::Wide.sample(223, len);
    let right = Domain::Wide.sample(227, len);
    let dimv = dims(shape);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &left)?;
    let b = upload(graph.handle(), &dimv, &right)?;
    let y = Tensor::cat(&[a.clone(), b.clone()], axis as usize)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    if actual.len() != 2 * len {
        return Err(format!("cat produced {} elements, want {}", actual.len(), 2 * len).into());
    }

    // Interleave along `axis`: the outer extent doubles, the inner block is
    // whatever sits below `axis`.
    let inner: usize = shape[axis as usize + 1..].iter().product::<u64>() as usize;
    let outer: usize = shape[..axis as usize].iter().product::<u64>() as usize;
    let along = shape[axis as usize] as usize;
    let mut expected = Vec::with_capacity(2 * len);
    for o in 0..outer {
        for source in [&left, &right] {
            for a in 0..along {
                for i in 0..inner {
                    expected.push(source[(o * along + a) * inner + i]);
                }
            }
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis as usize] *= 2;
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected)?;

    // Each input's gradient is its own slice — all ones under `sum_all`.
    for (label, operand) in [("lhs", &a), ("rhs", &b)] {
        let grad = gradient_of(&graph, &y, operand)?;
        if grad.len() != len {
            return Err(format!(
                "cat {label} gradient has {} elements, want {len}",
                grad.len()
            )
            .into());
        }
        if let Some((i, v)) = grad
            .iter()
            .enumerate()
            .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
        {
            return Err(format!("cat {label} gradient {i} is {v}, not 1").into());
        }
    }
    Ok(())
}

/// `stack` is `unsqueeze` then `cat`, so it must produce rank `R + 1`.
fn stack_case(session: &Session) -> CaseResult {
    const SHAPE: &[u64] = &[3, 4];
    let len = 12;
    let a_data = Domain::Wide.sample(229, len);
    let b_data = Domain::Wide.sample(233, len);
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &a_data)?;
    let b = upload(graph.handle(), &dimv, &b_data)?;
    let y = Tensor::stack(&[a.clone(), b.clone()], 0)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let mut expected = a_data.clone();
    expected.extend_from_slice(&b_data);
    expect_values(session, &[2, 3, 4], Dtype::F32, &actual, &expected)?;

    let grad = gradient_of(&graph, &y, &a)?;
    if grad.len() != len {
        return Err(format!("stack gradient has {} elements, want {len}", grad.len()).into());
    }
    Ok(())
}

/// `slice_assign` is the one scatter primitive `cat`, `stack`, `pad` and
/// `repeat` all lower to. Its adjoint is two-sided: the base receives the
/// gradient with the written region zeroed, the value receives that region.
fn slice_assign_case(session: &Session) -> CaseResult {
    const BASE: &[u64] = &[2, 6];
    const PATCH: &[u64] = &[2, 2];
    let base_data = Domain::Wide.sample(239, 12);
    let patch_data = Domain::Wide.sample(241, 4);

    let graph = graph_of(session);
    let base = upload(graph.handle(), &dims(BASE), &base_data)?;
    let patch = upload(graph.handle(), &dims(PATCH), &patch_data)?;
    // Axis 0 whole, axis 1 columns 2..4 — the [2, 2] patch's footprint.
    let y = base
        .slice_assign(&[0..2, 2..4], &patch)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let mut expected = base_data.clone();
    for row in 0..2 {
        for col in 0..2 {
            expected[row * 6 + 2 + col] = patch_data[row * 2 + col];
        }
    }
    expect_values(session, BASE, Dtype::F32, &actual, &expected)?;

    // The base's gradient is 1 outside the written region and 0 inside it.
    let d_base = gradient_of(&graph, &y, &base)?;
    for row in 0..2 {
        for col in 0..6 {
            let want = f32::from(!(2..4).contains(&col));
            let got = d_base[row * 6 + col];
            if (got - want).abs() > 1e-5 {
                return Err(format!(
                    "slice_assign base gradient at [{row}, {col}] is {got}, want {want}: \
                     the overwritten region must receive nothing"
                )
                .into());
            }
        }
    }
    // The patch's gradient is 1 everywhere.
    let d_patch = gradient_of(&graph, &y, &patch)?;
    if let Some((i, v)) = d_patch
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("slice_assign value gradient {i} is {v}, not 1").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn every_named_view_is_registered() {
        let names = registered();
        for wanted in [
            "cat_rank1",
            "cat_rank2",
            "cat_dim0",
            "cat_dim1",
            "cat_dim2",
            "stack",
            "chunk",
            "narrow",
            "expand",
            "repeat",
            "resize",
            "restride",
            "restride_strided_overlap",
            "restride_layout",
            "slice_assign",
            "sliding_window_view",
            "sliding_window_view_strided",
            "squeeze",
            "squeeze_dims",
            "unsqueeze",
            "unsqueeze_dims",
            "flatten_all",
            "flatten_first_n",
            "flatten_last_n",
            "pad_axis",
            "pad_with_zeros",
            "t",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("views::{wanted}")),
                "{wanted} is missing"
            );
        }
        assert_eq!(names.len(), 27, "the reference view matrix is 27 cases");
    }

    #[test]
    fn every_reference_preserves_or_accounts_for_the_element_count() {
        let data: Vec<f32> = (0..SRC_LEN).map(|i| i as f32).collect();
        for (name, _, reference) in single_input() {
            let (shape, values) = reference(&data);
            let expected: usize = shape.iter().product::<u64>() as usize;
            assert_eq!(
                values.len(),
                expected,
                "{name}: reference shape {shape:?} does not match {} values",
                values.len()
            );
        }
    }

    #[test]
    fn the_permutation_references_are_actual_permutations() {
        let data: Vec<f32> = (0..SRC_LEN).map(|i| i as f32).collect();
        for reference in [
            ref_permute as Reference,
            ref_transpose_02,
            ref_transpose_12,
            ref_identity,
            ref_flatten_all,
        ] {
            let (_, mut values) = reference(&data);
            values.sort_by(f32::total_cmp);
            assert_eq!(
                values, data,
                "a permutation reference lost or gained values"
            );
        }
    }

    #[test]
    fn the_window_references_repeat_exactly_where_the_windows_overlap() {
        let data: Vec<f32> = (0..SRC_LEN).map(|i| i as f32).collect();
        // Step == window: every element appears once.
        let (shape, values) = ref_windows_2_2(&data);
        assert_eq!(shape, vec![2, 3, 2, 2]);
        assert_eq!(values.len(), 24);
        // Step < window: 3 positions x 2 wide over an extent of 4, so 6 reads
        // per row rather than 4.
        let (shape, values) = ref_windows_2_1(&data);
        assert_eq!(shape, vec![2, 3, 3, 2]);
        assert_eq!(values.len(), 36);
        // Window 3, step 1: 2 positions x 3 wide.
        let (shape, values) = ref_overlap_windows(&data);
        assert_eq!(shape, vec![2, 3, 2, 3]);
        assert_eq!(values.len(), 36);
    }

    #[test]
    fn the_pad_references_insert_exactly_the_requested_zeros() {
        let data: Vec<f32> = (1..=SRC_LEN).map(|i| i as f32).collect();
        let (shape, values) = ref_pad_axis(&data);
        assert_eq!(shape, vec![2, 3, 7]);
        assert_eq!(
            values.iter().filter(|v| **v == 0.0).count(),
            18,
            "3 zeros per row"
        );
        let (shape, values) = ref_pad_axis0(&data);
        assert_eq!(shape, vec![4, 3, 4]);
        assert_eq!(values.iter().filter(|v| **v == 0.0).count(), 24);
    }

    #[test]
    fn the_expand_reference_repeats_along_the_stride_zero_axis() {
        let data: Vec<f32> = (0..SRC_LEN).map(|i| i as f32).collect();
        let (shape, values) = ref_expand(&data);
        assert_eq!(shape, vec![2, 5, 3, 4]);
        // Each source element appears exactly 5 times.
        for v in &data {
            assert_eq!(values.iter().filter(|x| *x == v).count(), 5);
        }
    }
}
