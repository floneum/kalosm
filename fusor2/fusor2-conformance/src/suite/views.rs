//! The ~22 view ops, all one `Restride` underneath, plus the `Scatter{Set}`
//! family (`cat`, `stack`, `pad`, `repeat`, `slice_assign`) and the two
//! `Window` forms.
//!
//! Every case checks the forward against a host-side reference **and** that
//! the adjoint routes each element's gradient back to the element it came
//! from — which for a broadcast axis is a sum and for an overlapping window is
//! a scatter. Shapes, slice bounds and window sizes are re-sampled per run;
//! every derived quantity comes from the sampled extents so it is always in
//! range.

use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fuzz_case};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_probe_loss, upload,
};

/// The rank-3 source spec. Every single-input case also runs a
/// finite-difference backward at every element, so the ceiling stays at 64
/// elements.
const SPEC3: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
];
/// Axis 2 must fit a window of at least 2.
const SPEC3_WIN: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(2, 4),
];
/// Axis 2 must fit a window strictly shorter than the extent (`c > w >= 2`),
/// so positions overlap and the adjoint is a `Scatter{Add}`.
const SPEC3_OVERLAP: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(3, 5),
];

/// `(outer, along, inner)` element counts around `axis`, row-major.
fn split(shape: &[u64], axis: usize) -> (usize, usize, usize) {
    let outer = shape[..axis].iter().product::<u64>() as usize;
    let along = shape[axis] as usize;
    let inner = shape[axis + 1..].iter().product::<u64>() as usize;
    (outer, along, inner)
}

fn ref_slice(
    d: &[f32],
    shape: &[u64],
    axis: usize,
    start: usize,
    len: usize,
) -> (Vec<u64>, Vec<f32>) {
    let (outer, along, inner) = split(shape, axis);
    let mut out = Vec::with_capacity(outer * len * inner);
    for o in 0..outer {
        for a in start..start + len {
            let base = (o * along + a) * inner;
            out.extend_from_slice(&d[base..base + inner]);
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len as u64;
    (out_shape, out)
}

fn ref_permute(d: &[f32], shape: &[u64], perm: &[usize]) -> (Vec<u64>, Vec<f32>) {
    let rank = shape.len();
    let mut in_strides = vec![1usize; rank];
    for a in (0..rank - 1).rev() {
        in_strides[a] = in_strides[a + 1] * shape[a + 1] as usize;
    }
    let out_shape: Vec<u64> = perm.iter().map(|&p| shape[p]).collect();
    let total: usize = shape.iter().product::<u64>() as usize;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let off: usize = (0..rank).map(|o| idx[o] * in_strides[perm[o]]).sum();
        out.push(d[off]);
        for o in (0..rank).rev() {
            idx[o] += 1;
            if idx[o] < out_shape[o] as usize {
                break;
            }
            idx[o] = 0;
        }
    }
    (out_shape, out)
}

/// `windows(axis 2, window, step)` over a rank-3 `[a, b, c]`: the axis becomes
/// `(c - window) / step + 1` positions and gains a trailing window axis.
fn ref_windows(d: &[f32], shape: &[u64], window: usize, step: usize) -> (Vec<u64>, Vec<f32>) {
    let (a, b, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let positions = (c - window) / step + 1;
    let mut out = Vec::with_capacity(a * b * positions * window);
    for i in 0..a {
        for j in 0..b {
            for p in 0..positions {
                for w in 0..window {
                    out.push(d[(i * b + j) * c + p * step + w]);
                }
            }
        }
    }
    (
        vec![shape[0], shape[1], positions as u64, window as u64],
        out,
    )
}

fn ref_pad(d: &[f32], shape: &[u64], axis: usize, lo: usize, hi: usize) -> (Vec<u64>, Vec<f32>) {
    let (outer, along, inner) = split(shape, axis);
    let mut out = Vec::with_capacity(outer * (lo + along + hi) * inner);
    for o in 0..outer {
        out.extend(std::iter::repeat_n(0.0f32, lo * inner));
        out.extend_from_slice(&d[o * along * inner..(o + 1) * along * inner]);
        out.extend(std::iter::repeat_n(0.0f32, hi * inner));
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] += (lo + hi) as u64;
    (out_shape, out)
}

/// Forward against the host reference, then a finite-difference backward.
///
/// The backward half is the point: every view is a `Restride`, and a
/// `Restride` whose adjoint forgot to sum over a stride-0 axis, or to
/// accumulate over overlapping window positions, produces the right forward
/// and the wrong gradient.
fn check_view(
    session: &Session,
    shape: &[u64],
    seed: u32,
    build: &dyn Fn(&Tensor) -> fusor2::Result<Tensor>,
    reference: &dyn Fn(&[f32]) -> (Vec<u64>, Vec<f32>),
) -> CaseResult {
    let len: usize = shape.iter().product::<u64>() as usize;
    let data = Domain::Wide.sample(seed, len);
    let dimv = dims(shape);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

    let (out_shape, expected) = reference(&data);
    let actual = read(&y)?;
    if actual.len() != expected.len() {
        return Err(format!(
            "produced {} elements, the reference view has {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_y = build(&probe_x).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[len], &data, &mut |probe| {
        read_probe_loss(&probe_x, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    cases.push_case(fuzz_case("views", "narrow", SPEC3, |s, shape, seed| {
        let c = shape[2];
        let mut rng = Rng::new(seed ^ 0x5eed);
        let len = rng.range(1, c) as usize;
        let start = rng.range(0, c - len as u64) as usize;
        check_view(s, shape, seed, &|x| x.narrow(2, start, len), &|d| {
            ref_slice(d, shape, 2, start, len)
        })
    }));

    cases.push_case(fuzz_case("views", "expand", SPEC3, |s, shape, seed| {
        let e = Rng::new(seed ^ 0x5eed).range(2, 5);
        let (a, b, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        check_view(
            s,
            shape,
            seed,
            &|x| {
                x.unsqueeze(1)?
                    .expand(&dims(&[shape[0], e, shape[1], shape[2]]))
            },
            &|d| {
                let mut out = Vec::with_capacity(a * e as usize * b * c);
                for i in 0..a {
                    for _ in 0..e {
                        out.extend_from_slice(&d[i * b * c..(i + 1) * b * c]);
                    }
                }
                (vec![shape[0], e, shape[1], shape[2]], out)
            },
        )
    }));

    cases.push_case(fuzz_case("views", "repeat", SPEC3, |s, shape, seed| {
        let r = Rng::new(seed ^ 0x5eed).range(2, 3) as usize;
        check_view(s, shape, seed, &|x| x.repeat(&[r, 1, 1]), &|d| {
            let mut out = Vec::with_capacity(r * d.len());
            for _ in 0..r {
                out.extend_from_slice(d);
            }
            (vec![r as u64 * shape[0], shape[1], shape[2]], out)
        })
    }));

    // Any factorization of the element count is a legal resize; `[c, a*b]`
    // differs from every flatten below.
    cases.push_case(fuzz_case("views", "resize", SPEC3, |s, shape, seed| {
        check_view(
            s,
            shape,
            seed,
            &|x| x.reshape_dims(&dims(&[shape[2], shape[0] * shape[1]])),
            &|d| (vec![shape[2], shape[0] * shape[1]], d.to_vec()),
        )
    }));

    cases.push_case(fuzz_case("views", "restride", SPEC3, |s, shape, seed| {
        check_view(s, shape, seed, &|x| x.permute(&[2, 0, 1]), &|d| {
            ref_permute(d, shape, &[2, 0, 1])
        })
    }));

    cases.push_case(fuzz_case(
        "views",
        "restride_strided_overlap",
        SPEC3_OVERLAP,
        |s, shape, seed| {
            let c = shape[2];
            let w = Rng::new(seed ^ 0x5eed).range(2, c - 1);
            check_view(s, shape, seed, &|x| x.windows(2, w as u32, 1), &|d| {
                ref_windows(d, shape, w as usize, 1)
            })
        },
    ));

    cases.push_case(fuzz_case(
        "views",
        "restride_layout",
        SPEC3,
        |s, shape, seed| {
            check_view(s, shape, seed, &|x| x.transpose(0, 2), &|d| {
                ref_permute(d, shape, &[2, 1, 0])
            })
        },
    ));

    // Step == window: every element appears exactly once.
    cases.push_case(fuzz_case(
        "views",
        "sliding_window_view",
        SPEC3,
        |s, shape, seed| {
            let w = Rng::new(seed ^ 0x5eed).range(1, shape[2]);
            check_view(
                s,
                shape,
                seed,
                &|x| x.windows(2, w as u32, w as u32),
                &|d| ref_windows(d, shape, w as usize, w as usize),
            )
        },
    ));

    cases.push_case(fuzz_case(
        "views",
        "sliding_window_view_strided",
        SPEC3_WIN,
        |s, shape, seed| {
            let mut rng = Rng::new(seed ^ 0x5eed);
            let w = rng.range(2, shape[2]);
            let step = rng.range(1, w);
            check_view(
                s,
                shape,
                seed,
                &|x| x.windows(2, w as u32, step as u32),
                &|d| ref_windows(d, shape, w as usize, step as usize),
            )
        },
    ));

    cases.push_case(fuzz_case("views", "squeeze", SPEC3, |s, shape, seed| {
        check_view(s, shape, seed, &|x| x.unsqueeze(1)?.squeeze(1), &|d| {
            (shape.to_vec(), d.to_vec())
        })
    }));

    cases.push_case(fuzz_case(
        "views",
        "squeeze_dims",
        SPEC3,
        |s, shape, seed| {
            check_view(
                s,
                shape,
                seed,
                &|x| x.unsqueeze(0)?.unsqueeze(2)?.squeeze(2)?.squeeze(0),
                &|d| (shape.to_vec(), d.to_vec()),
            )
        },
    ));

    cases.push_case(fuzz_case("views", "unsqueeze", SPEC3, |s, shape, seed| {
        check_view(s, shape, seed, &|x| x.unsqueeze(1), &|d| {
            (vec![shape[0], 1, shape[1], shape[2]], d.to_vec())
        })
    }));

    cases.push_case(fuzz_case(
        "views",
        "unsqueeze_dims",
        SPEC3,
        |s, shape, seed| {
            check_view(s, shape, seed, &|x| x.unsqueeze(0)?.unsqueeze(4), &|d| {
                (vec![1, shape[0], shape[1], shape[2], 1], d.to_vec())
            })
        },
    ));

    cases.push_case(fuzz_case(
        "views",
        "flatten_all",
        SPEC3,
        |s, shape, seed| {
            check_view(s, shape, seed, &|x| x.flatten_all(), &|d| {
                (vec![shape[0] * shape[1] * shape[2]], d.to_vec())
            })
        },
    ));

    cases.push_case(fuzz_case(
        "views",
        "flatten_first_n",
        SPEC3,
        |s, shape, seed| {
            check_view(s, shape, seed, &|x| x.flatten(0, 1), &|d| {
                (vec![shape[0] * shape[1], shape[2]], d.to_vec())
            })
        },
    ));

    cases.push_case(fuzz_case(
        "views",
        "flatten_last_n",
        SPEC3,
        |s, shape, seed| {
            check_view(s, shape, seed, &|x| x.flatten(1, 2), &|d| {
                (vec![shape[0], shape[1] * shape[2]], d.to_vec())
            })
        },
    ));

    cases.push_case(fuzz_case("views", "pad_axis", SPEC3, |s, shape, seed| {
        let mut rng = Rng::new(seed ^ 0x5eed);
        let lo = rng.range(0, 2) as usize;
        let hi = rng.range(0, 2) as usize;
        check_view(s, shape, seed, &|x| x.pad_axis(2, (lo, hi)), &|d| {
            ref_pad(d, shape, 2, lo, hi)
        })
    }));

    cases.push_case(fuzz_case(
        "views",
        "pad_with_zeros",
        SPEC3,
        |s, shape, seed| {
            let mut rng = Rng::new(seed ^ 0x5eed);
            let lo = rng.range(0, 2) as usize;
            let hi = rng.range(0, 2) as usize;
            check_view(s, shape, seed, &|x| x.pad_with_zeros(0, lo, hi), &|d| {
                ref_pad(d, shape, 0, lo, hi)
            })
        },
    ));

    cases.push_case(fuzz_case("views", "t", SPEC3, |s, shape, seed| {
        check_view(s, shape, seed, &|x| x.transpose(1, 2), &|d| {
            ref_permute(d, shape, &[0, 2, 1])
        })
    }));

    cases.push_case(fuzz_case("views", "chunk", SPEC3, |s, shape, seed| {
        let len = Rng::new(seed ^ 0x5eed).range(1, shape[1]) as usize;
        check_view(s, shape, seed, &|x| x.narrow(1, 0, len), &|d| {
            ref_slice(d, shape, 1, 0, len)
        })
    }));

    // `cat` along each of the three axes, at both ranks the reference covers.
    // The two operands' extents along the cat axis are sampled independently.
    const CAT1: &[FuzzDim] = &[FuzzDim::Range(1, 8)];
    const CAT2: &[FuzzDim] = &[FuzzDim::Range(1, 5), FuzzDim::Range(1, 5)];
    const CAT3: &[FuzzDim] = &[
        FuzzDim::Range(1, 4),
        FuzzDim::Range(1, 4),
        FuzzDim::Range(1, 4),
    ];
    cases.push_case(fuzz_case("views", "cat_rank1", CAT1, |s, shape, seed| {
        cat_case(s, shape, 0, seed)
    }));
    cases.push_case(fuzz_case("views", "cat_rank2", CAT2, |s, shape, seed| {
        cat_case(s, shape, 0, seed)
    }));
    cases.push_case(fuzz_case("views", "cat_dim0", CAT3, |s, shape, seed| {
        cat_case(s, shape, 0, seed)
    }));
    cases.push_case(fuzz_case("views", "cat_dim1", CAT3, |s, shape, seed| {
        cat_case(s, shape, 1, seed)
    }));
    cases.push_case(fuzz_case("views", "cat_dim2", CAT3, |s, shape, seed| {
        cat_case(s, shape, 2, seed)
    }));

    const STACK: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(1, 6)];
    cases.push_case(fuzz_case("views", "stack", STACK, stack_case));

    const SLICE_ASSIGN: &[FuzzDim] = &[FuzzDim::Range(2, 5), FuzzDim::Range(2, 8)];
    cases.push_case(fuzz_case(
        "views",
        "slice_assign",
        SLICE_ASSIGN,
        slice_assign_case,
    ));

    cases
}

/// `cat` is `Scatter{Set}` into a constant, and its adjoint hands each input
/// back its own slice of the gradient.
fn cat_case(session: &Session, shape: &[u64], axis: usize, seed: u32) -> CaseResult {
    let mut rshape = shape.to_vec();
    rshape[axis] = Rng::new(seed ^ 0x5eed).range(1, 5);
    let llen: usize = shape.iter().product::<u64>() as usize;
    let rlen: usize = rshape.iter().product::<u64>() as usize;
    let left = Domain::Wide.sample(seed, llen);
    let right = Domain::Wide.sample(seed ^ 0x9e37_79b9, rlen);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(shape), &left)?;
    let b = upload(graph.handle(), &dims(&rshape), &right)?;
    let y = Tensor::cat(&[a.clone(), b.clone()], axis)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    if actual.len() != llen + rlen {
        return Err(format!(
            "cat produced {} elements, want {}",
            actual.len(),
            llen + rlen
        )
        .into());
    }

    // Interleave along `axis`: per outer index, the lhs slab then the rhs
    // slab, each `along * inner` elements.
    let (outer, along_l, inner) = split(shape, axis);
    let along_r = rshape[axis] as usize;
    let mut expected = Vec::with_capacity(llen + rlen);
    for o in 0..outer {
        for (src, along) in [(&left, along_l), (&right, along_r)] {
            expected.extend_from_slice(&src[o * along * inner..(o + 1) * along * inner]);
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] += rshape[axis];
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected)?;

    // Each input's gradient is its own slice — all ones under `sum_all`.
    for (label, operand, len) in [("lhs", &a, llen), ("rhs", &b, rlen)] {
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
fn stack_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len: usize = shape.iter().product::<u64>() as usize;
    let a_data = Domain::Wide.sample(seed, len);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    let dimv = dims(shape);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &a_data)?;
    let b = upload(graph.handle(), &dimv, &b_data)?;
    let y = Tensor::stack(&[a.clone(), b.clone()], 0)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let mut expected = a_data.clone();
    expected.extend_from_slice(&b_data);
    expect_values(
        session,
        &[2, shape[0], shape[1]],
        Dtype::F32,
        &actual,
        &expected,
    )?;

    let grad = gradient_of(&graph, &y, &a)?;
    if grad.len() != len {
        return Err(format!("stack gradient has {} elements, want {len}", grad.len()).into());
    }
    Ok(())
}

/// `slice_assign` is the one scatter primitive `cat`, `stack`, `pad` and
/// `repeat` all lower to. Its adjoint is two-sided: the base receives the
/// gradient with the written region zeroed, the value receives that region.
fn slice_assign_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let len0 = rng.range(1, shape[0]) as usize;
    let start0 = rng.range(0, shape[0] - len0 as u64) as usize;
    let len1 = rng.range(1, shape[1]) as usize;
    let start1 = rng.range(0, shape[1] - len1 as u64) as usize;

    let base_data = Domain::Wide.sample(seed, rows * cols);
    let patch_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, len0 * len1);

    let graph = graph_of(session);
    let base = upload(graph.handle(), &dims(shape), &base_data)?;
    let patch = upload(
        graph.handle(),
        &dims(&[len0 as u64, len1 as u64]),
        &patch_data,
    )?;
    let y = base
        .slice_assign(&[start0..start0 + len0, start1..start1 + len1], &patch)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let mut expected = base_data.clone();
    for r in 0..len0 {
        for c in 0..len1 {
            expected[(start0 + r) * cols + start1 + c] = patch_data[r * len1 + c];
        }
    }
    expect_values(session, shape, Dtype::F32, &actual, &expected)?;

    // The base's gradient is 1 outside the written region and 0 inside it.
    let d_base = gradient_of(&graph, &y, &base)?;
    for row in 0..rows {
        for col in 0..cols {
            let inside =
                (start0..start0 + len0).contains(&row) && (start1..start1 + len1).contains(&col);
            let want = f32::from(!inside);
            let got = d_base[row * cols + col];
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
