//! `Gather` and all four `Scatter{Add}` lowerings, including duplicate
//! indices, which must accumulate.
//!
//! That is the load-bearing property in this area: `Scatter{Add}` is the
//! declared adjoint of `Gather`, so one token appearing twice in a batch must
//! receive the summed gradient. The duplicate cases draw their indices through
//! a modulus below both the index count and the row extent, so every run has a
//! repeated row *and* an unread row — an index set that happens to be a
//! permutation cannot tell a correct scatter-add from a scatter-set, and one
//! with full coverage cannot tell an explicit zero from a missing gradient.
//! Table extents and index counts are re-sampled per run; every index comes
//! from `fill_indices` over a sampled extent, so it is always in bounds.

use fusor2::{Dtype, Session};

use crate::harness::{
    CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fill_indices, from_u32, fuzz_case,
};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// The `[vocab, width]` table the gather/scatter cases read from.
const TABLE_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 8), FuzzDim::Range(1, 6)];

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// How many times each of `rows` rows appears in `idx`.
fn counts_of(idx: &[u32], rows: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; rows];
    for id in idx {
        counts[*id as usize] += 1.0;
    }
    counts
}

/// `count` indices whose modulus sits below both `count` and `rows`:
/// pigeonhole guarantees a repeated row, and the top row is never read.
/// Requires `count >= 2` and `rows >= 2`.
fn dup_indices(seed: u32, count: usize, rows: u64) -> Vec<u32> {
    let modulus = (rows - 1).min(count as u64 - 1).max(1) as u32;
    fill_indices(seed, count, modulus)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "index_select_dim0",
        TABLE_SPEC,
        |s, shape, seed| index_select_case(s, shape, seed, 0),
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "index_select_dim1",
        TABLE_SPEC,
        |s, shape, seed| index_select_case(s, shape, seed, 1),
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "index_select_duplicate_gradients_accumulate",
        TABLE_SPEC,
        dup_grad_case,
    ));
    // [vocab, width, batch, tokens].
    const EMBED_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 8),
        FuzzDim::Range(1, 6),
        FuzzDim::Range(1, 3),
        FuzzDim::Range(1, 3),
    ];
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "embedding",
        EMBED_SPEC,
        embedding_case,
    ));
    // `batch * tokens >= 2` so a duplicate can exist.
    const EMBED_DUP_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 8),
        FuzzDim::Range(1, 6),
        FuzzDim::Range(1, 3),
        FuzzDim::Range(2, 3),
    ];
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "embedding_backward_is_scatter_add",
        EMBED_DUP_SPEC,
        embedding_backward,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "gather_last",
        TABLE_SPEC,
        gather_last_case,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "gather_last_backward",
        TABLE_SPEC,
        gather_last_backward,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "scatter_add",
        TABLE_SPEC,
        scatter_add_case,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "scatter_add_duplicates_accumulate",
        TABLE_SPEC,
        scatter_add_dups,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "scatter_add_backward",
        TABLE_SPEC,
        scatter_add_backward,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "scatter_set_unique",
        TABLE_SPEC,
        scatter_set_case,
    ));
    cases.push(
        "indexing_scatter",
        "scatter_set_refuses_an_unproven_index",
        scatter_set_unproven,
    );
    const R2_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 5), FuzzDim::Range(1, 6)];
    const R3_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 4),
        FuzzDim::Range(2, 4),
        FuzzDim::Range(2, 5),
    ];
    const R4_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 3),
        FuzzDim::Range(2, 3),
        FuzzDim::Range(1, 4),
        FuzzDim::Range(2, 4),
    ];
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "i_rank2",
        R2_SPEC,
        index_rank2,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "i_rank3",
        R3_SPEC,
        index_rank3,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "i_rank4",
        R4_SPEC,
        index_rank4,
    ));
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "i_with_a_nonzero_pick",
        R3_SPEC,
        index_nonzero_pick,
    ));
    const BACKWARD_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 5), FuzzDim::Range(2, 6)];
    cases.push_case(fuzz_case(
        "indexing_scatter",
        "i_backward_zeroes_the_unselected_region",
        BACKWARD_SPEC,
        index_backward,
    ));
    cases
}

/// `index_select(dim, idx)`, forward and backward.
///
/// The backward is `Scatter{Add}`: under an all-ones seed, position `p` along
/// `dim` receives the number of times `p` appears in the index vector.
fn index_select_case(session: &Session, shape: &[u64], seed: u32, dim: usize) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let table = Domain::Wide.sample(seed, vocab * width);
    let n = Rng::new(seed ^ 0x5eed).range(1, 6) as usize;
    // The index stays inside the extent of the axis it selects along.
    let idx = fill_indices(seed ^ 0x9e37_79b9, n, shape[dim] as u32);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(shape), &table)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .index_select(dim, &i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (out_shape, expected) = if dim == 0 {
        let mut out = Vec::with_capacity(n * width);
        for id in &idx {
            let base = *id as usize * width;
            out.extend_from_slice(&table[base..base + width]);
        }
        (vec![n as u64, shape[1]], out)
    } else {
        let mut out = Vec::with_capacity(vocab * n);
        for r in 0..vocab {
            for id in &idx {
                out.push(table[r * width + *id as usize]);
            }
        }
        (vec![shape[0], n as u64], out)
    };
    expect_values(session, &out_shape, Dtype::F32, &read(&y)?, &expected)?;

    let grad = gradient_of(&graph, &y, &t)?;
    let mut want = vec![0.0f32; vocab * width];
    for id in &idx {
        if dim == 0 {
            for c in 0..width {
                want[*id as usize * width + c] += 1.0;
            }
        } else {
            for r in 0..vocab {
                want[r * width + *id as usize] += 1.0;
            }
        }
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[vocab, width],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// The property the whole area exists for, isolated: a row read twice gets
/// exactly twice the gradient, and a row never read gets an explicit zero
/// rather than no gradient at all.
fn dup_grad_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let table = Domain::Wide.sample(seed, vocab * width);
    let n = Rng::new(seed ^ 0x5eed).range(2, 8) as usize;
    let idx = dup_indices(seed ^ 0x9e37_79b9, n, shape[0]);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(shape), &table)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .index_select(0, &i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let counts = counts_of(&idx, vocab);
    for row in 0..vocab {
        for col in 0..width {
            let got = grad[row * width + col];
            if (got - counts[row]).abs() > 1e-5 {
                return Err(format!(
                    "row {row} was read {} time(s) but its gradient at column {col} is {got}: \
                     duplicate indices must accumulate and unread rows must receive an \
                     explicit zero",
                    counts[row]
                )
                .into());
            }
        }
    }
    Ok(())
}

/// `embedding(ids: [B, T] u32) -> [B, T, W]`.
fn embedding_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let (batch, tokens) = (shape[2], shape[3]);
    let table = Domain::Wide.sample(seed, vocab * width);
    let ids = fill_indices(
        seed ^ 0x9e37_79b9,
        (batch * tokens) as usize,
        shape[0] as u32,
    );

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&shape[..2]), &table)?;
    let ids_t = from_u32(graph.handle(), &dims(&[batch, tokens]), &ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .embedding(&ids_t)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(ids.len() * width);
    for id in &ids {
        let base = *id as usize * width;
        expected.extend_from_slice(&table[base..base + width]);
    }
    let out_shape = [batch, tokens, shape[1]];
    expect_values(session, &out_shape, Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Trainer constraint 3: there is no hand-written embedding backward. The
/// adjoint is `Scatter{Add}`, so the table's gradient is the per-row token
/// count.
fn embedding_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let (batch, tokens) = (shape[2], shape[3]);
    let table = Domain::Wide.sample(seed, vocab * width);
    let ids = dup_indices(seed ^ 0x9e37_79b9, (batch * tokens) as usize, shape[0]);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&shape[..2]), &table)?;
    let ids_t = from_u32(graph.handle(), &dims(&[batch, tokens]), &ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .embedding(&ids_t)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let counts = counts_of(&ids, vocab);
    let want: Vec<f32> = (0..vocab * width).map(|i| counts[i / width]).collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[vocab, width],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// `gather_last`: one column per row, picked by a rank-1 index as long as the
/// row count. The row-offset adjustment is the whole point — an
/// implementation that forgets it reads row 0 every time.
fn gather_last_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let table = Domain::Wide.sample(seed, vocab * width);
    let picks = fill_indices(seed ^ 0x9e37_79b9, vocab, shape[1] as u32);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(shape), &table)?;
    let i = from_u32(graph.handle(), &dims(&[shape[0]]), &picks)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .gather_last(&i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = picks
        .iter()
        .enumerate()
        .map(|(r, c)| table[r * width + *c as usize])
        .collect();
    expect_values(session, &[shape[0]], Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Its adjoint is a one-hot scatter: exactly one 1 per row, in the picked
/// column.
fn gather_last_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let table = Domain::Wide.sample(seed, vocab * width);
    let picks = fill_indices(seed ^ 0x9e37_79b9, vocab, shape[1] as u32);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(shape), &table)?;
    let i = from_u32(graph.handle(), &dims(&[shape[0]]), &picks)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .gather_last(&i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let mut want = vec![0.0f32; vocab * width];
    for (r, c) in picks.iter().enumerate() {
        want[r * width + *c as usize] = 1.0;
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[vocab, width],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// `scatter_add(axis, idx, updates)`: `base` plus the updates at their rows.
fn scatter_add_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let base = Domain::Wide.sample(seed, vocab * width);
    let n = Rng::new(seed ^ 0x5eed).range(1, 4) as usize;
    let idx = fill_indices(seed ^ 0x9e37_79b9, n, shape[0] as u32);
    let updates = Domain::Wide.sample(seed.wrapping_add(1), n * width);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(shape), &base)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(graph.handle(), &dims(&[n as u64, shape[1]]), &updates)?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (row_n, row) in idx.iter().enumerate() {
        for c in 0..width {
            expected[*row as usize * width + c] += updates[row_n * width + c];
        }
    }
    expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Several updates aimed at the same row must all land. All four lowerings —
/// atomic, sort-segment, workgroup-private merge, one-hot contraction — have
/// to agree here, and only a colliding index distinguishes them from a
/// last-writer-wins scatter. The modulus sits below the update count, so
/// every run has a collision.
fn scatter_add_dups(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let base = vec![0.0f32; vocab * width];
    let n = Rng::new(seed ^ 0x5eed).range(3, 6) as usize;
    let modulus = shape[0].min(n as u64 - 1) as u32;
    let idx = fill_indices(seed ^ 0x9e37_79b9, n, modulus);
    let updates: Vec<f32> = (0..n * width).map(|i| (i + 1) as f32).collect();

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(shape), &base)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(graph.handle(), &dims(&[n as u64, shape[1]]), &updates)?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (row_n, row) in idx.iter().enumerate() {
        for c in 0..width {
            expected[*row as usize * width + c] += updates[row_n * width + c];
        }
    }
    // The hottest row is hit at least twice — check it first, by name.
    let counts = counts_of(&idx, vocab);
    let hot = counts
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(r, _)| r)
        .unwrap_or(0);
    let got = read(&y)?;
    for c in 0..width {
        let want = expected[hot * width + c];
        let have = got.get(hot * width + c).copied().unwrap_or(f32::NAN);
        if (have - want).abs() > 1e-4 {
            return Err(format!(
                "row {hot} column {c} is {have}, want {want}: {} updates target row {hot} and \
                 all of them must be summed, not overwritten",
                counts[hot]
            )
            .into());
        }
    }
    expect_values(session, shape, Dtype::F32, &got, &expected)?;
    Ok(())
}

/// The adjoint of `Scatter{Add}` passes the gradient through to the base
/// unchanged and gathers it into the updates.
fn scatter_add_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let base = Domain::Wide.sample(seed, vocab * width);
    let n = Rng::new(seed ^ 0x5eed).range(1, 4) as usize;
    let idx = fill_indices(seed ^ 0x9e37_79b9, n, shape[0] as u32);
    let updates = Domain::Wide.sample(seed.wrapping_add(1), n * width);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(shape), &base)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(graph.handle(), &dims(&[n as u64, shape[1]]), &updates)?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let d_base = gradient_of(&graph, &y, &b)?;
    if let Some((row_n, v)) = d_base
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("scatter_add base gradient {row_n} is {v}, not 1").into());
    }
    let d_upd = gradient_of(&graph, &y, &u)?;
    if let Some((row_n, v)) = d_upd
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("scatter_add update gradient {row_n} is {v}, not 1").into());
    }
    Ok(())
}

/// `Scatter{Set}` overwrites, and the base's gradient is zero in the written
/// region. The uniqueness proof holds by construction: the sampled indices
/// are deduplicated before upload.
fn scatter_set_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, width) = (shape[0] as usize, shape[1] as usize);
    let base = Domain::Wide.sample(seed, vocab * width);
    let mut idx: Vec<u32> = Vec::new();
    for v in fill_indices(seed ^ 0x9e37_79b9, vocab, shape[0] as u32) {
        if !idx.contains(&v) {
            idx.push(v);
        }
    }
    let n = idx.len();
    let updates = Domain::Wide.sample(seed.wrapping_add(1), n * width);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(shape), &base)?;
    let i = from_u32(graph.handle(), &dims(&[n as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(graph.handle(), &dims(&[n as u64, shape[1]]), &updates)?;
    let y = b
        .scatter_set(0, &i, &u, true)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (row_n, row) in idx.iter().enumerate() {
        for c in 0..width {
            expected[*row as usize * width + c] = updates[row_n * width + c];
        }
    }
    expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)?;

    let d_base = gradient_of(&graph, &y, &b)?;
    for row in 0..vocab {
        let want = f32::from(!idx.contains(&(row as u32)));
        for c in 0..width {
            let got = d_base[row * width + c];
            if (got - want).abs() > 1e-5 {
                return Err(format!(
                    "scatter_set base gradient at [{row}, {c}] is {got}, want {want}: an \
                     overwritten element receives nothing"
                )
                .into());
            }
        }
    }
    Ok(())
}

/// `unique` is a caller-supplied proof and `verify_l0` rejects `Set` without
/// it — otherwise the result would depend on which lane wrote last. A refusal
/// path, so the shapes stay fixed.
fn scatter_set_unproven(session: &Session) -> CaseResult {
    let graph = graph_of(session);
    let b = upload(
        graph.handle(),
        &dims(&[5, 3]),
        &Domain::Wide.sample(1087, 15),
    )?;
    let i = from_u32(graph.handle(), &dims(&[2]), &[0u32, 4])
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(graph.handle(), &dims(&[2, 3]), &Domain::Wide.sample(1091, 6))?;
    if b.scatter_set(0, &i, &u, false).is_ok() {
        return Err(
            "scatter_set accepted an index without a uniqueness proof; the result \
                    would depend on lane order"
                .into(),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// i() / TensorIndex
// ---------------------------------------------------------------------------

/// `i((p, ..))` on a rank-2: exactly one bare index, and it removes its axis.
fn index_rank2(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (r, c) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, r * c);
    let p = Rng::new(seed ^ 0x5eed).range(0, shape[0] - 1) as usize;

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .i((p, ..))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    expect_values(
        session,
        &[shape[1]],
        Dtype::F32,
        &read(&y)?,
        &data[p * c..(p + 1) * c],
    )?;
    Ok(())
}

/// A bare index alongside `Full` and a `Range`.
fn index_rank3(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (a, b, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let data = Domain::Wide.sample(seed, a * b * c);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let p = rng.range(0, shape[1] - 1) as usize;
    let len = rng.range(1, shape[2]) as usize;
    let lo = rng.range(0, shape[2] - len as u64) as usize;

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .i((.., p, lo..lo + len))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::with_capacity(a * len);
    for i in 0..a {
        for k in lo..lo + len {
            expected.push(data[(i * b + p) * c + k]);
        }
    }
    expect_values(
        session,
        &[shape[0], len as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// Rank 4 with `RangeTo` and `RangeFrom` alongside the pick.
fn index_rank4(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (a, b, c, d) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );
    let data = Domain::Wide.sample(seed, a * b * c * d);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let hi = rng.range(1, shape[0]) as usize;
    let p = rng.range(0, shape[1] - 1) as usize;
    let q = rng.range(0, shape[3] - 1) as usize;

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .i((..hi, p, .., q..))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::with_capacity(hi * c * (d - q));
    for a2 in 0..hi {
        for c2 in 0..c {
            for d2 in q..d {
                expected.push(data[((a2 * b + p) * c + c2) * d + d2]);
            }
        }
    }
    expect_values(
        session,
        &[hi as u64, shape[2], (d - q) as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// A pick at a nonzero position with narrowed neighbours. This is the
/// two-node path (`slice` then `squeeze`): a `StrideSpec` offset rides on the
/// axis it names, and a dropped axis has no output axis left to carry it.
fn index_nonzero_pick(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (a, b, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let data = Domain::Wide.sample(seed, a * b * c);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let p = rng.range(1, shape[0] - 1) as usize;
    let len1 = rng.range(1, shape[1]) as usize;
    let lo1 = rng.range(0, shape[1] - len1 as u64) as usize;
    let len2 = rng.range(1, shape[2]) as usize;
    let lo2 = rng.range(0, shape[2] - len2 as u64) as usize;

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .i((p, lo1..lo1 + len1, lo2..lo2 + len2))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::with_capacity(len1 * len2);
    for j in lo1..lo1 + len1 {
        for k in lo2..lo2 + len2 {
            expected.push(data[(p * b + j) * c + k]);
        }
    }
    expect_values(
        session,
        &[len1 as u64, len2 as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// `i()` is a view, so its adjoint is ones in the selected region and zeros
/// everywhere else.
fn index_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (r, c) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, r * c);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let p = rng.range(0, shape[0] - 1) as usize;
    let len = rng.range(1, shape[1]) as usize;
    let lo = rng.range(0, shape[1] - len as u64) as usize;

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .i((p, lo..lo + len))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = (0..r * c)
        .map(|n| f32::from(n / c == p && (lo..lo + len).contains(&(n % c))))
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[r, c], &want, &grad, 1e-5, 1e-5)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    fn has(names: &[String], wanted: &str) -> bool {
        names
            .iter()
            .any(|n| n == &format!("indexing_scatter::{wanted}"))
    }

    #[test]
    fn every_named_gather_and_scatter_is_registered() {
        let names = registered();
        for wanted in [
            "index_select_dim0",
            "index_select_dim1",
            "embedding",
            "gather_last",
            "scatter_add",
            "scatter_set_unique",
            "i_rank2",
            "i_rank3",
            "i_rank4",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn the_duplicate_accumulation_cases_are_registered() {
        let names = registered();
        for wanted in [
            "index_select_duplicate_gradients_accumulate",
            "embedding_backward_is_scatter_add",
            "scatter_add_duplicates_accumulate",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn dup_indices_repeat_a_row_and_leave_one_unread() {
        // A permutation index cannot tell scatter-add from scatter-set, and a
        // fully covering one cannot tell an explicit zero from no gradient.
        for seed in [1u32, 7, 1009, 0xdead] {
            for (count, rows) in [(4usize, 5u64), (2, 2), (3, 3), (8, 4)] {
                let idx = dup_indices(seed, count, rows);
                assert!(idx.iter().all(|i| (*i as u64) < rows), "{idx:?}");
                let counts = counts_of(&idx, rows as usize);
                assert!(
                    counts.iter().any(|c| *c >= 2.0),
                    "no repeated row in {idx:?}"
                );
                assert!(
                    counts.iter().any(|c| *c == 0.0),
                    "no unread row in {idx:?}"
                );
                assert_eq!(counts.iter().sum::<f32>(), count as f32);
            }
        }
    }

    #[test]
    fn the_counts_are_the_row_multiplicities() {
        assert_eq!(
            counts_of(&[2, 0, 2, 3], 5),
            vec![1.0, 0.0, 2.0, 1.0, 0.0]
        );
    }
}
