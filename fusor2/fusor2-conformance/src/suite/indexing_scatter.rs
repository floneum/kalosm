//! `Gather` and all four `Scatter{Add}` lowerings, including duplicate
//! indices, which must accumulate.
//!
//! `Scatter{Add}` is the adjoint of `Gather`, so a token appearing twice in a
//! batch receives the summed gradient. Every gather case uses an index vector
//! with a repeat and an unread row: a permutation cannot distinguish
//! scatter-add from scatter-set, and full coverage cannot distinguish an
//! explicit zero from a missing gradient.

use fusor2::{Dtype, Session};

use crate::harness::{CaseError, CaseResult, Cases, dims, from_u32};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// The table every gather case reads from: `[VOCAB, WIDTH]`.
const VOCAB: usize = 5;
const WIDTH: usize = 3;
const TABLE_LEN: usize = VOCAB * WIDTH;

/// Not a permutation: `2` appears twice and `4` never.
const IDS: &[u32] = &[2, 0, 2, 3];

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// How many times each row of the table is read by [`IDS`].
fn id_counts() -> Vec<f32> {
    let mut counts = vec![0.0f32; VOCAB];
    for id in IDS {
        counts[*id as usize] += 1.0;
    }
    counts
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push("indexing_scatter", "index_select_dim0", |s| {
        index_select_case(s, 0)
    });
    cases.push("indexing_scatter", "index_select_dim1", |s| {
        index_select_case(s, 1)
    });
    cases.push(
        "indexing_scatter",
        "index_select_duplicate_gradients_accumulate",
        dup_grad_case,
    );
    cases.push("indexing_scatter", "embedding", embedding_case);
    cases.push(
        "indexing_scatter",
        "embedding_backward_is_scatter_add",
        embedding_backward,
    );
    cases.push("indexing_scatter", "gather_last", gather_last_case);
    cases.push(
        "indexing_scatter",
        "gather_last_backward",
        gather_last_backward,
    );
    cases.push("indexing_scatter", "scatter_add", scatter_add_case);
    cases.push(
        "indexing_scatter",
        "scatter_add_duplicates_accumulate",
        scatter_add_dups,
    );
    cases.push(
        "indexing_scatter",
        "scatter_add_backward",
        scatter_add_backward,
    );
    cases.push("indexing_scatter", "scatter_set_unique", scatter_set_case);
    cases.push(
        "indexing_scatter",
        "scatter_set_refuses_an_unproven_index",
        scatter_set_unproven,
    );
    cases.push("indexing_scatter", "i_rank2", index_rank2);
    cases.push("indexing_scatter", "i_rank3", index_rank3);
    cases.push("indexing_scatter", "i_rank4", index_rank4);
    cases.push(
        "indexing_scatter",
        "i_with_a_nonzero_pick",
        index_nonzero_pick,
    );
    cases.push(
        "indexing_scatter",
        "i_backward_zeroes_the_unselected_region",
        index_backward,
    );
    cases
}

/// `index_select(dim, idx)`, forward and backward.
///
/// The backward is `Scatter{Add}`: under an all-ones seed, position `p` along
/// `dim` receives the number of times `p` appears in the index vector.
fn index_select_case(session: &Session, dim: usize) -> CaseResult {
    let table = Domain::Wide.sample(1009, TABLE_LEN);
    // Along dim 1 the index must stay inside WIDTH; it still repeats.
    let idx: Vec<u32> = if dim == 0 {
        IDS.to_vec()
    } else {
        vec![1, 0, 1]
    };

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let i = from_u32(graph.handle(), &dims(&[idx.len() as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .index_select(dim, &i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (out_shape, expected) = if dim == 0 {
        let mut out = Vec::with_capacity(idx.len() * WIDTH);
        for id in &idx {
            let base = *id as usize * WIDTH;
            out.extend_from_slice(&table[base..base + WIDTH]);
        }
        (vec![idx.len() as u64, WIDTH as u64], out)
    } else {
        let mut out = Vec::with_capacity(VOCAB * idx.len());
        for r in 0..VOCAB {
            for id in &idx {
                out.push(table[r * WIDTH + *id as usize]);
            }
        }
        (vec![VOCAB as u64, idx.len() as u64], out)
    };
    expect_values(session, &out_shape, Dtype::F32, &read(&y)?, &expected)?;

    let grad = gradient_of(&graph, &y, &t)?;
    let mut want = vec![0.0f32; TABLE_LEN];
    for id in &idx {
        if dim == 0 {
            for c in 0..WIDTH {
                want[*id as usize * WIDTH + c] += 1.0;
            }
        } else {
            for r in 0..VOCAB {
                want[r * WIDTH + *id as usize] += 1.0;
            }
        }
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[VOCAB, WIDTH],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// The row read twice gets exactly twice the gradient; the row never read gets
/// an explicit zero.
fn dup_grad_case(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1013, TABLE_LEN);
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let i = from_u32(graph.handle(), &dims(&[IDS.len() as u64]), IDS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .index_select(0, &i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let counts = id_counts();
    for row in 0..VOCAB {
        for col in 0..WIDTH {
            let got = grad[row * WIDTH + col];
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
fn embedding_case(session: &Session) -> CaseResult {
    const BATCH: u64 = 2;
    const TOKENS: u64 = 2;
    let table = Domain::Wide.sample(1019, TABLE_LEN);

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let ids = from_u32(graph.handle(), &dims(&[BATCH, TOKENS]), IDS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .embedding(&ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(IDS.len() * WIDTH);
    for id in IDS {
        let base = *id as usize * WIDTH;
        expected.extend_from_slice(&table[base..base + WIDTH]);
    }
    let shape = [BATCH, TOKENS, WIDTH as u64];
    expect_values(session, &shape, Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Embedding has no hand-written backward: its adjoint is `Scatter{Add}`, so
/// the table's gradient is the per-row token count.
fn embedding_backward(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1021, TABLE_LEN);
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let ids = from_u32(graph.handle(), &dims(&[2, 2]), IDS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .embedding(&ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let counts = id_counts();
    let want: Vec<f32> = (0..TABLE_LEN).map(|i| counts[i / WIDTH]).collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[VOCAB, WIDTH],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// `gather_last`: one column per row, picked by a rank-1 index as long as the
/// row count. Each pick is offset by its row.
fn gather_last_case(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1031, TABLE_LEN);
    let picks: Vec<u32> = vec![2, 0, 1, 2, 0];

    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let i = from_u32(graph.handle(), &dims(&[VOCAB as u64]), &picks)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .gather_last(&i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = picks
        .iter()
        .enumerate()
        .map(|(r, c)| table[r * WIDTH + *c as usize])
        .collect();
    expect_values(session, &[VOCAB as u64], Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Its adjoint is a one-hot scatter: exactly one 1 per row, in the picked
/// column.
fn gather_last_backward(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1033, TABLE_LEN);
    let picks: Vec<u32> = vec![2, 0, 1, 2, 0];
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &table)?;
    let i = from_u32(graph.handle(), &dims(&[VOCAB as u64]), &picks)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = t
        .gather_last(&i)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let mut want = vec![0.0f32; TABLE_LEN];
    for (r, c) in picks.iter().enumerate() {
        want[r * WIDTH + *c as usize] = 1.0;
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[VOCAB, WIDTH],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// `scatter_add(axis, idx, updates)`: `base` plus the updates at their rows.
fn scatter_add_case(session: &Session) -> CaseResult {
    let base = Domain::Wide.sample(1039, TABLE_LEN);
    let idx: Vec<u32> = vec![1, 3];
    let updates = Domain::Wide.sample(1049, idx.len() * WIDTH);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &base)?;
    let i = from_u32(graph.handle(), &dims(&[idx.len() as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(
        graph.handle(),
        &dims(&[idx.len() as u64, WIDTH as u64]),
        &updates,
    )?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (n, row) in idx.iter().enumerate() {
        for c in 0..WIDTH {
            expected[*row as usize * WIDTH + c] += updates[n * WIDTH + c];
        }
    }
    expect_values(
        session,
        &[VOCAB as u64, WIDTH as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// Three updates aimed at the same row must all land, in every lowering
/// (atomic, sort-segment, workgroup-private merge, one-hot contraction).
fn scatter_add_dups(session: &Session) -> CaseResult {
    let base = vec![0.0f32; TABLE_LEN];
    let idx: Vec<u32> = vec![2, 2, 2, 0];
    let updates: Vec<f32> = (0..idx.len() * WIDTH).map(|i| (i + 1) as f32).collect();

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &base)?;
    let i = from_u32(graph.handle(), &dims(&[idx.len() as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(
        graph.handle(),
        &dims(&[idx.len() as u64, WIDTH as u64]),
        &updates,
    )?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (n, row) in idx.iter().enumerate() {
        for c in 0..WIDTH {
            expected[*row as usize * WIDTH + c] += updates[n * WIDTH + c];
        }
    }
    let got = read(&y)?;
    for c in 0..WIDTH {
        let want = expected[2 * WIDTH + c];
        let have = got.get(2 * WIDTH + c).copied().unwrap_or(f32::NAN);
        if (have - want).abs() > 1e-4 {
            return Err(format!(
                "row 2 column {c} is {have}, want {want}: three updates target row 2 and all \
                 three must be summed, not overwritten"
            )
            .into());
        }
    }
    expect_values(
        session,
        &[VOCAB as u64, WIDTH as u64],
        Dtype::F32,
        &got,
        &expected,
    )?;
    Ok(())
}

/// The adjoint of `Scatter{Add}` passes the gradient through to the base
/// unchanged and gathers it into the updates.
fn scatter_add_backward(session: &Session) -> CaseResult {
    let base = Domain::Wide.sample(1051, TABLE_LEN);
    let idx: Vec<u32> = vec![1, 3];
    let updates = Domain::Wide.sample(1061, idx.len() * WIDTH);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &base)?;
    let i = from_u32(graph.handle(), &dims(&[idx.len() as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(
        graph.handle(),
        &dims(&[idx.len() as u64, WIDTH as u64]),
        &updates,
    )?;
    let y = b
        .scatter_add(0, &i, &u)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let d_base = gradient_of(&graph, &y, &b)?;
    if let Some((n, v)) = d_base
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("scatter_add base gradient {n} is {v}, not 1").into());
    }
    let d_upd = gradient_of(&graph, &y, &u)?;
    if let Some((n, v)) = d_upd
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("scatter_add update gradient {n} is {v}, not 1").into());
    }
    Ok(())
}

/// `Scatter{Set}` overwrites, and the base's gradient is zero in the written
/// region.
fn scatter_set_case(session: &Session) -> CaseResult {
    let base = Domain::Wide.sample(1063, TABLE_LEN);
    let idx: Vec<u32> = vec![0, 4];
    let updates = Domain::Wide.sample(1069, idx.len() * WIDTH);

    let graph = graph_of(session);
    let b = upload(graph.handle(), &dims(&[VOCAB as u64, WIDTH as u64]), &base)?;
    let i = from_u32(graph.handle(), &dims(&[idx.len() as u64]), &idx)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(
        graph.handle(),
        &dims(&[idx.len() as u64, WIDTH as u64]),
        &updates,
    )?;
    let y = b
        .scatter_set(0, &i, &u, true)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = base.clone();
    for (n, row) in idx.iter().enumerate() {
        for c in 0..WIDTH {
            expected[*row as usize * WIDTH + c] = updates[n * WIDTH + c];
        }
    }
    expect_values(
        session,
        &[VOCAB as u64, WIDTH as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    let d_base = gradient_of(&graph, &y, &b)?;
    for row in 0..VOCAB {
        let want = f32::from(!idx.contains(&(row as u32)));
        for c in 0..WIDTH {
            let got = d_base[row * WIDTH + c];
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

/// `unique` is a caller-supplied proof; `verify_l0` rejects `Set` without it,
/// since the result would otherwise depend on which lane wrote last.
fn scatter_set_unproven(session: &Session) -> CaseResult {
    let graph = graph_of(session);
    let b = upload(
        graph.handle(),
        &dims(&[VOCAB as u64, WIDTH as u64]),
        &Domain::Wide.sample(1087, TABLE_LEN),
    )?;
    let i = from_u32(graph.handle(), &dims(&[2]), &[0u32, 4])
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let u = upload(
        graph.handle(),
        &dims(&[2, WIDTH as u64]),
        &Domain::Wide.sample(1091, 2 * WIDTH),
    )?;
    if b.scatter_set(0, &i, &u, false).is_ok() {
        return Err(
            "scatter_set accepted an index without a uniqueness proof; the result \
                    would depend on lane order"
                .into(),
        );
    }
    Ok(())
}

const R2: &[u64] = &[3, 4];
const R3: &[u64] = &[2, 3, 4];
const R4: &[u64] = &[2, 2, 3, 4];

/// `i((1, ..))` on a rank-2: exactly one bare index, and it removes its axis.
fn index_rank2(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1093, 12);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(R2), &data)?;
    let y = x
        .i((1usize, ..))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    expect_values(session, &[4], Dtype::F32, &read(&y)?, &data[4..8])?;
    Ok(())
}

/// A bare index alongside `Full` and a `Range`.
fn index_rank3(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1097, 24);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(R3), &data)?;
    let y = x
        .i((.., 1usize, 1..3))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::new();
    for i in 0..2 {
        for k in 1..3 {
            expected.push(data[(i * 3 + 1) * 4 + k]);
        }
    }
    expect_values(session, &[2, 2], Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// Rank 4 with `RangeTo` and `RangeFrom` alongside the pick.
fn index_rank4(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1103, 48);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(R4), &data)?;
    let y = x
        .i((..1usize, 1usize, .., 2usize..))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::new();
    for c in 0..3 {
        for d in 2..4 {
            // a = 0 is the only value `..1` admits; head 1 is the pick.
            expected.push(data[((0 * 2 + 1) * 3 + c) * 4 + d]);
        }
    }
    expect_values(session, &[1, 3, 2], Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// A pick at a nonzero position with narrowed neighbours, taking the two-node
/// path (`slice` then `squeeze`): a `StrideSpec` offset rides on the axis it
/// names, and a dropped axis has no output axis to carry it.
fn index_nonzero_pick(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1109, 24);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(R3), &data)?;
    let y = x
        .i((1usize, 1..3, 1..4))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mut expected = Vec::new();
    for j in 1..3 {
        for k in 1..4 {
            expected.push(data[(3 + j) * 4 + k]);
        }
    }
    expect_values(session, &[2, 3], Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// `i()` is a view, so its adjoint is ones in the selected region and zeros
/// everywhere else.
fn index_backward(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1117, 12);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(R2), &data)?;
    let y = x
        .i((1usize, 1..3))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = (0..12)
        .map(|n| f32::from(n / 4 == 1 && (1..3).contains(&(n % 4))))
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[3, 4], &want, &grad, 1e-5, 1e-5)?;
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
    fn the_index_vector_can_distinguish_add_from_set() {
        // A permutation index cannot tell scatter-add from scatter-set, and a
        // fully covering one cannot tell an explicit zero from no gradient.
        let counts = id_counts();
        assert!(
            counts.iter().any(|c| *c >= 2.0),
            "IDS must repeat at least one row: {IDS:?}"
        );
        assert!(
            counts.iter().any(|c| *c == 0.0),
            "IDS must leave at least one row unread: {IDS:?}"
        );
        assert_eq!(counts.iter().sum::<f32>(), IDS.len() as f32);
        assert!(IDS.iter().all(|i| (*i as usize) < VOCAB));
    }

    #[test]
    fn the_counts_are_the_row_multiplicities() {
        // IDS = [2, 0, 2, 3]: row 2 twice, rows 0 and 3 once, rows 1 and 4 never.
        assert_eq!(id_counts(), vec![1.0, 0.0, 2.0, 1.0, 0.0]);
    }
}
