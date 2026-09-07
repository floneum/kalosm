//! The device-side row machinery both samplers and `top_k_pairs` are built
//! from: a descending sort with the declared tie rule, prefix scans, and the
//! weighted pick.
//!
//! Every shape here is `[n, 1]`, `[1, n]` or `[n, n]`, and every "replicate a
//! value across an axis" step goes through a `Contract` against a dense
//! all-ones constant: both emitters index every operand with the flat output
//! index and ignore `Operand.layout`, so a stride-0 broadcast operand reads
//! the wrong element and runs off the end of its buffer. Until that is fixed
//! in the two `lower` crates, anything built on `broadcast_as`, `expand` or
//! `repeat` is silently wrong here.
//!
//! The cost is `O(V^2)` work for a vocabulary of `V`; see the note on
//! [`sort_desc`].

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use fusor_ir::egraph::Id;

use crate::graph::{GraphRef, WeakGraphRef};
use crate::tensor::Tensor;
use crate::{Dim, Dtype, Error, Result};

/// The sentinel a non-finite logit is replaced by. It sorts below every real
/// logit and its `exp` underflows to zero, so such a token can never be drawn.
pub(crate) const NEG_MAX: f32 = -f32::MAX;

/// Guards a division by an all-zero weight total.
pub(crate) const EPSILON: f32 = 1.0e-20;

/// How far back the repetition penalty looks.
pub(crate) const PREVIOUS_TOKENS: usize = 64;

pub(crate) fn dims(v: &[u64]) -> Vec<Dim> {
    v.iter().map(|&d| Dim::Const(d)).collect()
}

/// A dense f32 host constant. Dense because a splat leaf would reintroduce the
/// stride-0 operand this module exists to avoid.
pub(crate) fn konst(g: &GraphRef, shape: &[u64], data: &[f32]) -> Result<Tensor> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::from_slice(g, Dtype::F32, &dims(shape), &bytes)
}

/// The fixed matrices a draw needs. None depends on the logits, so each is
/// built once per graph and reused.
///
/// `Graph::constant_from_raw` mints a fresh buffer id every call, so a
/// sampler that rebuilds constants per draw grows the graph into the planner
/// cliff documented on [`cached`].
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Fixed {
    /// `[n, 1]` of ones.
    Ones(u64),
    /// `[1, n]` of ones.
    RowOnes(u64),
    /// `[n, 1]`, `1` at row 0.
    First(u64),
    /// `[n, 1]` of `0, 1, .. n-1`.
    Iota(u64),
    /// `[n, 1]` of [`NEG_MAX`].
    Sentinel(u64),
    /// `[n, n]` where `(r, c) -> r`.
    IdCol(u64),
    /// `[n, n]` where `(r, c) -> c`.
    IdRow(u64),
    /// `[n, n]` lower triangle including the diagonal.
    TriInclusive(u64),
    /// `[n, n]` strict lower triangle.
    TriExclusive(u64),
}

impl Fixed {
    fn build(self, g: &GraphRef) -> Result<Tensor> {
        match self {
            Fixed::Ones(n) => konst(g, &[n, 1], &vec![1.0; n as usize]),
            Fixed::RowOnes(n) => konst(g, &[1, n], &vec![1.0; n as usize]),
            Fixed::First(n) => {
                let mut v = vec![0.0; n as usize];
                if let Some(slot) = v.first_mut() {
                    *slot = 1.0;
                }
                konst(g, &[n, 1], &v)
            }
            Fixed::Iota(n) => {
                let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
                konst(g, &[n, 1], &v)
            }
            Fixed::Sentinel(n) => konst(g, &[n, 1], &vec![NEG_MAX; n as usize]),
            Fixed::IdCol(n) => grid(g, n, n, |r, _| r as f32),
            Fixed::IdRow(n) => grid(g, n, n, |_, c| c as f32),
            Fixed::TriInclusive(n) => grid(g, n, n, |r, c| if c <= r { 1.0 } else { 0.0 }),
            Fixed::TriExclusive(n) => grid(g, n, n, |r, c| if c < r { 1.0 } else { 0.0 }),
        }
    }
}

type Constants = Mutex<Vec<(WeakGraphRef, HashMap<Fixed, Id>)>>;

/// The per-graph constant pool.
///
/// Keyed by graph so a constant is only ever reused inside the graph that owns
/// it, and held as a `(Weak, Id)` pair so the pool never keeps a graph alive.
///
/// Pooling only mitigates a planner cliff, it does not fix it: resolving the
/// same growing graph over and over eventually returns wrong numbers (with one
/// graph reused across draws, every draw from the 77th on reads the one-hot
/// selector as all-ones). The break is a function of accumulated graph size —
/// a planner/extractor defect in `fusor-cost`.
fn constants() -> &'static Constants {
    static CONSTANTS: OnceLock<Constants> = OnceLock::new();
    CONSTANTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn cached(g: &GraphRef, want: Fixed) -> Result<Tensor> {
    let existing = {
        let Ok(mut pool) = constants().lock() else {
            return want.build(g);
        };
        pool.retain(|(weak, _)| weak.strong_count() > 0);
        let key = GraphRef::as_ptr(g);
        pool.iter()
            .find(|(weak, _)| weak.upgrade().is_some_and(|h| GraphRef::as_ptr(&h) == key))
            .and_then(|(_, map)| map.get(&want).copied())
    };
    if let Some(id) = existing {
        return Ok(g.tensor(id));
    }
    let built = want.build(g)?;
    if let Ok(mut pool) = constants().lock() {
        let key = GraphRef::as_ptr(g);
        match pool
            .iter_mut()
            .find(|(weak, _)| weak.upgrade().is_some_and(|h| GraphRef::as_ptr(&h) == key))
        {
            Some((_, map)) => {
                map.insert(want, built.id());
            }
            None => {
                let mut map = HashMap::new();
                map.insert(want, built.id());
                pool.push((GraphRef::downgrade(g), map));
            }
        }
    }
    Ok(built)
}

/// `[n, 1]` of ones.
pub(crate) fn ones(g: &GraphRef, n: u64) -> Result<Tensor> {
    cached(g, Fixed::Ones(n))
}

/// `[n, 1]` holding `1` at row 0 and `0` elsewhere.
pub(crate) fn first_only(g: &GraphRef, n: u64) -> Result<Tensor> {
    cached(g, Fixed::First(n))
}

/// Replicate a one-element tensor to `[n, 1]`.
///
/// This is `ones([n,1]) @ s([1,1])` rather than a broadcast, for the reason in
/// the module docs.
pub(crate) fn fanout(s: &Tensor, n: u64) -> Result<Tensor> {
    let g = s.graph();
    ones(g, n)?.matmul(&s.reshape_dims(&dims(&[1, 1]))?)
}

/// `[rows, cols]` where entry `(r, c)` is `f(r, c)`.
fn grid(g: &GraphRef, rows: u64, cols: u64, f: impl Fn(u64, u64) -> f32) -> Result<Tensor> {
    let mut data = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            data.push(f(r, c));
        }
    }
    konst(g, &[rows, cols], &data)
}

/// Inclusive prefix sum of an `[n, 1]` column: `out[r] = sum_{s <= r} x[s]`.
pub(crate) fn prefix_inclusive(x: &Tensor, n: u64) -> Result<Tensor> {
    cached(x.graph(), Fixed::TriInclusive(n))?.matmul(x)
}

/// Exclusive prefix sum of an `[n, 1]` column: `out[r] = sum_{s < r} x[s]`.
pub(crate) fn prefix_exclusive(x: &Tensor, n: u64) -> Result<Tensor> {
    cached(x.graph(), Fixed::TriExclusive(n))?.matmul(x)
}

/// The total of an `[n, 1]` column, as `[1, 1]`.
pub(crate) fn total_of(x: &Tensor) -> Result<Tensor> {
    x.sum_all()?.reshape_dims(&dims(&[1, 1]))
}

/// The number of elements in a logits row, rejecting anything batched.
///
/// Every dim but the last must be `1`: a real batched top-k would need one
/// independent sort per row and this module does not implement it.
pub(crate) fn row_len(x: &Tensor) -> Result<u64> {
    let shape = x.shape();
    if shape.is_empty() {
        return Err(Error::Shape(
            "sampling needs a logits row, not a scalar".into(),
        ));
    }
    let leading: Vec<Dim> = shape[..shape.len() - 1].to_vec();
    for d in &leading {
        match d {
            Dim::Const(1) => {}
            _ => {
                return Err(Error::Shape(format!(
                    "sampling reads one logits row; got shape {shape:?}. Batched sampling over \
                     a leading axis is not implemented — slice the row you want first."
                )));
            }
        }
    }
    match shape[shape.len() - 1] {
        Dim::Const(n) if n > 0 => Ok(n),
        other => Err(Error::Shape(format!(
            "the logits row has a non-constant or empty extent {other:?}"
        ))),
    }
}

/// Replace every non-finite logit with [`NEG_MAX`], as the reference's
/// `is_finite` guard does. Returns an `[n, 1]` column.
pub(crate) fn sanitized_column(x: &Tensor, n: u64) -> Result<Tensor> {
    let col = x.reshape_dims(&dims(&[n, 1]))?;
    let col = if col.dtype() == Dtype::F32 {
        col
    } else {
        col.to_f32()?
    };
    // NaN fails `x == x`; the infinities fail `|x| <= f32::MAX`.
    let not_nan = col.eq_tensor(&col)?;
    let bounded = col.abs()?.lte_scalar(f32::MAX)?;
    let finite = not_nan.mul(&bounded)?;
    let sentinel = cached(x.graph(), Fixed::Sentinel(n))?;
    finite.where_cond(&col, &sentinel)
}

/// A descending sort of one logits row.
///
/// Returns `(values, ids)` as `[n, 1]` f32 columns. The order is the rule the
/// reference's `better_candidate` declares and `top_k_pairs` documents:
/// **value descending, and on an exact tie the larger token id first**.
///
/// The sort is a rank-by-counting: `rank[i]` is how many tokens beat token
/// `i`, a full `n x n` comparison, and the sorted arrays are read out by
/// contracting against the `rank == r` indicator. `O(n^2)` work and memory —
/// exact and device-resident, but it does not scale to a real vocabulary.
pub(crate) fn sort_desc(x: &Tensor, n: u64) -> Result<(Tensor, Tensor)> {
    let g = x.graph();
    let col = sanitized_column(x, n)?; // [n,1], value[i]
    let row = col.reshape_dims(&dims(&[1, n]))?; // [1,n]

    // value_col[i][j] = value[i];  value_row[i][j] = value[j]
    let value_col = col.matmul(&cached(g, Fixed::RowOnes(n))?)?;
    let value_row = ones(g, n)?.matmul(&row)?;
    // id_col[i][j] = i;  id_row[i][j] = j
    let id_col = cached(g, Fixed::IdCol(n))?;
    let id_row = cached(g, Fixed::IdRow(n))?;

    // beats[i][j] = value[j] > value[i] || (value[j] == value[i] && j > i).
    // The two disjuncts are disjoint, so the sum is already 0/1.
    let greater = value_row.gt_tensor(&value_col)?;
    let tied = value_row.eq_tensor(&value_col)?;
    let larger_id = id_row.gt_tensor(&id_col)?;
    let beats = greater.add(&tied.mul(&larger_id)?)?;

    // rank[i] in 0..n, 0 = best. Every rank is distinct: the tie rule is a
    // total order on (value, id).
    let rank = beats.matmul(&ones(g, n)?)?; // [n,1]

    // indicator[r][i] = (rank[i] == r), so a contraction reads out position r.
    let rank_row = ones(g, n)?.matmul(&rank.reshape_dims(&dims(&[1, n]))?)?;
    let indicator = rank_row.eq_tensor(&id_col)?; // id_col[r][i] = r

    let values = indicator.matmul(&col)?;
    let ids = indicator.matmul(&cached(g, Fixed::Iota(n))?)?;
    Ok((values, ids))
}

/// `exp(sorted[r] - sorted[0])` — the reference's unnormalised weight, whose
/// first entry is exactly `1` and whose ratio to it is `p[r] / p_max`.
pub(crate) fn weights_of(sorted_values: &Tensor, k: u64) -> Result<Tensor> {
    let top = sorted_values.narrow(0, 0, 1)?;
    let top = fanout(&top, k)?;
    sorted_values.sub(&top)?.exp()
}

/// The reference's `weighted_pick`, as a one-hot `[k, 1]` selector.
///
/// `masked` is the candidate weight column with every rejected entry already
/// zeroed. The pick is the first `r` whose inclusive prefix reaches
/// `random * total`, which is what walking the row and breaking on
/// `cumulative >= threshold` computes.
pub(crate) fn pick_one_hot(masked: &Tensor, k: u64, random: f32) -> Result<Tensor> {
    let g = masked.graph();
    let total = total_of(masked)?;
    let threshold = fanout(&total, k)?.mul_scalar(random)?;

    let inclusive = prefix_inclusive(masked, k)?;
    let exclusive = prefix_exclusive(masked, k)?;
    let first = first_only(g, k)?;

    // Exactly one r has `exclusive < threshold <= inclusive`. Row 0 is forced
    // in so that a `threshold` of zero still selects the top candidate, which
    // is what the reference's walk does when `random` is 0.
    let after_start = exclusive.lt_tensor(&threshold)?.maximum(&first)?;
    let reached = inclusive.gte_tensor(&threshold)?;
    let chosen = reached.mul(&after_start)?;

    // If rounding in the scan left `threshold` just above the last prefix,
    // nothing would be selected; the reference defaults to the top candidate.
    let none = fanout(&total_of(&chosen)?, k)?.eq_scalar(0.0f32)?;
    chosen.add(&first.mul(&none)?)
}

/// Contract a one-hot `[k, 1]` selector against a `[k, 1]` column.
pub(crate) fn gather_one_hot(one_hot: &Tensor, column: &Tensor) -> Result<Tensor> {
    total_of(&one_hot.mul(column)?)
}

/// A one-element `u32` token tensor, the shape `GpuSampledToken` promises.
pub(crate) fn as_token(value: &Tensor) -> Result<Tensor> {
    value.reshape_dims(&dims(&[1]))?.to_u32()
}

/// `splitmix64`, then the top 24 bits as a float in `[0, 1)`.
///
/// Host-side: the reference also passes the draw in as a `random` uniform, so
/// a seed fully determines the token without the logits leaving the device.
pub(crate) fn unit_random(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Mix a call counter into a seed so successive draws from one sampler differ
/// while the sequence stays a pure function of the seed.
pub(crate) fn unit_random_at(seed: u64, step: u64) -> f32 {
    unit_random(seed ^ step.wrapping_mul(0xD1B5_4A32_D192_ED03))
}

/// Tokens this process has already drawn, per graph. The scope is the
/// [`crate::Graph`] the logits belong to, which for a decode loop is the loop
/// itself.
///
/// Entries hold the token's `Id` and a [`Weak`] graph handle rather than the
/// `Tensor`: a `Tensor` owns a `GraphRef`, so parking one here would keep its
/// graph alive forever and the pruning below would never fire.
type History = Mutex<Vec<(WeakGraphRef, Vec<Id>)>>;

fn history() -> &'static History {
    static HISTORY: OnceLock<History> = OnceLock::new();
    HISTORY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Record a drawn token against its graph. Nothing is resolved here — only the
/// node id is kept, so a decode loop that never asks for a penalty never pays
/// for one.
pub(crate) fn remember(graph: &GraphRef, token: Id) {
    let Ok(mut log) = history().lock() else {
        return;
    };
    log.retain(|(g, _)| g.strong_count() > 0);
    let key = GraphRef::as_ptr(graph);
    if let Some((_, ids)) = log
        .iter_mut()
        .find(|(g, _)| g.upgrade().is_some_and(|g| GraphRef::as_ptr(&g) == key))
    {
        ids.push(token);
        let overflow = ids.len().saturating_sub(PREVIOUS_TOKENS);
        ids.drain(..overflow);
    } else {
        log.push((GraphRef::downgrade(graph), vec![token]));
    }
}

/// The tokens already drawn on this graph, newest last.
///
/// This resolves the remembered token tensors, so it is only ever called when
/// a penalty above `1` is actually requested.
pub(crate) fn previous_tokens(graph: &GraphRef) -> Vec<u32> {
    let ids = remembered_ids(graph);
    ids.iter()
        .filter_map(|id| graph.tensor(*id).to_vec_u32().ok())
        .filter_map(|v| v.first().copied())
        .collect()
}

/// [`previous_tokens`], awaited.
pub(crate) async fn previous_tokens_async(graph: &GraphRef) -> Vec<u32> {
    let mut out = Vec::new();
    for id in remembered_ids(graph) {
        if let Ok(v) = graph.tensor(id).to_vec_u32_async().await
            && let Some(first) = v.first()
        {
            out.push(*first);
        }
    }
    out
}

/// The remembered token ids for `graph`, newest last, without resolving.
fn remembered_ids(graph: &GraphRef) -> Vec<Id> {
    {
        let Ok(log) = history().lock() else {
            return Vec::new();
        };
        let key = GraphRef::as_ptr(graph);
        log.iter()
            .find(|(g, _)| g.upgrade().is_some_and(|g| GraphRef::as_ptr(&g) == key))
            .map(|(_, ids)| ids.clone())
            .unwrap_or_default()
    }
}

/// Apply the repetition penalty to a `[n, 1]` logits column.
///
/// The reference's rule, verbatim: only for a token that has already been
/// drawn, and only when the penalty is above `1`, a non-positive logit is
/// *multiplied* by the penalty and a positive one *divided* by it. Both move
/// the logit down.
pub(crate) fn apply_repetition_penalty(
    column: &Tensor,
    n: u64,
    penalty: f32,
    previous: &[u32],
) -> Result<Tensor> {
    if penalty <= 1.0 || previous.is_empty() {
        return Ok(column.clone());
    }
    let mut seen = vec![0.0f32; n as usize];
    for token in previous {
        if let Some(slot) = seen.get_mut(*token as usize) {
            *slot = 1.0;
        }
    }
    if !seen.iter().any(|v| *v != 0.0) {
        return Ok(column.clone());
    }
    let seen = konst(column.graph(), &[n, 1], &seen)?;
    let scaled_up = column.mul_scalar(penalty)?;
    let scaled_down = column.div_scalar(penalty)?;
    let non_positive = column.lte_scalar(0.0f32)?;
    let penalized = non_positive.where_cond(&scaled_up, &scaled_down)?;
    seen.where_cond(&penalized, column)
}
