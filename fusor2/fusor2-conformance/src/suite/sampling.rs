//! Top-k and the two samplers, entering through `Launch::Ext`.
//!
//! Sampling is the one area whose output is not a function of its input alone,
//! so every case here pins something that *is* deterministic: the top-k
//! ordering and its tie rule, the fact that a zero-temperature sample is the
//! argmax, that a `top_k`/`top_p` filter can never return a token outside the
//! surviving set, and that the pending forms hand back a device tensor rather
//! than a host round trip.

use fusor2::sampling::mirostat2::Mirostat2Sampler;
use fusor2::sampling::standard::{StandardSamplerParams, sample};
use fusor2::sampling::top_k::top_k_pairs;
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dtype, Session};

use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fill_indices, fuzz_case};
use crate::suite::support::{Domain, expect_values, graph_of, read, upload};

/// The fixed vocabulary of the hand-authored tie table.
const VOCAB: usize = 16;

/// Vocabulary range for the sampler cases. The floor of 2 keeps a runner-up
/// available for the repetition penalty and the top-k filter.
const VOCAB_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 128)];

/// Vocabulary range for the top-k ordering cases; k = 1 is legal at vocab 1.
const TOP_K_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 64)];

/// A floor of 8 keeps "64 seeds all drew the same token" out of reach of an
/// honest sampler at temperature 2 over logits in [-3, 3].
const SEED_SPEC: &[FuzzDim] = &[FuzzDim::Range(8, 128)];

/// Random logits over a fuzzed vocabulary. Distinct LCG draws never tie, so
/// the host ordering is unambiguous; the tie rule has its own fixed case.
fn fuzzed_logits(seed: u32, vocab: usize) -> Vec<f32> {
    Domain::Custom(-3.0, 3.0).sample(seed, vocab)
}

/// `(value, token)` pairs sorted descending, ties broken by the larger token
/// id — the rule `top_k_pairs` declares.
fn host_top_k(values: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut pairs: Vec<(f32, u32)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i as u32))
        .collect();
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
    });
    pairs.truncate(k);
    pairs
}

fn upload_logits(session: &Session, values: &[f32]) -> Result<(fusor2::Graph, Tensor), CaseError> {
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[values.len() as u64]), values)?;
    Ok((graph, t))
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push_case(fuzz_case(
        "sampling",
        "top_k_pairs_k1",
        TOP_K_SPEC,
        |s, shape, seed| top_k_case(s, shape[0] as usize, 1, seed),
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "top_k_pairs_sampled_k",
        TOP_K_SPEC,
        |s, shape, seed| {
            let vocab = shape[0] as usize;
            let k = Rng::new(seed ^ 0x5eed).range(1, vocab as u64) as usize;
            top_k_case(s, vocab, k, seed)
        },
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "top_k_pairs_full_vocabulary",
        TOP_K_SPEC,
        |s, shape, seed| top_k_case(s, shape[0] as usize, shape[0] as usize, seed),
    ));
    cases.push(
        "sampling",
        "top_k_pairs_breaks_ties_by_larger_token_id",
        tie_rule,
    );
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token",
        VOCAB_SPEC,
        standard_case,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_at_zero_temperature_is_the_argmax",
        VOCAB_SPEC,
        greedy_case,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_respects_top_k",
        VOCAB_SPEC,
        top_k_filter,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_respects_top_p",
        VOCAB_SPEC,
        top_p_filter,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_respects_min_p",
        VOCAB_SPEC,
        min_p_filter,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_applies_the_repetition_penalty",
        VOCAB_SPEC,
        repetition_case,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_is_seed_deterministic",
        SEED_SPEC,
        seed_case,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_mirostat2_token",
        VOCAB_SPEC,
        mirostat_case,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_mirostat2_token_updates_mu",
        VOCAB_SPEC,
        mirostat_mu,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_standard_token_pending_stays_on_device",
        VOCAB_SPEC,
        pending_standard,
    ));
    cases.push_case(fuzz_case(
        "sampling",
        "sample_mirostat2_token_pending_stays_on_device",
        VOCAB_SPEC,
        pending_mirostat,
    ));
    cases
}

/// `top_k_pairs(k)` on a rank-1 f32 row: values descending, indices matching.
fn top_k_case(session: &Session, vocab: usize, k: usize, seed: u32) -> CaseResult {
    let values = fuzzed_logits(seed, vocab);
    let (_graph, t) = upload_logits(session, &values)?;
    let (got_values, got_indices) = top_k_pairs(&t, k as u32)
        .map_err(|e| -> CaseError { format!("top_k_pairs({k}): {e}").into() })?;

    if got_indices.dtype() != Dtype::U32 {
        return Err(format!("top_k indices are {:?}, want U32", got_indices.dtype()).into());
    }
    let want = host_top_k(&values, k);
    let want_values: Vec<f32> = want.iter().map(|(v, _)| *v).collect();
    let want_indices: Vec<f32> = want.iter().map(|(_, i)| *i as f32).collect();
    expect_values(
        session,
        &[k as u64],
        Dtype::F32,
        &read(&got_values)?,
        &want_values,
    )?;
    expect_values(
        session,
        &[k as u64],
        Dtype::U32,
        &read(&got_indices)?,
        &want_indices,
    )?;
    Ok(())
}

/// Two tokens with exactly equal logits: the larger id must come first.
fn tie_rule(session: &Session) -> CaseResult {
    let mut values = vec![0.0f32; VOCAB];
    values[3] = 2.0;
    values[9] = 2.0;
    let (_graph, t) = upload_logits(session, &values)?;
    let (_, indices) = top_k_pairs(&t, 2).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&indices)?;
    if got.first().copied() != Some(9.0) || got.get(1).copied() != Some(3.0) {
        return Err(format!(
            "tied logits produced the order {got:?}; the declared rule is larger token id \
             first, so it must be [9, 3]"
        )
        .into());
    }
    Ok(())
}

/// A sampled token is always a legal token id.
fn standard_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(seed, vocab);
    let (_graph, t) = upload_logits(session, &values)?;
    let params = StandardSamplerParams {
        temperature: 1.0,
        seed: 42,
        ..Default::default()
    };
    let token = sample(&t, params)
        .map_err(|e| -> CaseError { e.to_string().into() })?
        .to_u32()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    if token as usize >= vocab {
        return Err(format!("sampled token {token} is outside a vocabulary of {vocab}").into());
    }
    Ok(())
}

/// Temperature 0 collapses the distribution onto the argmax.
fn greedy_case(session: &Session, shape: &[u64], data_seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    let mut values = fuzzed_logits(data_seed, vocab);
    let argmax = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("a non-empty row");
    // The maximum must be unambiguous or the case pins nothing.
    values[argmax as usize] += 1.0;

    let (_graph, t) = upload_logits(session, &values)?;
    for seed in [0u64, 1, 12_345] {
        let params = StandardSamplerParams {
            temperature: 0.0,
            seed,
            ..Default::default()
        };
        let token = sample(&t, params)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        if token != argmax {
            return Err(format!(
                "temperature 0 with seed {seed} sampled {token}, not the argmax {argmax}"
            )
            .into());
        }
    }
    Ok(())
}

/// With `top_k = n`, only the n highest-scoring tokens can ever be returned.
fn top_k_filter(session: &Session, shape: &[u64], data_seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    // k < vocab, so the filter always excludes something.
    let k = Rng::new(data_seed ^ 0x5eed).range(1, vocab as u64 - 1) as usize;
    let values = fuzzed_logits(data_seed, vocab);
    let allowed: Vec<u32> = host_top_k(&values, k).into_iter().map(|(_, i)| i).collect();
    let (_graph, t) = upload_logits(session, &values)?;
    for seed in 0..16u64 {
        let params = StandardSamplerParams {
            temperature: 1.5,
            top_k: k as u32,
            seed,
            ..Default::default()
        };
        let token = sample(&t, params)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        if !allowed.contains(&token) {
            return Err(format!(
                "top_k = {k} sampled {token}, which is not in the surviving set {allowed:?}"
            )
            .into());
        }
    }
    Ok(())
}

/// Nucleus sampling: only the smallest prefix of the sorted distribution whose
/// mass reaches `top_p` may be sampled from.
fn top_p_filter(session: &Session, shape: &[u64], data_seed: u32) -> CaseResult {
    const P: f32 = 0.5;
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(data_seed, vocab);
    let allowed = nucleus(&values, P);
    let (_graph, t) = upload_logits(session, &values)?;
    for seed in 0..16u64 {
        let params = StandardSamplerParams {
            temperature: 1.0,
            top_p: P,
            seed,
            ..Default::default()
        };
        let token = sample(&t, params)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        if !allowed.contains(&token) {
            return Err(format!(
                "top_p = {P} sampled {token}, which is outside the nucleus {allowed:?}"
            )
            .into());
        }
    }
    Ok(())
}

/// The token ids in the smallest sorted prefix whose probability mass reaches
/// `p`.
fn nucleus(values: &[f32], p: f32) -> Vec<u32> {
    let sorted = host_top_k(values, values.len());
    let max = sorted[0].0;
    let exps: Vec<f32> = sorted.iter().map(|(v, _)| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let mut mass = 0.0f32;
    let mut out = Vec::new();
    for (e, (_, id)) in exps.iter().zip(&sorted) {
        out.push(*id);
        mass += e / total;
        if mass >= p {
            break;
        }
    }
    out
}

/// `min_p` drops every token whose probability is below `min_p * p_max`.
fn min_p_filter(session: &Session, shape: &[u64], data_seed: u32) -> CaseResult {
    const MIN_P: f32 = 0.2;
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(data_seed, vocab);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let pmax = exps.iter().copied().fold(0.0f32, f32::max) / total;
    let allowed: Vec<u32> = exps
        .iter()
        .enumerate()
        .filter(|(_, e)| *e / total >= MIN_P * pmax)
        .map(|(i, _)| i as u32)
        .collect();

    let (_graph, t) = upload_logits(session, &values)?;
    for seed in 0..16u64 {
        let params = StandardSamplerParams {
            temperature: 1.0,
            min_p: MIN_P,
            seed,
            ..Default::default()
        };
        let token = sample(&t, params)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        if !allowed.contains(&token) {
            return Err(format!(
                "min_p = {MIN_P} sampled {token}, which is below {MIN_P} * p_max; the \
                 surviving set is {allowed:?}"
            )
            .into());
        }
    }
    Ok(())
}

/// A repetition penalty of `r > 1` must make an already-seen token strictly
/// less likely — at temperature 0 that means the argmax moves off it.
fn repetition_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    // The penalized token and its runner-up are sampled ids; the offset keeps
    // them distinct.
    let seen = fill_indices(seed ^ 0x5eed, 1, vocab as u32)[0] as usize;
    let offset = fill_indices(seed ^ 0x9e37_79b9, 1, vocab as u32 - 1)[0] as usize;
    let runner = (seen + 1 + offset) % vocab;
    // The baseline stays below both peaks, so the argmax order is exact.
    let mut values = Domain::Custom(-1.0, 0.0).sample(seed, vocab);
    values[seen] = 2.0;
    values[runner] = 1.9;
    let (_graph, t) = upload_logits(session, &values)?;

    let plain = StandardSamplerParams {
        temperature: 0.0,
        seed: 5,
        ..Default::default()
    };
    let first = sample(&t, plain)
        .map_err(|e| -> CaseError { e.to_string().into() })?
        .to_u32()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    if first != seen as u32 {
        return Err(format!("the unpenalized argmax is {first}, want {seen}").into());
    }

    let penalized = StandardSamplerParams {
        temperature: 0.0,
        repetition_penalty: 4.0,
        seed: 5,
        ..Default::default()
    };
    let second = sample(&t, penalized)
        .map_err(|e| -> CaseError { e.to_string().into() })?
        .to_u32()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    if second == first {
        return Err(format!(
            "a repetition penalty of 4 left the argmax at {first}; the penalty must divide \
             the seen token's logit and there is a runner-up 0.1 below it"
        )
        .into());
    }
    Ok(())
}

/// The same seed must give the same token, and at least one other seed must
/// give a different one — otherwise the sampler is not sampling.
fn seed_case(session: &Session, shape: &[u64], data_seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(data_seed, vocab);
    let (_graph, t) = upload_logits(session, &values)?;
    let draw = |seed: u64| -> Result<u32, CaseError> {
        let params = StandardSamplerParams {
            temperature: 2.0,
            seed,
            ..Default::default()
        };
        sample(&t, params)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })
    };
    let a = draw(99)?;
    let b = draw(99)?;
    if a != b {
        return Err(format!("the same seed drew {a} then {b}").into());
    }
    let mut saw_other = false;
    for seed in 0..64u64 {
        if draw(seed)? != a {
            saw_other = true;
            break;
        }
    }
    if !saw_other {
        return Err(format!(
            "64 different seeds all drew token {a} at temperature 2; the seed is not \
             reaching the sampler"
        )
        .into());
    }
    Ok(())
}

/// Mirostat-2 returns a legal token and keeps its mu on device between calls.
fn mirostat_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(seed, vocab);
    let (_graph, t) = upload_logits(session, &values)?;
    let mut sampler = Mirostat2Sampler::new(5.0, 0.1);
    for _ in 0..4 {
        let token = sampler
            .sample(&t)
            .map_err(|e| -> CaseError { e.to_string().into() })?
            .to_u32()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        if token as usize >= vocab {
            return Err(format!("mirostat sampled the out-of-range token {token}").into());
        }
    }
    Ok(())
}

/// mu is state: it must move as the sampler observes surprise, or the target
/// perplexity is never reached.
fn mirostat_mu(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let vocab = shape[0] as usize;
    let values = fuzzed_logits(seed, vocab);
    let (_graph, t) = upload_logits(session, &values)?;
    let mut sampler = Mirostat2Sampler::new(3.0, 0.5);
    let start = sampler.mu;
    if (start - 6.0).abs() > 1e-6 {
        return Err(format!("mu starts at {start}, want 2 * tau = 6").into());
    }
    for _ in 0..8 {
        sampler
            .sample(&t)
            .map_err(|e| -> CaseError { e.to_string().into() })?;
    }
    if (sampler.mu - start).abs() < 1e-6 {
        return Err(format!(
            "mu is still {start} after eight draws; with eta = 0.5 it must track the \
             observed surprise"
        )
        .into());
    }
    Ok(())
}

/// The pending form hands back a `GpuSampledToken` whose `value` is a device
/// tensor. A decode loop reads it as an operand, so nothing in the step
/// touches the host.
fn pending_standard(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let values = fuzzed_logits(seed, shape[0] as usize);
    let (_graph, t) = upload_logits(session, &values)?;
    let pending = sample(
        &t,
        StandardSamplerParams {
            temperature: 1.0,
            seed: 7,
            ..Default::default()
        },
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;
    check_pending(&pending.value)
}

fn pending_mirostat(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let values = fuzzed_logits(seed, shape[0] as usize);
    let (_graph, t) = upload_logits(session, &values)?;
    let mut sampler = Mirostat2Sampler::new(5.0, 0.1);
    let pending = sampler
        .sample(&t)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    check_pending(&pending.value)
}

/// The token tensor must be a `u32` scalar usable as an operand without a
/// host round trip.
fn check_pending(token: &Tensor) -> CaseResult {
    if token.dtype() != Dtype::U32 {
        return Err(format!("the pending token tensor is {:?}, want U32", token.dtype()).into());
    }
    if token.rank() > 1 {
        return Err(format!(
            "the pending token tensor has rank {}; it names one token",
            token.rank()
        )
        .into());
    }
    // Usable as an operand without a readback.
    token
        .add_scalar(0u32)
        .map_err(|e| -> CaseError { format!("the pending token is not an operand: {e}").into() })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits with a clear ordering and one deliberate exact tie, so the tie
    /// rule — larger token id wins — is observable by the host helpers.
    fn logits() -> Vec<f32> {
        let mut v = Domain::Custom(-3.0, 3.0).sample(2101, VOCAB);
        // Tokens 4 and 11 tie exactly, at a value that is not the maximum.
        v[4] = 1.25;
        v[11] = 1.25;
        // One unambiguous maximum.
        v[7] = 5.0;
        v
    }

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    fn has(names: &[String], wanted: &str) -> bool {
        names.iter().any(|n| n == &format!("sampling::{wanted}"))
    }

    #[test]
    fn every_sampler_entry_point_is_registered() {
        let names = registered();
        for wanted in [
            "top_k_pairs_sampled_k",
            "sample_standard_token",
            "sample_mirostat2_token",
            "sample_standard_token_pending_stays_on_device",
            "sample_mirostat2_token_pending_stays_on_device",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn every_standard_sampler_knob_has_a_case() {
        let names = registered();
        for wanted in [
            "sample_standard_token_at_zero_temperature_is_the_argmax",
            "sample_standard_token_respects_top_k",
            "sample_standard_token_respects_top_p",
            "sample_standard_token_respects_min_p",
            "sample_standard_token_applies_the_repetition_penalty",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn the_host_top_k_sorts_descending_and_breaks_ties_by_larger_id() {
        let values = logits();
        let all = host_top_k(&values, VOCAB);
        assert_eq!(all.len(), VOCAB);
        for w in all.windows(2) {
            assert!(w[0].0 >= w[1].0, "{:?} then {:?}", w[0], w[1]);
        }
        // 7 is the unambiguous maximum.
        assert_eq!(all[0].1, 7);
        // 4 and 11 tie at 1.25; the larger id comes first.
        let four = all.iter().position(|(_, i)| *i == 4).unwrap();
        let eleven = all.iter().position(|(_, i)| *i == 11).unwrap();
        assert!(eleven < four, "the tie must resolve to the larger id first");
    }

    #[test]
    fn the_logits_row_actually_contains_a_tie() {
        // Otherwise `tie_rule` and the assertion above test nothing.
        let v = logits();
        assert_eq!(v[4], v[11]);
        assert!(v[7] > v[4]);
    }

    #[test]
    fn the_nucleus_is_a_prefix_of_the_sorted_order() {
        let values = logits();
        let sorted: Vec<u32> = host_top_k(&values, VOCAB)
            .into_iter()
            .map(|(_, i)| i)
            .collect();
        for p in [0.1f32, 0.5, 0.9, 1.0] {
            let n = nucleus(&values, p);
            assert!(!n.is_empty(), "p={p} admitted nothing");
            assert_eq!(n[..], sorted[..n.len()], "p={p} is not a prefix");
        }
        // A larger p can only admit more tokens.
        assert!(nucleus(&values, 0.9).len() >= nucleus(&values, 0.5).len());
    }

    #[test]
    fn a_nucleus_at_p_one_is_the_whole_vocabulary() {
        assert_eq!(nucleus(&logits(), 1.0).len(), VOCAB);
    }
}
