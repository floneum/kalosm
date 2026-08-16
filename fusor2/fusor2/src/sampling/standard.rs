//! The standard temperature / top-k / top-p / min-p sampler.

use crate::Result;
use crate::tensor::Tensor;

use super::row;
use super::top_k::GpuSampledToken;

#[derive(Copy, Clone, Debug)]
pub struct StandardSamplerParams {
    /// Divides the logits. **Exactly `0.0` means greedy**: the sampler returns
    /// the argmax and ignores the draw.
    pub temperature: f32,
    /// Keep only the `top_k` best-scoring tokens. `0` means the whole row.
    pub top_k: u32,
    /// Nucleus mass: keep the shortest prefix of the sorted distribution whose
    /// cumulative probability reaches `top_p`, including the token that
    /// crosses it. `1.0` keeps everything.
    pub top_p: f32,
    /// Drop every token whose probability is below `min_p * p_max`. `0.0`
    /// keeps everything.
    pub min_p: f32,
    /// Above `1.0`, push down the logit of a token already drawn on this
    /// graph. See [`row::apply_repetition_penalty`].
    pub repetition_penalty: f32,
    /// The draw is a pure function of this seed.
    pub seed: u64,
}

impl Default for StandardSamplerParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

/// Draw one token from a logits row.
///
/// Nothing is resolved: the returned token is a device tensor, and the logits
/// never reach the host. The only host input is the uniform draw.
///
/// The filters run in order: repetition penalty, temperature, sort, top-k,
/// min-p, top-p, weighted pick.
///
/// A temperature of `0` is greedy: the argmax is rank `0` of the sorted order
/// and survives every filter, so short-circuiting to it is equivalent to
/// sampling a distribution collapsed onto one token.
pub fn sample(logits: &Tensor, params: StandardSamplerParams) -> Result<GpuSampledToken> {
    let n = row::row_len(logits)?;
    let graph = logits.graph().clone();

    // Repetition penalty, then temperature — both on the raw row, before the
    // sort.
    let column = row::sanitized_column(logits, n)?;
    let previous = if params.repetition_penalty > 1.0 {
        row::previous_tokens(&graph)
    } else {
        Vec::new()
    };
    let column = row::apply_repetition_penalty(&column, n, params.repetition_penalty, &previous)?;
    let column = if params.temperature != 0.0 {
        column.div_scalar(params.temperature)?
    } else {
        column
    };

    let (sorted_values, sorted_ids) = row::sort_desc(&column, n)?;

    // top_k truncates the candidate list; 0 means the whole row.
    let k = match params.top_k {
        0 => n,
        requested => u64::from(requested).min(n),
    };
    let sorted_values = sorted_values.narrow(0, 0, k as usize)?;
    let sorted_ids = sorted_ids.narrow(0, 0, k as usize)?;

    let token = if params.temperature == 0.0 {
        // Greedy: rank 0 of an order that already breaks ties by larger id.
        sorted_ids.narrow(0, 0, 1)?
    } else {
        let one_hot = filtered_pick(&sorted_values, k, &params)?;
        row::gather_one_hot(&one_hot, &sorted_ids)?
    };

    let token = row::as_token(&token)?;
    row::remember(&graph, token.id());
    Ok(GpuSampledToken { value: token })
}

/// min-p, then top-p, then the weighted pick — as a one-hot `[k, 1]`.
fn filtered_pick(sorted_values: &Tensor, k: u64, params: &StandardSamplerParams) -> Result<Tensor> {
    let graph = sorted_values.graph().clone();
    // w[r] = exp(v[r] - v[0]); w[0] == 1 and w[r] == p[r] / p_max.
    let weights = row::weights_of(sorted_values, k)?;
    let first = row::first_only(&graph, k)?;

    // min-p compares that ratio directly against the knob.
    let keep = if params.min_p > 0.0 {
        weights.gte_scalar(params.min_p)?
    } else {
        row::ones(&graph, k)?
    };
    let survivors = weights.mul(&keep)?;

    // top-p over the min-p-filtered mass. Keeping every position whose
    // *exclusive* prefix is still below the target, including the token that
    // crosses.
    let keep = if params.top_p < 1.0 {
        let total = row::total_of(&survivors)?;
        let target = row::fanout(&total, k)?.mul_scalar(params.top_p.max(0.0))?;
        let before = row::prefix_exclusive(&survivors, k)?;
        let within = before.lt_tensor(&target)?;
        keep.mul(&within)?
    } else {
        keep
    };

    // Force a cutoff of at least one candidate when every filter rejected
    // everything.
    let keep = keep.maximum(&first)?;
    let masked = weights.mul(&keep)?;
    row::pick_one_hot(&masked, k, row::unit_random(params.seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::test_support::{conformance_row, cpu_row, host_sorted, nucleus, softmax};

    fn draw(t: &Tensor, params: StandardSamplerParams) -> u32 {
        sample(t, params).unwrap().to_u32().unwrap()
    }

    /// Temperature 0 is the argmax, whatever the seed.
    #[test]
    fn zero_temperature_is_the_argmax() {
        let values = conformance_row();
        let want = host_sorted(&values)[0].1;
        let (_s, _g, t) = cpu_row(&values);
        for seed in [0u64, 1, 12_345, u64::MAX] {
            let params = StandardSamplerParams {
                temperature: 0.0,
                seed,
                ..Default::default()
            };
            assert_eq!(draw(&t, params), want, "seed {seed}");
        }
    }

    /// Greedy on a tied maximum still follows the declared tie rule.
    #[test]
    fn zero_temperature_breaks_a_tied_maximum_by_larger_id() {
        let mut values = vec![0.0f32; 8];
        values[1] = 3.0;
        values[6] = 3.0;
        let (_s, _g, t) = cpu_row(&values);
        let params = StandardSamplerParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(draw(&t, params), 6);
    }

    /// top_k in isolation: nothing outside the k best is ever drawn, and with
    /// k = 1 the only legal answer is the argmax.
    #[test]
    fn top_k_confines_the_draw_to_the_k_best() {
        let values = conformance_row();
        let sorted = host_sorted(&values);
        let (_s, _g, t) = cpu_row(&values);
        for k in [1usize, 3, 5] {
            let allowed: Vec<u32> = sorted[..k].iter().map(|p| p.1).collect();
            for seed in 0..32u64 {
                let params = StandardSamplerParams {
                    temperature: 1.5,
                    top_k: k as u32,
                    seed,
                    ..Default::default()
                };
                let got = draw(&t, params);
                assert!(allowed.contains(&got), "k={k} seed={seed} drew {got}");
            }
        }
    }

    /// top_p in isolation: the draw stays inside the nucleus.
    #[test]
    fn top_p_confines_the_draw_to_the_nucleus() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        for p in [0.3f32, 0.5, 0.9] {
            let allowed = nucleus(&values, p);
            for seed in 0..32u64 {
                let params = StandardSamplerParams {
                    temperature: 1.0,
                    top_p: p,
                    seed,
                    ..Default::default()
                };
                let got = draw(&t, params);
                assert!(allowed.contains(&got), "p={p} seed={seed} drew {got}");
            }
        }
    }

    /// min_p in isolation: every token below `min_p * p_max` is unreachable.
    #[test]
    fn min_p_drops_everything_below_the_scaled_peak() {
        let values = conformance_row();
        let probs = softmax(&values);
        let peak = probs.iter().copied().fold(0.0f32, f32::max);
        let (_s, _g, t) = cpu_row(&values);
        for min_p in [0.05f32, 0.2, 0.6] {
            let allowed: Vec<u32> = probs
                .iter()
                .enumerate()
                .filter(|(_, p)| **p >= min_p * peak)
                .map(|(i, _)| i as u32)
                .collect();
            for seed in 0..32u64 {
                let params = StandardSamplerParams {
                    temperature: 1.0,
                    min_p,
                    seed,
                    ..Default::default()
                };
                let got = draw(&t, params);
                assert!(
                    allowed.contains(&got),
                    "min_p={min_p} seed={seed} drew {got}, allowed {allowed:?}"
                );
            }
        }
        // The filter must actually bite: at min_p = 0.6 only the peak survives.
        let params = StandardSamplerParams {
            temperature: 1.0,
            min_p: 0.6,
            seed: 3,
            ..Default::default()
        };
        assert_eq!(draw(&t, params), host_sorted(&values)[0].1);
    }

    /// The repetition penalty divides a positive logit of an already-drawn
    /// token, which is enough to move the greedy pick to the runner-up.
    #[test]
    fn the_repetition_penalty_moves_the_greedy_pick_off_a_drawn_token() {
        let mut values = vec![0.0f32; 16];
        values[7] = 2.0;
        values[2] = 1.9;
        let (_s, _g, t) = cpu_row(&values);

        let plain = StandardSamplerParams {
            temperature: 0.0,
            seed: 5,
            ..Default::default()
        };
        assert_eq!(draw(&t, plain), 7);

        let penalized = StandardSamplerParams {
            temperature: 0.0,
            repetition_penalty: 4.0,
            seed: 5,
            ..Default::default()
        };
        // 2.0 / 4 = 0.5 falls below the untouched 1.9.
        assert_eq!(draw(&t, penalized), 2);
    }

    /// A penalty of 1 is inert however much history has accumulated.
    #[test]
    fn a_penalty_of_one_never_changes_the_draw() {
        let mut values = vec![0.0f32; 16];
        values[7] = 2.0;
        values[2] = 1.9;
        let (_s, _g, t) = cpu_row(&values);
        let params = StandardSamplerParams {
            temperature: 0.0,
            repetition_penalty: 1.0,
            seed: 5,
            ..Default::default()
        };
        for _ in 0..6 {
            assert_eq!(draw(&t, params), 7);
        }
    }

    /// The same seed replays; different seeds do not all collapse to one token.
    #[test]
    fn the_draw_is_seed_deterministic_and_seed_sensitive() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let params = |seed| StandardSamplerParams {
            temperature: 2.0,
            seed,
            ..Default::default()
        };
        let a = draw(&t, params(99));
        for _ in 0..4 {
            assert_eq!(draw(&t, params(99)), a, "the same seed must replay");
        }
        let distinct: std::collections::BTreeSet<u32> =
            (0..64u64).map(|s| draw(&t, params(s))).collect();
        assert!(
            distinct.len() > 1,
            "64 seeds all drew {a}; the seed is not reaching the sampler"
        );
    }

    /// Every draw is a legal token id.
    #[test]
    fn every_draw_is_in_the_vocabulary() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        for seed in 0..64u64 {
            let got = draw(
                &t,
                StandardSamplerParams {
                    seed,
                    ..Default::default()
                },
            );
            assert!((got as usize) < values.len(), "drew {got}");
        }
    }

    /// The pending form must not resolve anything: the token is a one-element
    /// `U32` device tensor usable as an operand.
    #[test]
    fn the_pending_token_is_a_usable_device_operand() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let pending = sample(
            &t,
            StandardSamplerParams {
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pending.value.dtype(), crate::Dtype::U32);
        assert!(pending.value.rank() <= 1);
        assert!(pending.value.add_scalar(0u32).is_ok());
    }

    /// The filters compose: min-p first, then top-p over the surviving mass.
    #[test]
    fn min_p_and_top_p_compose() {
        let values = conformance_row();
        let probs = softmax(&values);
        let peak = probs.iter().copied().fold(0.0f32, f32::max);
        let min_p = 0.05f32;
        let allowed: Vec<u32> = probs
            .iter()
            .enumerate()
            .filter(|(_, p)| **p >= min_p * peak)
            .map(|(i, _)| i as u32)
            .collect();
        let (_s, _g, t) = cpu_row(&values);
        for seed in 0..32u64 {
            let got = draw(
                &t,
                StandardSamplerParams {
                    temperature: 1.0,
                    top_p: 0.8,
                    min_p,
                    seed,
                    ..Default::default()
                },
            );
            assert!(allowed.contains(&got), "seed={seed} drew {got}");
        }
    }
}
