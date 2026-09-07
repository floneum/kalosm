//! The standard temperature / top-k / top-p / min-p sampler.

use crate::Result;
use crate::tensor::Tensor;

use super::row;
use super::top_k::GpuSampledToken;

#[derive(Copy, Clone, Debug)]
/// Parameters for temperature, truncation, and repetition sampling.
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
    /// graph.
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
    let previous = if params.repetition_penalty > 1.0 {
        row::previous_tokens(logits.graph())
    } else {
        Vec::new()
    };
    sample_with_previous(logits, params, &previous)
}

/// [`sample`], awaited. The only readback a draw can need — the previously
/// drawn tokens, for the repetition penalty — is awaited, so this is the
/// form a browser can use.
pub async fn sample_async(
    logits: &Tensor,
    params: StandardSamplerParams,
) -> Result<GpuSampledToken> {
    let previous = if params.repetition_penalty > 1.0 {
        row::previous_tokens_async(logits.graph()).await
    } else {
        Vec::new()
    };
    sample_with_previous(logits, params, &previous)
}

fn sample_with_previous(
    logits: &Tensor,
    params: StandardSamplerParams,
    previous: &[u32],
) -> Result<GpuSampledToken> {
    let n = row::row_len(logits)?;
    let graph = logits.graph().clone();

    // Repetition penalty, then temperature — both on the raw row, before the
    // sort.
    let column = row::sanitized_column(logits, n)?;
    let column = row::apply_repetition_penalty(&column, n, params.repetition_penalty, previous)?;
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
