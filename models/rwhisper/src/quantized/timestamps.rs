// Based on an upstream Whisper DTW implementation with optimizations and refactoring.
// https://rtavenar.github.io/blog/dtw.html is a good resource for understanding the dtw algorithm

use fusor2::{stack, Minus2, Tensor};
use std::num::NonZeroUsize;

use crate::config::{HOP_LENGTH, N_FRAMES, SAMPLE_RATE};

/// Returns the token-level timestamps as `batch x timestamps`.
pub(super) async fn extract_timestamps(
    // A list of (layer, head) pairs to use for timestamp determination
    alignment_heads: &[[usize; 2]],
    // Per-layer `[batch, heads, tokens, frames]` cross-attention scores
    cross_attentions: &[Tensor<4>],
    filter_width: NonZeroUsize,
    n_frames: usize,
    mask: Vec<Vec<bool>>,
) -> Vec<Vec<f32>> {
    // Select relevant cross-attention heads
    let mut tensors_to_stack: Vec<Tensor<3>> = Vec::new();
    for [layer, head] in alignment_heads.iter().copied() {
        if let Some(attn) = cross_attentions.get(layer) {
            tensors_to_stack.push(attn.narrow(1, head, 1).squeeze(1));
        }
    }
    let stacked: Tensor<4> = stack(tensors_to_stack, 0);
    let permuted = stacked.permute([1, 0, 2, 3]);
    let weights = permuted.narrow(3, 0, n_frames.min(N_FRAMES) / 2);

    if weights.shape().contains(&0) {
        // No tokens to be aligned
        return Vec::new();
    }

    // Normalize
    let weights = weights.softmax_last_dim();

    // Smooth
    let var_sqrt = weights.var_keepdim(Minus2).pow_scalar(0.5);
    let weights = median_filter(
        filter_width,
        weights
            .sub_::<4, 4, _>(&weights.mean_keepdim(Minus2))
            .div_::<4, 4, _>(&var_sqrt),
    );

    let cost: Tensor<3> = weights.mean(1);

    // Do the timewarp
    let mut results = Vec::new();
    let [n_batch, _, tokens, frames] = weights.shape();
    for batch_idx in 0..n_batch {
        // Exclude any tokens in the mask
        let neg_cost = cost.mul_scalar(-1.0f32);
        let flat = neg_cost.narrow(0, batch_idx, 1).squeeze::<2>(0).to_flat();
        debug_assert_eq!(flat.len(), tokens * frames);
        let batch_index_cost = flat
            .chunks(frames)
            .enumerate()
            .filter_map(|(i, row)| {
                // Check bounds before accessing mask to avoid panics
                if i < mask.get(batch_idx).map(|m| m.len()).unwrap_or(0) && mask[batch_idx][i] {
                    Some(row.to_vec())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if batch_index_cost.is_empty() || batch_index_cost[0].is_empty() {
            return Vec::new();
        }
        let (text_indices, time_indices) = dynamic_time_warp(batch_index_cost);

        let jumps = std::iter::once(true)
            .chain(
                text_indices
                    .iter()
                    .zip(text_indices.iter().skip(1))
                    .map(|(a, b)| (b - a) as usize == 1),
            )
            .zip(time_indices)
            .filter_map(|(is_jump, time_index)| {
                if is_jump {
                    Some(time_index / (SAMPLE_RATE / (HOP_LENGTH * 2)) as f32)
                } else {
                    None
                }
            });

        results.push(jumps.collect())
    }
    results
}

/// Computes the lowest cost warping path through the provided cost matrix
fn dynamic_time_warp(matrix: Vec<Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
    #[derive(Debug, Clone, Copy)]
    enum Action {
        Match,
        Insert,
        Delete,
    }

    let n = matrix.len();
    let m = matrix[0].len();
    let mut cost = (0..n + 1)
        .map(|i| {
            (0..m + 1)
                .map(|j| if i == 0 && j == 0 { 0.0f32 } else { f32::INFINITY })
                .collect::<Box<[_]>>()
        })
        .collect::<Box<[_]>>();
    let mut trace = (0..n + 1)
        .map(|i| {
            (0..m + 1)
                .map(|_| {
                    if i == 0 {
                        Action::Delete
                    } else {
                        Action::Insert
                    }
                })
                .collect::<Box<[_]>>()
        })
        .collect::<Box<[_]>>();

    cost[0][0] = 0.0;
    for j in 1..m + 1 {
        for i in 1..n + 1 {
            let down_left = cost[i - 1][j - 1];
            let left = cost[i - 1][j];
            let down = cost[i][j - 1];
            let (min, action) = match (down_left < left, down_left < down, left < down) {
                // down_left < left and down_left < down
                (true, true, _) => (down_left, Action::Match),
                // left < down_left and left < down
                (false, _, true) => (left, Action::Insert),
                _ => (down, Action::Delete),
            };

            cost[i][j] = matrix[i - 1][j - 1] + min;
            trace[i][j] = action;
        }
    }

    let (mut i, mut j) = (trace.len() as u32 - 1, trace[0].len() as u32 - 1);

    let (mut xs, mut ys) = (vec![], vec![]);
    while i > 0 || j > 0 {
        xs.push(i.saturating_sub(1) as f32);
        ys.push(j.saturating_sub(1) as f32);
        match trace[i as usize][j as usize] {
            Action::Match => {
                i = i.saturating_sub(1);
                j = j.saturating_sub(1);
            }

            Action::Insert => {
                i = i.saturating_sub(1);
            }

            Action::Delete => {
                j = j.saturating_sub(1);
            }
        }
    }
    xs.reverse();
    ys.reverse();

    (xs, ys)
}

fn median_filter(_filter_width: NonZeroUsize, weights: Tensor<4>) -> Tensor<4> {
    // TODO: Implement proper median filtering for timestamp smoothing
    // For now, return the weights unchanged
    weights
}
