//! The window-attention permutation of merge blocks.
//!
//! Patches are attended within `window_size × window_size` pixel windows in
//! most blocks. Grouping each window's merge blocks contiguously turns the
//! block-diagonal attention into one dense attention per window.

/// `(window_index, cu_window_seqlens)` for one `[t, h, w]` patch grid.
///
/// `window_index` lists merge-block ids (a merge block is `merge × merge`
/// patches) in window order; `cu_window_seqlens` are the cumulative patch
/// counts at every window boundary, starting at 0, with empty windows
/// collapsed.
pub(crate) fn window_index(
    grid: [u32; 3],
    window_size: usize,
    merge: usize,
    patch_size: usize,
) -> (Vec<u32>, Vec<u32>) {
    let [t, h, w] = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
    let llm_h = h / merge;
    let llm_w = w / merge;
    let win = window_size / merge / patch_size;
    let unit = (merge * merge) as u32;
    // The padding the reference implementation applies: a full window's
    // worth when the grid already divides, which changes nothing here since
    // out-of-range cells are dropped anyway.
    let windows_h = llm_h.div_ceil(win);
    let windows_w = llm_w.div_ceil(win);

    let mut index = Vec::with_capacity(t * llm_h * llm_w);
    let mut cu = vec![0u32];
    for frame in 0..t {
        for wh in 0..windows_h {
            for ww in 0..windows_w {
                let mut count = 0u32;
                for i in 0..win {
                    for j in 0..win {
                        let (y, x) = (wh * win + i, ww * win + j);
                        if y < llm_h && x < llm_w {
                            index.push((frame * llm_h * llm_w + y * llm_w + x) as u32);
                            count += 1;
                        }
                    }
                }
                let last = *cu.last().expect("starts with 0");
                let next = last + count * unit;
                if next != last {
                    cu.push(next);
                }
            }
        }
    }
    (index, cu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_layout() {
        // window 2 patches wide over a 8x4 grid of 1-pixel patches, no merge
        // (merge 1 keeps every patch its own block, unit 1 * 1).
        let (index, cu) = window_index([1, 8, 4], 2, 1, 1);
        assert_eq!(
            index,
            vec![
                0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22,
                23, 24, 25, 28, 29, 26, 27, 30, 31
            ]
        );
        assert_eq!(cu, vec![0, 4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn merge_blocks_scale_the_boundaries() {
        // 112-pixel windows of 14-pixel patches merged 2x2: 4 blocks a side
        // over a 6x4 block grid is one full window and one 2x4 window.
        let (index, cu) = window_index([1, 12, 8], 112, 2, 14);
        assert_eq!(index.len(), 6 * 4);
        assert_eq!(cu, vec![0, 64, 96]);
        // Row-major within the window: the first window walks rows 0..4.
        assert_eq!(&index[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&index[16..20], &[16, 17, 18, 19]);
    }
}
