//! Multi-axis rope positions for a token sequence with images in it.
//!
//! Text tokens advance one shared counter; an image's tokens hold the
//! counter on the time axis and spread over the height and width axes, and
//! the text after it resumes from the largest position the image reached
//! plus one. The reference implementation's `get_rope_index` for stills.

/// One token's `(time, height, width)` rope position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RopePosition(pub(crate) [u32; 3]);

impl RopePosition {
    pub(crate) fn text(p: u32) -> Self {
        Self([p; 3])
    }

    /// `Some(p)` when every axis agrees, the case a plain rope table serves.
    pub(crate) fn scalar(self) -> Option<u32> {
        let [t, h, w] = self.0;
        (t == h && h == w).then_some(t)
    }
}

/// Positions for `tokens`, where every run of `image_pad` tokens after a
/// `vision_start` token is one image whose merged grid comes from
/// `grids` in order. `start` is the counter's value for the first token;
/// the returned counter is the value for the token after the last.
pub(crate) fn rope_index(
    tokens: &[u32],
    vision_start: u32,
    image_pad: u32,
    grids: &[[u32; 2]],
    start: u32,
) -> (Vec<RopePosition>, u32) {
    let mut out = Vec::with_capacity(tokens.len());
    let mut cur = start;
    let mut grids = grids.iter();
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        out.push(RopePosition::text(cur));
        cur += 1;
        i += 1;
        if token == vision_start && tokens.get(i) == Some(&image_pad) {
            let Some(&[h, w]) = grids.next() else {
                continue;
            };
            let base = cur;
            for y in 0..h {
                for x in 0..w {
                    if tokens.get(i) != Some(&image_pad) {
                        break;
                    }
                    out.push(RopePosition([base, base + y, base + x]));
                    i += 1;
                }
            }
            cur = base + h.max(w);
        }
    }
    (out, cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_counts_up() {
        let (pos, next) = rope_index(&[5, 6, 7], 1, 2, &[], 10);
        assert_eq!(
            pos,
            vec![
                RopePosition::text(10),
                RopePosition::text(11),
                RopePosition::text(12)
            ]
        );
        assert_eq!(next, 13);
    }

    #[test]
    fn an_image_spreads_over_its_grid() {
        // text, start, 2x3 image, end-of-vision text, text
        let tokens = [9, 1, 2, 2, 2, 2, 2, 2, 8, 9];
        let (pos, next) = rope_index(&tokens, 1, 2, &[[2, 3]], 0);
        assert_eq!(pos[0], RopePosition::text(0));
        assert_eq!(pos[1], RopePosition::text(1));
        assert_eq!(pos[2], RopePosition([2, 2, 2]));
        assert_eq!(pos[3], RopePosition([2, 2, 3]));
        assert_eq!(pos[4], RopePosition([2, 2, 4]));
        assert_eq!(pos[5], RopePosition([2, 3, 2]));
        assert_eq!(pos[7], RopePosition([2, 3, 4]));
        // The text after resumes at base + max(h, w) = 2 + 3.
        assert_eq!(pos[8], RopePosition::text(5));
        assert_eq!(pos[9], RopePosition::text(6));
        assert_eq!(next, 7);
        // The image's first token sits at the base on every axis; the rest
        // do not.
        assert_eq!(pos[2].scalar(), Some(2));
        assert_eq!(pos[3].scalar(), None);
        assert_eq!(pos[8].scalar(), Some(5));
    }
}
