//! Upsampling operations for spatial tensors.

use crate::{ConcreteTensor, SimdElement, Tensor};
use fusor_core::DataType;

impl<D> Tensor<4, D>
where
    D: SimdElement + DataType + Default,
{
    /// Upsample a 4D tensor (B, C, H, W) using nearest-neighbor interpolation.
    ///
    /// Scales the spatial dimensions by integer scale factors.
    /// Each pixel is repeated `scale_h` by `scale_w` times.
    ///
    /// # Panics
    /// Panics if `scale_h == 0` or `scale_w == 0` (a zero scale would produce
    /// a degenerate zero-element broadcast).
    pub fn upsample_nearest2d(
        &self,
        scale_h: usize,
        scale_w: usize,
    ) -> Tensor<4, D, ConcreteTensor<D, 4>> {
        assert!(
            scale_h >= 1 && scale_w >= 1,
            "upsample_nearest2d scales must be >= 1, got ({scale_h}, {scale_w})"
        );
        let [b, c, h, w] = self.shape();
        // (B, C, H, W) -> (B, C, H, 1, W, 1)
        let expanded: Tensor<6, D, _> = self.reshape([b, c, h, 1, w, 1]);
        // Broadcast to (B, C, H, scale_h, W, scale_w)
        let broadcast = expanded.broadcast_as([b, c, h, scale_h, w, scale_w]);
        // Reshape to (B, C, H * scale_h, W * scale_w)
        broadcast
            .reshape([b, c, h * scale_h, w * scale_w])
            .to_concrete()
    }
}
