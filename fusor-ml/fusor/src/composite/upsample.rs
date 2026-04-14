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
    /// Each pixel is repeated `scale_h × scale_w` times.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_upsample_nearest2d_2x() {
        // Input: (1, 1, 2, 2) =
        //   1 2
        //   3 4
        // Expected (1, 1, 4, 4):
        //   1 1 2 2
        //   1 1 2 2
        //   3 3 4 4
        //   3 3 4 4
        let device = Device::Cpu;
        let t: Tensor<4, f32> = Tensor::from_slice(&device, [1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let up = t.upsample_nearest2d(2, 2);
        let slice = up.as_slice().await.unwrap();
        assert_eq!(slice.shape(), &[1, 1, 4, 4]);
        let expected = [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert!((slice[[0, 0, r, c]] - expected[r][c]).abs() < 1e-6);
            }
        }
    }

    #[tokio::test]
    #[should_panic(expected = "scales must be >= 1")]
    async fn test_upsample_rejects_zero_scale() {
        let device = Device::Cpu;
        let t: Tensor<4, f32> = Tensor::from_slice(&device, [1, 1, 2, 2], &[1.0; 4]);
        let _ = t.upsample_nearest2d(0, 1);
    }
}
