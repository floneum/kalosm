//! Regression tests for view folding interacting with reduce fusion: a
//! folded keepdim view reads a lower-rank input pointwise, which must not be
//! mistaken for a shape-preserving unary chain.

#[test]
fn keepdim_chain_keeps_rank() {
    pollster::block_on(async {
        let Ok(device) = fusor_core::Device::new().await else {
            return;
        };
        let t =
            fusor_core::Tensor::new::<f32, 2, _>(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let s = t.sum_keepdim(1);
        let slice = s.as_slice::<2, f32>().await.unwrap();
        assert_eq!(slice.shape(), &[2, 1]);
        assert_eq!(slice[[0, 0]], 6.0);
        assert_eq!(slice[[1, 0]], 15.0);

        // mean-style chain: an elementwise op on top of the keepdim view, so
        // the view folds into the nary and the nary must NOT fuse into the
        // reduce (that would drop the keepdim rank).
        let m = t.sum_keepdim(1) / 3.0;
        let slice = m.as_slice::<2, f32>().await.unwrap();
        assert_eq!(slice.shape(), &[2, 1]);
        assert_eq!(slice[[0, 0]], 2.0);
        assert_eq!(slice[[1, 0]], 5.0);
    });
}
