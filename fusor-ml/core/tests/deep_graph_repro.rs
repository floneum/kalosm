//! Standalone reproduction of the wgpu `encode_commands` slowdown seen in SAM's
//! ImageEncoderViT. Narrows down to `cat` / `slice_assign`.

use fusor_core::{Device, Tensor};
use std::time::Instant;

#[tokio::test]
async fn wide_concat_timings() {
    let device = Device::new().await.unwrap();
    let a: Tensor<2, f32> = Tensor::new(&device, &[[1.0f32; 16]; 16]);
    let b: Tensor<2, f32> = Tensor::new(&device, &[[1.0f32; 16]; 16]);
    for &n in &[8usize, 16, 32, 64, 128, 256] {
        let parts: Vec<Tensor<2, f32>> = (0..n).map(|_| a.mat_mul(&b)).collect();
        let cat = Tensor::cat(parts, 0);
        let t = Instant::now();
        let _ = cat.as_slice().await.unwrap();
        eprintln!("wide cat  n={n:>4}  as_slice={:?}", t.elapsed());
    }
}

#[tokio::test]
async fn two_input_cat_in_chain() {
    // What SAM actually does: many *binary* cat ops buried in a forward pass.
    // pad_with_zeros uses cat([left, self, right]) — up to 3 inputs per cat.
    let device = Device::new().await.unwrap();
    let a: Tensor<2, f32> = Tensor::new(&device, &[[1.0f32; 16]; 16]);
    for &n in &[1usize, 4, 16, 64, 256] {
        let mut x = a.clone();
        for _ in 0..n {
            // simulate pad: cat([pad_left, x, pad_right])
            let pad = Tensor::new(&device, &[[0.0f32; 16]; 1]);
            x = Tensor::cat([pad.clone(), x, pad], 0);
        }
        let t = Instant::now();
        let _ = x.as_slice().await.unwrap();
        eprintln!(
            "binary cat chain n={n:>4}  total={:?}  final_rows={}",
            t.elapsed(),
            x.shape()[0]
        );
    }
}

#[tokio::test]
async fn pure_slice_assign_cost() {
    // Measure whether the slice_assign itself is the quadratic term.
    let device = Device::new().await.unwrap();
    for &(size, n) in &[
        (1024usize, 8usize),
        (1024, 32),
        (1024, 64),
        (1024, 128),
        (1024, 256),
    ] {
        let mut buf: Tensor<1, f32> = Tensor::zeros(&device, [size * n]);
        let piece: Tensor<1, f32> = Tensor::new(&device, vec![1.0f32; size].as_slice());
        let t = Instant::now();
        for i in 0..n {
            let range = [(i * size)..((i + 1) * size)];
            buf = buf.slice_assign(range, &piece);
        }
        let build = t.elapsed();
        let t = Instant::now();
        let _ = buf.as_slice().await.unwrap();
        eprintln!(
            "slice_assign n={n:>4} size={size}  build={build:?}  as_slice={:?}",
            t.elapsed()
        );
    }
}
