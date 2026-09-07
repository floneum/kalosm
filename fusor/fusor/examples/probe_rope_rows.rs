//! Probe: `rope_pair` over a one-row table at offset 0 must equal the full
//! table at that row's position, and `rope_pair_at` with a position leaf.
use fusor::{Device, Dim, Tensor};

fn main() {
    let device = pollster::block_on(Device::gpu()).expect("gpu");
    let (heads, hd, half, ctx) = (2usize, 8usize, 4usize, 64usize);
    let q: Vec<f32> = (0..heads * hd).map(|i| (i as f32 * 0.37).sin()).collect();
    let k: Vec<f32> = (0..heads * hd).map(|i| (i as f32 * 0.11).cos()).collect();
    let q = Tensor::<4>::from_slice(&device, [1, heads, 1, hd], &q);
    let k = Tensor::<4>::from_slice(&device, [1, heads, 1, hd], &k);
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10_000f32.powf(2.0 * i as f32 / hd as f32))
        .collect();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in 0..ctx {
        for f in &inv {
            cos.push((p as f32 * f).cos());
            sin.push((p as f32 * f).sin());
        }
    }
    let cos_t = Tensor::<2>::from_slice(&device, [ctx, half], &cos);
    let sin_t = Tensor::<2>::from_slice(&device, [ctx, half], &sin);
    let p = 22usize;
    let (qa, ka) = q.rope_pair(&k, &cos_t, &sin_t, p as u64);
    let row_c = Tensor::<2>::from_slice(&device, [1, half], &cos[p * half..(p + 1) * half]);
    let row_s = Tensor::<2>::from_slice(&device, [1, half], &sin[p * half..(p + 1) * half]);
    let (qb, kb) = q.rope_pair(&k, &row_c, &row_s, 0);
    let pos = Tensor::<1, u32>::leaf(&device, [Dim::Const(1)]);
    pos.set_elements(&[p as u32]);
    let (qc, kc) = q.rope_pair_at(&k, &cos_t, &sin_t, &pos);
    let diff = |a: &Tensor<4>, b: &Tensor<4>| {
        a.to_flat()
            .iter()
            .zip(b.to_flat())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "q: table@p vs row@0 {:.6}, table@p vs leaf {:.6}",
        diff(&qa, &qb),
        diff(&qa, &qc)
    );
    println!(
        "k: table@p vs row@0 {:.6}, table@p vs leaf {:.6}",
        diff(&ka, &kb),
        diff(&ka, &kc)
    );
    println!("q table@p {:?}", &qa.to_flat()[..8]);
    println!("q row@0   {:?}", &qb.to_flat()[..8]);
}
