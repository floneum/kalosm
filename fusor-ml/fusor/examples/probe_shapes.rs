//! Single-op probes at the exact shapes where the batch=64 transformer
//! example wedges the GPU. Each case is one op family in its own process:
//! run `probe_shapes <case>`, one bounded submission, full CPU verification.
//!
//! Cases cross the 65535-workgroup / 16.78M-element dispatch boundary that
//! batch=32 stays under and batch=64 exceeds.

use fusor::{Device, Tensor, ToVec};

const B: usize = 64;
const H: usize = 6;
const S: usize = 256;
const HD: usize = 64;
const ELEMS: usize = B * H * S * S; // 25_165_824 > 65535 * 256

fn fill(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 251) as f32) * 0.01 - 1.25).collect()
}

async fn read_flat<const R: usize>(t: &Tensor<R, f32>, len: usize) -> Vec<f32> {
    t.reshape([len]).as_slice().await.unwrap().to_vec()
}

fn check(name: &str, got: &[f32], want: impl Fn(usize) -> f32, tol: f32) {
    let mut bad = 0usize;
    let mut first = None;
    for (i, &g) in got.iter().enumerate() {
        let w = want(i);
        if (g - w).abs() > tol * w.abs().max(1.0) {
            bad += 1;
            if first.is_none() {
                first = Some((i, g, w));
            }
        }
    }
    if bad == 0 {
        println!("{name}: PASS ({} elements)", got.len());
    } else {
        println!(
            "{name}: FAIL {bad}/{} mismatched, first at {:?}",
            got.len(),
            first
        );
        std::process::exit(1);
    }
}

#[tokio::main]
async fn main() {
    let case = std::env::args().nth(1).expect("usage: probe_shapes <case>");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
        )
        .init();
    let device = Device::gpu().await.unwrap();
    let start = std::time::Instant::now();
    println!("SUBMITTING {case}");

    match case.as_str() {
        // nary elementwise over 25.17M elements
        "add25m" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y = x.add_::<4, 4, _>(&x);
            let got = read_flat(&y, ELEMS).await;
            check("add25m", &got, |i| 2.0 * data[i], 1e-6);
        }
        // reduce over the last axis: 98304 rows of 256
        "sum25m" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y: Tensor<3, f32> = x.sum(3);
            let got = read_flat(&y, ELEMS / S).await;
            let rows: Vec<f32> = (0..ELEMS / S)
                .map(|r| data[r * S..(r + 1) * S].iter().sum())
                .collect();
            check("sum25m", &got, |r| rows[r], 1e-4);
        }
        // row-program softmax over 98304 rows (> 65535 workgroups)
        "softmax98k" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y = x.softmax_last_dim::<3>();
            let got = read_flat(&y, ELEMS).await;
            let mut refs = vec![0.0f32; ELEMS];
            for r in 0..ELEMS / S {
                let row = &data[r * S..(r + 1) * S];
                let max = row.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
                let denom: f32 = exps.iter().sum();
                for (c, e) in exps.iter().enumerate() {
                    refs[r * S + c] = e / denom;
                }
            }
            check("softmax98k", &got, |i| refs[i], 1e-4);
        }
        // big-M coop matmul: [16384, 384] x [384, 384]
        "matmul16k" => {
            let (m, k, n) = (B * S, 384usize, 384usize);
            let a = fill(m * k);
            let b = fill(k * n);
            let at = Tensor::from_slice(&device, [m, k], &a);
            let bt = Tensor::from_slice(&device, [k, n], &b);
            let y = at.mat_mul(&bt);
            let got = read_flat(&y, m * n).await;
            // verify rows around the tile-grid extremes
            for row in [0usize, 127, 8191, 8192, 16256, m - 1] {
                let mut want = vec![0.0f32; n];
                for kk in 0..k {
                    let av = a[row * k + kk];
                    for j in 0..n {
                        want[j] += av * b[kk * n + j];
                    }
                }
                for j in 0..n {
                    let g = got[row * n + j];
                    let w = want[j];
                    if (g - w).abs() > 1e-2 * w.abs().max(1.0) {
                        println!("matmul16k: FAIL at row {row} col {j}: got {g} want {w}");
                        std::process::exit(1);
                    }
                }
            }
            println!("matmul16k: PASS (6 rows exact)");
        }
        // batched attention-shape matmuls
        "qkt" => {
            let a = fill(B * H * S * HD);
            let b = fill(B * H * S * HD);
            let at = Tensor::from_slice(&device, [B, H, S, HD], &a);
            let bt = Tensor::from_slice(&device, [B, H, HD, S], &b);
            let y = at.mat_mul(&bt);
            let got = read_flat(&y, ELEMS).await;
            // verify one full matrix at the start, middle, and last batch
            for batch in [0usize, 192, B * H - 1] {
                let a0 = &a[batch * S * HD..(batch + 1) * S * HD];
                let b0 = &b[batch * HD * S..(batch + 1) * HD * S];
                for i in [0usize, 255] {
                    for j in 0..S {
                        let mut w = 0.0f32;
                        for kk in 0..HD {
                            w += a0[i * HD + kk] * b0[kk * S + j];
                        }
                        let g = got[batch * S * S + i * S + j];
                        if (g - w).abs() > 1e-3 * w.abs().max(1.0) {
                            println!("qkt: FAIL batch {batch} [{i},{j}]: got {g} want {w}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            println!("qkt: PASS (sampled)");
        }
        "av" => {
            let a = fill(ELEMS);
            let v = fill(B * H * S * HD);
            let at = Tensor::from_slice(&device, [B, H, S, S], &a);
            let vt = Tensor::from_slice(&device, [B, H, S, HD], &v);
            let y = at.mat_mul(&vt);
            let got = read_flat(&y, B * H * S * HD).await;
            for batch in [0usize, 192, B * H - 1] {
                let a0 = &a[batch * S * S..(batch + 1) * S * S];
                let v0 = &v[batch * S * HD..(batch + 1) * S * HD];
                for i in [0usize, 255] {
                    for j in 0..HD {
                        let mut w = 0.0f32;
                        for kk in 0..S {
                            w += a0[i * S + kk] * v0[kk * HD + j];
                        }
                        let g = got[batch * S * HD + i * HD + j];
                        if (g - w).abs() > 1e-3 * w.abs().max(1.0) {
                            println!("av: FAIL batch {batch} [{i},{j}]: got {g} want {w}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            println!("av: PASS (sampled)");
        }
        // rank-2 adds bracketing the 65535-workgroup dispatch boundary
        "add65535" | "add65536" => {
            let rows: usize = if case == "add65535" { 65535 } else { 65536 };
            let len = rows * 256;
            let data = fill(len);
            let x = Tensor::from_slice(&device, [rows, 256], &data);
            let y = x.add_::<2, 2, _>(&x);
            let got = y.as_slice().await.unwrap().to_vec();
            let mut bad = 0usize;
            let mut first = None;
            for r in 0..rows {
                for c in 0..256 {
                    let w = 2.0 * data[r * 256 + c];
                    let g = got[r][c];
                    if (g - w).abs() > 1e-6 * w.abs().max(1.0) {
                        bad += 1;
                        if first.is_none() {
                            first = Some((r, c, g, w));
                        }
                    }
                }
            }
            if bad == 0 {
                println!("{case}: PASS ({len} elements)");
            } else {
                println!("{case}: FAIL {bad}/{len} mismatched, first at {first:?}");
                std::process::exit(1);
            }
        }
        // full failing size, rank-2, no reshape before readback
        "add25m2d" => {
            let rows = ELEMS / 256;
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [rows, 256], &data);
            let y = x.add_::<2, 2, _>(&x);
            let got = y.as_slice().await.unwrap().to_vec();
            let mut bad = 0usize;
            let mut first = None;
            for r in 0..rows {
                for c in 0..256 {
                    let w = 2.0 * data[r * 256 + c];
                    let g = got[r][c];
                    if (g - w).abs() > 1e-6 * w.abs().max(1.0) {
                        bad += 1;
                        if first.is_none() {
                            first = Some((r, c, g, w));
                        }
                    }
                }
            }
            if bad == 0 {
                println!("add25m2d: PASS ({ELEMS} elements)");
            } else {
                println!("add25m2d: FAIL {bad}/{ELEMS} mismatched, first at {first:?}");
                std::process::exit(1);
            }
        }
        // region map of the failing case: which flat ranges were written
        "map25m" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y = x.add_::<4, 4, _>(&x);
            let got = read_flat(&y, ELEMS).await;
            let state = |i: usize| -> u8 {
                let w = 2.0 * data[i];
                let g = got[i];
                if (g - w).abs() <= 1e-6 * w.abs().max(1.0) {
                    1 // correct
                } else if g == 0.0 {
                    0 // untouched
                } else {
                    2 // garbage
                }
            };
            let mut runs: Vec<(u8, usize, usize)> = Vec::new();
            let mut cur = state(0);
            let mut start = 0usize;
            let mut counts = [0usize; 3];
            for i in 0..ELEMS {
                let s = state(i);
                counts[s as usize] += 1;
                if s != cur {
                    runs.push((cur, start, i));
                    cur = s;
                    start = i;
                }
            }
            runs.push((cur, start, ELEMS));
            println!(
                "map25m: untouched={} correct={} garbage={} runs={}",
                counts[0],
                counts[1],
                counts[2],
                runs.len()
            );
            for (s, a, b) in runs.iter().take(24) {
                let label = ["ZERO", "OK", "GARBAGE"][*s as usize];
                println!(
                    "  {label:8} [{a:>9}..{b:>9}) len {:>9}  wg [{}..{}]",
                    b - a,
                    a / 256,
                    (b - 1) / 256
                );
            }
        }
        // rank-2 add read through a reshape view (same readback as add25m)
        "add25m2d_reshaped" => {
            let rows = ELEMS / 256;
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [rows, 256], &data);
            let y = x.add_::<2, 2, _>(&x);
            let got = read_flat(&y, ELEMS).await;
            check("add25m2d_reshaped", &got, |i| 2.0 * data[i], 1e-6);
        }
        // rank-4 add read back directly, no reshape
        "add25m4d_direct" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y = x.add_::<4, 4, _>(&x);
            let slice = y.as_slice().await.unwrap();
            let mut bad = 0usize;
            let mut first = None;
            for b in 0..B {
                for h in 0..H {
                    for i in 0..S {
                        for j in 0..S {
                            let idx = ((b * H + h) * S + i) * S + j;
                            let w = 2.0 * data[idx];
                            let g = slice[[b, h, i, j]];
                            if (g - w).abs() > 1e-6 * w.abs().max(1.0) {
                                bad += 1;
                                if first.is_none() {
                                    first = Some((idx, g, w));
                                }
                            }
                        }
                    }
                }
            }
            if bad == 0 {
                println!("add25m4d_direct: PASS ({ELEMS} elements)");
            } else {
                println!("add25m4d_direct: FAIL {bad}/{ELEMS}, first {first:?}");
                std::process::exit(1);
            }
        }
        // generic rank-4 add over a shape given as 4 extra args
        "addshape" => {
            let dims: Vec<usize> = std::env::args()
                .skip(2)
                .map(|a| a.parse().unwrap())
                .collect();
            let [d0, d1, d2, d3] = dims[..] else {
                panic!("addshape needs 4 dims")
            };
            let len = d0 * d1 * d2 * d3;
            let data = fill(len);
            let x = Tensor::from_slice(&device, [d0, d1, d2, d3], &data);
            let y = x.add_::<4, 4, _>(&x);
            let slice = y.as_slice().await.unwrap();
            let mut bad = 0usize;
            let mut first = None;
            let mut idx = 0usize;
            'outer: for a in 0..d0 {
                for b in 0..d1 {
                    for i in 0..d2 {
                        for j in 0..d3 {
                            let w = 2.0 * data[idx];
                            let g = slice[[a, b, i, j]];
                            if (g - w).abs() > 1e-6 * w.abs().max(1.0) {
                                bad += 1;
                                if first.is_none() {
                                    first = Some((idx, g, w));
                                }
                                if bad > 5_000_000 {
                                    break 'outer;
                                }
                            }
                            idx += 1;
                        }
                    }
                }
            }
            if bad == 0 {
                println!("addshape {dims:?}: PASS ({len} elements)");
            } else {
                println!("addshape {dims:?}: FAIL {bad}+/{len}, first {first:?}");
                std::process::exit(1);
            }
        }
        // empirical write map: which source value lands at which target slot
        "writemap" => {
            let dims: Vec<usize> = std::env::args()
                .skip(2)
                .map(|a| a.parse().unwrap())
                .collect();
            let [d0, d1, d2, d3] = dims[..] else {
                panic!("writemap needs 4 dims")
            };
            let len = d0 * d1 * d2 * d3;
            // encode source index as (quotient+1, remainder+1) across two runs
            let dq: Vec<f32> = (0..len).map(|i| (i / 4096 + 1) as f32).collect();
            let dr: Vec<f32> = (0..len).map(|i| (i % 4096 + 1) as f32).collect();
            let read4 = |data: &[f32]| {
                let x = Tensor::from_slice(&device, [d0, d1, d2, d3], data);
                let y = x.add_::<4, 4, _>(&x);
                async move { read_flat(&y, len).await }
            };
            let gq = read4(&dq).await;
            let gr = read4(&dr).await;
            let mut untouched = 0usize;
            let mut identity = 0usize;
            let mut moved = 0usize;
            let mut samples: Vec<(usize, usize)> = Vec::new();
            for t in 0..len {
                if gq[t] == 0.0 && gr[t] == 0.0 {
                    untouched += 1;
                    continue;
                }
                let q = (gq[t] / 2.0 - 1.0) as usize;
                let r = (gr[t] / 2.0 - 1.0) as usize;
                let s = q * 4096 + r;
                if s == t {
                    identity += 1;
                } else {
                    moved += 1;
                    if samples.len() < 30 {
                        samples.push((t, s));
                    }
                }
            }
            println!(
                "writemap {dims:?}: untouched={untouched} identity={identity} moved={moved}"
            );
            for (t, s) in samples {
                let tc = (
                    t / (d1 * d2 * d3),
                    (t / (d2 * d3)) % d1,
                    (t / d3) % d2,
                    t % d3,
                );
                let sc = (
                    s / (d1 * d2 * d3),
                    (s / (d2 * d3)) % d1,
                    (s / d3) % d2,
                    s % d3,
                );
                println!("  t={t} {tc:?}  <-  s={s} {sc:?}  (s-t={})", s as i64 - t as i64);
            }
        }
        // same failing add, but materialize + sleep before reading back:
        // complete data => submit/map race; still partial => work lost on GPU
        "addsleep" => {
            let data = fill(ELEMS);
            let x = Tensor::from_slice(&device, [B, H, S, S], &data);
            let y = x.add_::<4, 4, _>(&x);
            y.materialize().await;
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            let got = read_flat(&y, ELEMS).await;
            check("addsleep", &got, |i| 2.0 * data[i], 1e-6);
        }
        // raw wgpu micro-test: does the GPU compute the delinearize/relinearize
        // identity exactly? Bounds checks left ON; independent of fusor kernels.
        "divident" => {
            let fusor::Device::Gpu(core_device) = &device else {
                panic!("gpu device required")
            };
            let wgpu_device = core_device.wgpu_device();
            let queue = core_device.wgpu_queue();
            let total: u32 = ELEMS as u32; // 25_165_824
            let shader = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wg.x + wg.y * 65535u;
    let f = group * 256u + lane;
    if (f < arrayLength(&out)) {
        let a = (f / 393216u) % 64u;
        let b = (f / 65536u) % 6u;
        let i = (f / 256u) % 256u;
        let j = f % 256u;
        out[f] = a * 393216u + b * 65536u + i * 256u + j;
    }
}
"#;
            let module = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("divident"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
            let out = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: total as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let pipeline =
                wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let bind = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                }],
            });
            let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: total as u64 * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = wgpu_device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(65535, 2, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, total as u64 * 4);
            queue.submit(Some(encoder.finish()));
            let (sender, receiver) = futures_channel::oneshot::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = sender.send(r);
                });
            core_device.poll_wait();
            receiver.await.unwrap().unwrap();
            let view = staging.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            let mut bad = 0usize;
            let mut samples = Vec::new();
            for (f, &got) in words.iter().enumerate() {
                if got != f as u32 {
                    bad += 1;
                    if samples.len() < 20 {
                        samples.push((f, got));
                    }
                }
            }
            println!("divident: {bad}/{total} wrong");
            for (f, got) in samples {
                println!(
                    "  f={f} -> {got} (a={} b={} i={} j={})",
                    (f as u32 / 393216) % 64,
                    (f as u32 / 65536) % 6,
                    (f as u32 / 256) % 256,
                    f as u32 % 256
                );
            }
            if bad > 0 {
                std::process::exit(1);
            }
        }
        // same expression, 1-D grid only (covers f < 16.78M)
        "divident1d" | "divident2" => {
            let fusor::Device::Gpu(core_device) = &device else {
                panic!("gpu device required")
            };
            let wgpu_device = core_device.wgpu_device();
            let queue = core_device.wgpu_queue();
            let one_d = case == "divident1d";
            let total: u32 = if one_d { 16_776_960 } else { ELEMS as u32 };
            // divident2 keeps the 2-D dispatch but derives the group id from a
            // flat linearization that the compiler can't fold with wg.x alone
            let group_expr = if one_d {
                "let group = wg.x;"
            } else {
                "let group = wg.x + wg.y * 65535u;"
            };
            let shader = format!(
                r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{
    {group_expr}
    let f = group * 256u + lane;
    if (f < arrayLength(&out)) {{
        let q = f / 65536u;
        let b = q % 6u;
        out[f] = b;
    }}
}}
"#
            );
            let module = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(case.as_str()),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
            let out = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: total as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let pipeline =
                wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let bind = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                }],
            });
            let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: total as u64 * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = wgpu_device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                if one_d {
                    pass.dispatch_workgroups(65535, 1, 1);
                } else {
                    pass.dispatch_workgroups(65535, 2, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, total as u64 * 4);
            queue.submit(Some(encoder.finish()));
            let (sender, receiver) = futures_channel::oneshot::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = sender.send(r);
                });
            core_device.poll_wait();
            receiver.await.unwrap().unwrap();
            let view = staging.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&view);
            let mut bad = 0usize;
            let mut samples = Vec::new();
            for (f, &got) in words.iter().enumerate() {
                let want = (f as u32 / 65536) % 6;
                if got != want {
                    bad += 1;
                    if samples.len() < 8 {
                        samples.push((f, got, want));
                    }
                }
            }
            println!("{case}: {bad}/{total} wrong");
            for (f, got, want) in samples {
                println!("  f={f} got={got} want={want}");
            }
        }
        other => {
            eprintln!("unknown case {other}");
            std::process::exit(2);
        }
    }
    println!("{case} elapsed: {:?}", start.elapsed());
}
