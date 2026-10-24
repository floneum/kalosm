//! Ordinary device acquisition and tensor algebra, with no optimization preset.
#![cfg(all(feature = "kernel-observer", not(target_arch = "wasm32")))]
use fusor::{Device, Tensor};
use std::sync::{Arc, Mutex};

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} != {e}"
        );
    }
}
fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + seed * 13) % 101) as f32 / 37.0 - 1.1)
        .collect()
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn indexed_elementwise_pipeline_fuses_and_replays() {
    let device = pollster::block_on(Device::gpu()).unwrap();
    let target = device.backend().gpu_target().unwrap();
    let launches = Arc::new(Mutex::new(Vec::new()));
    let seen = launches.clone();
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        *seen.lock().unwrap() = commands
            .iter()
            .filter_map(|command| match command {
                fusor_gpu::launch::CommandRecord::Dispatch { name, .. } => Some(name.to_string()),
                _ => None,
            })
            .collect();
    })));
    for width in [17, 128] {
        let values = data(3 * width, 23);
        let input = Tensor::from_slice(&device, [3, width], &values);
        let indices = Tensor::from_slice(&device, [width], &vec![0u32; width]);
        let gathered = input.as_dyn().index_select(1, indices.as_dyn()).unwrap();
        let output = gathered
            .mul_scalar(0.5)
            .unwrap()
            .sub(&input.as_dyn().mul_scalar(0.25).unwrap())
            .unwrap();
        for step in 0..3 {
            let picks: Vec<_> = (0..width)
                .map(|i| ((i * 7 + step) % width) as u32)
                .collect();
            indices.set_elements(&picks);
            output.clear_device_buf();
            device
                .session()
                .resolve(std::slice::from_ref(&output))
                .unwrap();
            device.session().wait().unwrap();
            let actual = output.to_vec_f32().unwrap();
            let expected: Vec<_> = (0..3 * width)
                .map(|i| {
                    values[i / width * width + picks[i % width] as usize] * 0.5 - values[i] * 0.25
                })
                .collect();
            close(&actual, &expected, 1e-6);
            let names = launches.lock().unwrap();
            assert_eq!(
                names.len(),
                1,
                "gather epilogue materialized an intermediate: {names:?}"
            );
            assert_eq!(names[0], "kgather");
        }
        assert_eq!(input.to_vec_f32(), values);
    }
}

#[cfg(feature = "cpu")]
#[test]
fn indexed_elementwise_pipeline_preserves_cpu_values() {
    let device = Device::cpu();
    let values: Vec<_> = (0..35).map(|i| i as f32 - 10.).collect();
    let input = Tensor::from_slice(&device, [5, 7], &values);
    let index = Tensor::from_slice(&device, [3], &[0u32, 2, 4]);
    let bias = Tensor::from_slice(&device, [3, 7], &[0.5f32; 21]);
    let result = input
        .as_dyn()
        .index_select(0, index.as_dyn())
        .unwrap()
        .mul_scalar(0.25)
        .unwrap()
        .sub(bias.as_dyn())
        .unwrap();
    for picks in [[0u32, 2, 4], [4, 1, 1]] {
        index.set_elements(&picks);
        result.clear_device_buf();
        let expected: Vec<_> = picks
            .into_iter()
            .flat_map(|row| {
                values[row as usize * 7..(row as usize + 1) * 7]
                    .iter()
                    .map(|x| x * 0.25 - 0.5)
            })
            .collect();
        assert_eq!(result.to_vec_f32().unwrap(), expected);
    }
    assert_eq!(input.to_vec_f32(), values);
}
fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    [batch, heads, kv_heads, keys, width]: [usize; 5],
) -> Vec<f32> {
    let mut expected = vec![0.; q.len()];
    for b in 0..batch {
        for h in 0..heads {
            let qr = &q[(b * heads + h) * width..(b * heads + h + 1) * width];
            let base = (b * kv_heads + h / (heads / kv_heads)) * keys * width;
            let scores: Vec<_> = (0..keys)
                .map(|t| {
                    qr.iter()
                        .zip(&k[base + t * width..base + (t + 1) * width])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / (width as f32).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities: Vec<_> = scores.into_iter().map(|s| (s - max).exp()).collect();
            let sum = probabilities.iter().sum::<f32>();
            for d in 0..width {
                expected[(b * heads + h) * width + d] = (0..keys)
                    .map(|t| probabilities[t] * v[base + t * width + d])
                    .sum::<f32>()
                    / sum;
            }
        }
    }
    expected
}
#[test]
#[ignore = "requires a live WebGPU device"]
fn shared_fusions_and_transfers_match_host_results() {
    let device = pollster::block_on(Device::gpu()).unwrap();
    let target = device.backend().gpu_target().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let capture = seen.clone();
    target.set_kernel_observer(Some(Arc::new(move |ir, _, _| {
        capture.lock().unwrap().push(ir.name.to_string())
    })));
    for (rows, width) in [(7, 37), (2, 128), (5, 257), (1, 4096), (4, 8192)] {
        let values = data(rows * width, 3);
        let weights = data(width, 5);
        let x = Tensor::from_slice(&device, [rows, width], &values);
        let w = Tensor::from_slice(&device, [width], &weights);
        let y = x.as_dyn().rms_norm(w.as_dyn(), 1e-5).unwrap();
        let mut expected = values.clone();
        for (out, row) in expected.chunks_mut(width).zip(values.chunks(width)) {
            let inv = (row.iter().map(|x| x * x).sum::<f32>() / width as f32 + 1e-5)
                .sqrt()
                .recip();
            for (i, o) in out.iter_mut().enumerate() {
                *o = row[i] * inv * weights[i];
            }
        }
        close(&y.to_vec_f32().unwrap(), &expected, 2e-5);
    }
    for (rows, width) in [(2, 513), (1, 4096), (2, 8192)] {
        let values = data(rows * width, 11);
        let residual = data(rows * width, 19);
        let weights = data(width, 7);
        let x = Tensor::from_slice(&device, [rows, width], &values);
        let r = Tensor::from_slice(&device, [rows, width], &residual);
        let w = Tensor::from_slice(&device, [width], &weights);
        let y = x
            .as_dyn()
            .add(r.as_dyn())
            .unwrap()
            .rms_norm(w.as_dyn(), 1e-5)
            .unwrap();
        let mut expected: Vec<f32> = values.iter().zip(&residual).map(|(a, b)| a + b).collect();
        for row in expected.chunks_mut(width) {
            let inv = (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / width as f64
                + f64::from(1e-5f32))
            .sqrt()
            .recip();
            for (out, &weight) in row.iter_mut().zip(&weights) {
                *out = (f64::from(*out) * inv * f64::from(weight)) as f32;
            }
        }
        close(&y.to_vec_f32().unwrap(), &expected, 2e-5);
        assert_eq!(x.to_vec_f32(), values);
        assert_eq!(r.to_vec_f32(), residual);
    }
    for (batch, heads, kv_heads, keys, width) in [
        (1, 3, 3, 17, 32),
        (1, 3, 3, 23, 37),
        (2, 4, 2, 33, 64),
        (1, 2, 1, 257, 96),
        (2, 2, 2, 513, 128),
        (1, 2, 1, 19, 256),
    ] {
        let q = data(batch * heads * width, 1);
        let k = data(batch * kv_heads * keys * width, 2);
        let v = data(k.len(), 3);
        let qt = Tensor::from_slice(&device, [batch, heads, 1, width], &q);
        let kt = Tensor::from_slice(&device, [batch, kv_heads, keys, width], &k);
        let vt = Tensor::from_slice(&device, [batch, kv_heads, keys, width], &v);
        let y = fusor::composite::attention(
            qt.as_dyn(),
            kt.as_dyn(),
            vt.as_dyn(),
            fusor_ir::ir::launch::MaskKind::None,
            None,
        )
        .unwrap();
        let expected = attention_reference(&q, &k, &v, [batch, heads, kv_heads, keys, width]);
        close(&y.to_vec_f32().unwrap(), &expected, 3e-4);
    }
    let compiles = target.launcher().pipeline_compiles();
    for width in [3, 7, 33, 65] {
        let values = data(2 * width * 4, 7);
        let x = Tensor::from_slice(&device, [2, width, 4], &values);
        let padded = x
            .as_dyn()
            .pad_with_zeros(1, 2, 3)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let mut expected = vec![0.; 2 * (width + 5) * 4];
        for row in 0..2 {
            expected[(row * (width + 5) + 2) * 4..(row * (width + 5) + 2 + width) * 4]
                .copy_from_slice(&values[row * width * 4..(row + 1) * width * 4]);
        }
        assert_eq!(padded, expected);
        assert_eq!(x.to_vec_f32(), values);
    }
    assert_eq!(
        target.launcher().pipeline_compiles(),
        compiles,
        "copies should not compile shaders"
    );
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter().any(|n| n == "rms_norm"),
        "normalization fusion was not selected: {seen:?}"
    );
    assert!(
        seen.iter().any(|n| n == "rms_norm_residual"),
        "residual normalization was not fused: {seen:?}"
    );
    if device
        .session()
        .caps()
        .subgroups
        .is_some_and(|s| s.is_fixed())
    {
        assert!(
            seen.iter()
                .any(|n| n.starts_with("decode_attention_online")),
            "attention fusion was not selected: {seen:?}"
        );
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn shared_quantized_contractions_match_dequantized_reference() {
    use fusor_ir::dtype::{QFmt, QLayout};
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut fixed_subgroup = false;
    for (fmt, layout) in QFmt::ALL
        .into_iter()
        .flat_map(|fmt| [QLayout::Native, QLayout::F32Scales].map(|layout| (fmt, layout)))
    {
        for n in [1024usize, 1027] {
            // Each case releases its graph, compiled plans, and GPU pool.
            let device = pollster::block_on(Device::gpu()).unwrap();
            fixed_subgroup |= device
                .session()
                .caps()
                .subgroups
                .is_some_and(|s| s.is_fixed() && s.assumed() <= 64);
            let graph = device.graph();
            let capture = seen.clone();
            device
                .backend()
                .gpu_target()
                .unwrap()
                .set_kernel_observer(Some(Arc::new(move |ir, _, _| {
                    capture.lock().unwrap().push(ir.name.to_string());
                })));
            let k = 4096usize;
            let block_bytes = fmt.block_bytes(layout) as usize;
            let block_elements = fmt.block_elements() as usize;
            let fields = fusor_gguf::blocks::block_fields(fmt, layout);
            let mut bytes: Vec<u8> = (0..n * k / block_elements * block_bytes)
                .map(|i| ((i * 71 + 13) % 251) as u8)
                .collect();
            for block in bytes.chunks_mut(block_bytes) {
                for (offset, value) in [(Some(fields.scale), 0.03125f32), (fields.min, 0.015625f32)]
                {
                    if let Some(offset) = offset {
                        let offset = offset as usize;
                        if fields.scale_is_f16 {
                            block[offset..offset + 2]
                                .copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
                        } else {
                            block[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
            let mut dense = vec![0.; n * k];
            for (src, dst) in bytes
                .chunks(block_bytes)
                .zip(dense.chunks_mut(block_elements))
            {
                fusor_gguf::blocks::cpu_dequantize_block(fmt, layout, src, dst);
            }
            let weight = fusor::QMatrix::from_raw_bytes(
                graph,
                fmt,
                layout,
                [fusor::Dim::Const(n as u64), fusor::Dim::Const(k as u64)],
                &bytes,
            )
            .unwrap();
            for rows in [1, 3] {
                for exact in [false, true] {
                    // The dyadic case makes products and partial sums exactly
                    // representable. It detects decode/addressing/algebra bugs
                    // with zero tolerance, independently of reduction order.
                    let values = if exact {
                        (0..rows * k)
                            .map(|i| ((i % 3) as f32 - 1.) / 128.)
                            .collect()
                    } else {
                        data(rows * k, 4)
                    };
                    let input = Tensor::from_slice(&device, [rows, k], &values);
                    let actual = input.q_mat_mul(&weight).to_vec_f32();
                    let expected: Vec<(f64, f64)> = (0..rows)
                        .flat_map(|r| (0..n).map(move |c| (r, c)))
                        .map(|(r, c)| {
                            values[r * k..(r + 1) * k]
                                .iter()
                                .zip(&dense[c * k..(c + 1) * k])
                                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                                .fold((0., 0.), |(sum, magnitude), product| {
                                    (sum + product, magnitude + product.abs())
                                })
                        })
                        .collect();
                    assert_eq!(actual.len(), expected.len());
                    for (i, (&actual, &(expected, magnitude))) in
                        actual.iter().zip(&expected).enumerate()
                    {
                        let tolerance = if exact {
                            // All fixture products are integer multiples of
                            // 1/8192, and their total magnitude fits 24 bits.
                            assert!(magnitude * 8192. <= f64::from(1u32 << 24));
                            assert_eq!(f64::from(expected as f32), expected);
                            0.
                        } else {
                            // Near cancellation, scale the roundoff allowance
                            // by the products as well as the final result. A
                            // result-only check rejects even a scalar f32/FMA
                            // reference for this fixture. This is a test budget,
                            // not a universal bound on arbitrary reductions.
                            (5e-4 * (1. + expected.abs())).max(f64::from(f32::EPSILON) * magnitude)
                        };
                        assert!(
                            actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
                            "{fmt:?}/{layout:?} M={rows} N={n} K={k}, exact={exact}, \
                             element {i}: {actual} != {expected}; \
                             sum_abs_products={magnitude}, tolerance={tolerance}"
                        );
                    }
                }
            }
        }
    }

    let seen = seen.lock().unwrap();
    if fixed_subgroup {
        assert!(
            seen.iter().any(|n| n == "sgemv_cols"),
            "ordinary contraction lowering was not selected: {seen:?}"
        );
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn ordinary_attention_replays_across_cache_growth() {
    use fusor::cache::KvCache;
    let device = pollster::block_on(Device::gpu()).unwrap();
    let target = device.backend().gpu_target().unwrap();
    let (batch, heads, kv_heads, width) = (2, 4, 2, 64);
    let q = data(batch * heads * width, 17);
    let query = Tensor::from_slice(&device, [batch, heads, 1, width], &q);
    let initial_k = data(batch * kv_heads * 255 * width, 19);
    let initial_v = data(initial_k.len(), 23);
    let mut k_rows: Vec<Vec<f32>> = initial_k.chunks(255 * width).map(<[f32]>::to_vec).collect();
    let mut v_rows: Vec<Vec<f32>> = initial_v.chunks(255 * width).map(<[f32]>::to_vec).collect();
    let mut cache = KvCache::with_capacity(2, 256);
    cache.append(
        &Tensor::from_slice(&device, [batch, kv_heads, 255, width], &initial_k),
        &Tensor::from_slice(&device, [batch, kv_heads, 255, width], &initial_v),
    );
    cache.detach();
    let k_input = Tensor::from_slice(
        &device,
        [batch, kv_heads, 1, width],
        &vec![0.; batch * kv_heads * width],
    );
    let v_input = Tensor::from_slice(
        &device,
        [batch, kv_heads, 1, width],
        &vec![0.; batch * kv_heads * width],
    );
    let mut output = None;
    let mut warm_compiles = 0;
    for step in 0..5 {
        KvCache::reserve_all(std::slice::from_mut(&mut cache), 1).unwrap();
        let k = data(batch * kv_heads * width, 29 + step);
        let v = data(k.len(), 37 + step);
        k_input.set_elements(&k);
        v_input.set_elements(&v);
        for row in 0..batch * kv_heads {
            k_rows[row].extend_from_slice(&k[row * width..(row + 1) * width]);
            v_rows[row].extend_from_slice(&v[row * width..(row + 1) * width]);
        }
        if step == 0 {
            cache.append(&k_input, &v_input);
            output = Some(
                fusor::composite::attention(
                    query.as_dyn(),
                    cache.k().unwrap().as_dyn(),
                    cache.v().unwrap().as_dyn(),
                    fusor_ir::ir::launch::MaskKind::None,
                    None,
                )
                .unwrap(),
            );
        } else {
            assert!(cache.can_replay(1));
            cache.replay_append(1).unwrap();
        }
        let output = output.as_ref().unwrap();
        output.clear_device_buf();
        KvCache::resolve_and_commit_all(
            std::slice::from_mut(&mut cache),
            std::slice::from_ref(output),
        )
        .unwrap();
        let k: Vec<_> = k_rows.iter().flatten().copied().collect();
        let v: Vec<_> = v_rows.iter().flatten().copied().collect();
        let expected = attention_reference(&q, &k, &v, [batch, heads, kv_heads, 256 + step, width]);
        close(&output.to_vec_f32().unwrap(), &expected, 3e-4);
        if step == 0 {
            warm_compiles = target.launcher().pipeline_compiles();
        } else {
            assert_eq!(
                target.launcher().pipeline_compiles(),
                warm_compiles,
                "cache growth/replay compiled new shaders at step {step}"
            );
        }
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn full_size_views_replay_their_producer() {
    let device = pollster::block_on(Device::gpu()).unwrap();
    let input = Tensor::from_slice(&device, [2, 3], &[0.; 6]);
    let output = input.mul_scalar(3.).transpose(0, 1);
    for step in 1..=3 {
        let values: Vec<_> = (0..6).map(|i| (i + step * 7) as f32).collect();
        input.set_elements(&values);
        output.as_dyn().clear_device_buf();
        let expected: Vec<_> = [0, 3, 1, 4, 2, 5].map(|i| values[i] * 3.).to_vec();
        assert_eq!(output.to_vec_f32(), expected);
    }
    let broadcast_input = Tensor::from_slice(&device, [1, 3], &[0.; 3]);
    let broadcast = broadcast_input
        .as_dyn()
        .mul_scalar(3.)
        .unwrap()
        .broadcast_as(&[fusor::Dim::Const(2), fusor::Dim::Const(3)])
        .unwrap();
    for step in 1..=3 {
        let values: Vec<f32> = (0..3).map(|i| (i + step * 7) as f32).collect();
        broadcast_input.set_elements(&values);
        broadcast.clear_device_buf();
        let expected: Vec<f32> = values.iter().cycle().take(6).map(|x| x * 3.).collect();
        assert_eq!(broadcast.to_vec_f32().unwrap(), expected);
    }
    let graph = device.graph();
    let rows = graph.sym("view_rows");
    let input = graph
        .leaf(
            "view_input",
            &[rows, fusor::Dim::Const(3)],
            fusor::Dtype::F32,
        )
        .unwrap();
    let output = input.mul_scalar(3.).unwrap().transpose(0, 1).unwrap();
    for rows in [2usize, 4, 3] {
        graph.bind("view_rows", rows as u64);
        let values: Vec<f32> = (0..rows * 3).map(|i| i as f32 + rows as f32).collect();
        input
            .set_bytes(bytemuck::cast_slice(&values).to_vec())
            .unwrap();
        output.clear_device_buf();
        let expected: Vec<f32> = (0..3)
            .flat_map(|col| (0..rows).map(move |row| row * 3 + col))
            .map(|i| values[i] * 3.)
            .collect();
        device
            .session()
            .resolve(std::slice::from_ref(&output))
            .unwrap();
        assert_eq!(
            output.narrow(1, 0, rows).unwrap().to_vec_f32().unwrap(),
            expected
        );
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn scatter_preserves_inputs_and_retained_views() {
    let device = pollster::block_on(Device::gpu()).unwrap();
    for host_backed in [false, true] {
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&device, [4, 3], &values);
        if host_backed {
            x.as_dyn()
                .set_bytes(bytemuck::cast_slice(&values).to_vec())
                .unwrap();
        }
        let view = x.narrow(0, 1, 2);
        let idx = Tensor::from_slice(&device, [1], &[2u32]);
        let update = Tensor::from_slice(&device, [1, 3], &[21., 22., 23.]);
        let y = x
            .as_dyn()
            .scatter_set(0, idx.as_dyn(), update.as_dyn(), true)
            .unwrap();
        let mut expected = values.clone();
        expected[6..9].copy_from_slice(&[21., 22., 23.]);
        assert_eq!(y.to_vec_f32().unwrap(), expected);
        assert_eq!(view.to_vec_f32(), values[3..9]);
        assert_eq!(x.to_vec_f32(), values);
        y.clear_device_buf();
        assert_eq!(y.to_vec_f32().unwrap(), expected);
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn reductions_keep_view_index_spaces() {
    for case in 0..4 {
        let device = pollster::block_on(Device::gpu()).unwrap();
        let values: Vec<f32> = (0..24).map(|i| i as f32 - 9.).collect();
        let input = Tensor::from_slice(&device, [2, 3, 4], &values);
        let x = input.as_dyn();
        let view = match case {
            0 => x.transpose(0, 2).unwrap(),
            1 => x.permute(&[2, 0, 1]).unwrap(),
            2 => x
                .unsqueeze(1)
                .unwrap()
                .broadcast_as(&[
                    fusor::Dim::Const(2),
                    fusor::Dim::Const(3),
                    fusor::Dim::Const(3),
                    fusor::Dim::Const(4),
                ])
                .unwrap(),
            _ => x.transpose(0, 2).unwrap().narrow(1, 1, 1).unwrap(),
        };
        let expected: f32 = values
            .iter()
            .enumerate()
            .filter(|(i, _)| case != 3 || (i / 4) % 3 == 1)
            .map(|(_, v)| v * if case == 2 { 3. } else { 1. })
            .sum();
        let loss = view.sum_all().unwrap();
        close(&loss.to_vec_f32().unwrap(), &[expected], 1e-6);
        let grads = device
            .graph()
            .backward_with(&loss, std::slice::from_ref(x))
            .unwrap();
        let expected_grad: Vec<f32> = (0..24)
            .map(|i| {
                if case == 3 && (i / 4) % 3 != 1 {
                    0.
                } else if case == 2 {
                    3.
                } else {
                    1.
                }
            })
            .collect();
        close(
            &grads.get(x).unwrap().to_vec_f32().unwrap(),
            &expected_grad,
            1e-6,
        );
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn softmax_gradients_after_view_forwarding() {
    for (rows, width) in [(5, 56), (2, 37), (7, 65)] {
        let device = pollster::block_on(Device::gpu()).unwrap();
        let values = data(rows * width, 101);
        let weights = data(rows * width, 103);
        let input = Tensor::from_slice(&device, [rows, width], &values);
        let weight = Tensor::from_slice(&device, [rows, width], &weights);
        let y = input
            .as_dyn()
            .softmax_last_dim()
            .unwrap()
            .mul(weight.as_dyn())
            .unwrap();
        let loss = y.sum_all().unwrap();
        let grads = device
            .graph()
            .backward_with(&loss, std::slice::from_ref(input.as_dyn()))
            .unwrap();
        let mut expected = Vec::new();
        for (x, w) in values.chunks(width).zip(weights.chunks(width)) {
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            let dot: f32 = p.iter().zip(w).map(|(p, w)| p * w).sum();
            expected.extend(p.iter().zip(w).map(|(p, w)| p * (w - dot)));
        }
        let gradient = grads.get(input.as_dyn()).unwrap();
        close(&gradient.to_vec_f32().unwrap(), &expected, 1e-4);
        gradient.clear_device_buf();
        close(&gradient.to_vec_f32().unwrap(), &expected, 1e-4);
    }
}

#[test]
#[ignore = "requires a live WebGPU device"]
fn ordinary_dense_contractions_cover_tails_and_replay() {
    for (rows, k, n) in [(1, 37, 7), (3, 256, 31), (1, 513, 33)] {
        let device = pollster::block_on(Device::gpu()).unwrap();
        let a = data(rows * k, 3);
        let w = data(n * k, 5);
        let x = Tensor::from_slice(&device, [rows, k], &a);
        let weights = Tensor::from_slice(&device, [n, k], &w);
        let y = x.matmul(&weights.t());
        let reference: Vec<f32> = (0..rows)
            .flat_map(|r| (0..n).map(move |c| (r, c)))
            .map(|(r, c)| {
                a[r * k..(r + 1) * k]
                    .iter()
                    .zip(&w[c * k..(c + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        close(&y.to_vec_f32(), &reference, 2e-4);
        y.as_dyn().clear_device_buf();
        close(&y.to_vec_f32(), &reference, 2e-4);
    }
}
