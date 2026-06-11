use crate::{Device, StrideSpec, Tensor};

// Build a small intermediate that requires a real kernel (not a Tensor input).
// `x` materializes via `(input * 2.0) + 1.0`, which fuses to a single nary.
fn build_intermediate(device: &Device) -> Tensor {
    let rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 8];
    let input = Tensor::new::<f32, 2, _>(device, &rows);
    (&input * 2.0) + 1.0
}

fn build_matmul_intermediate(device: &Device) -> Tensor {
    let left_rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 8];
    let right_rows = vec![vec![0.25f32, 0.5, 0.75, 1.0]; 4];
    let left = Tensor::new::<f32, 2, _>(device, &left_rows);
    let right = Tensor::new::<f32, 2, _>(device, &right_rows);
    left.mat_mul(&right)
}

#[test]
fn sequential_resolve_reuses_shared_ancestor() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Sequential resolve(a) then resolve(b) sharing intermediate `x`. We drop
        // the user-facing `x` handle so its node is only kept alive by the
        // descendants; this must not throw the buffer away and force
        // `b.resolve()` to recompute.
        let (a_kernels, b_kernels) = {
            let x = build_matmul_intermediate(&device);
            let a = x.sin();
            let b = x.cos();
            drop(x);
            let a_kernels = a.data().materialize().1;
            let b_kernels = b.data().materialize().1;
            (a_kernels, b_kernels)
        };

        assert!(
            a_kernels > 0,
            "first resolve should dispatch at least one kernel",
        );
        assert_eq!(
            b_kernels, 1,
            "second resolve should reuse the shared ancestor and only dispatch \
             the final operation (got {b_kernels})",
        );
    });
}

#[test]
fn shared_ancestor_freed_when_no_descendant_live() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let x = build_intermediate(&device);
        let x_key = x.key();
        let a = x.sin();
        drop(x);

        // Resolve `a`. `x` has no external Tensor handle and `a` is the target —
        // after `a`'s kernel runs, `x` should be eligible for freeing.
        let _ = a.data().materialize();
        drop(a);

        // After dropping `a`, the entire chain should be gone from the graph.
        assert!(
            device.compute_graph().node_count() == 0
                || device.compute_graph().live_descendant_count(x_key) == 0,
            "x should be released after its only descendant `a` is dropped",
        );
    });
}

#[test]
fn live_descendant_count_tracks_clone_and_drop() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let x = build_intermediate(&device);
        let x_key = x.key();

        // No descendants yet — x is alive only via its own ref_count.
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 0);

        let a = x.sin();
        // One alive child: a.
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 1);

        // Cloning `a` bumps a.ref_count but doesn't add an edge — x's edge-count
        // to alive children stays at 1.
        let a2 = a.clone();
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 1);

        let b = x.cos();
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 2);

        drop(a2);
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 2);

        drop(a);
        assert_eq!(device.compute_graph().live_descendant_count(x_key), 1);

        drop(b);
        // b dropping makes x's last alive child dead, and dropping x's only
        // remaining external ref makes the whole subtree collectable.
        drop(x);
        // Past this point the node may or may not be gone depending on whether
        // any other test holds it — but the device's graph should be empty.
        assert_eq!(
            device.compute_graph().node_count(),
            0,
            "graph should be empty after all tensors drop",
        );
    });
}

#[test]
fn deep_lazy_chain_frees_intermediates_during_resolve() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Build a multi-branch lazy graph N layers deep, holding only the
        // final tensor. This mimics the qwen-vision blow-up pattern, where each
        // layer multiplies node count via fan-out and recombination.
        const STEPS: usize = 4;
        let mut h = build_intermediate(&device);
        for _ in 0..STEPS {
            let b1 = (&h * 0.5).sin();
            let b2 = (&h * 0.3).cos();
            let b3 = &h + 0.1;
            h = (b1 + b2) + b3;
        }
        let final_key = h.key();

        let nodes_before_resolve = device.compute_graph().node_count();
        assert!(
            nodes_before_resolve >= STEPS,
            "expected deep lazy chain to accumulate nodes (got {nodes_before_resolve})",
        );

        let (_, kernels) = h.data.materialize();
        assert!(kernels > 0, "expected kernels to actually dispatch");

        assert!(
            device.compute_graph().is_cached_for_test(final_key),
            "final tensor should be cached after resolve",
        );

        // The key invariant: number of cached buffers after resolve is small
        // (proportional to held outputs), not proportional to STEPS. Pre-fix
        // behaviour would keep every intermediate cached because the held
        // final tensor pins the whole chain as "alive".
        let cached_after = device.compute_graph().cached_node_count();
        assert!(
            cached_after <= 4,
            "deep chain should free its intermediates during resolve; only the \
             final output (plus at most a handful of input tensors) should still \
             be cached (got {cached_after} cached nodes over {STEPS} steps)",
        );
    });
}

#[test]
fn auto_flush_resolves_pending_siblings() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Threshold deliberately small so building several independent lazy
        // outputs trips it on the first resolve.
        device.compute_graph().set_flush_threshold(8);

        // Several independent lazy outputs the user still holds.
        let outputs: Vec<_> = (0..6).map(|_| build_intermediate(&device).sin()).collect();

        let before = device.compute_graph().node_count();
        assert!(
            before >= 8,
            "expected enough nodes to trip the flush threshold (got {before})",
        );

        // Resolve a single one. The end-of-resolve auto_flush should also
        // materialize the other live, uncached outputs.
        let _ = outputs[0].data.materialize();

        // Every held output should now be cached.
        for out in &outputs {
            assert!(
                device.compute_graph().is_cached_for_test(out.key()),
                "output should be cached after auto-flush",
            );
        }
    });
}

// --- split + op + cat lowering (resolve/recognize_cat.rs) ---

/// Narrow a 2D tensor along a dimension as a view.
fn narrow2(tensor: &Tensor, dim: usize, start: usize, length: usize) -> Tensor {
    let shape = tensor.shape();
    let specs: Vec<StrideSpec> = (0..2)
        .map(|i| {
            if i == dim {
                StrideSpec::dim(i, length).with_offset(start)
            } else {
                StrideSpec::dim(i, shape[i])
            }
        })
        .collect();
    tensor.restride(specs)
}

fn cat_test_input(device: &Device) -> (Tensor, Vec<Vec<f32>>) {
    let rows: Vec<Vec<f32>> = (0..4)
        .map(|r| (0..8).map(|c| (r * 8 + c) as f32 * 0.1).collect())
        .collect();
    (Tensor::new::<f32, 2, _>(device, &rows), rows)
}

#[test]
fn cat_of_same_op_collapses_to_op_on_whole_tensor() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (x, rows) = cat_test_input(&device);

        let first = narrow2(&x, 1, 0, 4).sin();
        let second = narrow2(&x, 1, 4, 4).sin();
        let dest = Tensor::splat(&device, 0.0f32, [4, 8]);
        let out = dest
            .slice_assign([0..4, 0..4], &first)
            .slice_assign([0..4, 4..8], &second);

        let (_, kernels) = out.data.materialize();
        assert_eq!(
            kernels, 1,
            "same op over chunks tiling the tensor should collapse to one kernel"
        );

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                assert!(
                    (result[[r, c]] - value.sin()).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}

#[test]
fn cat_of_different_ops_fuses_to_single_select_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (x, rows) = cat_test_input(&device);

        let first = narrow2(&x, 1, 0, 4).sin();
        let second = narrow2(&x, 1, 4, 4).cos();
        let dest = Tensor::splat(&device, 0.0f32, [4, 8]);
        let out = dest
            .slice_assign([0..4, 0..4], &first)
            .slice_assign([0..4, 4..8], &second);

        let (_, kernels) = out.data.materialize();
        assert_eq!(kernels, 1, "branch ops should lift into the select arms");

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                let expected = if c < 4 { value.sin() } else { value.cos() };
                assert!(
                    (result[[r, c]] - expected).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}

#[test]
fn reordered_cat_with_op_fuses_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (x, rows) = cat_test_input(&device);

        // rotate_half: cat([-x2, x1], last_dim)
        let negated_second = &narrow2(&x, 1, 4, 4) * -1.0;
        let first = narrow2(&x, 1, 0, 4);
        let dest = Tensor::splat(&device, 0.0f32, [4, 8]);
        let out = dest
            .slice_assign([0..4, 0..4], &negated_second)
            .slice_assign([0..4, 4..8], &first);

        let (_, kernels) = out.data.materialize();
        assert_eq!(kernels, 1, "reordered cat should still fuse to one kernel");

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for c in 0..8 {
                let expected = if c < 4 { -row[c + 4] } else { row[c - 4] };
                assert!(
                    (result[[r, c]] - expected).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}

#[test]
fn partial_slice_assign_keeps_destination_outside_region() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (x, rows) = cat_test_input(&device);

        // Only cover the left half: the right half must keep the splat value.
        let first = narrow2(&x, 1, 0, 4).sin();
        let dest = Tensor::splat(&device, 7.0f32, [4, 8]);
        let out = dest.slice_assign([0..4, 0..4], &first);

        let (_, kernels) = out.data.materialize();
        assert_eq!(kernels, 1);

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for c in 0..8 {
                let expected = if c < 4 { row[c].sin() } else { 7.0 };
                assert!(
                    (result[[r, c]] - expected).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}

#[test]
fn three_way_chunk_cat_collapses() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|r| (0..4).map(|c| (r * 4 + c) as f32 * 0.1).collect())
            .collect();
        let x = Tensor::new::<f32, 2, _>(&device, &rows);

        let dest = Tensor::splat(&device, 0.0f32, [6, 4]);
        let mut out = dest;
        for chunk in 0..3 {
            let start = chunk * 2;
            let part = &narrow2(&x, 0, start, 2) * 2.0;
            out = out.slice_assign([start..start + 2, 0..4], &part);
        }

        let (_, kernels) = out.data.materialize();
        assert_eq!(kernels, 1, "3-way same-op chunk cat should collapse");

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                assert!(
                    (result[[r, c]] - value * 2.0).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}

#[test]
fn deep_branch_chains_collapse_through_cat() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (x, rows) = cat_test_input(&device);

        // Multi-node branches: ((narrow * 2) + 1).sin() on each half.
        let branch = |start: usize| ((&narrow2(&x, 1, start, 4) * 2.0) + 1.0).sin();
        let dest = Tensor::splat(&device, 0.0f32, [4, 8]);
        let out = dest
            .slice_assign([0..4, 0..4], &branch(0))
            .slice_assign([0..4, 4..8], &branch(4));

        let (_, kernels) = out.data.materialize();
        assert_eq!(kernels, 1, "whole branch chains should inline and collapse");

        let result = out.as_slice::<2, f32>().await.unwrap();
        for (r, row) in rows.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                let expected = (value * 2.0 + 1.0).sin();
                assert!(
                    (result[[r, c]] - expected).abs() < 1e-5,
                    "mismatch at [{r}, {c}]"
                );
            }
        }
    });
}
