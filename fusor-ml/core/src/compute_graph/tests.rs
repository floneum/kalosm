use crate::{Device, Tensor};

// Build a small intermediate that requires a real kernel (not a Tensor input).
// `x` materializes via `(input * 2.0) + 1.0`, which fuses to a single nary.
fn build_intermediate(device: &Device) -> Tensor {
    let rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 8];
    let input = Tensor::new::<f32, 2, _>(device, &rows);
    (&input * 2.0) + 1.0
}

#[test]
fn sequential_resolve_reuses_shared_ancestor() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Sequential resolve(a) then resolve(b) sharing intermediate `x`. We drop
        // the user-facing `x` handle so its node is only kept alive by the
        // descendants — exactly the case where the old freeing predicate would
        // throw the buffer away and force `b.resolve()` to recompute.
        let seq_total = {
            let x = build_intermediate(&device);
            let a = x.sin();
            let b = x.cos();
            drop(x);
            let a_kernels = a.data().materialize().1;
            let b_kernels = b.data().materialize().1;
            a_kernels + b_kernels
        };

        let batch_total = {
            let x = build_intermediate(&device);
            let a = x.sin();
            let b = x.cos();
            drop(x);
            device.resolve_batch(&[a.key(), b.key()])
        };

        assert_eq!(
            seq_total, batch_total,
            "sequential resolve should reuse shared ancestors and dispatch the \
             same number of kernels as resolve_batch (got seq={seq_total}, \
             batch={batch_total})",
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
        // final tensor. This mimics the qwen-vision blow-up pattern
        // (each layer multiplies node count via fan-out and recombination)
        // the `FLUSH_EVERY = 4` workaround used to handle.
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
