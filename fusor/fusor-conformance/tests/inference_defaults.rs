//! Exercise the shared compiler path without diagnostic optimization flags.
#![cfg(not(target_arch = "wasm32"))]

#[test]
#[ignore = "requires a live GPU; run explicitly with --ignored"]
fn ordinary_device_preserves_normalization_attention_and_cache_snapshots() {
    use fusor::{Session, session::Backend};
    use fusor_conformance::{harness, suite};

    let _guard = harness::gpu_test_guard();
    let backend =
        pollster::block_on(Backend::gpu()).expect("GPU required for shared-compiler regression");
    let session = Session::new(backend).unwrap();
    eprintln!("{}: {:?}", session.caps().name, session.caps().kind);
    let names = [
        "rms_norm_decode_rows",
        "padded_cache_decode_prefix",
        "padded_cache_growth_replay",
        "cache_attention_grouped_heads",
        "padded_cache_tail_batches",
        "fixed_cache_commit_all_replay_ring",
        "fixed_cache_commit_all_shared_store",
        "rope_pair_with_position",
        "rope_interleaved_pair_with_position",
    ];
    let cases = suite::registry();
    for name in names {
        let case = cases
            .iter()
            .find(|case| case.short() == name)
            .expect("registered case");
        let outcome = harness::run_one(case, &session);
        eprintln!("{name}: {outcome:?}");
        assert!(outcome.is_pass(), "{name}: {outcome:?}");
    }
}
