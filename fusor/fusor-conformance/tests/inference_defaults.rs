//! Exercise production inference defaults without diagnostic optimization flags.
#![cfg(not(target_arch = "wasm32"))]

#[test]
#[ignore = "requires a live GPU; run explicitly with --ignored"]
fn inference_profile_preserves_normalization_attention_and_cache_snapshots() {
    use fusor::{Session, session::Backend};
    use fusor_conformance::{harness, suite};
    use fusor_ir::execution::ExecutionProfile;

    let _guard = harness::gpu_test_guard();
    let backend = pollster::block_on(Backend::gpu_with_profile(ExecutionProfile::LowLatency))
        .expect("GPU required for inference-profile regression");
    let session = Session::new(backend).unwrap();
    eprintln!(
        "{}: {:?}",
        session.caps().name,
        session.caps().execution_profile
    );
    let names = [
        "rms_norm_decode_rows",
        "padded_cache_decode_prefix",
        "padded_cache_llama_geometry",
        "padded_cache_tail_batches",
        "fixed_cache_donation_replay_ring",
        "fixed_cache_donation_shared_store",
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
