
#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn immutable_storage_footprint_fixture() {
    use fusor_ir::dtype::{QFmt, QLayout};
    let session = Session::new(Backend::gpu_blocking().unwrap()).unwrap();
    let graph = Graph::new(&session);
    let h = graph.handle();
    let target = session.device().gpu_target().unwrap();
    // A bounded 72 MiB byte stream, far smaller than any complete model.
    let size = 4096 * (32768 / 32) * QFmt::Q4_0.block_bytes(QLayout::Native) as usize;
    let source = h.tensor(h.quantized_leaf(QFmt::Q4_0, QLayout::Native,
        [Dim::Const(4096), Dim::Const(32768)], vec![0x65; size]).unwrap());
    let words = h.tensor(h.words_leaf_of(source.id).unwrap().unwrap());
    let before = target.pool().counters().live_bytes;
    source.upload().unwrap();
    let first = target.pool().counters().live_bytes;
    words.upload().unwrap();
    let second = target.pool().counters().live_bytes;
    let reused = h.device_buf(source.id).unwrap().addr() == h.device_buf(words.id).unwrap().addr();
    println!("STORAGE_FOOTPRINT bytes={size} first_upload_bytes={} second_upload_bytes={} total_upload_bytes={} reused={reused}",
        first-before, second-first, second-before);
    let expected_reuse = std::env::var("EXPECT_STORAGE_REUSE").unwrap() == "1";
    assert_eq!(reused, expected_reuse);
    assert_eq!(second-first, if expected_reuse { 0 } else { first-before });
    for id in [source.id, words.id] {
        let copied = pollster::block_on(session.device().download(&h.device_buf(id).unwrap(), size as u64)).unwrap();
        assert_eq!(copied.len(), size);
        assert!(copied.iter().all(|&byte| byte == 0x65));
    }
    println!("STORAGE_VERIFIED bytes={}", size * 2);
}
