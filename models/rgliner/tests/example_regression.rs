use fusor::Device;
use kalosm_model_types::FileSource;
use rgliner::{Gliner, GlinerSource};

fn remote_edge_source() -> GlinerSource {
    GlinerSource::custom(
        FileSource::huggingface(
            "Demonthos/gliner-gguf".to_string(),
            "main".to_string(),
            "gliner-bi-edge-v2.0-Q4_K.gguf".to_string(),
        ),
        FileSource::huggingface(
            "Demonthos/gliner-gguf".to_string(),
            "main".to_string(),
            "gliner-bi-edge-v2.0-Q4_K-label-encoder.gguf".to_string(),
        ),
        FileSource::huggingface(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ),
        FileSource::huggingface(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ),
        FileSource::huggingface(
            "knowledgator/gliner-bi-edge-v2.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ),
        FileSource::huggingface(
            "knowledgator/gliner-bi-edge-v2.0".to_string(),
            "main".to_string(),
            "gliner_config.json".to_string(),
        ),
    )
}

#[test]
fn remote_edge_cached_labels_match_uncached_extract() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            // Keep the regression on an explicit CPU backend; libtest's panic-hook
            // environment can interfere with the auto-device probe, while the plain
            // example covers the user-facing default path separately.
            let mut gliner = Gliner::builder()
                .with_source(remote_edge_source())
                .with_device(Device::cpu())
                .build()
                .await?;

            let labels = ["person", "organization", "location"];
            let cases = [
                "Apple Inc. was founded by Steve Jobs in California.",
                "Microsoft Corporation is headquartered in Seattle.",
                "Elon Musk is the CEO of Tesla.",
                "Google was founded in Mountain View.",
            ];

            for text in cases {
                let uncached_entities = gliner.extract(text, &labels).await?;
                let uncached: Vec<(String, String, usize, usize, f32)> = uncached_entities
                    .iter()
                    .map(|entity| {
                        (
                            entity.label.clone(),
                            entity.text.clone(),
                            entity.start_char,
                            entity.end_char,
                            entity.score,
                        )
                    })
                    .collect();

                gliner.cache_labels(&labels).await?;
                let entities = gliner.extract_with_cached_labels(text).await?;
                let cached: Vec<(String, String, usize, usize, f32)> = entities
                    .iter()
                    .map(|entity| {
                        (
                            entity.label.clone(),
                            entity.text.clone(),
                            entity.start_char,
                            entity.end_char,
                            entity.score,
                        )
                    })
                    .collect();

                assert_eq!(uncached.len(), cached.len(), "entity count mismatch for input: {text}");
                for (uncached_entity, cached_entity) in uncached.iter().zip(&cached) {
                    assert_eq!(uncached_entity.0, cached_entity.0, "label mismatch for input: {text}");
                    assert_eq!(uncached_entity.1, cached_entity.1, "text mismatch for input: {text}");
                    assert_eq!(uncached_entity.2, cached_entity.2, "start mismatch for input: {text}");
                    assert_eq!(uncached_entity.3, cached_entity.3, "end mismatch for input: {text}");
                    assert!(
                        (uncached_entity.4 - cached_entity.4).abs() < 1e-5,
                        "score mismatch for input: {text}: uncached={:.6} cached={:.6}",
                        uncached_entity.4,
                        cached_entity.4
                    );
                }
            }

            Ok(())
        })
}

#[test]
fn edge_extract_batch_matches_serial_extract() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let mut gliner = Gliner::builder()
                .with_source(remote_edge_source())
                .with_device(Device::cpu())
                .build()
                .await?;

            let labels = ["person", "organization", "location"];
            let texts = [
                "Apple Inc. was founded by Steve Jobs in California.",
                "Microsoft Corporation is headquartered in Seattle.",
                "",
                "Google was founded in Mountain View.",
            ];

            let mut serial = Vec::with_capacity(texts.len());
            for text in texts.iter().copied() {
                let entities = gliner.extract(text, &labels).await?;
                serial.push(
                    entities
                        .into_iter()
                        .map(|entity| {
                            (
                                entity.label,
                                entity.text,
                                entity.start_char,
                                entity.end_char,
                                entity.score,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }

            let batched = gliner.extract_batch(&texts, &labels).await?;
            assert_eq!(batched.len(), texts.len());

            for (serial_entities, batched_entities) in serial.iter().zip(&batched) {
                let batched: Vec<(String, String, usize, usize, f32)> = batched_entities
                    .iter()
                    .map(|entity| {
                        (
                            entity.label.clone(),
                            entity.text.clone(),
                            entity.start_char,
                            entity.end_char,
                            entity.score,
                        )
                    })
                    .collect();

                assert_eq!(serial_entities.len(), batched.len());
                for (serial_entity, batched_entity) in serial_entities.iter().zip(&batched) {
                    assert_eq!(serial_entity.0, batched_entity.0);
                    assert_eq!(serial_entity.1, batched_entity.1);
                    assert_eq!(serial_entity.2, batched_entity.2);
                    assert_eq!(serial_entity.3, batched_entity.3);
                    assert!(
                        (serial_entity.4 - batched_entity.4).abs() < 1e-5,
                        "score mismatch: serial={:.6} batched={:.6}",
                        serial_entity.4,
                        batched_entity.4
                    );
                }
            }

            Ok(())
        })
}

#[test]
#[ignore = "cache the remote edge checkpoint and sidecars"]
fn cache_remote_edge_checkpoint() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let _gliner = Gliner::builder()
                .with_source(remote_edge_source())
                .with_device(Device::cpu())
                .build()
                .await?;
            Ok(())
        })
}
