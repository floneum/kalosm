use std::path::PathBuf;

use fusor::Device;
use rgliner::{Gliner, GlinerSource};

fn local_edge_source() -> GlinerSource {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_dir = manifest_dir.join("weights");

    GlinerSource::local(
        weights_dir.join("gliner-edge.gguf"),
        weights_dir.join("gliner-edge-label-encoder.gguf"),
    )
}

#[test]
fn edge_example_sentences_regression() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            // Keep the regression on an explicit CPU backend; libtest's panic-hook
            // environment can interfere with the auto-device probe, while the plain
            // example covers the user-facing default path separately.
            let mut gliner = Gliner::builder()
                .with_source(local_edge_source())
                .with_device(Device::cpu())
                .build()
                .await?;

            let labels = ["person", "organization", "location"];
            let cases = [
                (
                    "Apple Inc. was founded by Steve Jobs in California.",
                    vec![
                        ("organization", "Apple Inc."),
                        ("person", "Steve Jobs"),
                        ("location", "California"),
                    ],
                ),
                (
                    "Microsoft Corporation is headquartered in Seattle.",
                    vec![
                        ("organization", "Microsoft Corporation"),
                        ("location", "Seattle"),
                    ],
                ),
                (
                    "Elon Musk is the CEO of Tesla.",
                    vec![("person", "Elon Musk"), ("organization", "Tesla")],
                ),
                (
                    "Google was founded in Mountain View.",
                    vec![("organization", "Google"), ("location", "Mountain View")],
                ),
            ];

            for (text, expected) in cases {
                let uncached_entities = gliner.extract(text, &labels).await?;
                let uncached: Vec<(&str, &str)> = uncached_entities
                    .iter()
                    .map(|entity| (entity.label.as_str(), entity.text.as_str()))
                    .collect();

                assert_eq!(
                    uncached, expected,
                    "unexpected uncached entities for input: {text}"
                );

                gliner.cache_labels(&labels).await?;
                let entities = gliner.extract_with_cached_labels(text).await?;
                let actual: Vec<(&str, &str)> = entities
                    .iter()
                    .map(|entity| (entity.label.as_str(), entity.text.as_str()))
                    .collect();

                assert_eq!(actual, expected, "unexpected entities for input: {text}");
                assert!(
                    entities.iter().all(|entity| entity.score >= 0.5),
                    "all expected entities should remain above the default threshold for input: {text}"
                );
            }

            Ok(())
        })
}
