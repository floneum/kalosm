use fusor::Device;
use kalosm_model_types::FileSource;
use rgliner::relation_decoding::Relation;
use rgliner::relex::{GlinerRelEx, GlinerRelExSource};
use rgliner::{Entity, Gliner, GlinerSource};

const SMOKE_TEXT: &str = "Apple was founded by Steve Jobs in California.";
const ENTITY_LABELS: &[&str] = &["person", "organization", "location"];
const RELATION_LABELS: &[&str] = &["founded by", "located in"];

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

fn entity_overlaps_text(entity: &Entity, text: &str, expected_text: &str) -> bool {
    let expected_start = text
        .find(expected_text)
        .unwrap_or_else(|| panic!("{expected_text:?} must appear in {text:?}"));
    let expected_end = expected_start + expected_text.len();
    entity.start_char < expected_end && entity.end_char > expected_start
}

fn assert_contains_entity(
    model_name: &str,
    entities: &[Entity],
    text: &str,
    expected_text: &str,
    expected_label: &str,
) {
    assert!(
        entities.iter().any(|entity| {
            entity.label == expected_label && entity_overlaps_text(entity, text, expected_text)
        }),
        "{model_name} did not extract {expected_text:?} as {expected_label:?}; entities: {entities:#?}"
    );
}

fn assert_contains_relation(
    relations: &[Relation],
    text: &str,
    lhs_text: &str,
    expected_relation: &str,
    rhs_text: &str,
) {
    assert!(
        relations.iter().any(|relation| {
            relation.relation == expected_relation
                && ((entity_overlaps_text(&relation.head, text, lhs_text)
                    && entity_overlaps_text(&relation.tail, text, rhs_text))
                    || (entity_overlaps_text(&relation.head, text, rhs_text)
                        && entity_overlaps_text(&relation.tail, text, lhs_text)))
        }),
        "RelEx did not extract a {expected_relation:?} relation between {lhs_text:?} and {rhs_text:?}; relations: {relations:#?}"
    );
}

#[test]
fn remote_edge_and_relex_models_extract_expected_entities() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let mut gliner = Gliner::builder()
                .with_source(remote_edge_source())
                .with_device(Device::cpu())
                .with_threshold(0.01)
                .build_with_loading_handler(|_| {})
                .await?;

            let gliner_entities = gliner.extract(SMOKE_TEXT, ENTITY_LABELS).await?;
            assert_contains_entity(
                "GLiNER edge",
                &gliner_entities,
                SMOKE_TEXT,
                "Apple",
                "organization",
            );
            assert_contains_entity(
                "GLiNER edge",
                &gliner_entities,
                SMOKE_TEXT,
                "Steve Jobs",
                "person",
            );

            let relex = GlinerRelEx::builder()
                .with_source(GlinerRelExSource::relex_multi())
                .with_device(Device::cpu())
                .with_entity_threshold(0.2)
                .with_relation_threshold(0.2)
                .build_with_loading_handler(|_| {})
                .await?;

            let (relex_entities, relex_relations) = relex
                .extract(SMOKE_TEXT, ENTITY_LABELS, RELATION_LABELS)
                .await?;
            assert_contains_entity(
                "GLiNER-RelEx multi",
                &relex_entities,
                SMOKE_TEXT,
                "Apple",
                "organization",
            );
            assert_contains_entity(
                "GLiNER-RelEx multi",
                &relex_entities,
                SMOKE_TEXT,
                "Steve Jobs",
                "person",
            );
            assert_contains_relation(
                &relex_relations,
                SMOKE_TEXT,
                "Apple",
                "founded by",
                "Steve Jobs",
            );

            Ok(())
        })
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

                assert_eq!(
                    uncached.len(),
                    cached.len(),
                    "entity count mismatch for input: {text}"
                );
                for (uncached_entity, cached_entity) in uncached.iter().zip(&cached) {
                    assert_eq!(
                        uncached_entity.0, cached_entity.0,
                        "label mismatch for input: {text}"
                    );
                    assert_eq!(
                        uncached_entity.1, cached_entity.1,
                        "text mismatch for input: {text}"
                    );
                    assert_eq!(
                        uncached_entity.2, cached_entity.2,
                        "start mismatch for input: {text}"
                    );
                    assert_eq!(
                        uncached_entity.3, cached_entity.3,
                        "end mismatch for input: {text}"
                    );
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
