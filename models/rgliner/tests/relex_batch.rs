//! Batched-vs-serial parity for the GLiNER-RelEx remote checkpoints. These
//! download model weights, so they only pass with network (and exercise the
//! GPU when one is available, otherwise CPU).

use fusor::Device;
use rgliner::relation_decoding::Relation;
use rgliner::relex::{GlinerRelEx, GlinerRelExSource};
use rgliner::{Entity, GlinerLoadingError};

const ENTITY_LABELS: &[&str] = &["organization", "person", "location"];
const RELATION_LABELS: &[&str] = &["founded by", "located in"];

async fn load_relex(
    source: GlinerRelExSource,
    device: Device,
) -> Result<GlinerRelEx, GlinerLoadingError> {
    GlinerRelEx::builder()
        .with_source(source)
        .with_device(device)
        .build_with_loading_handler(|_| {})
        .await
}

fn entity_signature(entities: &[Entity]) -> Vec<(String, String, usize, usize, usize, usize)> {
    entities
        .iter()
        .map(|entity| {
            (
                entity.label.clone(),
                entity.text.clone(),
                entity.start_char,
                entity.end_char,
                entity.start_word,
                entity.end_word,
            )
        })
        .collect()
}

fn relation_signature(
    relations: &[Relation],
) -> Vec<(String, String, String, usize, usize, usize, usize)> {
    relations
        .iter()
        .map(|relation| {
            (
                relation.head.text.clone(),
                relation.tail.text.clone(),
                relation.relation.clone(),
                relation.head.start_char,
                relation.head.end_char,
                relation.tail.start_char,
                relation.tail.end_char,
            )
        })
        .collect()
}

async fn assert_batch_matches_serial_extract(
    variant: &'static str,
    source: GlinerRelExSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::gpu().await.unwrap_or_else(|_| Device::cpu());
    let texts = [
        "Apple was founded by Steve Jobs.",
        "Google was founded by Larry Page in Mountain View.",
    ];
    let model = load_relex(source, device).await?;

    let mut serial_results = Vec::with_capacity(texts.len());
    for text in texts.iter().copied() {
        serial_results.push(model.extract(text, ENTITY_LABELS, RELATION_LABELS).await?);
    }
    let batched_results = model
        .extract_batch(&texts, ENTITY_LABELS, RELATION_LABELS)
        .await?;

    assert_eq!(
        serial_results.len(),
        batched_results.len(),
        "batch size mismatch for {variant}"
    );

    for ((serial_entities, serial_relations), (batched_entities, batched_relations)) in
        serial_results.iter().zip(&batched_results)
    {
        assert_eq!(
            entity_signature(serial_entities),
            entity_signature(batched_entities),
            "entity mismatch for {variant}"
        );
        assert_eq!(
            relation_signature(serial_relations),
            relation_signature(batched_relations),
            "relation mismatch for {variant}"
        );
        assert_eq!(
            serial_entities.len(),
            batched_entities.len(),
            "entity count mismatch for {variant}"
        );
        assert_eq!(
            serial_relations.len(),
            batched_relations.len(),
            "relation count mismatch for {variant}"
        );

        for (serial_entity, batched_entity) in serial_entities.iter().zip(batched_entities.iter()) {
            assert!(
                (serial_entity.score - batched_entity.score).abs() < 1e-5,
                "entity score mismatch for {variant}: serial={:.6} batched={:.6}",
                serial_entity.score,
                batched_entity.score
            );
        }
        for (serial_relation, batched_relation) in
            serial_relations.iter().zip(batched_relations.iter())
        {
            assert!(
                (serial_relation.score - batched_relation.score).abs() < 1e-5,
                "relation score mismatch for {variant}: serial={:.6} batched={:.6}",
                serial_relation.score,
                batched_relation.score
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn extract_batch_matches_serial_extract_for_remote_multi(
) -> Result<(), Box<dyn std::error::Error>> {
    assert_batch_matches_serial_extract("multi", GlinerRelExSource::relex_multi()).await
}

#[tokio::test]
async fn extract_batch_matches_serial_extract_for_remote_large(
) -> Result<(), Box<dyn std::error::Error>> {
    assert_batch_matches_serial_extract("large", GlinerRelExSource::relex_large()).await
}
