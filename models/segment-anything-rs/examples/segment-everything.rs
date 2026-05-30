use segment_anything_rs::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder()
        .build()
        .await
        .expect("Failed to load model");

    let image_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
    let image = image::open(&image_path).unwrap();
    let masks = model.segment_everything(image.clone()).await.unwrap();

    for (i, mask) in masks.iter().enumerate() {
        mask.save(format!("{i}.png")).unwrap();
    }
}
