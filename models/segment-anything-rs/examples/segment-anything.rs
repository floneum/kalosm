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
    // Point at center-x, quarter-y (in the sky above the building)
    // Coordinates are normalized to [0, 1]
    let mask = model
        .segment_from_points(
            SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25),
        )
        .await
        .unwrap();

    mask.save("out.png").unwrap();
    println!("Saved mask to out.png");
}
