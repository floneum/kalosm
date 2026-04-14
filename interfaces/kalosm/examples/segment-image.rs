use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder().build().await.unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let images = model
        .segment_from_points(
            SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25),
        )
        .await
        .unwrap();

    images.save("out.png").unwrap();
}
