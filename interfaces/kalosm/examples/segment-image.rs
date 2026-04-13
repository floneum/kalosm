use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder().build().await.unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let x = 0.5;
    let y = 0.25;
    let images = model
        .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point(x, y))
        .await
        .unwrap();

    images.save("out.png").unwrap();
}
