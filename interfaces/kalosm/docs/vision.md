# Kalosm Vision

Kalosm Vision is a collection of image models and utilities for the Kalosm framework. It includes utilities for segmenting images into objects.

## Image Segmentation

Kalosm supports image segmentation with the [`SegmentAnything`] model. You can use the [`SegmentAnything::segment_everything`] method to segment an image into objects or the [`SegmentAnything::segment_from_points`] method to segment an image into objects at specific points:

```rust, no_run
use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder().build().await.unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let images = model
        .segment_from_points(
            SegmentAnythingInferenceSettings::new(image)
                .add_goal_point_normalized(0.5, 0.25),
        )
        .await
        .unwrap();

    images.save("out.png").unwrap();
}
```
