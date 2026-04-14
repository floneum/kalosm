# Kalosm Vision

Kalosm Vision is a collection of image models and utilities for the Kalosm framework. It includes utilities for generating images from text, and segmenting images into objects.

## Image Generation

You can use the [`Wuerstchen`] model to generate images from text:

```rust, no_run
use futures_util::StreamExt;
use kalosm_vision::{Wuerstchen, WuerstchenInferenceSettings};

#[tokio::main]
async fn main() {
    let model = Wuerstchen::builder().build().await.unwrap();
    let settings = WuerstchenInferenceSettings::new(
        "a cute cat with a hat in a room covered with fur with incredible detail",
    );

    let mut images = model.run(settings);
    while let Some(image) = images.next().await {
        if let Some(buf) = image.generated_image() {
            buf.save(&format!("{}.png", image.sample_num())).unwrap();
        }
    }
}
```

## Image Segmentation

Kalosm supports image segmentation with the [`SegmentAnything`] model. You can use the [`SegmentAnything::segment_everything`] method to segment an image into objects or the [`SegmentAnything::segment_from_points`] method to segment an image into objects at specific points:

```rust, no_run
use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder().build().await.unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let x = image.width() / 2;
    let y = image.height() / 4;
    let images = model
        .segment_from_points(
            SegmentAnythingInferenceSettings::new(image)
                .add_goal_point(x, y),
        )
        .await
        .unwrap();

    images.save("out.png").unwrap();
}
```