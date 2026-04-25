//! Times the public `segment_from_points` for one point on the tiny model.
//! The mask decoder for 1 point is cheap, so any large time is the image encoder.

use fusor::Device;
use segment_anything_rs::*;
use std::time::Instant;

#[tokio::test]
async fn time_one_point_segment() {
    let t = Instant::now();
    let model = SegmentAnything::builder().build().await.unwrap();
    eprintln!("builder.build: {:?}", t.elapsed());

    let image_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
    let image = image::open(&image_path).unwrap();
    let settings =
        SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25);

    let t = Instant::now();
    let _mask = model.segment_from_points(settings).await.unwrap();
    eprintln!("segment_from_points (encoder + decoder): {:?}", t.elapsed());
}

#[tokio::test]
#[ignore = "slow end-to-end GPU smoke test"]
async fn gpu_smoke_one_point_segment() {
    let t = Instant::now();
    let model = SegmentAnything::builder()
        .device(Device::new().await.unwrap())
        .build()
        .await
        .unwrap();
    eprintln!("builder.build (gpu): {:?}", t.elapsed());

    let image_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
    let image = image::open(&image_path).unwrap();
    let (w, h) = (image.width(), image.height());
    let settings =
        SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25);

    let t = Instant::now();
    let mask = model.segment_from_points(settings).await.unwrap();
    eprintln!(
        "segment_from_points gpu (encoder + decoder): {:?}",
        t.elapsed()
    );

    assert_eq!(mask.width(), w);
    assert_eq!(mask.height(), h);

    let mask_rgb = mask.to_rgb8();
    let pixels: &[u8] = mask_rgb.as_raw();
    let total = pixels.len();
    let white_count = pixels.iter().filter(|&&v| v == 255).count();
    let black_count = pixels.iter().filter(|&&v| v == 0).count();
    let white_frac = white_count as f64 / total as f64;
    let black_frac = black_count as f64 / total as f64;
    assert!(white_frac > 0.01, "Mask is all black");
    assert!(black_frac > 0.01, "Mask is all white");
}
