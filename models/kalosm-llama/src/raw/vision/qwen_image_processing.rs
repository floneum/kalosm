use fusor::{Device, Tensor};
use image::DynamicImage;

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_image(
    image: &DynamicImage,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
    image_mean: &[f32],
    image_std: &[f32],
    device: &Device,
) -> fusor::Result<(Tensor<2, f32>, [u32; 3])> {
    let patch_size_u32 = patch_size as u32;
    let merge_size_u32 = merge_size as u32;
    let merge_patch = patch_size_u32 * merge_size_u32;
    let resized = normalize_image_shape(
        [merge_patch; 2],
        min_pixels.unwrap_or(4 * 28 * 28),
        max_pixels.unwrap_or(512 * 28 * 28),
        image,
    );

    assert!(resized.height().is_multiple_of(merge_patch));
    assert!(resized.width().is_multiple_of(merge_patch));
    let rgb = image_to_rgb(&resized, device)?;
    // Normalize the image
    let mean_tensor = Tensor::new(device, image_mean);
    let required_rgb_mean = mean_tensor.reshape([1, 3, 1, 1]);
    let std_tensor = Tensor::new(device, image_std);
    let required_rgb_std = std_tensor.reshape([1, 3, 1, 1]);
    let rgb = rgb
        .sub_(&required_rgb_mean)
        .div_(&required_rgb_std)
        .to_concrete();

    let grid_t = 1;
    let grid_h = resized.height() as usize / patch_size;
    let grid_w = resized.width() as usize / patch_size;
    let rgb = rgb.reshape([
        1,                                         // time size
        1,                                         // temporal patch size
        3,                                         // channels
        (resized.height() / merge_patch) as usize, // height patches
        merge_size,                                // height merge size
        patch_size,                                // height patch size
        (resized.width() / merge_patch) as usize,  // width patches
        merge_size,                                // width merge size
        patch_size,                                // width patch size
    ]);

    let rgb = Tensor::cat(vec![rgb; temporal_patch_size], 1);

    // Move the time, height, and width dimensions to the start
    let rgb = rgb.permute([0, 3, 6, 4, 7, 2, 1, 5, 8]);
    // Reshape to [patch count, patch data]
    let rgb = rgb.reshape([
        // patch count
        grid_h * grid_w,
        // patch data
        3 * patch_size * patch_size * temporal_patch_size,
    ]);
    Ok((
        rgb.to_concrete(),
        [grid_t as u32, grid_h as u32, grid_w as u32],
    ))
}

fn normalize_image_shape(
    patch_size: [u32; 2],
    min_pixels: u32,
    max_pixels: u32,
    image: &DynamicImage,
) -> DynamicImage {
    let mut width = image.width();
    let mut height = image.height();

    // Round to the nearest multiple of the patch size
    width = (width as f64 / patch_size[0] as f64).ceil() as u32 * patch_size[0];
    height = (height as f64 / patch_size[1] as f64).ceil() as u32 * patch_size[1];

    if width * height > max_pixels {
        // Scale down the image while keeping the aspect ratio to at most the maximum pixels
        let scale_down_by = ((width * height) as f64 / max_pixels as f64).sqrt();
        width =
            (width as f64 / scale_down_by / patch_size[0] as f64).floor() as u32 * patch_size[0];
        height =
            (height as f64 / scale_down_by / patch_size[1] as f64).floor() as u32 * patch_size[1];
    } else if width * height < min_pixels {
        // Scale up the image while keeping the aspect ratio to at least the minimum pixels
        let scale_up_by = (min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * scale_up_by / patch_size[0] as f64).ceil() as u32 * patch_size[0];
        height = (height as f64 * scale_up_by / patch_size[1] as f64).ceil() as u32 * patch_size[1];
    }
    // Ensure the new dimensions are not zero
    if width == 0 || height == 0 {
        width = patch_size[0];
        height = patch_size[1];
    }

    // Finally, resize the image to the new dimensions
    image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
}

fn image_to_rgb(image: &DynamicImage, device: &Device) -> Result<Tensor<4, f32>, fusor::Error> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let rgb = image.to_rgb8();
    let as_u32 = rgb
        .into_raw()
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    let data_tensor = Tensor::new(device, &as_u32);
    let data = data_tensor.reshape([height, width, 3]);
    let img = data.permute([2, 0, 1]).cast::<f32>() * (1.0 / 255.0);

    Ok(img.unsqueeze(0).to_concrete())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_image() {
        // Skip in CI
        if std::env::var("CI").is_ok() {
            return;
        }

        let device = Device::new().await.unwrap();
        // download image from https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
        let image_bytes = reqwest::get(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        )
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
        let image = image::load_from_memory(&image_bytes).unwrap();
        let spacial_merge_size = 2;
        let patch_size = 14;
        let temporal_patch_size = 2;
        let resized = normalize_image_shape(
            [patch_size as u32 * spacial_merge_size as u32; 2],
            256 * 28 * 28,
            512 * 28 * 28,
            &image,
        );
        println!(
            "Resized image size: {:?}",
            [resized.height(), resized.width()]
        );
        assert_eq!(resized.height(), 504);
        assert_eq!(resized.width(), 756);
        let (rgb, [grid_t, grid_h, grid_w]) = process_image(
            &image,
            patch_size,
            spacial_merge_size,
            temporal_patch_size,
            Some(256 * 28 * 28),
            Some(512 * 28 * 28),
            &[0.5, 0.5, 0.5],
            &[0.5, 0.5, 0.5],
            &device,
        )
        .unwrap();
        println!("RGB shape: {:?}", rgb.shape());
        println!("Grid shape: {:?}", [grid_t, grid_h, grid_w]);
        assert_eq!(rgb.shape(), [1944, 1176]);
        assert_eq!([grid_t, grid_h, grid_w], [1, 36, 54]);
    }
}
