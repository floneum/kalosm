use futures_util::StreamExt;
use kalosm_vision::{Wuerstchen, WuerstchenInferenceSettings};

fn main() {
    pollster::block_on(async {
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
    });
}
