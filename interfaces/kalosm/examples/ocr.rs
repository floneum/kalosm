use kalosm::vision::*;

fn main() {
    pollster::block_on(async {
        let mut model = Ocr::builder().build().await.unwrap();
        let image = image::open("examples/ocr.png").unwrap();
        let text = model
            .recognize_text(OcrInferenceSettings::new(image))
            .unwrap();

        println!("{text}");
    });
}
