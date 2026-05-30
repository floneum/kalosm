use kalosm_ocr::*;

fn main() {
    pollster::block_on(async {
        {
            let mut model = Ocr::builder().build().await.unwrap();
            let image = image::open("examples/written.png").unwrap();
            let text = model
                .recognize_text(OcrInferenceSettings::new(image))
                .unwrap();

            println!("{text}");
        }
        {
            let mut model = Ocr::builder()
                .with_source(OcrSource::base_printed())
                .build()
                .await
                .unwrap();
            let image = image::open("examples/printed.png").unwrap();
            let text = model
                .recognize_text(OcrInferenceSettings::new(image))
                .unwrap();

            println!("{text}");
        }
    });
}
