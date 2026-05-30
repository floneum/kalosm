use fusor_core::*;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt().init();
        use crate::Device;
        use fusor_gguf::GgufMetadata;

        let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
        let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();

        let random_data: Vec<Vec<f32>> = (0..1)
            .map(|_| (0..1536).map(|_| rand::random()).collect())
            .collect();

        let device = Device::new().await.unwrap();

        let mut reader = std::io::Cursor::new(&bytes);
        let metadata = GgufMetadata::read(&mut reader).unwrap();
        let q_matrix_metadata = metadata.tensor_infos.get("blk.3.ffn_down.weight").unwrap();
        println!("Q matrix metadata: {q_matrix_metadata:?}");

        let q_matrix = QMatrix::read(
            &device,
            q_matrix_metadata,
            &mut reader,
            metadata.tensor_data_offset,
        )
        .unwrap();

        let device = device.clone();
        let random_data = random_data.clone();
        let tensor = Tensor::new(&device, &random_data);
        _ = tensor.as_slice::<2, f32>().await.unwrap();
        for _ in 0..1 {
            let new = tensor.q_mat_mul(&q_matrix);
            new.materialize().await;
        }
    });
}
