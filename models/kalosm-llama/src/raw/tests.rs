use super::*;

fn tiny_model() -> (Model, Device, LlamaCache) {
    let device = Device::try_cpu().unwrap();
    let config = Arc::new(LlamaConfig {
        rope_freq_weight: None,
        rope_theta: 10_000.0,
        context_length: 128,
        head_dimension: 4,
        n_head: 4,
        n_layer: 2,
        start_token_string: String::new(),
        stop_token: 0,
        stop_token_string: String::new(),
        chat_template: None,
        rope_scaling: None,
        sliding_window_type: None,
        sliding_window_size: None,
        mrope_sections: None,
        vision_start_token: None,
        image_pad_token: None,
    });
    let weight = |rows, cols, seed| {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 + seed) % 41) as f32 * 0.015 - 0.3)
            .collect();
        Weight::Dense(Tensor::from_slice(&device, [rows, cols], &values).into_dyn())
    };
    let norm = || RmsNorm::new(Some(Tensor::from_slice(&device, [16], &[1.0; 16])), 1e-5);
    let layers = (0..2)
        .map(|layer| {
            let seed = layer * 7;
            LlamaAttention {
                attention_variant: AttentionVariant::Separate(Box::new(SeparateAttention {
                    attention_wq: weight(16, 16, seed + 1),
                    attention_qkv: None,
                    attention_q_norm: None,
                    attention_wk: weight(8, 16, seed + 2),
                    attention_k_norm: None,
                    attention_wv: weight(8, 16, seed + 3),
                    bias: None,
                    interleaved_rope: false,
                })),
                attention_wo: weight(16, 16, seed + 4),
                attention_norm: norm(),
                post_attention_norm: None,
                feed_forward_variant: FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new(
                    weight(20, 16, seed + 5),
                    weight(16, 20, seed + 6),
                    weight(20, 16, seed + 7),
                ))),
                ffn_norm: norm(),
                post_ffn_norm: None,
                n_head: 4,
                n_kv_head: 2,
                head_dim: 4,
                hidden_size: 16,
                rope_cache: RopeImplementation::new(&config, config.rope_theta, &device),
                sliding_window_size: None,
            }
        })
        .collect();
    let mut cache = LlamaCache::new(&config);
    cache.blocks = (0..2).map(|_| KvCache::with_capacity(2, 8)).collect();
    let model = Model {
        tok_embeddings: weight(13, 16, 23),
        tok_embedding_scale: None,
        layers,
        norm: norm(),
        output: weight(13, 16, 31),
        masks: Mutex::new(MaskCache::new()),
        step_inputs: Default::default(),
        embed_inputs: Default::default(),
        config,
        #[cfg(feature = "vision")]
        vision_encoder: None,
    };
    (model, device, cache)
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "{context}, at {i}: {a} != {b}"
        );
    }
}

#[test]
fn batched_prefill_matches_single_token_cache_and_logits() {
    let (batched, batch_device, mut batch_cache) = tiny_model();
    let (single, single_device, mut single_cache) = tiny_model();
    let mut position = 0;
    for length in [27, 1, 19, 1, 10, 1] {
        let tokens: Vec<u32> = (position..position + length)
            .map(|i| (i * 7 % 13) as u32)
            .collect();
        let actual = batched
            .forward(&tokens, &[], &batch_device, Some(&mut batch_cache))
            .unwrap();
        let mut expected = None;
        for &token in &tokens {
            expected = Some(
                single
                    .forward(&[token], &[], &single_device, Some(&mut single_cache))
                    .unwrap(),
            );
        }
        position += length;
        assert_close(
            &actual.to_vec_f32(),
            &expected.unwrap().to_vec_f32(),
            &format!("logits after {position} tokens"),
        );
        assert_eq!(batch_cache.tokens, single_cache.tokens);
        assert_eq!(batch_cache.rope_position, position as u32);
        for (layer, (a, b)) in batch_cache
            .blocks
            .iter()
            .zip(&single_cache.blocks)
            .enumerate()
        {
            let keys = a.k().unwrap().narrow(2, 0, position).to_vec_f32();
            assert_eq!(keys.len(), 2 * position * 4);
            assert_close(
                &keys,
                &b.k().unwrap().narrow(2, 0, position).to_vec_f32(),
                &format!("layer {layer} keys after {position} tokens"),
            );
            assert_close(
                &a.v().unwrap().narrow(2, 0, position).to_vec_f32(),
                &b.v().unwrap().narrow(2, 0, position).to_vec_f32(),
                &format!("layer {layer} values after {position} tokens"),
            );
        }
    }
}
