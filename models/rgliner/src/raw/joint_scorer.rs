//! Joint scorer for GLiNER-RelEx.
//!
//! The joint scorer projects token and label embeddings, then uses an MLP
//! to score (token, label) pairs. Outputs 3 classes per pair.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

/// Joint scorer for token-label pair classification.
///
/// Architecture (GLiNER token-level scoring):
/// 1. Project label embeddings: proj_label(label_embs) -> [n_labels, proj_dim]
/// 2. Concatenate token embeddings (hidden) with projected labels (proj_dim)
/// 3. MLP: concat(token, proj_label) -> fc1 -> GELU -> fc2 -> scores
///
/// Note: proj_token exists in weights but the actual forward pass concatenates
/// raw token embeddings with projected labels for the MLP input.
pub struct JointScorer {
    #[allow(dead_code)]
    proj_token: Linear<f32>, // Not used in main scoring path
    proj_label: Linear<f32>,
    out_fc1: Linear<f32>,
    out_fc2: Linear<f32>,
}

impl JointScorer {
    /// Load joint scorer from GGUF.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let proj_token = Linear::load(device, &mut vb.pp("proj_token"))?;
        let proj_label = Linear::load(device, &mut vb.pp("proj_label"))?;
        let out_fc1 = Linear::load(device, &mut vb.pp("out_mlp.0"))?;
        let out_fc2 = Linear::load(device, &mut vb.pp("out_mlp.3"))?;

        Ok(Self {
            proj_token,
            proj_label,
            out_fc1,
            out_fc2,
        })
    }

    /// Score token-label pairs using bilinear interaction.
    ///
    /// # Arguments
    /// * `token_embs` - Token embeddings [batch, seq_len, hidden_dim]
    /// * `label_embs` - Label embeddings [n_labels, hidden_dim]
    ///
    /// # Returns
    /// Scores [batch, seq_len, n_labels, 3] (3 classes: O, B, I)
    ///
    /// # Architecture
    /// The scorer uses bilinear interaction following GLiNER's design:
    /// 1. Project both tokens and labels: hidden_dim -> hidden_dim * 2
    /// 2. Split each projection into two halves (first, second)
    /// 3. MLP input = concat(token_first, label_first, token_second * label_second)
    /// 4. This enables complex token-label interactions through the element-wise product
    pub async fn forward(
        &self,
        token_embs: &Tensor<3, f32>,
        label_embs: &Tensor<2, f32>,
    ) -> Tensor<4, f32> {
        let [batch_size, seq_len, _hidden_dim] = token_embs.shape();
        let [n_labels, _] = label_embs.shape();

        // Project both token and label embeddings
        // token: [batch, seq, hidden] -> [batch, seq, hidden*2]
        let proj_tokens = self.proj_token.forward(token_embs);
        let [_, _, proj_dim] = proj_tokens.shape();
        let half_proj = proj_dim / 2;

        // label: [n_labels, hidden] -> [n_labels, hidden*2]
        let label_embs_3d: Tensor<3, f32> = label_embs.unsqueeze(0).to_concrete();
        let proj_labels = self.proj_label.forward(&label_embs_3d);
        let proj_labels: Tensor<2, f32> = proj_labels.squeeze(0).to_concrete();

        // Split and combine: token_first + label_first + (token_second * label_second)
        // MLP input dimension = half_proj + half_proj + half_proj = 3 * half_proj
        let mlp_input_dim = 3 * half_proj;

        // Get raw data slices (without expansion - we'll handle broadcast manually)
        // proj_tokens shape: [batch, seq, proj_dim]
        // proj_labels shape: [n_labels, proj_dim]
        let tokens_data = proj_tokens.clone().as_slice().await.unwrap();
        let labels_data = proj_labels.clone().as_slice().await.unwrap();

        let tokens_slice = tokens_data.as_slice(); // [batch * seq * proj_dim]
        let labels_slice = labels_data.as_slice(); // [n_labels * proj_dim]

        // Build combined features with manual broadcasting
        // Output: [batch, seq, n_labels, mlp_input_dim]
        let total_elements = batch_size * seq_len * n_labels;
        let mut combined_data = vec![0.0f32; total_elements * mlp_input_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for l in 0..n_labels {
                    // Token features for (b, s): at index (b * seq_len + s) * proj_dim
                    let tok_base = (b * seq_len + s) * proj_dim;
                    // Label features for l: at index l * proj_dim
                    let lab_base = l * proj_dim;
                    // Output index for (b, s, l)
                    let out_idx = (b * seq_len * n_labels + s * n_labels + l) * mlp_input_dim;

                    // token_first (first half of token projection)
                    for i in 0..half_proj {
                        combined_data[out_idx + i] = tokens_slice[tok_base + i];
                    }
                    // label_first (first half of label projection)
                    for i in 0..half_proj {
                        combined_data[out_idx + half_proj + i] = labels_slice[lab_base + i];
                    }
                    // element-wise product of second halves
                    for i in 0..half_proj {
                        let tok_second = tokens_slice[tok_base + half_proj + i];
                        let lab_second = labels_slice[lab_base + half_proj + i];
                        combined_data[out_idx + 2 * half_proj + i] = tok_second * lab_second;
                    }
                }
            }
        }

        let device = token_embs.device();
        let combined: Tensor<3, f32> = Tensor::new(&device, &combined_data)
            .reshape([1, total_elements, mlp_input_dim])
            .to_concrete();

        // Apply MLP: fc1 -> ReLU -> fc2
        let hidden = self.out_fc1.forward(&combined);
        let hidden = hidden.relu();
        let output = self.out_fc2.forward(&hidden);

        // Reshape back: [batch, seq, n_labels, 3]
        output
            .reshape([batch_size, seq_len, n_labels, 3])
            .to_concrete()
    }

    /// Score with sigmoid for entity predictions.
    ///
    /// Returns the 3 per-class sigmoid scores (start, end, inside) for each
    /// (token, label) pair, shape [batch, seq_len, n_labels, 3].
    ///
    /// The 3 channels are: [start, end, inside] (NOT OBI).
    /// Each channel is passed through independent sigmoid.
    pub async fn forward_entity_scores(
        &self,
        token_embs: &Tensor<3, f32>,
        label_embs: &Tensor<2, f32>,
    ) -> Tensor<4, f32> {
        let logits = self.forward(token_embs, label_embs).await;
        let logits_data = logits.clone().as_slice().await.unwrap();

        // Apply sigmoid to each value independently (NOT softmax).
        let data = logits_data.as_slice();
        let sigmoid_data: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

        let device = logits.device();
        Tensor::new(&device, &sigmoid_data)
            .reshape(logits.shape())
            .to_concrete()
    }
}

/// Prompt representation layer for entity/relation labels.
///
/// Projects label embeddings through a 2-layer FFN.
pub struct PromptRepLayer {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

impl PromptRepLayer {
    /// Load from GGUF.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let fc1 = Linear::load(device, &mut vb.pp("0"))?;
        let fc2 = Linear::load(device, &mut vb.pp("3"))?;

        Ok(Self { fc1, fc2 })
    }

    /// Project label embeddings.
    ///
    /// # Arguments
    /// * `label_embs` - Label embeddings from encoder [n_labels, hidden]
    ///
    /// # Returns
    /// Projected embeddings [n_labels, hidden]
    pub fn forward(&self, label_embs: &Tensor<2, f32>) -> Tensor<2, f32> {
        // Wrap as 3D for Linear::forward
        let label_3d: Tensor<3, f32> = label_embs.unsqueeze(0).to_concrete();
        let hidden = self.fc1.forward(&label_3d);
        let hidden = hidden.relu();
        let output = self.fc2.forward(&hidden);
        output.squeeze(0).to_concrete()
    }

    /// Forward for 3D tensor [batch, n_labels, hidden].
    pub fn forward_3d(&self, label_embs: &Tensor<3, f32>) -> Tensor<3, f32> {
        let hidden = self.fc1.forward(label_embs);
        let hidden = hidden.relu();
        self.fc2.forward(&hidden)
    }
}
