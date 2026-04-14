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
    pub fn forward(
        &self,
        token_embs: &Tensor<3, f32>,
        label_embs: &Tensor<2, f32>,
    ) -> Tensor<4, f32> {
        let [batch_size, seq_len, _hidden_dim] = token_embs.shape();
        let [n_labels, _] = label_embs.shape();

        // Project tokens: [batch, seq, hidden] -> [batch, seq, 2*half]
        let proj_tokens = self.proj_token.forward(token_embs);
        let [_, _, proj_dim] = proj_tokens.shape();
        let half = proj_dim / 2;

        // Project labels: [n_labels, hidden] -> [n_labels, 2*half]
        let label_embs_3d: Tensor<3, f32> = label_embs.unsqueeze(0).to_concrete();
        let proj_labels = self.proj_label.forward(&label_embs_3d);
        let proj_labels: Tensor<2, f32> = proj_labels.squeeze(0).to_concrete();

        // Split projections into first/second halves along the feature dim.
        let tokens_first = proj_tokens.narrow(2, 0, half).to_concrete(); // [b, s, half]
        let tokens_second = proj_tokens.narrow(2, half, half).to_concrete(); // [b, s, half]
        let labels_first = proj_labels.narrow(1, 0, half).to_concrete(); // [n, half]
        let labels_second = proj_labels.narrow(1, half, half).to_concrete(); // [n, half]

        // Broadcast to [batch, seq, n_labels, half] and build the three concatenation parts:
        //   [token_first, label_first, token_second * label_second]
        let target = [batch_size, seq_len, n_labels, half];
        let tok_first_4d: Tensor<4, f32> =
            tokens_first.unsqueeze(2).broadcast_as(target).to_concrete();
        let tok_second_4d: Tensor<4, f32> = tokens_second
            .unsqueeze(2)
            .broadcast_as(target)
            .to_concrete();
        let lab_first_4d: Tensor<4, f32> = labels_first
            .unsqueeze(0)
            .unsqueeze(0)
            .broadcast_as(target)
            .to_concrete();
        let lab_second_4d: Tensor<4, f32> = labels_second
            .unsqueeze(0)
            .unsqueeze(0)
            .broadcast_as(target)
            .to_concrete();

        let prod_4d: Tensor<4, f32> = (tok_second_4d * lab_second_4d).to_concrete();

        // Concat along the last dim -> [batch, seq, n_labels, 3*half]
        let combined: Tensor<4, f32> = Tensor::cat([tok_first_4d, lab_first_4d, prod_4d], 3);

        // Linear::forward takes 3D input, so fold (batch, seq, n_labels) into one dim.
        let mlp_in_dim = 3 * half;
        let flat: Tensor<3, f32> = combined
            .reshape([1, batch_size * seq_len * n_labels, mlp_in_dim])
            .to_concrete();

        let hidden = self.out_fc1.forward(&flat).relu();
        let output = self.out_fc2.forward(&hidden);
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
    pub fn forward_entity_scores(
        &self,
        token_embs: &Tensor<3, f32>,
        label_embs: &Tensor<2, f32>,
    ) -> Tensor<4, f32> {
        let logits = self.forward(token_embs, label_embs);
        // sigmoid(x) = 0.5 * (tanh(x / 2) + 1); stays on-device and avoids needing
        // scalar-left division or a `recip` primitive.
        let half_logits: Tensor<4, f32> = (logits * 0.5f32).to_concrete();
        let tanh = half_logits.tanh();
        ((tanh + 1.0f32) * 0.5f32).to_concrete()
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

    /// Forward for 3D tensor [batch, n_labels, hidden].
    pub fn forward_3d(&self, label_embs: &Tensor<3, f32>) -> Tensor<3, f32> {
        let hidden = self.fc1.forward(label_embs);
        let hidden = hidden.relu();
        self.fc2.forward(&hidden)
    }
}
