//! Entity pair projector for relation classification.
//!
//! Projects concatenated head and tail entity embeddings to a space
//! suitable for scoring against relation label embeddings.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

/// Entity pair projector.
///
/// Architecture: Linear(hidden*2 -> hidden) -> ReLU -> Dropout -> Linear(hidden -> hidden)
///
/// Takes concatenated head and tail entity embeddings and produces a pair representation
/// that can be scored against relation label embeddings.
pub struct PairProjector {
    linear1: Linear<f32>,
    linear2: Linear<f32>,
}

impl PairProjector {
    /// Load the pair projector from GGUF weights.
    ///
    /// The GGUF weights use numeric indices (0, 3) for the layers
    /// corresponding to the PyTorch Sequential layer indices.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let linear1 = Linear::load(device, &mut vb.pp("0"))?;
        let linear2 = Linear::load(device, &mut vb.pp("3"))?;

        Ok(Self { linear1, linear2 })
    }

    /// Project entity pairs to relation space.
    ///
    /// # Arguments
    /// * `head_embeddings` - Head entity embeddings [num_pairs, hidden_size]
    /// * `tail_embeddings` - Tail entity embeddings [num_pairs, hidden_size]
    ///
    /// # Returns
    /// Pair representations [num_pairs, hidden_size]
    pub fn forward(
        &self,
        head_embeddings: &Tensor<2, f32>,
        tail_embeddings: &Tensor<2, f32>,
    ) -> Tensor<2, f32> {
        let [_num_pairs, _hidden_size] = head_embeddings.shape();

        // Expand to 3D for cat operation, then squeeze back
        let head_3d: Tensor<3, f32> = head_embeddings.unsqueeze(0).to_concrete();
        let tail_3d: Tensor<3, f32> = tail_embeddings.unsqueeze(0).to_concrete();

        // Concatenate head and tail: [1, num_pairs, hidden_size * 2]
        let concatenated = Tensor::cat([head_3d, tail_3d], 2);

        // First layer: Linear -> ReLU
        let hidden = self.linear1.forward(&concatenated).relu();

        // Second layer: Linear -> squeeze back to 2D
        let result = self.linear2.forward(&hidden);
        result.squeeze(0).to_concrete()
    }
}
