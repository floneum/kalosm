//! ModernBERT GeGLU Feed Forward Network.

use fusor::{Device, QMatrix, Result, Tensor, VarBuilder};

/// GeGLU Feed Forward Network with fused gate+up projection.
///
/// Formula: GeGLU(x) = GELU(gate) * up @ down
/// where [gate, up] = x @ fused_gate_up
///
/// This differs from Qwen's SiLU-gated FFN by using GELU instead of SiLU.
pub struct GeGluFeedForward {
    /// Fused gate+up projection: [2 * intermediate_size, hidden_size]
    gate_up: QMatrix,
    down: QMatrix,
    intermediate_size: usize,
}

impl GeGluFeedForward {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let gate_up = vb.get("ffn_gate_up.weight", device)?;
        let down = vb.get("ffn_down.weight", device)?;

        // Determine intermediate size from fused weight dimensions
        // gate_up is [2 * intermediate_size, hidden_size]
        let intermediate_size = gate_up.shape()[0] / 2;

        Ok(Self {
            gate_up,
            down,
            intermediate_size,
        })
    }

    pub fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        // Compute fused gate+up: [batch, seq_len, 2 * intermediate_size]
        let gate_up = x.q_mat_mul(&self.gate_up).to_concrete();

        // Split into gate and up
        let gate = gate_up.narrow(2, 0, self.intermediate_size).to_concrete();
        let up = gate_up.narrow(2, self.intermediate_size, self.intermediate_size);

        // GeGLU: GELU(gate) * up, then project down
        gate.gelu().mul_(&up).q_mat_mul(&self.down)
    }
}
