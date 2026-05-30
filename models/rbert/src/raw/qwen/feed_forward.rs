use fusor::{Device, QMatrix, Result, Tensor, VarBuilder};

/// Qwen-style Feed Forward Network with gate/up/down projections
/// Formula: SiLU(x @ gate) * (x @ up) @ down
pub struct QwenFeedForward {
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
}

impl QwenFeedForward {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let gate = vb.get("ffn_gate.weight", device)?;
        let up = vb.get("ffn_up.weight", device)?;
        let down = vb.get("ffn_down.weight", device)?;

        Ok(Self { gate, up, down })
    }

    pub fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let gate = x.q_mat_mul(&self.gate);
        let up = x.q_mat_mul(&self.up);
        // SiLU(gate) * up, then project down
        gate.silu().mul_(&up).q_mat_mul(&self.down)
    }
}
