use fusor::{Device, Result, Tensor, VarBuilder};

use super::QLinear;

/// Qwen-style Feed Forward Network with gate/up/down projections
/// Formula: SiLU(x @ gate) * (x @ up) @ down
pub struct QwenFeedForward {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl QwenFeedForward {
    pub fn load(device: &Device, vb: &VarBuilder) -> Result<Self> {
        let gate = QLinear::load(vb, device, "ffn_gate.weight")?;
        let up = QLinear::load(vb, device, "ffn_up.weight")?;
        let down = QLinear::load(vb, device, "ffn_down.weight")?;

        Ok(Self { gate, up, down })
    }

    pub fn forward(&self, x: &Tensor<3>) -> Tensor<3> {
        let gate = self.gate.forward(x);
        let up = self.up.forward(x);
        // SiLU(gate) * up, then project down
        self.down.forward(&gate.silu().mul(&up))
    }
}
