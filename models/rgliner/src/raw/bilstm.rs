//! Bidirectional LSTM implementation for GLiNER token representation.
//!
//! The BiLSTM processes encoder output to capture bidirectional context
//! before span representation computation. The timestep loop is inherently
//! sequential, but every tensor operation inside the loop runs on the active
//! device — no `.as_slice()` round-trips or scalar Rust gate math.

use fusor::{Device, Result, Tensor, VarBuilder};

/// Per-direction LSTM parameters pre-arranged for matrix multiplication.
struct LstmDir {
    // Shape [input_size, 4 * hidden] — transpose of the GGUF `weight_ih_l0*` layout.
    w_ih_t: Tensor<2, f32>,
    // Shape [hidden_size, 4 * hidden] — transpose of the GGUF `weight_hh_l0*` layout.
    w_hh_t: Tensor<2, f32>,
    // Shape [4 * hidden] — pre-summed `bias_ih + bias_hh`.
    bias: Tensor<1, f32>,
}

impl LstmDir {
    fn load(device: &Device, vb: &mut VarBuilder, suffix: &str) -> Result<Self> {
        let w_ih: Tensor<2, f32> = vb
            .get(&format!("weight_ih_l0{suffix}"), device)?
            .dequantize();
        let w_hh: Tensor<2, f32> = vb
            .get(&format!("weight_hh_l0{suffix}"), device)?
            .dequantize();
        let b_ih: Tensor<1, f32> = vb.get(&format!("bias_ih_l0{suffix}"), device)?.dequantize();
        let b_hh: Tensor<1, f32> = vb.get(&format!("bias_hh_l0{suffix}"), device)?.dequantize();

        let w_ih_t = w_ih.transpose(0, 1).to_concrete();
        let w_hh_t = w_hh.transpose(0, 1).to_concrete();
        let bias = (b_ih + b_hh).to_concrete();

        Ok(Self {
            w_ih_t,
            w_hh_t,
            bias,
        })
    }
}

/// Bidirectional LSTM layer for token representation.
///
/// Processes transformer encoder output through forward and backward LSTMs
/// and concatenates the outputs.
pub struct BiLstm {
    forward: LstmDir,
    backward: LstmDir,
    hidden_size: usize,
}

impl BiLstm {
    /// Load BiLSTM weights from GGUF.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let forward = LstmDir::load(device, vb, "")?;
        let backward = LstmDir::load(device, vb, "_reverse")?;

        // The forward weight matrix is [4*hidden, input_size] in GGUF layout, which
        // after transpose becomes [input_size, 4*hidden]. The hidden dim is the
        // last axis divided by 4.
        let hidden_size = forward.w_ih_t.shape()[1] / 4;

        Ok(Self {
            forward,
            backward,
            hidden_size,
        })
    }

    /// Forward pass through BiLSTM.
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, seq_len, input_size]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, 2*hidden_size]
    pub async fn forward(&self, input: &Tensor<3, f32>) -> Tensor<3, f32> {
        let [batch, seq_len, _input_size] = input.shape();
        let device = input.device();

        let fwd_out = run_direction(input, &self.forward, self.hidden_size, &device, false);
        let bwd_out = run_direction(input, &self.backward, self.hidden_size, &device, true);

        // Concatenate forward and backward along the feature dim -> [batch, seq, 2*hidden]
        Tensor::cat([fwd_out, bwd_out], 2)
            .reshape([batch, seq_len, 2 * self.hidden_size])
            .to_concrete()
    }
}

/// Run one direction of the LSTM. Sequential over time; every timestep's gate
/// math stays on-device.
fn run_direction(
    input: &Tensor<3, f32>,
    dir: &LstmDir,
    hidden_size: usize,
    device: &Device,
    reverse: bool,
) -> Tensor<3, f32> {
    let [batch, seq_len, _] = input.shape();

    let mut h: Tensor<2, f32> = Tensor::zeros(device, [batch, hidden_size]);
    let mut c: Tensor<2, f32> = Tensor::zeros(device, [batch, hidden_size]);

    // outputs[t] holds the hidden state at timestep t, already unsqueezed on
    // dim 1 so that a final cat along dim 1 yields [batch, seq_len, hidden].
    let mut outputs: Vec<Tensor<3, f32>> = Vec::with_capacity(seq_len);
    outputs.resize_with(seq_len, || Tensor::zeros(device, [batch, 1, hidden_size]));

    let bias_broadcast: Tensor<2, f32> = dir
        .bias
        .unsqueeze(0)
        .broadcast_as([batch, 4 * hidden_size])
        .to_concrete();

    let iter: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..seq_len).rev())
    } else {
        Box::new(0..seq_len)
    };

    for t in iter {
        let x_t: Tensor<2, f32> = input
            .narrow(1, t, 1)
            .reshape([batch, input.shape()[2]])
            .to_concrete();

        // gates_pre = x_t @ W_ih^T + h @ W_hh^T + bias, shape [batch, 4*hidden]
        let gates_pre: Tensor<2, f32> =
            (x_t.mat_mul(&dir.w_ih_t) + h.mat_mul(&dir.w_hh_t) + bias_broadcast.clone())
                .to_concrete();

        let i_raw: Tensor<2, f32> = gates_pre.narrow(1, 0, hidden_size).to_concrete();
        let f_raw: Tensor<2, f32> = gates_pre.narrow(1, hidden_size, hidden_size).to_concrete();
        let g_raw: Tensor<2, f32> = gates_pre
            .narrow(1, 2 * hidden_size, hidden_size)
            .to_concrete();
        let o_raw: Tensor<2, f32> = gates_pre
            .narrow(1, 3 * hidden_size, hidden_size)
            .to_concrete();

        let i_gate = sigmoid_2d(&i_raw);
        let f_gate = sigmoid_2d(&f_raw);
        let g_gate = g_raw.tanh();
        let o_gate = sigmoid_2d(&o_raw);

        c = (f_gate * c + i_gate * g_gate).to_concrete();
        h = (o_gate * c.clone().tanh()).to_concrete();

        outputs[t] = h.clone().unsqueeze(1).to_concrete();
    }

    Tensor::cat(outputs, 1)
        .reshape([batch, seq_len, hidden_size])
        .to_concrete()
}

/// sigmoid via `0.5 * (tanh(x / 2) + 1)` — avoids needing scalar-left division
/// or a `recip` primitive, and keeps the computation on-device.
fn sigmoid_2d(x: &Tensor<2, f32>) -> Tensor<2, f32> {
    let half = (x.clone() * 0.5f32).to_concrete();
    ((half.tanh() + 1.0f32) * 0.5f32).to_concrete()
}
