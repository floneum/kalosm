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
        let [_batch, seq_len, _input_size] = input.shape();
        let lengths = vec![seq_len; input.shape()[0]];
        self.forward_with_lengths(input, &lengths).await
    }

    /// Forward pass through BiLSTM with explicit per-item sequence lengths.
    ///
    /// Padded timesteps are masked out so shorter sequences in a batch do not
    /// corrupt the backward direction state.
    pub async fn forward_with_lengths(
        &self,
        input: &Tensor<3, f32>,
        lengths: &[usize],
    ) -> Tensor<3, f32> {
        let [batch, seq_len, _input_size] = input.shape();
        assert_eq!(lengths.len(), batch, "lengths must match batch size");
        let device = input.device();

        let fwd_out = run_direction(
            input,
            &self.forward,
            self.hidden_size,
            &device,
            false,
            lengths,
        );
        let bwd_out = run_direction(
            input,
            &self.backward,
            self.hidden_size,
            &device,
            true,
            lengths,
        );

        // Concatenate forward and backward along the feature dim -> [batch, seq, 2*hidden]
        Tensor::cat([fwd_out, bwd_out], 2)
            .reshape([batch, seq_len, 2 * self.hidden_size])
            .to_concrete()
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn debug_forward_direction(
        &self,
        input: &Tensor<3, f32>,
        lengths: &[usize],
        reverse: bool,
    ) -> Tensor<3, f32> {
        run_direction(
            input,
            if reverse {
                &self.backward
            } else {
                &self.forward
            },
            self.hidden_size,
            &input.device(),
            reverse,
            lengths,
        )
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn debug_first_step_gates(
        &self,
        input: &Tensor<3, f32>,
        reverse: bool,
    ) -> Tensor<2, f32> {
        let [batch, seq_len, input_size] = input.shape();
        let hidden_size = self.hidden_size;
        let dir = if reverse {
            &self.backward
        } else {
            &self.forward
        };
        let t = if reverse { seq_len - 1 } else { 0 };
        let h: Tensor<2, f32> = Tensor::zeros(&input.device(), [batch, hidden_size]);
        let x_t: Tensor<2, f32> = input
            .narrow(1, t, 1)
            .reshape([batch, input_size])
            .to_concrete();
        let bias_broadcast: Tensor<2, f32> = dir
            .bias
            .unsqueeze(0)
            .broadcast_as([batch, 4 * hidden_size])
            .to_concrete();
        (x_t.mat_mul(&dir.w_ih_t) + h.mat_mul(&dir.w_hh_t) + bias_broadcast).to_concrete()
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn debug_forward_direction_state_only(
        &self,
        input: &Tensor<3, f32>,
        lengths: &[usize],
        reverse: bool,
    ) -> Tensor<2, f32> {
        let [batch, seq_len, input_size] = input.shape();
        let hidden_size = self.hidden_size;
        let device = input.device();
        let dir = if reverse {
            &self.backward
        } else {
            &self.forward
        };

        let mut h: Tensor<2, f32> = Tensor::zeros(&device, [batch, hidden_size]);
        let mut c: Tensor<2, f32> = Tensor::zeros(&device, [batch, hidden_size]);
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
                .reshape([batch, input_size])
                .to_concrete();
            let gates_pre: Tensor<2, f32> =
                (x_t.mat_mul(&dir.w_ih_t) + h.mat_mul(&dir.w_hh_t) + bias_broadcast.clone())
                    .to_concrete();

            let i_raw: Tensor<2, f32> = gates_pre.narrow(1, 0, hidden_size).to_concrete();
            let f_raw: Tensor<2, f32> =
                gates_pre.narrow(1, hidden_size, hidden_size).to_concrete();
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

            let next_c = (f_gate * c.clone() + i_gate * g_gate).to_concrete();
            let next_h = (o_gate * next_c.clone().tanh()).to_concrete();

            let active_mask_2d = timestep_mask_2d(&device, batch, hidden_size, lengths, t);
            c = active_mask_2d.where_cond(&next_c, &c).to_concrete();
            h = active_mask_2d.where_cond(&next_h, &h).to_concrete();
        }

        h
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
    lengths: &[usize],
) -> Tensor<3, f32> {
    let [batch, seq_len, _] = input.shape();

    let mut h: Tensor<2, f32> = Tensor::zeros(device, [batch, hidden_size]);
    let mut c: Tensor<2, f32> = Tensor::zeros(device, [batch, hidden_size]);
    let mut outputs: Tensor<3, f32> = Tensor::zeros(device, [batch, seq_len, hidden_size]);

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

        let next_c = (f_gate * c.clone() + i_gate * g_gate).to_concrete();
        let next_h = (o_gate * next_c.clone().tanh()).to_concrete();

        let active_mask_2d = timestep_mask_2d(device, batch, hidden_size, lengths, t);
        c = active_mask_2d.where_cond(&next_c, &c).to_concrete();
        h = active_mask_2d.where_cond(&next_h, &h).to_concrete();

        let output_t = h.clone().unsqueeze(1).to_concrete().materialized();
        outputs = outputs
            .slice_assign([0..batch, t..(t + 1), 0..hidden_size], &output_t)
            .materialized();
    }

    let all_active = lengths.iter().all(|&length| length >= seq_len);
    if all_active {
        outputs
    } else {
        let mask_data: Vec<f32> = lengths
            .iter()
            .flat_map(|&length| (0..seq_len).map(move |t| if t < length { 1.0 } else { 0.0 }))
            .collect();
        let mask: Tensor<3, f32> = Tensor::new(device, &mask_data)
            .reshape([batch, seq_len, 1])
            .broadcast_as([batch, seq_len, hidden_size])
            .to_concrete();
        let zeros: Tensor<3, f32> = Tensor::zeros(device, [batch, seq_len, hidden_size]);
        mask.where_cond(&outputs, &zeros).to_concrete()
    }
}

fn timestep_mask_2d(
    device: &Device,
    batch: usize,
    hidden_size: usize,
    lengths: &[usize],
    timestep: usize,
) -> Tensor<2, f32> {
    let mask_data: Vec<f32> = lengths
        .iter()
        .map(|&length| if timestep < length { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(device, &mask_data)
        .reshape([batch, 1])
        .broadcast_as([batch, hidden_size])
        .to_concrete()
}

/// sigmoid via `0.5 * (tanh(x / 2) + 1)` — avoids needing scalar-left division
/// or a `recip` primitive, and keeps the computation on-device.
fn sigmoid_2d(x: &Tensor<2, f32>) -> Tensor<2, f32> {
    let half = (x.clone() * 0.5f32).to_concrete();
    ((half.tanh() + 1.0f32) * 0.5f32).to_concrete()
}
