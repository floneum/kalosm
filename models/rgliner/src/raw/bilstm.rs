//! Bidirectional LSTM implementation for GLiNER token representation.
//!
//! The BiLSTM processes encoder output to capture bidirectional context
//! before span representation computation.

use fusor::{Device, Result, Tensor, VarBuilder};

/// Bidirectional LSTM layer for token representation.
///
/// Processes transformer encoder output through forward and backward LSTMs
/// and concatenates the outputs.
pub struct BiLstm {
    // Forward LSTM weights
    weight_ih_f: Tensor<2, f32>, // [4*hidden, input_size]
    weight_hh_f: Tensor<2, f32>, // [4*hidden, hidden_size]
    bias_ih_f: Tensor<1, f32>,   // [4*hidden]
    bias_hh_f: Tensor<1, f32>,   // [4*hidden]
    // Backward LSTM weights
    weight_ih_b: Tensor<2, f32>,
    weight_hh_b: Tensor<2, f32>,
    bias_ih_b: Tensor<1, f32>,
    bias_hh_b: Tensor<1, f32>,
    hidden_size: usize,
}

impl BiLstm {
    /// Load BiLSTM weights from GGUF.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight_ih_f: Tensor<2, f32> = vb.get("weight_ih_l0", device)?.dequantize();
        let weight_hh_f: Tensor<2, f32> = vb.get("weight_hh_l0", device)?.dequantize();
        let bias_ih_f: Tensor<1, f32> = vb.get("bias_ih_l0", device)?.dequantize();
        let bias_hh_f: Tensor<1, f32> = vb.get("bias_hh_l0", device)?.dequantize();

        let weight_ih_b: Tensor<2, f32> = vb.get("weight_ih_l0_reverse", device)?.dequantize();
        let weight_hh_b: Tensor<2, f32> = vb.get("weight_hh_l0_reverse", device)?.dequantize();
        let bias_ih_b: Tensor<1, f32> = vb.get("bias_ih_l0_reverse", device)?.dequantize();
        let bias_hh_b: Tensor<1, f32> = vb.get("bias_hh_l0_reverse", device)?.dequantize();

        // hidden_size is 4*hidden (for i,f,g,o gates), so actual hidden = shape[0]/4
        let hidden_size = weight_ih_f.shape()[0] / 4;

        Ok(Self {
            weight_ih_f,
            weight_hh_f,
            bias_ih_f,
            bias_hh_f,
            weight_ih_b,
            weight_hh_b,
            bias_ih_b,
            bias_hh_b,
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
        let [batch_size, seq_len, input_size] = input.shape();
        let device = input.device();
        let output_size = 2 * self.hidden_size;

        // Get all weight data upfront
        let input_data = input.clone().as_slice().await.unwrap();
        let w_ih_f = self.weight_ih_f.clone().as_slice().await.unwrap();
        let w_hh_f = self.weight_hh_f.clone().as_slice().await.unwrap();
        let b_ih_f = self.bias_ih_f.clone().as_slice().await.unwrap();
        let b_hh_f = self.bias_hh_f.clone().as_slice().await.unwrap();
        let w_ih_b = self.weight_ih_b.clone().as_slice().await.unwrap();
        let w_hh_b = self.weight_hh_b.clone().as_slice().await.unwrap();
        let b_ih_b = self.bias_ih_b.clone().as_slice().await.unwrap();
        let b_hh_b = self.bias_hh_b.clone().as_slice().await.unwrap();

        let mut output_data = vec![0.0f32; batch_size * seq_len * output_size];

        for b in 0..batch_size {
            // Forward LSTM
            let forward_out = self.lstm_direction(
                input_data.as_slice(),
                b,
                seq_len,
                input_size,
                w_ih_f.as_slice(),
                w_hh_f.as_slice(),
                b_ih_f.as_slice(),
                b_hh_f.as_slice(),
                false,
            );

            // Backward LSTM
            let backward_out = self.lstm_direction(
                input_data.as_slice(),
                b,
                seq_len,
                input_size,
                w_ih_b.as_slice(),
                w_hh_b.as_slice(),
                b_ih_b.as_slice(),
                b_hh_b.as_slice(),
                true,
            );

            // Concatenate forward and backward outputs
            for t in 0..seq_len {
                for i in 0..self.hidden_size {
                    let out_idx = b * seq_len * output_size + t * output_size;
                    output_data[out_idx + i] = forward_out[t * self.hidden_size + i];
                    output_data[out_idx + self.hidden_size + i] =
                        backward_out[t * self.hidden_size + i];
                }
            }
        }

        Tensor::new(&device, &output_data)
            .reshape([batch_size, seq_len, output_size])
            .to_concrete()
    }

    /// Single direction LSTM pass.
    fn lstm_direction(
        &self,
        input_data: &[f32],
        batch_idx: usize,
        seq_len: usize,
        input_size: usize,
        w_ih: &[f32],
        w_hh: &[f32],
        b_ih: &[f32],
        b_hh: &[f32],
        reverse: bool,
    ) -> Vec<f32> {
        let hidden_size = self.hidden_size;
        let mut h = vec![0.0f32; hidden_size];
        let mut c = vec![0.0f32; hidden_size];
        let mut outputs = vec![0.0f32; seq_len * hidden_size];

        // Process sequence in order (or reverse)
        let indices: Vec<usize> = if reverse {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for (out_idx, &t) in indices.iter().enumerate() {
            // Get input at time t for this batch
            let x_start = batch_idx * seq_len * input_size + t * input_size;
            let x = &input_data[x_start..x_start + input_size];

            // Compute gates: i, f, g, o
            let mut gates = vec![0.0f32; 4 * hidden_size];

            for g in 0..(4 * hidden_size) {
                let mut sum = b_ih[g] + b_hh[g];

                // Input contribution: x @ W_ih^T
                for i in 0..input_size {
                    sum += x[i] * w_ih[g * input_size + i];
                }

                // Hidden contribution: h @ W_hh^T
                for j in 0..hidden_size {
                    sum += h[j] * w_hh[g * hidden_size + j];
                }

                gates[g] = sum;
            }

            // Apply activations and compute new h, c
            for i in 0..hidden_size {
                let i_gate = sigmoid(gates[i]);
                let f_gate = sigmoid(gates[hidden_size + i]);
                let g_gate = tanh(gates[2 * hidden_size + i]);
                let o_gate = sigmoid(gates[3 * hidden_size + i]);

                c[i] = f_gate * c[i] + i_gate * g_gate;
                h[i] = o_gate * tanh(c[i]);
            }

            // Store output in correct position
            let store_pos = if reverse {
                seq_len - 1 - out_idx
            } else {
                out_idx
            };
            for i in 0..hidden_size {
                outputs[store_pos * hidden_size + i] = h[i];
            }
        }

        outputs
    }

    /// Get output dimension (2 * hidden_size for bidirectional).
    pub fn output_dim(&self) -> usize {
        2 * self.hidden_size
    }

    /// Get the hidden size of a single direction.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_tanh() {
        assert!(tanh(0.0).abs() < 1e-6);
        assert!(tanh(10.0) > 0.99);
        assert!(tanh(-10.0) < -0.99);
    }
}
