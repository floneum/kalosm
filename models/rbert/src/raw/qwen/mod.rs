mod attention;
mod feed_forward;
mod layer;
mod model;

pub use model::QwenEmbeddingModel;

use fusor2::device::Device;
use fusor2::layers::Linear;
use fusor2::tensor::Tensor;
use fusor2::{Dim, Dtype, QMatrix, Result, VarBuilder};

/// A matmul weight that is either a block-quantized [`QMatrix`] read in place
/// or a dense `[out, in]` tensor, depending on what the GGUF ships.
pub(crate) enum QLinear {
    Quantized(QMatrix),
    Dense(Linear),
}

impl QLinear {
    pub(crate) fn load(vb: &VarBuilder, device: &Device, name: &str) -> Result<Self> {
        let raw = vb.get_raw(name)?;
        // `fusor2_gguf` reverses GGUF's fastest-varying-first dims at read, so
        // `raw.shape` is already row-major `[rows, cols]`.
        match raw.fmt {
            Dtype::Q(fmt) => {
                let (rows, cols) = match raw.shape.as_slice() {
                    [cols] => (1, *cols),
                    [rows, cols] => (*rows, *cols),
                    other => {
                        return Err(fusor2::Error::Shape(format!(
                            "{name} has GGUF shape {other:?}; a QMatrix is rank 1 or 2"
                        )));
                    }
                };
                let q = QMatrix::from_raw_bytes(
                    device.graph(),
                    fmt,
                    raw.layout,
                    [Dim::Const(rows), Dim::Const(cols)],
                    &raw.bytes,
                )?;
                Ok(Self::Quantized(q))
            }
            _ => {
                let shape: Vec<Dim> = raw.shape.iter().map(|d| Dim::Const(*d)).collect();
                let dense = Tensor::from_slice(
                    device.graph().handle(),
                    raw.fmt,
                    &shape,
                    &raw.bytes,
                )?;
                let dense = match raw.fmt {
                    Dtype::F32 => dense,
                    _ => dense.cast(Dtype::F32)?,
                };
                Ok(Self::Dense(Linear::new(dense, None)))
            }
        }
    }

    /// `x @ W^T`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => q.q_mat_mul(x),
            Self::Dense(linear) => linear.forward(x),
        }
    }
}
