mod attention;
mod feed_forward;
mod layer;
mod model;

pub use model::QwenEmbeddingModel;

use fusor2::layers::Linear;
use fusor2::{Device, Dim, Dtype, Error, QMatrix, Result, Tensor, VarBuilder};

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
        let [rows, cols] = match raw.shape.as_slice() {
            [cols] => [Dim::Const(1), Dim::Const(*cols)],
            [rows, cols] => [Dim::Const(*rows), Dim::Const(*cols)],
            other => {
                return Err(Error::Shape(format!(
                    "{name} has GGUF shape {other:?}; a matmul weight is rank 1 or 2"
                )));
            }
        };
        match raw.fmt {
            Dtype::Q(fmt) => {
                let q =
                    QMatrix::from_raw_bytes(device.graph(), fmt, raw.layout, [rows, cols], &raw.bytes)?;
                Ok(Self::Quantized(q))
            }
            // `from_raw_bytes` reads at the file's dtype and casts to the
            // graph's, so an f16 checkpoint needs no cast written here.
            _ => {
                let weight = Tensor::<2>::from_raw_bytes(device, raw.fmt, [rows, cols], &raw.bytes);
                Ok(Self::Dense(Linear::new(weight, None)))
            }
        }
    }

    /// `x @ W^T`.
    pub(crate) fn forward<const R: usize>(&self, x: &Tensor<R>) -> Tensor<R> {
        match self {
            Self::Quantized(q) => x.q_mat_mul(q),
            Self::Dense(linear) => linear.forward(x),
        }
    }
}
