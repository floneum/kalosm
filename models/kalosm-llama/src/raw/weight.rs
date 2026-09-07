//! A linear weight as the GGUF stores it.
//!
//! Most checkpoints quantize every matmul weight, and those rows are read in
//! place by the quantized contraction. Some keep one matrix dense — Gemma 3's
//! QAT files store `token_embd.weight` as f16, and an f16 export is dense
//! throughout — and a dense weight contracts in its own dtype with the
//! activation cast to match, never decoded to a second copy.

use fusor::quantized::{contract_rows, QMatrix};
use fusor::tensor::Dyn;
use fusor::{Dim, Dtype, Element, Graph, Result, Tensor};
use fusor_gguf::RawTensorBytes;

#[derive(Clone)]
pub(crate) enum Weight {
    /// Block-quantized rows, read in place.
    Quantized(QMatrix),
    /// A dense `[rows, cols]` matrix in the file's own dtype (f16 or f32).
    Dense(Dyn),
}

impl Weight {
    /// The `[rows, cols]` weight a GGUF tensor denotes. `fusor_gguf` already
    /// reverses GGUF's fastest-varying-first dims at read, so `raw.shape` is
    /// row-major `[rows, cols]` as-is; a rank-1 tensor is a single row.
    pub(crate) fn from_raw(graph: &Graph, raw: &RawTensorBytes) -> Result<Self> {
        let (rows, cols) = match raw.shape.as_slice() {
            [cols] => (1, *cols),
            [rows, cols] => (*rows, *cols),
            other => {
                return Err(fusor::Error::Shape(format!(
                    "{} has GGUF shape {other:?}; a weight matrix is rank 1 or 2",
                    raw.name
                )));
            }
        };
        let shape = [Dim::Const(rows), Dim::Const(cols)];
        match raw.fmt {
            Dtype::Q(fmt) => Ok(Self::Quantized(QMatrix::from_raw_bytes(
                graph, fmt, raw.layout, shape, &raw.bytes,
            )?)),
            Dtype::F16 | Dtype::F32 | Dtype::BF16 => Ok(Self::Dense(Dyn::from_slice(
                graph.handle(),
                raw.fmt,
                &shape,
                &raw.bytes,
            )?)),
            other => Err(fusor::Error::Dtype(format!(
                "{} has dtype {other:?}, which is not a weight matrix dtype",
                raw.name
            ))),
        }
    }

    /// The runtime tensor underneath.
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    fn tensor(&self) -> &Dyn {
        match self {
            Self::Quantized(q) => &q.tensor,
            Self::Dense(d) => d,
        }
    }

    /// A leaf of the same dtype, format and shape with no contents of its
    /// own: the slot a replayed graph binds each block's weight into.
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) fn slot(&self, graph: &Graph) -> Result<Self> {
        match self {
            Self::Dense(d) => Ok(Self::Dense(graph.leaf("", &d.shape(), d.dtype())?)),
            Self::Quantized(q) => {
                let (Dim::Const(rows), Dim::Const(cols)) = (q.rows, q.cols) else {
                    return Err(fusor::Error::Shape(
                        "a quantized weight slot needs constant extents".into(),
                    ));
                };
                let blocks = rows * (cols / u64::from(q.fmt.block_elements()).max(1));
                let bytes = vec![0u8; (blocks * u64::from(q.fmt.block_bytes(q.layout))) as usize];
                Ok(Self::Quantized(QMatrix::from_raw_bytes(
                    graph,
                    q.fmt,
                    q.layout,
                    [q.rows, q.cols],
                    &bytes,
                )?))
            }
        }
    }

    /// Put the weight on the device now, so a slot can adopt its buffer.
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) fn upload(&self) -> Result<()> {
        self.tensor().upload()
    }

    /// Rebind this slot to `from`'s device buffer; see
    /// [`fusor::tensor::Dyn::adopt_buffer`].
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) fn adopt(&self, from: &Self) -> Result<()> {
        self.tensor().adopt_buffer(from.tensor())
    }

    /// The quantized matrix, when the weight is one.
    pub(crate) fn quantized(&self) -> Option<&QMatrix> {
        match self {
            Self::Quantized(q) => Some(q),
            Self::Dense(_) => None,
        }
    }

    /// Row count.
    pub(crate) fn rows(&self) -> Dim {
        match self {
            Self::Quantized(q) => q.rows,
            Self::Dense(w) => w.dim(0),
        }
    }

    /// `x @ self^T`: `[.., k]` in, `[.., rows]` out, in the activation's dtype.
    #[track_caller]
    pub(crate) fn mat_mul<const R: usize, T: Element>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        match self {
            Self::Quantized(q) => x.q_mat_mul(q),
            Self::Dense(w) => {
                let out = (|| {
                    let act = if x.as_dyn().dtype() == w.dtype() {
                        x.as_dyn().clone()
                    } else {
                        x.as_dyn().cast(w.dtype())?
                    };
                    let out = contract_rows(&act, w, w.dim(0))?;
                    if out.dtype() == T::DTYPE {
                        Ok(out)
                    } else {
                        out.cast(T::DTYPE)
                    }
                })();
                Tensor::<R, T>::from_dyn(out.expect("dense weight contraction"))
            }
        }
    }

    /// The rows named by `ids`, as f32. `[n]` in, `[n, cols]` out.
    #[track_caller]
    pub(crate) fn rows_at(&self, ids: &Tensor<1, u32>) -> Tensor<2, f32> {
        match self {
            Self::Quantized(q) => q.rows_at(ids),
            Self::Dense(w) => {
                let picked = w.index_select(0, ids.as_dyn()).and_then(|t| {
                    if t.dtype() == Dtype::F32 {
                        Ok(t)
                    } else {
                        t.cast(Dtype::F32)
                    }
                });
                Tensor::<2, f32>::from_dyn(picked.expect("dense embedding rows"))
            }
        }
    }

    /// One matrix stacked from `parts` along rows, when every part is
    /// quantized alike; a dense part keeps the projections separate.
    pub(crate) fn concat_rows(parts: &[&Self]) -> Option<Self> {
        let qs: Option<Vec<&QMatrix>> = parts.iter().map(|p| p.quantized()).collect();
        QMatrix::concat_rows(&qs?).ok().map(Self::Quantized)
    }
}
