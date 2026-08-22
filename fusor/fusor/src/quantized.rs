//! `QMatrix`: a quantized weight matrix as a leaf.
//!
//! Both storage layouts are legal inputs everywhere; moving between them is
//! the priced `qrepack` rewrite, so layout never feeds back into routing
//! through format variants.

use fusor_gguf::VarBuilder;
use fusor_ir::dtype::{Dtype, QFmt, QLayout};
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;

use crate::graph::{Graph, GraphRef};
use crate::tensor::Tensor;
use crate::{Error, Result};

/// A block-quantized `[rows, cols]` weight matrix.
#[derive(Clone)]
pub struct QMatrix {
    /// The runtime-rank quantized tensor.
    pub tensor: Tensor,
    /// Quantization format.
    pub fmt: QFmt,
    /// Packed storage layout.
    pub layout: QLayout,
    /// Matrix row extent.
    pub rows: Dim,
    /// Matrix column extent.
    pub cols: Dim,
}

impl QMatrix {
    /// The `QMatrix` a quantized *value* denotes, or `None` when the tensor
    /// is not one.
    ///
    /// Recovers `(fmt, layout, shape)` from the `LeafKind::Quantized` node
    /// itself, so any quantized tensor gets the same [`Self::dequantize`]
    /// class. A quantized value that is not a leaf returns `None`.
    pub fn of_tensor(t: &Tensor) -> Option<Self> {
        if !t.dtype().is_quantized() {
            return None;
        }
        let (fmt, layout, shape) = t
            .graph()
            .with_egraph(|g| {
                Ok(match &g.node(t.id()).op {
                    Op::Logical(Logical::Leaf(LeafKind::Quantized {
                        fmt, layout, shape, ..
                    })) => Some((*fmt, *layout, shape.clone())),
                    _ => None,
                })
            })
            .ok()??;
        let [rows, cols] = [*shape.first()?, *shape.get(1)?];
        Some(Self {
            tensor: t.clone(),
            fmt,
            layout,
            rows,
            cols,
        })
    }

    /// A `QMatrix` over raw block bytes, with no file behind it.
    ///
    /// `shape` is `[rows, cols]` in elements, not blocks; `bytes` is the
    /// packed block stream for `(fmt, layout)` in row-major block order.
    /// The byte count is checked against the format table: a short buffer
    /// decodes out of bounds on device with no diagnostic.
    pub fn from_raw_bytes(
        graph: &Graph,
        fmt: QFmt,
        layout: QLayout,
        shape: [Dim; 2],
        bytes: &[u8],
    ) -> Result<Self> {
        Self::from_raw_bytes_in(graph.handle(), fmt, layout, shape, bytes)
    }

    /// [`QMatrix::from_raw_bytes`] against a graph handle rather than a
    /// [`Graph`].
    fn from_raw_bytes_in(
        graph: &GraphRef,
        fmt: QFmt,
        layout: QLayout,
        shape: [Dim; 2],
        bytes: &[u8],
    ) -> Result<Self> {
        let [rows, cols] = shape;
        let elements = fmt.block_elements() as u64;
        if let Dim::Const(c) = cols
            && (elements == 0 || c % elements != 0)
        {
            return Err(Error::Shape(format!(
                "a {fmt:?} matrix needs its inner extent a multiple of {elements}, got {c}"
            )));
        }
        if let (Dim::Const(r), Dim::Const(c)) = (rows, cols) {
            let blocks = r * (c / elements.max(1));
            let want = blocks * fmt.block_bytes(layout) as u64;
            if bytes.len() as u64 != want {
                return Err(Error::Shape(format!(
                    "{fmt:?}/{layout:?} [{r}, {c}] is {want} bytes of blocks, got {}",
                    bytes.len()
                )));
            }
        }
        let id = graph.add_logical(Logical::Leaf(LeafKind::Quantized {
            name: graph.fresh_buffer_id(),
            fmt,
            layout,
            shape: shape.into_iter().collect(),
        }))?;
        graph.set_leaf_bytes(id, bytes.to_vec());
        Ok(Self {
            tensor: graph.tensor(id),
            fmt,
            layout,
            rows,
            cols,
        })
    }

    /// The `[rows, cols]` quantized tensor named `name` under `vb`.
    ///
    /// `raw.shape` is already row-major: the GGUF parser reverses the file's
    /// fastest-varying-first dimension order at read (see
    /// [`fusor_gguf::GgufTensor`]), so reversing again here would hand back
    /// a transposed matrix. A rank-1 tensor loads as a single row.
    /// The layout is whatever the file holds, always [`QLayout::Native`].
    pub fn load(vb: &VarBuilder, graph: &Graph, name: &str) -> Result<Self> {
        let raw = vb.get_raw(name)?;
        let Dtype::Q(fmt) = raw.fmt else {
            return Err(Error::Dtype(format!(
                "{name} has dtype {:?}, which is not a block-quantized format",
                raw.fmt
            )));
        };
        let (rows, cols) = match raw.shape.as_slice() {
            [cols] => (1, *cols),
            [rows, cols] => (*rows, *cols),
            other => {
                return Err(Error::Shape(format!(
                    "{name} has GGUF shape {other:?}; a QMatrix is rank 1 or 2"
                )));
            }
        };
        Self::from_raw_bytes(
            graph,
            fmt,
            raw.layout,
            [Dim::Const(rows), Dim::Const(cols)],
            &raw.bytes,
        )
    }

    /// Materialize the dequantized matrix. Almost always the wrong thing —
    /// `q_mat_mul` keeps the weights quantized inside the kernel.
    ///
    /// The sugar node and its definitional `Restride` + `Map` expansion are
    /// unioned into one class here, so there is nothing to recognize later.
    /// Formats that still need a block program keep only the sugar node.
    pub fn dequantize(&self) -> Result<Tensor> {
        let graph = self.tensor.graph();
        // The sugar is minted first, so it takes the lower id and lands in
        // operand 0 of the `Union` — the node carrying the "quantized weights
        // are not trainable" refusal. Building the defn first would silently
        // route a gradient into the unpack `Map` and its `U32` leaves.
        let sugar = graph.add_logical(Logical::Dequant {
            fmt: self.fmt,
            layout: self.layout,
            x: self.tensor.id(),
        })?;
        let Some(defn) = crate::composite::quantized::dequant_defn(self)? else {
            return Ok(graph.tensor(sugar));
        };
        graph.with_egraph(|g| {
            g.mark_defn(defn);
            Ok(())
        })?;
        // See `composite::macro_op`: a stable first-union root, so a decode
        // loop's rebuild keeps one name.
        let root = graph.union_stable(sugar, defn)?;
        Ok(graph.tensor(root))
    }

    /// The `Restride` + `Map` expansion alone, with no `Logical::Dequant` in the
    /// class — the forced definitional spelling.
    ///
    /// The extractor has no alternative here, so a test against this proves
    /// the bit arithmetic rather than proving which class member happened to
    /// win.
    pub fn dequantize_slow(&self) -> Result<Tensor> {
        let graph = self.tensor.graph();
        match crate::composite::quantized::dequant_defn(self)? {
            Some(id) => Ok(graph.tensor(id)),
            None => Err(Error::Dtype(format!(
                "{:?}/{:?} has no Map-spelled decode: a decode is a `Restride` \
                 over the block stream read as `u32` words, and this block's \
                 stride is not a whole number of words",
                self.fmt, self.layout
            ))),
        }
    }

    /// `act @ self^T`: the activation contracts against the quantized rows,
    /// which is the orientation a GGUF weight is stored in. `[.., k]` in,
    /// `[.., rows]` out.
    ///
    /// A rank-1 activation routes through a `[1, k]` view and reshapes back.
    /// A rank-3-or-higher activation folds leading axes into the row axis
    /// (since the weight is rank 2) and restores them on the way out.
    pub fn q_mat_mul(&self, act: &Tensor) -> Result<Tensor> {
        match act.rank() {
            1 => {
                let k = act.dim(0);
                let out = act
                    .reshape_dims(&[Dim::Const(1), k])?
                    .matmul_t(&self.tensor)?;
                out.reshape_dims(&[self.rows])
            }
            2 => act.matmul_t(&self.tensor),
            _ => {
                let shape = act.shape();
                let (lead, k) = shape.split_at(shape.len() - 1);
                let mut rows: u64 = 1;
                for d in lead {
                    let Dim::Const(n) = d else {
                        return Err(Error::Shape(format!(
                            "a rank-{} activation folds its leading axes into the row axis, \
                             which needs them constant; {d} is symbolic",
                            act.rank()
                        )));
                    };
                    rows *= n;
                }
                let flat = act.reshape_dims(&[Dim::Const(rows), k[0]])?;
                let out = flat.matmul_t(&self.tensor)?;
                let mut back: Vec<Dim> = lead.to_vec();
                back.push(self.rows);
                out.reshape_dims(&back)
            }
        }
    }

    /// The rows named by `idx`, decoded to `F32`. `[n]` in, `[n, cols]` out.
    pub fn index_select_rows(&self, idx: &Tensor) -> Result<Tensor> {
        self.index_select_rows_to(idx, Dtype::F32)
    }

    /// The rows named by `idx`, decoded to `dtype`.
    ///
    /// `Dequant` then `Gather`: the decode is a value, the row pick is a
    /// value, and which program computes them is the extractor's decision.
    ///
    /// The fused form is a *member*, not a spelling: `GATHER_QUANTIZED_ROWS`
    /// matches this `Gather`-of-`Dequant` pair and mints a float-typed
    /// [`GatherMode::QuantizedRows`] `Gather` reading the quantized leaf
    /// directly.
    ///
    /// [`GatherMode::QuantizedRows`]: fusor_ir::ir::launch::GatherMode::QuantizedRows
    pub fn index_select_rows_to(&self, idx: &Tensor, dtype: Dtype) -> Result<Tensor> {
        if idx.rank() != 1 {
            return Err(Error::Shape(format!(
                "index_select_rows indices must be rank 1, not rank {}",
                idx.rank()
            )));
        }
        if !matches!(idx.dtype(), Dtype::U32 | Dtype::I32) {
            return Err(Error::Dtype(format!(
                "index_select_rows indices must be U32 or I32, not {:?}",
                idx.dtype()
            )));
        }
        if dtype.is_quantized() {
            return Err(Error::Dtype(format!(
                "index_select_rows decodes to a dense dtype, not {dtype:?}"
            )));
        }
        let picked = self.dequantize()?.index_select(0, idx)?;
        if dtype == Dtype::F32 {
            Ok(picked)
        } else {
            picked.cast(dtype)
        }
    }

    /// One matrix stacked from `parts` along rows, without decoding.
    ///
    /// A fused QKV projection is three `[rows_i, cols]` weights read as one
    /// `[sum rows_i, cols]` weight: the block stream is row-major in blocks,
    /// so the concatenation is a byte append and the result decodes to the
    /// concatenation of the parts. Format, storage layout and column count
    /// must agree — a repack is the priced rewrite, not something a concat
    /// performs silently.
    pub fn concat_rows(parts: &[&Self]) -> Result<Self> {
        let Some(first) = parts.first().copied() else {
            return Err(Error::Shape("concat_rows needs at least one matrix".into()));
        };
        if parts.len() == 1 {
            return Ok(first.clone());
        }
        let graph = first.tensor.graph();
        let (fmt, layout, cols) = (first.fmt, first.layout, first.cols);
        let Dim::Const(cols_n) = cols else {
            return Err(Error::Shape(
                "concat_rows needs a constant column extent; the block stride is not \
                 known otherwise"
                    .into(),
            ));
        };

        let mut bytes = Vec::new();
        let mut rows: u64 = 0;
        for (i, m) in parts.iter().enumerate() {
            if !GraphRef::ptr_eq(graph, m.tensor.graph()) {
                return Err(Error::Shape(format!(
                    "concat_rows part {i} lives in a different graph"
                )));
            }
            if m.fmt != fmt || m.layout != layout {
                return Err(Error::Dtype(format!(
                    "concat_rows part {i} is {:?}/{:?}, the first is {fmt:?}/{layout:?}; \
                     moving between them is the priced qrepack rewrite",
                    m.fmt, m.layout
                )));
            }
            if m.cols != cols {
                return Err(Error::Shape(format!(
                    "concat_rows part {i} has {:?} columns, the first has {cols:?}",
                    m.cols
                )));
            }
            let Dim::Const(r) = m.rows else {
                return Err(Error::Shape(format!(
                    "concat_rows part {i} has a symbolic row extent"
                )));
            };
            let Some(part) = graph.leaf_bytes(m.tensor.id()) else {
                return Err(Error::Shape(format!(
                    "concat_rows part {i} has no host bytes; only a quantized leaf \
                     built from bytes can be concatenated"
                )));
            };
            rows = rows.saturating_add(r);
            bytes.extend_from_slice(&part);
        }
        Self::from_raw_bytes_in(
            graph,
            fmt,
            layout,
            [Dim::Const(rows), Dim::Const(cols_n)],
            &bytes,
        )
    }
}
