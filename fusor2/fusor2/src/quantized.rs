//! `QMatrix`: a quantized weight matrix as a leaf.
//!
//! Both storage layouts are legal inputs everywhere; moving between them is
//! the priced `qrepack` rewrite, so layout never feeds back into routing
//! through format variants.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;
use fusor2_ir::dtype::{Dtype, QFmt, QLayout};
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::shape::Dim;

use crate::graph::{Graph, GraphRef};
use crate::tensor::Tensor;
use crate::{Error, Result};

/// A block-quantized `[rows, cols]` weight matrix.
#[derive(Clone)]
pub struct QMatrix {
    pub tensor: Tensor,
    pub fmt: QFmt,
    pub layout: QLayout,
    pub rows: Dim,
    pub cols: Dim,
}

impl QMatrix {
    /// A `QMatrix` over raw block bytes, with no file behind it.
    ///
    /// `shape` is `[rows, cols]` **in elements**, not blocks; `bytes` is the
    /// packed block stream for `(fmt, layout)` in row-major block order.
    /// The byte count is checked against the format table rather than
    /// trusted, because a short buffer decodes out of bounds on device with
    /// no diagnostic.
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
    /// [`Graph`]. `concat_rows` builds its result in the graph its inputs
    /// already live in, and a `QMatrix` only carries the handle.
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
        let id = graph.add_l0(L0::Leaf(LeafKind::Quantized {
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
    /// GGUF stores a matrix as `[cols, rows]` — the fastest-varying extent
    /// first — so the shape is reversed here, matching the reference loader.
    /// The layout is whatever the file holds, always [`QLayout::Native`];
    /// moving to `F32Scales` is the priced `qrepack` rewrite and not a
    /// loader decision.
    pub fn load(vb: &VarBuilder, graph: &Graph, name: &str) -> Result<Self> {
        let raw = vb.get_raw(name)?;
        let Dtype::Q(fmt) = raw.fmt else {
            return Err(Error::Dtype(format!(
                "{name} has dtype {:?}, which is not a block-quantized format",
                raw.fmt
            )));
        };
        let (cols, rows) = match raw.shape.as_slice() {
            [cols] => (*cols, 1),
            [cols, rows] => (*cols, *rows),
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
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::emit(
            self.tensor.graph(),
            L0::Dequant {
                fmt: self.fmt,
                layout: self.layout,
                x: self.tensor.id(),
            },
        )
    }

    /// `act @ self^T`: the activation contracts against the quantized rows,
    /// which is the orientation a GGUF weight is stored in. `[.., k]` in,
    /// `[.., rows]` out.
    ///
    /// A rank-1 activation is one matrix row, so it routes through a
    /// `[1, k]` view and reshapes back — the same promotion the reference
    /// makes.
    pub fn q_mat_mul(&self, act: &Tensor) -> Result<Tensor> {
        if act.rank() == 1 {
            let k = act.dim(0);
            let out = act
                .reshape_dims(&[Dim::Const(1), k])?
                .matmul_t(&self.tensor)?;
            return out.reshape_dims(&[self.rows]);
        }
        act.matmul_t(&self.tensor)
    }

    /// The rows named by `idx`, decoded to `F32`. `[n]` in, `[n, cols]` out.
    pub fn index_select_rows(&self, idx: &Tensor) -> Result<Tensor> {
        self.index_select_rows_to(idx, Dtype::F32)
    }

    /// The rows named by `idx`, decoded to `dtype`.
    ///
    /// `Dequant` then `Gather`, which is the reference's spelling: the decode
    /// is a value, the row pick is a value, and which program computes them is
    /// the extractor's decision rather than this method's.
    ///
    /// **The fused form is not reachable from here yet, measured.** The
    /// obvious alternative — one `L0::Gather` on axis 0 over the *quantized*
    /// leaf, then a `Dequant` — is what `fusor2_tile::rules::gather`'s
    /// `GATHER_QUANTIZED_ROWS` matches (it requires operand 0 quantized), and
    /// it computes the wrong numbers: `infer_l1` gives the minted `KGather`
    /// its source's dtype, so the class is `Q(fmt)` while both backends'
    /// `KGather` bodies *already* decode through `operand_src`. The consuming
    /// `Dequant` then decodes the decoded f32 a second time and `Q8_0` row 0
    /// column 0 reads -0.0 against 1.484375. Reaching
    /// [`GatherMode::QuantizedRows`] needs that rule to match the
    /// `Dequant`-of-`Gather` pair and mint a float-typed node — a tile-rule
    /// change, not a frontend one — and until it does, spelling the fused form
    /// here would be a wrong answer wearing a fused program's clothes.
    ///
    /// [`GatherMode::QuantizedRows`]: fusor2_ir::ir::level1::GatherMode::QuantizedRows
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
            return Err(Error::Shape(
                "concat_rows needs at least one matrix".into(),
            ));
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
            if !std::sync::Arc::ptr_eq(graph, m.tensor.graph()) {
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
        Self::from_raw_bytes_in(graph, fmt, layout, [Dim::Const(rows), Dim::Const(cols_n)], &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Device, Session};
    use fusor2_gguf::blocks::{block_fields, cpu_dequantize_block};
    use fusor2_gguf::repack;
    use half::f16;

    const ROWS: u64 = 3;

    fn graph() -> Graph {
        Graph::new(&Session::new(Device::cpu().unwrap()).unwrap())
    }

    /// A well-formed block: an explicit finite scale (and min, where the
    /// format carries one) plus a deterministic payload. A random f16 scale
    /// is NaN or Inf about once in 2000, and a NaN compares unequal to
    /// itself.
    fn make_block(fmt: QFmt, layout: QLayout, seed: u32) -> Vec<u8> {
        let fields = block_fields(fmt, layout);
        let mut block = vec![0u8; fmt.block_bytes(layout) as usize];
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for slot in block.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = (state >> 24) as u8;
        }
        let mut write = |at: u32, value: f32| {
            let at = at as usize;
            if fields.scale_is_f16 {
                block[at..at + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes());
            } else {
                block[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        };
        write(fields.scale, 0.015_625);
        if let Some(min) = fields.min {
            write(min, 0.003_906_25);
        }
        block
    }

    /// `ROWS` blocks and the values the scalar reference decodes them to.
    fn rows(fmt: QFmt, layout: QLayout) -> (Vec<u8>, Vec<f32>) {
        let stride = fmt.block_bytes(layout) as usize;
        let elements = fmt.block_elements() as usize;
        let mut bytes = Vec::with_capacity(ROWS as usize * stride);
        let mut values = vec![0.0f32; ROWS as usize * elements];
        for r in 0..ROWS as usize {
            let block = make_block(fmt, layout, 4001 + r as u32);
            cpu_dequantize_block(
                fmt,
                layout,
                &block,
                &mut values[r * elements..(r + 1) * elements],
            );
            bytes.extend_from_slice(&block);
        }
        (bytes, values)
    }

    #[test]
    fn every_format_and_layout_dequantizes_to_the_reference_decoder() {
        let g = graph();
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let (bytes, want) = rows(fmt, layout);
                let qm = QMatrix::from_raw_bytes(
                    &g,
                    fmt,
                    layout,
                    [
                        Dim::Const(ROWS),
                        Dim::Const(u64::from(fmt.block_elements())),
                    ],
                    &bytes,
                )
                .unwrap();
                assert_eq!(qm.tensor.dtype(), Dtype::Q(fmt));

                let dense = qm.dequantize().unwrap();
                assert_eq!(dense.dtype(), Dtype::F32);
                let got = dense.to_vec_f32().unwrap();
                assert_eq!(got.len(), want.len(), "{fmt:?}/{layout:?}");
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g, w, "{fmt:?}/{layout:?} element {i}");
                }
                // Not a comparison of two zero vectors.
                assert!(want.iter().any(|v| *v != 0.0), "{fmt:?}/{layout:?}");
            }
        }
    }

    #[test]
    fn the_two_layouts_of_one_matrix_decode_identically() {
        // Layout is a priced operand attribute, not a format variant, so a
        // repack must not move a single decoded value.
        let g = graph();
        for fmt in QFmt::ALL {
            let (native, _) = rows(fmt, QLayout::Native);
            let mut widened = Vec::new();
            repack(fmt, QLayout::Native, QLayout::F32Scales, &native, &mut widened).unwrap();

            let shape = [
                Dim::Const(ROWS),
                Dim::Const(u64::from(fmt.block_elements())),
            ];
            let a = QMatrix::from_raw_bytes(&g, fmt, QLayout::Native, shape, &native)
                .unwrap()
                .dequantize()
                .unwrap()
                .to_vec_f32()
                .unwrap();
            let b = QMatrix::from_raw_bytes(&g, fmt, QLayout::F32Scales, shape, &widened)
                .unwrap()
                .dequantize()
                .unwrap()
                .to_vec_f32()
                .unwrap();
            assert_eq!(a, b, "{fmt:?}");
        }
    }

    #[test]
    fn a_short_or_misshapen_byte_stream_is_refused() {
        let g = graph();
        let fmt = QFmt::Q4_0;
        let (bytes, _) = rows(fmt, QLayout::Native);
        let cols = Dim::Const(u64::from(fmt.block_elements()));

        // One byte short.
        assert!(
            QMatrix::from_raw_bytes(
                &g,
                fmt,
                QLayout::Native,
                [Dim::Const(ROWS), cols],
                &bytes[..bytes.len() - 1]
            )
            .is_err()
        );
        // The other layout's block size against these bytes.
        assert!(
            QMatrix::from_raw_bytes(
                &g,
                fmt,
                QLayout::F32Scales,
                [Dim::Const(ROWS), cols],
                &bytes
            )
            .is_err()
        );
        // An inner extent that is not a whole number of blocks.
        assert!(
            QMatrix::from_raw_bytes(
                &g,
                fmt,
                QLayout::Native,
                [Dim::Const(ROWS), Dim::Const(17)],
                &bytes
            )
            .is_err()
        );
    }

    #[test]
    fn q_mat_mul_orients_the_weight_as_gguf_stores_it() {
        // `[.., k] @ [rows, k]^T -> [.., rows]`, and a rank-1 activation
        // comes back rank 1.
        let g = graph();
        let fmt = QFmt::Q8_0;
        let k = u64::from(fmt.block_elements());
        let (bytes, _) = rows(fmt, QLayout::Native);
        let qm = QMatrix::from_raw_bytes(
            &g,
            fmt,
            QLayout::Native,
            [Dim::Const(ROWS), Dim::Const(k)],
            &bytes,
        )
        .unwrap();

        let act2 = g
            .leaf("a", &[Dim::Const(2), Dim::Const(k)], Dtype::F32)
            .unwrap();
        let y2 = qm.q_mat_mul(&act2).unwrap();
        assert_eq!(&y2.shape()[..], &[Dim::Const(2), Dim::Const(ROWS)]);
        assert_eq!(y2.dtype(), Dtype::F32);

        let act1 = g.leaf("b", &[Dim::Const(k)], Dtype::F32).unwrap();
        let y1 = qm.q_mat_mul(&act1).unwrap();
        assert_eq!(&y1.shape()[..], &[Dim::Const(ROWS)]);

        // A contraction extent that does not match the weight's is refused.
        let bad = g
            .leaf("c", &[Dim::Const(2), Dim::Const(k + 32)], Dtype::F32)
            .unwrap();
        assert!(qm.q_mat_mul(&bad).is_err());
    }

    /// `u32` indices as a leaf.
    fn index_leaf(g: &Graph, idx: &[u32]) -> Tensor {
        let mut bytes = Vec::with_capacity(idx.len() * 4);
        for i in idx {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        g.constant_from_raw(Dtype::U32, &[Dim::Const(idx.len() as u64)], &bytes)
            .unwrap()
    }

    #[test]
    fn index_select_rows_decodes_only_the_selected_rows() {
        let g = graph();
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let (bytes, want) = rows(fmt, layout);
                let elements = fmt.block_elements() as usize;
                let qm = QMatrix::from_raw_bytes(
                    &g,
                    fmt,
                    layout,
                    [Dim::Const(ROWS), Dim::Const(elements as u64)],
                    &bytes,
                )
                .unwrap();
                let picks = [2u32, 0, 2, 1];
                let idx = index_leaf(&g, &picks);
                let out = qm.index_select_rows(&idx).unwrap();
                assert_eq!(
                    &out.shape()[..],
                    &[Dim::Const(picks.len() as u64), Dim::Const(elements as u64)]
                );
                let got = out.to_vec_f32().unwrap();
                for (r, p) in picks.iter().enumerate() {
                    for c in 0..elements {
                        assert_eq!(
                            got[r * elements + c],
                            want[*p as usize * elements + c],
                            "{fmt:?}/{layout:?} row {r} (source {p}) column {c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn concat_rows_is_a_byte_append_that_decodes_to_the_parts() {
        let g = graph();
        let fmt = QFmt::Q4K;
        let layout = QLayout::Native;
        let elements = fmt.block_elements() as u64;
        let shape = [Dim::Const(ROWS), Dim::Const(elements)];
        let (bytes, values) = rows(fmt, layout);
        let a = QMatrix::from_raw_bytes(&g, fmt, layout, shape, &bytes).unwrap();
        let b = QMatrix::from_raw_bytes(&g, fmt, layout, shape, &bytes).unwrap();
        let cat = QMatrix::concat_rows(&[&a, &b]).unwrap();
        assert_eq!(cat.rows, Dim::Const(2 * ROWS));
        assert_eq!(cat.cols, Dim::Const(elements));

        let got = cat.dequantize().unwrap().to_vec_f32().unwrap();
        let mut want = values.clone();
        want.extend_from_slice(&values);
        assert_eq!(got, want);

        // A column-count or format disagreement is refused rather than
        // silently repacked.
        let narrow = QMatrix::from_raw_bytes(
            &g,
            fmt,
            layout,
            [Dim::Const(1), Dim::Const(elements)],
            &bytes[..fmt.block_bytes(layout) as usize],
        )
        .unwrap();
        assert!(QMatrix::concat_rows(&[&a, &narrow]).is_ok());
        let other = QMatrix::from_raw_bytes(
            &g,
            QFmt::Q8_0,
            layout,
            [
                Dim::Const(ROWS),
                Dim::Const(u64::from(QFmt::Q8_0.block_elements())),
            ],
            &rows(QFmt::Q8_0, layout).0,
        )
        .unwrap();
        assert!(QMatrix::concat_rows(&[&a, &other]).is_err());
        assert!(QMatrix::concat_rows(&[]).is_err());
    }

    #[test]
    fn two_quantized_operands_are_refused() {
        let g = graph();
        let fmt = QFmt::Q8_0;
        let k = u64::from(fmt.block_elements());
        let (bytes, _) = rows(fmt, QLayout::Native);
        let shape = [Dim::Const(ROWS), Dim::Const(k)];
        let a = QMatrix::from_raw_bytes(&g, fmt, QLayout::Native, shape, &bytes).unwrap();
        let b = QMatrix::from_raw_bytes(&g, fmt, QLayout::Native, shape, &bytes).unwrap();
        assert!(a.tensor.matmul_t(&b.tensor).is_err());
    }
}
