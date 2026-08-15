//! `QMatrix`: a quantized weight matrix as a leaf.
//!
//! Both storage layouts are legal inputs everywhere; moving between them is
//! the priced `qrepack` rewrite, so layout never feeds back into routing
//! through format variants.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;
use fusor2_ir::dtype::{Dtype, QFmt, QLayout};
use fusor2_ir::ir::Op;
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
    /// The `QMatrix` a quantized *value* denotes, or `None` when the tensor
    /// is not one.
    ///
    /// Recovers `(fmt, layout, shape)` from the `LeafKind::Quantized` node
    /// itself, so any quantized tensor — `Graph::quantized`, a GGUF load, a
    /// concat — gets the same [`Self::dequantize`] class without its caller
    /// having carried a `QMatrix` around. A quantized value that is not a
    /// leaf (nothing mints one today) returns `None` and stays on the raw
    /// path.
    pub fn of_tensor(t: &Tensor) -> Option<Self> {
        if !t.dtype().is_quantized() {
            return None;
        }
        let (fmt, layout, shape) = t
            .graph()
            .with_egraph(|g| {
                Ok(match &g.node(t.id()).op {
                    Op::L0(L0::Leaf(LeafKind::Quantized {
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
    /// `raw.shape` is already row-major: the GGUF parser reverses the file's
    /// fastest-varying-first dimension order at read (see
    /// [`fusor2_gguf::GgufTensor`]), so reversing again here would hand back
    /// a transposed matrix. A rank-1 tensor loads as a single row.
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
    /// unioned into one class here, so there is nothing to recognize later:
    /// see [`crate::composite::quantized::dequant_defn`], which returns `None`
    /// for the `(fmt, layout)` pairs that still need a block program.
    pub fn dequantize(&self) -> Result<Tensor> {
        let graph = self.tensor.graph();
        // The sugar is minted **first**, so it takes the lower id and lands in
        // operand 0 of the `Union`. Every other composite does the reverse,
        // and for the reverse reason: there only the `defn` is
        // differentiable, whereas here it is the *sugar* that carries the
        // intentional "quantized weights are not trainable" refusal. Building
        // the defn first would silently route a gradient into the unpack
        // `Map` and its `U32` leaves.
        let sugar = graph.add_l0(L0::Dequant {
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

    /// The `Restride` + `Map` expansion alone, with no `L0::Dequant` in the
    /// class — the `*_slow` spelling [`crate::composite::core_op`] documents.
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
    /// A rank-1 activation is one matrix row, so it routes through a
    /// `[1, k]` view and reshapes back — the same promotion the reference
    /// makes. A rank-3-or-higher activation is the *same* promotion in the
    /// other direction: a weight is rank 2 and [`Tensor::matmul_t`] shares no
    /// batch rank with it, so the leading axes fold into the row axis and are
    /// restored on the way out. Both `kalosm-llama` and `rwhisper` carried a
    /// byte-identical private helper doing exactly this because the method
    /// stopped at rank 2; the views it builds are the views they built.
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
    /// `Dequant` then `Gather`, which is the reference's spelling: the decode
    /// is a value, the row pick is a value, and which program computes them is
    /// the extractor's decision rather than this method's.
    ///
    /// The fused form is a *member*, not a spelling: `GATHER_QUANTIZED_ROWS`
    /// matches this `Gather`-of-`Dequant` pair and mints a float-typed
    /// [`GatherMode::QuantizedRows`] `KGather` reading the quantized leaf
    /// directly (`infer_l1` gives the mode `F32`, so nothing decodes twice —
    /// the wrong-values trap an earlier gather-of-quantized-leaf spelling
    /// fell into). The extractor picks it on cost, which is what deleted the
    /// 2.1 GB dense-table launch an 8B model's per-token lookup paid.
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
    use crate::session::{Backend, Session};
    use fusor2_gguf::blocks::{block_fields, cpu_dequantize_block};
    use fusor2_gguf::repack;
    use half::f16;

    const ROWS: u64 = 3;

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
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

    /// A `QMatrix` over `rows(fmt, layout)`.
    fn matrix(g: &Graph, fmt: QFmt, layout: QLayout) -> (QMatrix, Vec<f32>) {
        let (bytes, want) = rows(fmt, layout);
        let qm = QMatrix::from_raw_bytes(
            g,
            fmt,
            layout,
            [
                Dim::Const(ROWS),
                Dim::Const(u64::from(fmt.block_elements())),
            ],
            &bytes,
        )
        .unwrap();
        (qm, want)
    }

    /// The `defn` alone, forced: the extractor cannot fall back to the block
    /// program, so this is a statement about the bit arithmetic and not about
    /// which class member happened to win.
    ///
    /// Exact equality, not a tolerance: every one of these decodes is an
    /// integer widened to f32 and multiplied by the block's f32 scale, which
    /// is bit-for-bit what the scalar reference decoder does.
    ///
    /// Both layouts: an f16 scale is decoded by `f16_lane`'s bit arithmetic,
    /// which is exact against `f16::to_f32`, so `Native` is held to the same
    /// bit-for-bit bar. What is left of the old layout restriction is the
    /// **block stride**: a decode reads the stream as `u32` words, so a block
    /// whose stride is not a whole number of words has no expansion, and
    /// `word_aligned` is that predicate. It is asserted in both directions so
    /// a format silently losing its expansion fails here.
    #[test]
    fn the_dequant_defn_decodes_exactly_as_the_reference_block_decoder() {
        let g = graph();
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let (qm, want) = matrix(&g, fmt, layout);
                let slow = qm.dequantize_slow();
                if !fusor2_gguf::blocks::word_aligned(fmt, layout) {
                    assert!(slow.is_err(), "{fmt:?}/{layout:?} is not word-aligned");
                    continue;
                }
                let got = slow.unwrap().to_vec_f32().unwrap();
                assert_eq!(got.len(), want.len(), "{fmt:?}/{layout:?}");
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(a, b, "{fmt:?}/{layout:?} element {i}");
                }
                assert!(want.iter().any(|v| *v != 0.0), "{fmt:?}/{layout:?}");
            }
        }
    }

    /// The class shape every other composite is tested for: the sugar and a
    /// marked `defn`, both in one class.
    #[test]
    fn a_dequant_class_holds_both_the_sugar_and_a_marked_defn() {
        use fusor2_ir::ir::Op;
        let g = graph();
        let shape = |qm: &QMatrix| {
            let y = qm.dequantize().unwrap();
            g.handle()
                .with_egraph(|eg| {
                    let ms = eg.members(eg.class_of(y.id()));
                    let sugars = ms
                        .iter()
                        .filter(|m| matches!(eg.node(**m).op, Op::L0(L0::Dequant { .. })))
                        .count();
                    let defns = ms.iter().filter(|m| eg.is_defn(**m)).count();
                    Ok((ms.len(), sugars, defns))
                })
                .unwrap()
        };
        let (qm, _) = matrix(&g, QFmt::Q8_0, QLayout::F32Scales);
        let (members, sugars, defns) = shape(&qm);
        assert!(members >= 2, "expected sugar + defn, got {members}");
        assert_eq!((sugars, defns), (1, 1));

        // ... and the same shape at `Native`, wherever the block stride tiles
        // the word stream: the f16 scales decode through `f16_lane`, so this
        // class holds a real alternative to the block program rather than the
        // bare sugar it used to.
        let (q4k, _) = matrix(&g, QFmt::Q4K, QLayout::Native);
        let (members, sugars, defns) = shape(&q4k);
        assert!(members >= 2, "expected sugar + defn, got {members}");
        assert_eq!((sugars, defns), (1, 1));

        // Q8_0's native block is 34 bytes, so it does not tile the `u32` word
        // stream a `Restride` reads: the class is the bare sugar and there is
        // nothing to force. (Q4K/Q5K native *are* word-aligned and do get a
        // defn — the layout is not what decides this.)
        let (native, _) = matrix(&g, QFmt::Q8_0, QLayout::Native);
        assert!(native.dequantize_slow().is_err());
        let bare = native.dequantize().unwrap();
        g.handle()
            .with_egraph(|eg| {
                assert!(matches!(eg.node(bare.id()).op, Op::L0(L0::Dequant { .. })));
                Ok(())
            })
            .unwrap();
    }

    /// The gradient still stops at the quantized leaf. The `Union`'s adjoint
    /// routes to operand 0, which is the lower id, which is the sugar — the
    /// one node carrying the refusal. Build the `defn` first and this silently
    /// becomes a gradient into a `Map` over `U32` leaves.
    #[test]
    fn a_dequantize_with_a_defn_still_refuses_a_gradient() {
        let g = graph();
        let (qm, _) = matrix(&g, QFmt::Q8_0, QLayout::F32Scales);
        let y = qm.dequantize().unwrap();
        assert!(g.backward_with(&y, &[qm.tensor.clone()]).is_err());
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

    #[test]
    fn q_mat_mul_offers_the_word_aligned_twin_exactly_once() {
        // Q6K's native block is 210 bytes — not a whole number of words — so
        // `contract_2d` unions a third spelling over the `F32Scales` twin
        // (`qrepack`'s consuming half). The twin is a *separate leaf*, minted
        // at most once per source, holding exactly the repacked bytes; an
        // aligned format (Q4K, 144 B) never mints one — its twin would be the
        // same addressing arithmetic plus two dead bytes per block.
        let g = graph();
        let fmt = QFmt::Q6K;
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
        let act = g
            .leaf("a", &[Dim::Const(2), Dim::Const(k)], Dtype::F32)
            .unwrap();
        let y = qm.q_mat_mul(&act).unwrap();

        let src = qm.tensor.id();
        let twin = g
            .handle()
            .repacked_leaf_of(src)
            .unwrap()
            .expect("a misaligned native block mints a twin");
        assert_eq!(
            g.handle().repacked_leaf_of(src).unwrap(),
            Some(twin),
            "the twin is memoized, not re-minted"
        );
        let mut want = Vec::new();
        repack::repack(fmt, QLayout::Native, QLayout::F32Scales, &bytes, &mut want).unwrap();
        assert_eq!(
            g.handle().leaf_bytes(twin).unwrap(),
            want,
            "twin bytes are exactly the Native -> F32Scales repack"
        );
        // The twin spelling is a member of the matmul's value class: a
        // `Contract` whose weight side reads the twin leaf.
        let unioned = g
            .handle()
            .with_egraph(|eg| {
                let class = eg.class_of(y.id());
                let twin_class = eg.class_of(twin);
                Ok(eg.class_ids(class).into_iter().any(|m| match &eg.node(m).op {
                    fusor2_ir::ir::Op::L0(L0::Contract { b, .. }) => {
                        eg.class_of(*b) == twin_class
                    }
                    _ => false,
                }))
            })
            .unwrap();
        assert!(unioned, "the contraction over the twin joins the class");

        // Aligned native blocks stay single-spelled.
        let fmt = QFmt::Q4K;
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
        assert_eq!(g.handle().repacked_leaf_of(qm.tensor.id()).unwrap(), None);
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
