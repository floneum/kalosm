//! Six formats x two layouts, plus `qrepack` round-tripping.
//!
//! The oracle everywhere in this area is
//! [`fusor2_gguf::blocks::cpu_dequantize_block`]: one raw block in, its
//! elements out.
//!
//! Both layouts are legal inputs for all six formats, so every value case
//! runs twice and the repack between them has to be byte-exact in both
//! directions.

use fusor2::{Dtype, QMatrix, Session};
use fusor2_gguf::blocks::{block_fields, cpu_dequantize_block, word_aligned};
use fusor2_gguf::repack;
use fusor2_ir::dtype::{QFmt, QLayout};
use half::f16;

use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, from_u32, fuzz_case};
use crate::suite::support::{Domain, expect_values, graph_of, read, upload};

/// Rows of the quantized weight in the fixed cases. One block per row keeps
/// the host reference a straight `chunks(block_bytes)` walk.
const ROWS: u64 = 3;

/// `[rows, blocks-per-row]` for the fuzzed decode cases. `k` is always
/// `blocks * block_elements()`, so the K axis stays block-aligned by
/// construction rather than by a sampled extent happening to divide.
const DEQUANT_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 4)];

/// `[batch, rows, blocks-per-row]` for the fuzzed matmul cases.
const QMATMUL_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 6),
    FuzzDim::Range(1, 4),
];

/// `[blocks]` for the repack round trips: the repack walks a flat block
/// stream, so the only extent is how many blocks it crosses.
const REPACK_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8)];

/// `[rows]` for the layout-agreement case, one block per row.
const LAYOUTS_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4)];

/// `[batch, rows, blocks-per-row]` for the backward case; small, because the
/// host reference sums over every row per activation element.
const BACKWARD_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 2),
];

/// `fuzz_case` keys its per-run seed off the case name for the life of the
/// process, so a formatted per-format name is leaked once at registration.
fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// A well-formed block: an explicit finite scale (and min, where the format
/// carries one) plus a deterministic quant payload. A random f16 scale is NaN
/// or Inf about 1 time in 2000, and a NaN compares unequal to itself.
fn make_block(fmt: QFmt, layout: QLayout, seed: u32) -> Vec<u8> {
    let fields = block_fields(fmt, layout);
    let bytes = fmt.block_bytes(layout) as usize;
    let mut block = vec![0u8; bytes];

    // Payload first, so the scale writes below cannot be overwritten.
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for slot in block.iter_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *slot = (state >> 24) as u8;
    }

    let write_scale = |block: &mut [u8], at: u32, value: f32| {
        let at = at as usize;
        if fields.scale_is_f16 {
            block[at..at + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes());
        } else {
            block[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    };
    write_scale(&mut block, fields.scale, 0.015_625);
    if let Some(min) = fields.min {
        write_scale(&mut block, min, 0.003_906_25);
    }
    block
}

/// `rows` blocks of `(fmt, layout)` from `seed`, and the f32 values the
/// scalar reference decodes them to. One block per row keeps the host
/// reference a straight `chunks(block_bytes)` walk.
fn block_rows(fmt: QFmt, layout: QLayout, seed: u32, rows: usize) -> (Vec<u8>, Vec<f32>) {
    let block_bytes = fmt.block_bytes(layout) as usize;
    let elements = fmt.block_elements() as usize;
    let mut bytes = Vec::with_capacity(rows * block_bytes);
    let mut values = vec![0.0f32; rows * elements];
    for r in 0..rows {
        let block = make_block(fmt, layout, seed + r as u32);
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

/// `ROWS` blocks of `(fmt, layout)` and the f32 values they decode to.
fn quantized_rows(fmt: QFmt, layout: QLayout) -> (Vec<u8>, Vec<f32>) {
    block_rows(fmt, layout, 2003, ROWS as usize)
}

/// A `QMatrix` over `bytes`, built from parts rather than from a GGUF file.
fn matrix_from_parts(
    graph: &fusor2::Graph,
    fmt: QFmt,
    layout: QLayout,
    bytes: &[u8],
    rows: u64,
    cols: u64,
) -> Result<QMatrix, CaseError> {
    let cols = fusor2::Dim::Const(cols);
    let rows = fusor2::Dim::Const(rows);
    let tensor = graph
        .quantized(fmt, layout, [rows, cols], bytes)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    Ok(QMatrix {
        tensor,
        fmt,
        layout,
        rows,
        cols,
    })
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let name = leak(format!("dequantize_{}_{}", fmt_name(fmt), layout_name(layout)));
            cases.push_case(fuzz_case(
                "quantized",
                name,
                DEQUANT_SPEC,
                move |s, shape, seed| dequantize_case(s, fmt, layout, shape, seed),
            ));
        }
    }
    // The same decode forced through the `Restride` + `Map` expansion, with
    // no `Logical::Dequant` in the class for the extractor to fall back to;
    // the sugared case above proves only that some class member is right.
    // Only where the block stride is a whole number of `u32` words: a block
    // that straddles a word is not addressable by a `Restride` over the word
    // stream.
    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            if !word_aligned(fmt, layout) {
                continue;
            }
            let name = format!("dequantize_defn_{}_{}", fmt_name(fmt), layout_name(layout));
            cases.push("quantized", name, move |s| {
                dequantize_defn_case(s, fmt, layout)
            });
        }
    }
    for fmt in QFmt::ALL {
        let name = leak(format!("qmatmul_{}", fmt_name(fmt)));
        cases.push_case(fuzz_case(
            "quantized",
            name,
            QMATMUL_SPEC,
            move |s, shape, seed| qmatmul_case(s, fmt, shape, seed),
        ));
    }
    for fmt in QFmt::ALL {
        let name = leak(format!("repack_round_trip_{}", fmt_name(fmt)));
        cases.push_case(fuzz_case(
            "quantized",
            name,
            REPACK_SPEC,
            move |s, shape, seed| repack_case(s, fmt, shape, seed),
        ));
    }

    cases.push_case(fuzz_case(
        "quantized",
        "both_layouts_decode_to_the_same_values",
        LAYOUTS_SPEC,
        layouts_agree,
    ));
    cases.push_case(fuzz_case(
        "quantized",
        "q_mat_mul_backward_reaches_the_activation_only",
        BACKWARD_SPEC,
        qmatmul_backward,
    ));
    for fmt in QFmt::ALL {
        let name = format!("qmatmul_coop_shape_{}", fmt_name(fmt));
        cases.push("quantized", name, move |s| qmatmul_coop_shape(s, fmt));
    }
    // ... and the same geometry with the defn as the *only* class member, so
    // the unpack `Map` really does run inside the staging fill.
    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            if !word_aligned(fmt, layout) {
                continue;
            }
            let name = format!(
                "dequantize_defn_coop_shape_{}_{}",
                fmt_name(fmt),
                layout_name(layout)
            );
            cases.push("quantized", name, move |s| defn_coop_shape(s, fmt, layout));
        }
    }
    cases.push(
        "quantized",
        "qgemv_grid_past_the_dimension_cap",
        qgemv_grid_past_the_dimension_cap,
    );
    cases.push("quantized", "index_select_rows", index_select_rows);
    cases.push("quantized", "concat_rows", concat_rows);
    cases.push(
        "quantized",
        "qmatrix_load_orientation",
        qmatrix_load_orientation,
    );
    cases
}

/// `QMatrix::load` of a `[rows, cols]` GGUF weight must agree with
/// `from_raw_bytes` over the same block stream — shape included.
///
/// The GGUF parser already reverses the file's fastest-varying-first extents
/// into row-major at read, so the loader must not reverse again: a double
/// reverse hands back a transposed `[cols, rows]` matrix whose block stream
/// still decodes to the same flat values, and only the dims betray the swap.
fn qmatrix_load_orientation(session: &Session) -> CaseResult {
    use fusor2::Dim;
    use fusor2_gguf::{Gguf, GgmlType, GgufMetadata, GgufTensor, VarBuilder};

    let fmt = QFmt::Q4K;
    let layout = QLayout::Native;
    let be = fmt.block_elements() as u64;
    // Rectangular on purpose: rows != cols is what makes a transpose visible.
    let (rows, cols) = (ROWS, be * 2);
    let blocks = ((rows * cols) / be) as usize;
    let (bytes, expected) = block_rows(fmt, layout, 4243, blocks);

    // A synthetic single-tensor GGUF. `GgufTensor::shape` is row-major: the
    // writer reverses to the wire order and the reader reverses back.
    let meta = GgufMetadata {
        tensors: vec![GgufTensor {
            name: "w".into(),
            ty: GgmlType::Q4K,
            shape: [rows, cols].into_iter().collect(),
            offset: 0,
            bytes: bytes.len() as u64,
        }],
        ..Default::default()
    };
    let mut file = std::io::Cursor::new(Vec::new());
    meta.write(&mut file, [("w", bytes.as_slice())])
        .map_err(|e| -> CaseError { format!("gguf write: {e}").into() })?;
    let gguf = Gguf::from_bytes(file.into_inner())
        .map_err(|e| -> CaseError { format!("gguf parse: {e}").into() })?;
    let vb = VarBuilder::new(std::sync::Arc::new(gguf));

    let graph = graph_of(session);
    let loaded = QMatrix::load(&vb, &graph, "w")
        .map_err(|e| -> CaseError { format!("QMatrix::load: {e}").into() })?;
    if (loaded.rows, loaded.cols) != (Dim::Const(rows), Dim::Const(cols)) {
        return Err(format!(
            "QMatrix::load read a [{rows}, {cols}] weight as [{:?}, {:?}]",
            loaded.rows, loaded.cols
        )
        .into());
    }

    let reference = QMatrix::from_raw_bytes(
        &graph,
        fmt,
        layout,
        [Dim::Const(rows), Dim::Const(cols)],
        &bytes,
    )
    .map_err(|e| -> CaseError { format!("from_raw_bytes: {e}").into() })?;

    let shape = [rows, cols];
    let via_load = read(
        &loaded
            .dequantize()
            .map_err(|e| -> CaseError { format!("dequantize(load): {e}").into() })?,
    )?;
    let via_parts = read(
        &reference
            .dequantize()
            .map_err(|e| -> CaseError { format!("dequantize(parts): {e}").into() })?,
    )?;
    // Same class either way once the shapes agree, so also pin both against
    // the scalar host decode of the same blocks.
    expect_values(session, &shape, Dtype::F32, &via_load, &via_parts)?;
    expect_values(session, &shape, Dtype::F32, &via_load, &expected)?;
    Ok(())
}


/// A quantized matmul at a shape the cooperative-matrix path can take; the
/// small-shape cases favor other contraction lowerings and do not exercise
/// the cooperative path.
///
/// The oracle is the dequantized dense matmul of the same weight. A mismatch
/// means the staging-fill decode addresses the block differently from the
/// reference decode.
fn qmatmul_coop_shape(session: &Session, fmt: QFmt) -> CaseResult {
    const M: u64 = 64;
    const N: u64 = 64;
    let layout = QLayout::Native;
    let be = fmt.block_elements() as u64;
    // Eight blocks deep, so `k` clears the smallest coop `bk` several times
    // over and still divides the block extent exactly.
    let k = be * 8;

    let graph = graph_of(session);
    // Valid blocks, not random bytes: arbitrary bytes decode f16 scales to
    // NaN, which compares unequal to itself.
    let blocks = ((k / be) * N) as usize;
    let (bytes, _) = block_rows(fmt, layout, 3301, blocks);
    let w = graph
        .quantized(fmt, layout, [fusor2::Dim::Const(N), fusor2::Dim::Const(k)], &bytes)
        .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;

    let act = Domain::Wide.sample(3301, (M * k) as usize);
    let a = upload(graph.handle(), &dims(&[M, k]), &act)?;

    let got = read(&a.matmul_t(&w).map_err(|e| -> CaseError { e.to_string().into() })?)?;

    // The oracle: the same contraction against the dequantized weight.
    let qm = QMatrix {
        tensor: w,
        fmt,
        layout,
        rows: fusor2::Dim::Const(N),
        cols: fusor2::Dim::Const(k),
    };
    let dense = qm
        .dequantize()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let want = read(&a.matmul_t(&dense).map_err(|e| -> CaseError { e.to_string().into() })?)?;

    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[M as usize, N as usize],
        &want,
        &got,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

/// The `Map`-spelled decode driven through a cooperative contraction:
/// `dequantize_slow` leaves the `Restride` + `Map` as the only member of the
/// class, so the unpack bit-arithmetic has to survive being substituted into
/// the staging fill at a geometry several `bk` deep.
///
/// The oracle is the host decode uploaded dense, so a mismatch means the
/// staging fill addresses the block stream differently from
/// `cpu_dequantize_block`.
fn defn_coop_shape(session: &Session, fmt: QFmt, layout: QLayout) -> CaseResult {
    const M: u64 = 64;
    const N: u64 = 64;
    let be = fmt.block_elements() as u64;
    let k = be * 8;

    let graph = graph_of(session);
    let blocks = ((k / be) * N) as usize;
    // `block_rows` emits blocks back to back, which is exactly a `[N, k]`
    // row-major quantized matrix of `k / be` blocks per row.
    let (bytes, decoded) = block_rows(fmt, layout, 3301, blocks);
    let w = graph
        .quantized(
            fmt,
            layout,
            [fusor2::Dim::Const(N), fusor2::Dim::Const(k)],
            &bytes,
        )
        .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;
    let qm = QMatrix {
        tensor: w,
        fmt,
        layout,
        rows: fusor2::Dim::Const(N),
        cols: fusor2::Dim::Const(k),
    };
    let dense = qm
        .dequantize_slow()
        .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;

    let act = Domain::Wide.sample(3301, (M * k) as usize);
    let a = upload(graph.handle(), &dims(&[M, k]), &act)?;
    let got = read(
        &a.matmul_t(&dense)
            .map_err(|e| -> CaseError { e.to_string().into() })?,
    )?;

    let oracle = upload(graph.handle(), &dims(&[N, k]), &decoded)?;
    let want = read(
        &a.matmul_t(&oracle)
            .map_err(|e| -> CaseError { e.to_string().into() })?,
    )?;

    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[M as usize, N as usize],
        &want,
        &got,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

fn fmt_name(fmt: QFmt) -> &'static str {
    match fmt {
        QFmt::Q4_0 => "q4_0",
        QFmt::Q5_0 => "q5_0",
        QFmt::Q8_0 => "q8_0",
        QFmt::Q4K => "q4k",
        QFmt::Q5K => "q5k",
        QFmt::Q6K => "q6k",
    }
}

fn layout_name(layout: QLayout) -> &'static str {
    match layout {
        QLayout::Native => "native",
        QLayout::F32Scales => "f32_scales",
    }
}

/// The device dequantize against the scalar reference decoder. `shape` is
/// `[rows, blocks-per-row]` from [`DEQUANT_SPEC`]; the block stream is
/// row-major in blocks, so `rows * blocks` consecutive blocks decode to a
/// `[rows, blocks * block_elements]` matrix.
fn dequantize_case(
    session: &Session,
    fmt: QFmt,
    layout: QLayout,
    shape: &[u64],
    seed: u32,
) -> CaseResult {
    let (rows, blocks) = (shape[0], shape[1]);
    let k = blocks * fmt.block_elements() as u64;
    let (bytes, expected) = block_rows(fmt, layout, seed, (rows * blocks) as usize);
    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes, rows, k)?;
    let dense = qm
        .dequantize()
        .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;

    let shape = [rows, k];
    let got = read(&dense)?;
    if got.len() != expected.len() {
        return Err(format!(
            "{fmt:?}/{layout:?} decoded {} elements, the reference decodes {}",
            got.len(),
            expected.len()
        )
        .into());
    }
    expect_values(session, &shape, Dtype::F32, &got, &expected)?;
    Ok(())
}

/// The decode spelled as a `Restride` over the block stream read as `u32`
/// words plus a `Map` of unpack bit-arithmetic — `Logical::Dequant`'s
/// definitional expansion, with the sugar absent. `dequantize` unions the
/// two and extraction is free to pick either; `dequantize_slow` builds the
/// expansion alone.
fn dequantize_defn_case(session: &Session, fmt: QFmt, layout: QLayout) -> CaseResult {
    let (bytes, expected) = quantized_rows(fmt, layout);
    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes, ROWS, fmt.block_elements() as u64)?;
    let dense = qm
        .dequantize_slow()
        .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;

    let shape = [ROWS, fmt.block_elements() as u64];
    let got = read(&dense)?;
    if got.len() != expected.len() {
        return Err(format!(
            "{fmt:?}/{layout:?} decoded {} elements, the reference decodes {}",
            got.len(),
            expected.len()
        )
        .into());
    }
    expect_values(session, &shape, Dtype::F32, &got, &expected)?;
    Ok(())
}

/// An activation against a quantized weight. The result must equal the dense
/// matmul against the reference-decoded weight, whichever program extraction
/// picked.
fn qmatmul_case(session: &Session, fmt: QFmt, shape: &[u64], seed: u32) -> CaseResult {
    let (batch, rows, blocks) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let layout = QLayout::Native;
    let k = blocks * fmt.block_elements() as usize;
    let (bytes, weights) = block_rows(fmt, layout, seed, rows * blocks);
    let act = Domain::Wide.sample(seed ^ 0x9e37_79b9, batch * k);

    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes, rows as u64, k as u64)?;
    let a = upload(graph.handle(), &dims(&[batch as u64, k as u64]), &act)?;
    // The weight is `[rows, k]`, so the contraction is against its transpose.
    let y = a
        .matmul_t(&qm.tensor)
        .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;

    let mut expected = vec![0.0f32; batch * rows];
    for b in 0..batch {
        for r in 0..rows {
            expected[b * rows + r] = (0..k).map(|t| act[b * k + t] * weights[r * k + t]).sum();
        }
    }
    // A quantized dot accumulates in f32 over up to 256 terms, so the bar is
    // relative to the result rather than absolute.
    let got = read(&y)?;
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[batch, rows],
        &expected,
        &got,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

/// A quantized matvec whose output needs more workgroups than one dispatch
/// dimension holds (65,535), so `distribute_workgroups` folds the grid onto a
/// second slab. An sgemv addressing that fold with a raw `ProgramId(X)` would
/// recompute slab 0's rows and never write past the fold. The harness sets
/// `FUSOR2_VERIFY_MEMBERS`, so this geometry races every contraction family.
fn qgemv_grid_past_the_dimension_cap(session: &Session) -> CaseResult {
    // Two slabs, with a remainder so the fold is not exact: 65,792 groups
    // dispatch as [32896, 2, 1].
    const MANY_ROWS: usize = 65_792;
    let fmt = QFmt::Q4K;
    let layout = QLayout::Native;
    let k = fmt.block_elements() as usize; // one block per row
    let (bytes, weights) = block_rows(fmt, layout, 4407, MANY_ROWS);

    let graph = graph_of(session);
    let rows = fusor2::Dim::Const(MANY_ROWS as u64);
    let cols = fusor2::Dim::Const(k as u64);
    let w = graph
        .quantized(fmt, layout, [rows, cols], &bytes)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let act = Domain::Wide.sample(4407, k);
    let a = upload(graph.handle(), &dims(&[1, k as u64]), &act)?;
    let y = a
        .matmul_t(&w)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&y)?;

    let mut expected = vec![0.0f32; MANY_ROWS];
    for (r, slot) in expected.iter_mut().enumerate() {
        *slot = (0..k).map(|t| act[t] * weights[r * k + t]).sum();
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[1, MANY_ROWS],
        &expected,
        &got,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

/// Gradients flow to the activation only; the quantized weight is
/// non-trainable through this route (QAT keeps a separate f32 master).
fn qmatmul_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (batch, rows, blocks) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let fmt = QFmt::Q8_0;
    let layout = QLayout::Native;
    let k = blocks * fmt.block_elements() as usize;
    let (bytes, weights) = block_rows(fmt, layout, seed, rows * blocks);
    let act = Domain::Wide.sample(seed ^ 0x9e37_79b9, batch * k);

    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes, rows as u64, k as u64)?;
    let a = upload(graph.handle(), &dims(&[batch as u64, k as u64]), &act)?;
    let y = a
        .matmul_t(&qm.tensor)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // d_act[b, t] = sum over rows of w[r, t], under an all-ones seed.
    let d_a = crate::suite::support::gradient_of(&graph, &y, &a)?;
    let want: Vec<f32> = (0..batch * k)
        .map(|n| (0..rows).map(|r| weights[r * k + n % k]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[batch, k],
        &want,
        &d_a,
        1e-3,
        1e-3,
    )?;

    // And nothing must claim to differentiate the quantized weight itself.
    if crate::suite::support::gradient_of(&graph, &y, &qm.tensor).is_ok() {
        return Err(
            "a gradient was produced for a quantized weight; that route is not \
                    trainable and QAT keeps a separate f32 master"
                .into(),
        );
    }
    Ok(())
}

/// `Native -> F32Scales -> Native` must be byte-identical, and the forward
/// direction must decode to the same values.
fn repack_case(_session: &Session, fmt: QFmt, shape: &[u64], seed: u32) -> CaseResult {
    let blocks = shape[0] as usize;
    let (native, values) = block_rows(fmt, QLayout::Native, seed, blocks);

    let mut widened = Vec::new();
    repack(
        fmt,
        QLayout::Native,
        QLayout::F32Scales,
        &native,
        &mut widened,
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;
    let want_len = blocks * fmt.block_bytes(QLayout::F32Scales) as usize;
    if widened.len() != want_len {
        return Err(format!(
            "{fmt:?} repacked to {} bytes, want {want_len}",
            widened.len()
        )
        .into());
    }

    // The widened blocks decode to the same values.
    let elements = fmt.block_elements() as usize;
    let stride = fmt.block_bytes(QLayout::F32Scales) as usize;
    let mut decoded = vec![0.0f32; blocks * elements];
    for r in 0..blocks {
        cpu_dequantize_block(
            fmt,
            QLayout::F32Scales,
            &widened[r * stride..(r + 1) * stride],
            &mut decoded[r * elements..(r + 1) * elements],
        );
    }
    crate::compare::exact_eq("host", &[decoded.len()], &values, &decoded).map_err(
        |e| -> CaseError {
            format!("{fmt:?}: widening the scales changed a decoded value: {e}").into()
        },
    )?;

    // And back. Lossless, because the f32 came from an f16 in the first place.
    let mut narrowed = Vec::new();
    repack(
        fmt,
        QLayout::F32Scales,
        QLayout::Native,
        &widened,
        &mut narrowed,
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;
    crate::compare::assert_bytes_eq(&native, &narrowed)
        .map_err(|e| -> CaseError { format!("{fmt:?} repack round trip: {e}").into() })?;
    Ok(())
}

/// Both layouts are legal inputs everywhere, so a matrix uploaded in either
/// one must produce the same numbers on the same device.
fn layouts_agree(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let rows = shape[0];
    for fmt in QFmt::ALL {
        let be = fmt.block_elements() as u64;
        let (native, _) = block_rows(fmt, QLayout::Native, seed, rows as usize);
        let mut widened = Vec::new();
        repack(
            fmt,
            QLayout::Native,
            QLayout::F32Scales,
            &native,
            &mut widened,
        )
        .map_err(|e| -> CaseError { e.to_string().into() })?;

        let graph = graph_of(session);
        let a = matrix_from_parts(&graph, fmt, QLayout::Native, &native, rows, be)?
            .dequantize()
            .map_err(|e| -> CaseError { format!("{fmt:?} native: {e}").into() })?;
        let b = matrix_from_parts(&graph, fmt, QLayout::F32Scales, &widened, rows, be)?
            .dequantize()
            .map_err(|e| -> CaseError { format!("{fmt:?} f32 scales: {e}").into() })?;
        let (va, vb) = (read(&a)?, read(&b)?);
        crate::compare::exact_eq(backend_of(session), &[va.len()], &va, &vb).map_err(
            |e| -> CaseError {
                format!(
                    "{fmt:?}: the two layouts disagree ({e}). Layout is a priced operand \
                     attribute, not a format variant."
                )
                .into()
            },
        )?;
    }
    Ok(())
}

/// The shared table's declared block size is the format's. A row that
/// disagrees would address every block at the wrong stride.
/// A quantized embedding lookup. Row `i` of the result must be exactly what
/// the scalar reference decodes source row `idx[i]` to — for every format and
/// both layouts, with the picks out of order and one row repeated.
///
/// The lookup is `Dequant` then `Gather`; with `FUSOR2_VERIFY_MEMBERS` this
/// case value-checks the fused `GatherMode::QuantizedRows` member against the
/// decode-then-pick members.
fn index_select_rows(session: &Session) -> CaseResult {
    // Out of order, one repeat, and neither end of the table first.
    const PICKS: &[u32] = &[2, 0, 2, 1];

    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let cols = fmt.block_elements() as usize;
            let (bytes, decoded) = quantized_rows(fmt, layout);
            let graph = graph_of(session);
            let qm = matrix_from_parts(&graph, fmt, layout, &bytes, ROWS, cols as u64)?;
            let idx = from_u32(graph.handle(), &dims(&[PICKS.len() as u64]), PICKS)
                .map_err(|e| -> CaseError { e.to_string().into() })?;

            let picked = qm
                .index_select_rows(&idx)
                .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;
            let want_shape = dims(&[PICKS.len() as u64, cols as u64]);
            if picked.shape().as_slice() != want_shape.as_slice() {
                return Err(format!(
                    "{fmt:?}/{layout:?}: index_select_rows returned {:?}, want {want_shape:?}",
                    picked.shape()
                )
                .into());
            }

            let want: Vec<f32> = PICKS
                .iter()
                .flat_map(|p| {
                    let at = *p as usize * cols;
                    decoded[at..at + cols].iter().copied()
                })
                .collect();
            let got = read(&picked)?;
            expect_values(
                session,
                &[PICKS.len() as u64, cols as u64],
                Dtype::F32,
                &got,
                &want,
            )?;

            // The picks are not the identity, so a lookup that ignored `idx`
            // would have to be caught above. Say so, rather than trusting it.
            if want[..cols] == decoded[..cols] {
                return Err(format!(
                    "{fmt:?}/{layout:?}: the picks start with source row 0, which makes \
                     the comparison blind to a lookup that ignores its indices"
                )
                .into());
            }

            // `index_select_rows_to` is the same lookup at a narrower output
            // dtype, compared at that dtype's tolerance.
            let half = qm
                .index_select_rows_to(&idx, Dtype::F16)
                .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?} -> f16: {e}").into() })?;
            if half.dtype() != Dtype::F16 {
                return Err(format!(
                    "{fmt:?}/{layout:?}: index_select_rows_to(.., F16) returned {:?}",
                    half.dtype()
                )
                .into());
            }
            let got_half = read(&half)?;
            expect_values(
                session,
                &[PICKS.len() as u64, cols as u64],
                Dtype::F16,
                &got_half,
                &want,
            )?;
        }
    }

    // A rank-2 or float index is refused rather than reinterpreted.
    let graph = graph_of(session);
    let fmt = QFmt::Q8_0;
    let (bytes, _) = quantized_rows(fmt, QLayout::Native);
    let qm = matrix_from_parts(
        &graph,
        fmt,
        QLayout::Native,
        &bytes,
        ROWS,
        fmt.block_elements() as u64,
    )?;
    let square = from_u32(graph.handle(), &dims(&[2, 2]), &[0, 1, 2, 0])
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    if qm.index_select_rows(&square).is_ok() {
        return Err("index_select_rows accepted a rank-2 index run".into());
    }
    let floats = upload(graph.handle(), &dims(&[2]), &[0.0, 1.0])?;
    if qm.index_select_rows(&floats).is_ok() {
        return Err("index_select_rows accepted an F32 index run".into());
    }
    Ok(())
}

/// Fused QKV projections concatenate three quantized weights row-wise without
/// decoding them.
///
/// Two claims: the byte-appended stream must decode to the concatenation of
/// the parts, and `q_mat_mul` against the concatenated weight must equal the
/// three separate products stacked — both against the host reference.
fn concat_rows(session: &Session) -> CaseResult {
    // Unequal row counts: equal parts would hide an offset bug in the byte
    // append.
    const PARTS: [(u32, usize); 3] = [(3_100, 3), (3_200, 2), (3_300, 4)];
    let total: u64 = PARTS.iter().map(|(_, r)| *r as u64).sum();

    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let cols = fmt.block_elements() as u64;
            let graph = graph_of(session);
            let (mats, want) = qkv_parts(&graph, fmt, layout, &PARTS)?;
            let refs: Vec<&QMatrix> = mats.iter().collect();
            let cat = QMatrix::concat_rows(&refs)
                .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;

            if cat.rows != fusor2::Dim::Const(total) || cat.cols != fusor2::Dim::Const(cols) {
                return Err(format!(
                    "{fmt:?}/{layout:?}: concat_rows is [{:?}, {:?}], want [{total}, {cols}]",
                    cat.rows, cat.cols
                )
                .into());
            }
            let dense = cat
                .dequantize()
                .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;
            let got = read(&dense)?;
            expect_values(session, &[total, cols], Dtype::F32, &got, &want)?;
        }
    }

    // The fused projection. `Q8_0` carries one scale per block, `Q4K` a
    // super-block with group scales, so the two exercise different block
    // programs behind the same concatenation.
    const BATCH: usize = 2;
    for fmt in [QFmt::Q8_0, QFmt::Q4K] {
        let layout = QLayout::Native;
        let k = fmt.block_elements() as usize;
        let act = Domain::Wide.sample(2029, BATCH * k);

        let graph = graph_of(session);
        let (mats, weights) = qkv_parts(&graph, fmt, layout, &PARTS)?;
        let a = upload(graph.handle(), &dims(&[BATCH as u64, k as u64]), &act)?;
        let refs: Vec<&QMatrix> = mats.iter().collect();
        let cat = QMatrix::concat_rows(&refs)
            .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;

        let fused = read(
            &cat.q_mat_mul(&a)
                .map_err(|e| -> CaseError { format!("{fmt:?} fused: {e}").into() })?,
        )?;

        // The same products, one projection per part, written into the
        // columns the concatenation puts them in.
        let mut separate = vec![0.0f32; BATCH * total as usize];
        let mut at = 0usize;
        for (m, (_, rows)) in mats.iter().zip(PARTS) {
            let y = read(
                &m.q_mat_mul(&a)
                    .map_err(|e| -> CaseError { format!("{fmt:?} part: {e}").into() })?,
            )?;
            for b in 0..BATCH {
                for r in 0..rows {
                    separate[b * total as usize + at + r] = y[b * rows + r];
                }
            }
            at += rows;
        }

        // Both against the reference-decoded weight, at `qmatmul_case`'s bar.
        let mut expected = vec![0.0f32; BATCH * total as usize];
        for b in 0..BATCH {
            for r in 0..total as usize {
                expected[b * total as usize + r] =
                    (0..k).map(|t| act[b * k + t] * weights[r * k + t]).sum();
            }
        }
        let shape = [BATCH, total as usize];
        crate::compare::approx_or_relative_eq(
            backend_of(session),
            &shape,
            &expected,
            &fused,
            1e-3,
            1e-3,
        )
        .map_err(|e| -> CaseError {
            format!("{fmt:?}: the fused projection disagrees with the reference: {e}").into()
        })?;
        crate::compare::approx_or_relative_eq(
            backend_of(session),
            &shape,
            &separate,
            &fused,
            1e-3,
            1e-3,
        )
        .map_err(|e| -> CaseError {
            format!(
                "{fmt:?}: one projection against the concatenated weight disagrees with \
                 three against the parts: {e}"
            )
            .into()
        })?;
    }
    Ok(())
}

/// The `(seed, rows)` parts as `QMatrix`es, and the values they decode to,
/// concatenated in the same order.
fn qkv_parts(
    graph: &fusor2::Graph,
    fmt: QFmt,
    layout: QLayout,
    parts: &[(u32, usize)],
) -> Result<(Vec<QMatrix>, Vec<f32>), CaseError> {
    let cols = fusor2::Dim::Const(fmt.block_elements() as u64);
    let mut mats = Vec::with_capacity(parts.len());
    let mut values = Vec::new();
    for (seed, rows) in parts {
        let (bytes, decoded) = block_rows(fmt, layout, *seed, *rows);
        let m = QMatrix::from_raw_bytes(
            graph,
            fmt,
            layout,
            [fusor2::Dim::Const(*rows as u64), cols],
            &bytes,
        )
        .map_err(|e| -> CaseError { format!("{fmt:?}/{layout:?}: {e}").into() })?;
        mats.push(m);
        values.extend_from_slice(&decoded);
    }
    Ok((mats, values))
}
