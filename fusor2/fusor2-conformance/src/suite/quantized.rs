//! Six formats x two layouts x two activation packings, plus `qrepack`
//! round-tripping.
//!
//! The oracle everywhere in this area is
//! [`fusor2_gguf::blocks::cpu_dequantize_block`]: one raw block in, its
//! elements out. Comparing a device dequantize against that scalar decoder is
//! what makes "the GPU decode program and the CPU decode program agree" a
//! statement about the format rather than about two implementations agreeing
//! with each other.
//!
//! `QLayout` is deliberately *not* a property of a format: both layouts are
//! legal inputs for all six, so every value case runs twice and the repack
//! between them has to be byte-exact in both directions.
//!
//! Owned by W14.

use fusor2::{Dtype, QMatrix, Session};
use fusor2_gguf::blocks::{block_fields, cpu_dequantize_block};
use fusor2_gguf::repack;
use fusor2_ir::dtype::{QAct, QFmt, QLayout};
use half::f16;

use crate::harness::{CaseError, CaseResult, Cases, dims, from_u32};
use crate::suite::support::{Domain, expect_values, graph_of, read, upload};

/// Rows of the quantized weight. One block per row keeps the host reference a
/// straight `chunks(block_bytes)` walk.
const ROWS: u64 = 3;

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// A well-formed block: an explicit finite scale (and min, where the format
/// carries one) plus a deterministic quant payload.
///
/// Random bytes would work for a decode comparison — the host oracle reads the
/// same bytes — but a random f16 scale is NaN or Inf about 1 time in 2000, and
/// a NaN compares unequal to itself. So the scale fields are written and the
/// rest is filled.
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
) -> Result<QMatrix, CaseError> {
    let cols = fusor2::Dim::Const(fmt.block_elements() as u64);
    let rows = fusor2::Dim::Const(ROWS);
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
            let name = format!("dequantize_{}_{}", fmt_name(fmt), layout_name(layout));
            cases.push("quantized", name, move |s| dequantize_case(s, fmt, layout));
        }
    }
    for fmt in QFmt::ALL {
        let name = format!("qmatmul_{}", fmt_name(fmt));
        cases.push("quantized", name, move |s| qmatmul_case(s, fmt));
    }
    for fmt in QFmt::ALL {
        let name = format!("repack_round_trip_{}", fmt_name(fmt));
        cases.push("quantized", name, move |s| repack_case(s, fmt));
    }

    cases.push(
        "quantized",
        "both_layouts_decode_to_the_same_values",
        layouts_agree,
    );
    cases.push(
        "quantized",
        "every_format_declares_both_activation_paths",
        activations,
    );
    cases.push("quantized", "block_fields_tile_the_block", fields_tile);
    cases.push(
        "quantized",
        "q_mat_mul_backward_reaches_the_activation_only",
        qmatmul_backward,
    );
    cases.push("quantized", "index_select_rows", index_select_rows);
    cases.push("quantized", "concat_rows", concat_rows);
    cases
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

/// The device dequantize against the scalar reference decoder.
fn dequantize_case(session: &Session, fmt: QFmt, layout: QLayout) -> CaseResult {
    let (bytes, expected) = quantized_rows(fmt, layout);
    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes)?;
    let dense = qm
        .dequantize()
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
/// matmul against the *reference-decoded* weight — whether extraction picked
/// `QAct::F32` or `QAct::Q8Dp4a` is invisible here, which is the point.
fn qmatmul_case(session: &Session, fmt: QFmt) -> CaseResult {
    const BATCH: usize = 2;
    let layout = QLayout::Native;
    let k = fmt.block_elements() as usize;
    let (bytes, weights) = quantized_rows(fmt, layout);
    let act = Domain::Wide.sample(2011, BATCH * k);

    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes)?;
    let a = upload(graph.handle(), &dims(&[BATCH as u64, k as u64]), &act)?;
    // The weight is `[rows, k]`, so the contraction is against its transpose.
    let y = a
        .matmul_t(&qm.tensor)
        .map_err(|e| -> CaseError { format!("{fmt:?}: {e}").into() })?;

    let mut expected = vec![0.0f32; BATCH * ROWS as usize];
    for b in 0..BATCH {
        for r in 0..ROWS as usize {
            expected[b * ROWS as usize + r] =
                (0..k).map(|t| act[b * k + t] * weights[r * k + t]).sum();
        }
    }
    // A quantized dot accumulates in f32 over up to 256 terms, so the bar is
    // relative to the result rather than absolute.
    let got = read(&y)?;
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[BATCH, ROWS as usize],
        &expected,
        &got,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

/// Gradients flow to the activation only. The weight is quantized and
/// non-trainable through this route — that is exactly why QAT keeps a separate
/// f32 master rather than a quantized backward kernel.
fn qmatmul_backward(session: &Session) -> CaseResult {
    const BATCH: usize = 2;
    let fmt = QFmt::Q8_0;
    let layout = QLayout::Native;
    let k = fmt.block_elements() as usize;
    let (bytes, weights) = quantized_rows(fmt, layout);
    let act = Domain::Wide.sample(2017, BATCH * k);

    let graph = graph_of(session);
    let qm = matrix_from_parts(&graph, fmt, layout, &bytes)?;
    let a = upload(graph.handle(), &dims(&[BATCH as u64, k as u64]), &act)?;
    let y = a
        .matmul_t(&qm.tensor)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // d_act[b, t] = sum over rows of w[r, t], under an all-ones seed.
    let d_a = crate::suite::support::gradient_of(&graph, &y, &a)?;
    let want: Vec<f32> = (0..BATCH * k)
        .map(|n| (0..ROWS as usize).map(|r| weights[r * k + n % k]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[BATCH, k],
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
fn repack_case(_session: &Session, fmt: QFmt) -> CaseResult {
    let (native, values) = quantized_rows(fmt, QLayout::Native);

    let mut widened = Vec::new();
    repack(
        fmt,
        QLayout::Native,
        QLayout::F32Scales,
        &native,
        &mut widened,
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;
    let want_len = ROWS as usize * fmt.block_bytes(QLayout::F32Scales) as usize;
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
    let mut decoded = vec![0.0f32; ROWS as usize * elements];
    for r in 0..ROWS as usize {
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
fn layouts_agree(session: &Session) -> CaseResult {
    for fmt in QFmt::ALL {
        let (native, _) = quantized_rows(fmt, QLayout::Native);
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
        let a = matrix_from_parts(&graph, fmt, QLayout::Native, &native)?
            .dequantize()
            .map_err(|e| -> CaseError { format!("{fmt:?} native: {e}").into() })?;
        let b = matrix_from_parts(&graph, fmt, QLayout::F32Scales, &widened)?
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

/// Both activation packings are legal for every format. `Q8Dp4a` is not
/// expressible as dequantize-then-dot, so a table that offers only `F32` has
/// silently deleted a candidate the cost model is supposed to weigh.
fn activations(_session: &Session) -> CaseResult {
    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let spec = fusor2_gguf::block_spec(fmt, layout);
            for wanted in [QAct::F32, QAct::Q8Dp4a] {
                if !spec.activation.contains(&wanted) {
                    return Err(format!(
                        "{fmt:?}/{layout:?} does not offer {wanted:?}; both packings are \
                         legal everywhere and the cost model decides"
                    )
                    .into());
                }
            }
            if spec.bytes as u32 != fmt.block_bytes(layout) {
                return Err(format!(
                    "{fmt:?}/{layout:?} declares {} bytes but the format says {}",
                    spec.bytes,
                    fmt.block_bytes(layout)
                )
                .into());
            }
        }
    }
    Ok(())
}

/// The fields tile the block exactly: no gaps, no overlap, and the last one
/// ends at `block_bytes`.
fn fields_tile(_session: &Session) -> CaseResult {
    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let f = block_fields(fmt, layout);
            let total = fmt.block_bytes(layout);
            if f.scale_is_f16 != matches!(layout, QLayout::Native) {
                return Err(format!(
                    "{fmt:?}/{layout:?}: scale_is_f16 must be exactly `layout == Native`"
                )
                .into());
            }
            if f.ql >= total {
                return Err(format!(
                    "{fmt:?}/{layout:?}: the low-bit plane starts at {} in a {total}-byte block",
                    f.ql
                )
                .into());
            }
            for (label, offset) in [("min", f.min), ("qh", f.qh)] {
                if let Some(o) = offset
                    && o >= total
                {
                    return Err(
                        format!("{fmt:?}/{layout:?}: {label} starts at {o} of {total}").into(),
                    );
                }
            }
            if let Some((gs, len)) = f.group_scales
                && gs + len > total
            {
                return Err(format!(
                    "{fmt:?}/{layout:?}: the group scales run to {} of {total}",
                    gs + len
                )
                .into());
            }
        }
    }
    Ok(())
}

/// A quantized embedding lookup. Row `i` of the result must be exactly what
/// the scalar reference decodes source row `idx[i]` to — for every format and
/// both layouts, with the picks out of order and one row repeated, so a
/// lookup that quietly returned rows `0..n` would not survive.
///
/// The lookup is `Dequant` then `Gather`, so nothing here claims the dense
/// table is skipped: `GatherMode::QuantizedRows` is minted only from a
/// `Gather` whose *operand* is quantized, and that node's class carries the
/// source's `Q(fmt)` dtype while both backends' gather bodies already decode,
/// so the consuming `Dequant` decodes twice and the values are wrong. The
/// fused mode is a tile-rule change away; the values are not waiting on it.
fn index_select_rows(session: &Session) -> CaseResult {
    // Out of order, one repeat, and neither end of the table first.
    const PICKS: &[u32] = &[2, 0, 2, 1];

    for fmt in QFmt::ALL {
        for layout in [QLayout::Native, QLayout::F32Scales] {
            let cols = fmt.block_elements() as usize;
            let (bytes, decoded) = quantized_rows(fmt, layout);
            let graph = graph_of(session);
            let qm = matrix_from_parts(&graph, fmt, layout, &bytes)?;
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
    let qm = matrix_from_parts(&graph, fmt, QLayout::Native, &bytes)?;
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
/// Two claims. The block stream is row-major in blocks, so the concatenation
/// is a byte append and the result must decode to the concatenation of the
/// parts — checked for every format and both layouts. And the whole reason to
/// concatenate is to issue one projection instead of three, so
/// `q_mat_mul` against the concatenated weight must equal the three separate
/// products stacked, and both must equal the host reference built from the
/// reference decoder.
fn concat_rows(session: &Session) -> CaseResult {
    // Unequal row counts, because a QKV fusion with grouped-query attention
    // has them: equal parts would hide an offset bug in the byte append.
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

        // Both against the reference-decoded weight. A quantized dot
        // accumulates in f32 over up to 256 terms, so the bar is the one
        // `qmatmul_case` uses.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn every_format_and_layout_has_a_dequantize_case() {
        let names = registered();
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let wanted = format!(
                    "quantized::dequantize_{}_{}",
                    fmt_name(fmt),
                    layout_name(layout)
                );
                assert!(names.iter().any(|n| *n == wanted), "{wanted} is missing");
            }
        }
    }

    #[test]
    fn every_format_has_a_matmul_and_a_repack_case() {
        let names = registered();
        for fmt in QFmt::ALL {
            for prefix in ["qmatmul", "repack_round_trip"] {
                let wanted = format!("quantized::{prefix}_{}", fmt_name(fmt));
                assert!(names.iter().any(|n| *n == wanted), "{wanted} is missing");
            }
        }
    }

    #[test]
    fn the_six_format_names_are_distinct() {
        let mut names: Vec<&str> = QFmt::ALL.iter().copied().map(fmt_name).collect();
        assert_eq!(names.len(), 6);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn a_generated_block_is_the_declared_length_and_decodes_finitely() {
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let block = make_block(fmt, layout, 11);
                assert_eq!(block.len(), fmt.block_bytes(layout) as usize);
                let mut out = vec![0.0f32; fmt.block_elements() as usize];
                cpu_dequantize_block(fmt, layout, &block, &mut out);
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "{fmt:?}/{layout:?} decoded a non-finite value"
                );
                // The payload is not all zeros, so the case is not comparing
                // two zero vectors.
                assert!(
                    out.iter().any(|v| *v != 0.0),
                    "{fmt:?}/{layout:?} decoded to all zeros"
                );
            }
        }
    }

    #[test]
    fn the_two_layouts_of_a_generated_block_are_a_repack_apart() {
        // Guards `layouts_agree`: the widened bytes really do decode to the
        // same values, independently of any device.
        for fmt in QFmt::ALL {
            let (native, values) = quantized_rows(fmt, QLayout::Native);
            let mut widened = Vec::new();
            repack(
                fmt,
                QLayout::Native,
                QLayout::F32Scales,
                &native,
                &mut widened,
            )
            .unwrap();
            let elements = fmt.block_elements() as usize;
            let stride = fmt.block_bytes(QLayout::F32Scales) as usize;
            let mut decoded = vec![0.0f32; ROWS as usize * elements];
            for r in 0..ROWS as usize {
                cpu_dequantize_block(
                    fmt,
                    QLayout::F32Scales,
                    &widened[r * stride..(r + 1) * stride],
                    &mut decoded[r * elements..(r + 1) * elements],
                );
            }
            assert_eq!(values, decoded, "{fmt:?}");
        }
    }

    #[test]
    fn the_repack_round_trip_is_byte_exact() {
        for fmt in QFmt::ALL {
            let (native, _) = quantized_rows(fmt, QLayout::Native);
            let mut widened = Vec::new();
            repack(
                fmt,
                QLayout::Native,
                QLayout::F32Scales,
                &native,
                &mut widened,
            )
            .unwrap();
            let mut back = Vec::new();
            repack(
                fmt,
                QLayout::F32Scales,
                QLayout::Native,
                &widened,
                &mut back,
            )
            .unwrap();
            assert_eq!(native, back, "{fmt:?}");
        }
    }
}
