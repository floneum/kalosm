//! End-to-end pins for four lowering defects, each of which returned a
//! plausible wrong number rather than an error.
//!
//! Every expectation here is hand-computed, and every one runs the real
//! resolve path on the CPU backend. The conformance suite covers these too;
//! these exist because each defect is a one-line regression away and the
//! suite takes minutes.

// `Device` and `Tensor` are spelled by module path rather than through the
// crate root, because the root pair is exactly what `typed-api` swaps. These
// pins drive the runtime-rank lowering path, which both configurations build;
// naming it directly is what makes this target compile under either feature
// set. `Graph` and `Session` are the same type in both, so the root is fine.
use fusor2::session::Device;
use fusor2::tensor::Tensor;
use fusor2::{Graph, Session};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::shape::{Dim, StrideSpec};

fn session() -> Session {
    Session::new(Device::cpu().expect("cpu device")).expect("session")
}

fn up(g: &Graph, shape: &[u64], data: &[f32]) -> Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
    Tensor::from_slice(g.handle(), Dtype::F32, &dims, &bytes).expect("upload")
}

fn up_u32(g: &Graph, len: u64, data: &[u32]) -> Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::from_slice(g.handle(), Dtype::U32, &[Dim::Const(len)], &bytes).expect("upload")
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4,
            "element {i}: got {a}, want {b}\n  got  {got:?}\n  want {want:?}"
        );
    }
}

/// A gather whose index vector is not as long as the axis it indexes.
///
/// The source's outer coordinate has to step by the *source's* stride. Both
/// backends scaled it by the *output's*, so row 1 of a `[3,2]` table gathered
/// to `[3,4]` read row 2 and row 2 read off the end. That is the rope table
/// expansion, and equally `index_select`, upsample and embedding.
#[test]
fn a_gather_on_an_inner_axis_uses_the_sources_stride() {
    let s = session();
    let g = Graph::new(&s);
    let table = up(&g, &[3, 2], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let idx = up_u32(&g, 4, &[0, 1, 0, 1]);
    let got = table.index_select(1, &idx).unwrap().to_vec_f32().unwrap();
    close(
        &got,
        &[10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0, 30.0, 31.0, 30.0, 31.0],
    );
}

/// Gathering the outermost axis still works, and widening it is not special.
#[test]
fn a_gather_on_the_outer_axis_may_repeat_rows() {
    let s = session();
    let g = Graph::new(&s);
    let table = up(&g, &[3, 2], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let idx = up_u32(&g, 4, &[2, 0, 2, 1]);
    let got = table.index_select(0, &idx).unwrap().to_vec_f32().unwrap();
    close(&got, &[30.0, 31.0, 10.0, 11.0, 30.0, 31.0, 20.0, 21.0]);
}

/// A view's base offset must survive being folded into its reader's index map.
///
/// `MultiFlattenMap` is a sum of stride terms with no constant slot, so
/// `Operand::address_map` takes the offset from the operand's layout.
/// `fold_views_into_index` replaced that layout with a contiguous one, which
/// says offset 0 — a `table[2..]` narrow silently became the whole table.
/// That is `rope`'s sequence offset.
#[test]
fn a_narrowed_view_keeps_its_offset_through_a_consumer() {
    let s = session();
    let g = Graph::new(&s);
    let t = up(
        &g,
        &[5, 2],
        &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
    );
    let rows = t
        .restride(&[
            StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(2)),
            StrideSpec::dim(1, Dim::Const(2)),
        ])
        .unwrap();
    // Broadcast it up a rank and consume it, which is what makes the reader
    // fold the view rather than materialize it.
    let b = rows
        .restride(&[
            StrideSpec::broadcast(Dim::Const(2)),
            StrideSpec::dim(0, Dim::Const(3)),
            StrideSpec::dim(1, Dim::Const(2)),
        ])
        .unwrap();
    let ones = up(&g, &[2, 3, 2], &[1.0; 12]);
    let got = b.mul(&ones).unwrap().to_vec_f32().unwrap();
    close(
        &got,
        &[20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
    );
}

/// An operand broadcast against its reader's index space may not be folded.
///
/// The folded map's extents are what the divisors are derived from, so a
/// `[rows, 1]` view read over a `[rows, cols]` space addressed
/// `flat % rows` where `flat / cols` belonged. `x * rowsum(x)` came back with
/// the wrong row's sum in every column but the first.
#[test]
fn a_broadcast_operand_reads_one_value_per_row() {
    let s = session();
    let (rows, cols) = (3usize, 5usize);
    let xd: Vec<f32> = (0..rows * cols).map(|i| 0.3 + i as f32 * 0.11).collect();
    let g = Graph::new(&s);
    let x = up(&g, &[rows as u64, cols as u64], &xd);
    let rs = x.sum_keepdim(1).unwrap();
    let rb = rs
        .broadcast_as(&[Dim::Const(rows as u64), Dim::Const(cols as u64)])
        .unwrap();
    let got = x.mul(&rb).unwrap().to_vec_f32().unwrap();

    let sums: Vec<f32> = (0..rows)
        .map(|r| xd[r * cols..(r + 1) * cols].iter().sum())
        .collect();
    let want: Vec<f32> = (0..rows * cols).map(|i| xd[i] * sums[i / cols]).collect();
    close(&got, &want);
}

/// `sum(softmax(x))` is exactly the row count, in one resolve.
///
/// This is the same defect seen from the loss side: with the broadcast row
/// sum misread, the softmax rows did not sum to one and every finite
/// difference taken through them was wrong.
#[test]
fn softmax_rows_sum_to_one_without_materializing_anything() {
    let s = session();
    let (rows, cols) = (3usize, 5usize);
    let xd: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
    let g = Graph::new(&s);
    let x = up(&g, &[rows as u64, cols as u64], &xd);
    let total = x.softmax(1).unwrap().sum_all().unwrap().to_vec_f32().unwrap();
    close(&total, &[rows as f32]);
}

/// A contraction reads each operand through the spec's labels.
///
/// `KContract` records only the products m, n, k and batch, so it cannot tell
/// a transposed rhs from a plain one; the generic floor aliased each operand's
/// own dense layout, which says "space axis i is operand axis i". Both read a
/// `[m, k]` activation as `[k, m]` under `d_rhs`.
#[test]
fn a_transposed_contraction_reads_the_right_operand_axes() {
    let s = session();
    let (m, k, n) = (3usize, 4usize, 5usize);
    let ad: Vec<f32> = (0..m * k).map(|i| 0.25 + i as f32 * 0.13).collect();
    let bd: Vec<f32> = (0..n * k).map(|i| -0.4 + i as f32 * 0.07).collect();

    let g = Graph::new(&s);
    let a = up(&g, &[m as u64, k as u64], &ad);
    let b = up(&g, &[n as u64, k as u64], &bd);
    let got = a.matmul_t(&b).unwrap().to_vec_f32().unwrap();

    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|t| ad[i * k + t] * bd[j * k + t]).sum();
        }
    }
    close(&got, &want);
}

/// The rhs adjoint of a plain matmul is `A^T @ grad`, itself a non-canonical
/// contraction. Under `sum_all` the seed is all ones, so `dB[t, j]` is the
/// column sum of `A`.
#[test]
fn the_matmul_rhs_adjoint_sums_the_right_axis() {
    let s = session();
    let (m, k, n) = (3usize, 4usize, 5usize);
    let ad: Vec<f32> = (0..m * k).map(|i| 0.25 + i as f32 * 0.13).collect();
    let bd: Vec<f32> = (0..k * n).map(|i| -0.4 + i as f32 * 0.07).collect();

    let g = Graph::new(&s);
    let a = up(&g, &[m as u64, k as u64], &ad);
    let b = up(&g, &[k as u64, n as u64], &bd);
    let y = a.matmul(&b).unwrap();
    let loss = y.sum_all().unwrap();
    let grads = g.backward_with(&loss, &[a.clone(), b.clone()]).unwrap();

    let da = grads.get(&a).unwrap().to_vec_f32().unwrap();
    let db = grads.get(&b).unwrap().to_vec_f32().unwrap();
    let want_a: Vec<f32> = (0..m * k)
        .map(|i| (0..n).map(|j| bd[(i % k) * n + j]).sum())
        .collect();
    let want_b: Vec<f32> = (0..k * n)
        .map(|i| (0..m).map(|r| ad[r * k + i / n]).sum())
        .collect();
    close(&da, &want_a);
    close(&db, &want_b);
}

/// `relu` is flat at the kink. The reference differentiates
/// `max_elementwise(rhs)` to `grad * input.mt(rhs)` — strictly greater — so a
/// tie sends the gradient nowhere.
#[test]
fn relu_has_no_gradient_at_zero() {
    let s = session();
    let data = [-1.0f32, -0.5, 0.0, 0.5, 1.0, 0.0];
    let g = Graph::new(&s);
    let x = up(&g, &[6], &data);
    let y = x.relu().unwrap();
    let loss = y.sum_all().unwrap();
    let grad = g
        .backward_with(&loss, std::slice::from_ref(&x))
        .unwrap()
        .get(&x)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    close(&grad, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
}

/// A backward pass may not cross graphs. It used to panic indexing one
/// e-graph's arena with the other's `Id`.
#[test]
fn a_loss_from_another_graph_is_refused_not_panicked() {
    let s = session();
    let a = Graph::new(&s);
    let b = Graph::new(&s);
    let x = up(&a, &[4], &[1.0, 2.0, 3.0, 4.0]);
    let other = up(&b, &[4], &[1.0, 2.0, 3.0, 4.0]);
    let loss = x.sqr().unwrap().sum_all().unwrap();
    assert!(b.backward_with(&loss, std::slice::from_ref(&other)).is_err());
}
