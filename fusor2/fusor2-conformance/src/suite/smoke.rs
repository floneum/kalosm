//! Smoke cases: upload-and-read-back, one elementwise op, one matmul, one
//! repeated read. Every other area assumes these work, so they are registered
//! first in `suite::registry`.
//!
//! Each asserts the actual numbers, on every backend `sessions()` offers.

use fusor2::{Graph, Session};

use crate::compare;
use crate::harness::{Cases, dims, from_f32};

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    // No ops at all. Four floats up, four floats back.
    cases.push("smoke", "upload_and_read_back_four_f32", |s: &Session| {
        let g = Graph::new(s);
        let data = [1.0f32, -2.5, 3.25, 4.0];
        let x = from_f32(g.handle(), &dims(&[4]), &data)?;
        let got = x.to_vec_f32()?;
        compare::assert_close(&got, &data, 0.0, 0.0)?;
        Ok(())
    });

    // One elementwise op, hand-computed.
    cases.push("smoke", "add_one_to_four_f32", |s: &Session| {
        let g = Graph::new(s);
        let data = [1.0f32, -2.5, 3.25, 4.0];
        let x = from_f32(g.handle(), &dims(&[4]), &data)?;
        let y = x.add_scalar(1.0f32)?;
        let got = y.to_vec_f32()?;
        let want = [2.0f32, -1.5, 4.25, 5.0];
        compare::assert_close(&got, &want, 1e-6, 1e-6)?;
        Ok(())
    });

    // [2,3] @ [3,2], hand-computed:
    // row0 = [1,2,3] . cols([[7,8],[9,10],[11,12]]) = [58, 64]
    // row1 = [4,5,6] . same                         = [139, 154]
    cases.push("smoke", "matmul_2x3_by_3x2", |s: &Session| {
        let g = Graph::new(s);
        let a = from_f32(g.handle(), &dims(&[2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let b = from_f32(
            g.handle(),
            &dims(&[3, 2]),
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )?;
        let c = a.matmul(&b)?;
        let got = c.to_vec_f32()?;
        let want = [58.0f32, 64.0, 139.0, 154.0];
        compare::assert_close(&got, &want, 1e-4, 1e-5)?;
        Ok(())
    });

    // A f32 leaf uploaded once and read twice: the second read must not depend
    // on a buffer the first resolve recycled.
    cases.push("smoke", "reading_twice_is_stable", |s: &Session| {
        let g = Graph::new(s);
        let data = [0.5f32, 1.5, 2.5, 3.5];
        let x = from_f32(g.handle(), &dims(&[4]), &data)?;
        let y = x.add_scalar(-0.5f32)?;
        let first = y.to_vec_f32()?;
        let second = y.to_vec_f32()?;
        compare::assert_close(&second, &first, 0.0, 0.0)?;
        let want = [0.0f32, 1.0, 2.0, 3.0];
        compare::assert_close(&first, &want, 1e-6, 1e-6)?;
        Ok(())
    });

    cases
}
