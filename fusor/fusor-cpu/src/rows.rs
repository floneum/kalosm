//! A gather of whole rows: when the gathered axis' inner extent is one
//! contiguous run of 4-byte elements, each selected row is a block copy.

use fusor_ir::Result;
use fusor_ir::error::Error;

use crate::emit::RawBuf;

/// `out[o, g, ..] = src[o, idx[g], ..]` over `outer` leading blocks.
#[derive(Clone, Debug)]
pub struct GatherRows {
    pub out: usize,
    pub src: usize,
    pub idx: usize,
    pub outer: usize,
    pub count: usize,
    pub inner: usize,
    /// The source's extent along the gathered axis.
    pub src_axis: usize,
}

impl GatherRows {
    pub(crate) fn name(&self) -> String {
        format!(
            "cpu_gather_rows:{},{},{},{},{},{},{}",
            self.out, self.src, self.idx, self.outer, self.count, self.inner, self.src_axis
        )
    }

    pub fn parse(name: &str) -> Option<Self> {
        let values: Vec<usize> = name
            .strip_prefix("cpu_gather_rows:")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        let [out, src, idx, outer, count, inner, src_axis] = values[..] else {
            return None;
        };
        Some(Self {
            out,
            src,
            idx,
            outer,
            count,
            inner,
            src_axis,
        })
    }

    pub(crate) fn run(&self, bufs: &[RawBuf]) -> Result<()> {
        let missing = || Error::Device("a row gather's binding is missing".into());
        let out = bufs.get(self.out).ok_or_else(missing)?;
        let src = bufs.get(self.src).ok_or_else(missing)?;
        let idx = bufs.get(self.idx).ok_or_else(missing)?;
        let src_len = src.bytes / 4;
        if out.bytes / 4 < self.outer * self.count * self.inner || idx.bytes / 4 < self.count {
            return Err(Error::Device("a row gather exceeds its binding".into()));
        }
        // SAFETY: the output and index extents were checked above, and every
        // source row is checked against the source's length before its copy.
        unsafe {
            let rows = std::slice::from_raw_parts(idx.ptr as *const u32, self.count);
            for o in 0..self.outer {
                for (g, &row) in rows.iter().enumerate() {
                    let from = (o * self.src_axis + row as usize) * self.inner;
                    let to = (out.ptr as *mut u32).add((o * self.count + g) * self.inner);
                    if from + self.inner <= src_len {
                        std::ptr::copy_nonoverlapping(
                            (src.ptr as *const u32).add(from),
                            to,
                            self.inner,
                        );
                    } else {
                        // An index past the source reads as zero, element by
                        // element, as a masked load does.
                        for e in 0..self.inner {
                            *to.add(e) = if from + e < src_len {
                                *(src.ptr as *const u32).add(from + e)
                            } else {
                                0
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_copied_and_out_of_range_rows_are_zero() {
        let src: Vec<f32> = (0..2 * 4 * 3).map(|i| i as f32).collect();
        let idx = [3u32, 0, 9];
        let mut out = vec![-1f32; 2 * 3 * 3];
        let raw = |ptr: *const u8, bytes: usize| RawBuf {
            ptr: ptr as *mut u8,
            bytes,
        };
        let bufs = [
            raw(out.as_mut_ptr().cast(), out.len() * 4),
            raw(src.as_ptr().cast(), src.len() * 4),
            raw(idx.as_ptr().cast(), 12),
        ];
        let spec = GatherRows {
            out: 0,
            src: 1,
            idx: 2,
            outer: 2,
            count: 3,
            inner: 3,
            src_axis: 4,
        };
        assert_eq!(GatherRows::parse(&spec.name()).unwrap().src_axis, 4);
        spec.run(&bufs).unwrap();
        // Block 0: rows 3 and 0, then row 9, which lands in block 1's rows
        // (index 9 of the flat source rows is past the end: zero).
        assert_eq!(&out[..6], &[9., 10., 11., 0., 1., 2.]);
        assert_eq!(&out[6..9], &[0., 0., 0.]);
        assert_eq!(&out[9..15], &[21., 22., 23., 12., 13., 14.]);
    }
}
