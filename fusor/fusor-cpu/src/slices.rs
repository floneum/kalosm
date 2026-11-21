//! Elementwise f32 expressions evaluated a slice at a time: each step is one
//! tight loop over a chunk, which the compiler vectorizes on every target.
//! Used for the transforms fused around a contraction and for whole map
//! kernels over plain buffers.

use fusor_ir::Result;
use fusor_ir::dtype::{Dtype, Splat};
use fusor_ir::error::Error;
use fusor_ir::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp};

use crate::emit::RawBuf;
use crate::emit::expr::{NumTy, apply_bin, apply_un};
use crate::helpers::{decode_bin, decode_un};

/// One step of a fused elementwise transform, in postfix order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SliceOp {
    Arg(u32),
    Const(f32),
    Un(UnOp),
    Bin(BinOp),
}

/// Deepest operand stack a transform may need.
const STACK: usize = 8;

/// `expr` as a postfix program over its first `args` f32 arguments, as text
/// for a kernel name; empty for the identity of argument 0, None when it is
/// not such a function.
pub(crate) fn encode(expr: &ScalarExpr, args: u32) -> Option<String> {
    if matches!(expr.kind(), ScalarKind::Arg(0)) {
        return Some(String::new());
    }
    fn walk(expr: &ScalarExpr, args: u32, out: &mut Vec<String>, depth: usize) -> Option<usize> {
        if expr.dtype() != Dtype::F32 {
            return None;
        }
        match expr.kind() {
            ScalarKind::Arg(i) if *i < args => out.push(format!("x{i}")),
            ScalarKind::Lit(lit) => match lit.0 {
                Splat::F32(v) => out.push(format!("c{:x}", v.to_bits())),
                _ => return None,
            },
            ScalarKind::Un { op, x } => {
                let need = walk(x, args, out, depth)?;
                out.push(format!("u{}", *op as u32));
                return Some(need);
            }
            ScalarKind::Bin { op, a, b } => {
                let left = walk(a, args, out, depth)?;
                let right = walk(b, args, out, depth + 1)?;
                out.push(format!("b{}", *op as u32));
                return Some(left.max(right));
            }
            _ => return None,
        }
        Some(depth + 1)
    }
    let mut tokens = Vec::new();
    let need = walk(expr, args, &mut tokens, 0)?;
    (need <= STACK).then(|| tokens.join("."))
}

pub(crate) fn decode(text: &str) -> Option<Vec<SliceOp>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    text.split('.')
        .map(|token| {
            let (kind, rest) = token.split_at(1);
            Some(match kind {
                "x" => SliceOp::Arg(rest.parse().ok()?),
                "c" => SliceOp::Const(f32::from_bits(u32::from_str_radix(rest, 16).ok()?)),
                "u" => SliceOp::Un(decode_un(rest.parse().ok()?)),
                "b" => SliceOp::Bin(decode_bin(rest.parse().ok()?)),
                _ => return None,
            })
        })
        .collect()
}

/// One argument of an expression: `len` elements at a pointer (which may be
/// the output itself), or a value broadcast over the output.
#[derive(Clone, Copy)]
pub(crate) enum Operand {
    Slice(*const f32),
    Scalar(f32),
}

const CHUNK: usize = 64;

/// Operand-stack slots `ops` needs.
fn depth(ops: &[SliceOp]) -> usize {
    let (mut top, mut most) = (0usize, 0);
    for (at, op) in ops.iter().enumerate() {
        match op {
            SliceOp::Const(_) if top > 0 && matches!(ops.get(at + 1), Some(SliceOp::Bin(_))) => {}
            SliceOp::Arg(_) | SliceOp::Const(_) => top += 1,
            SliceOp::Un(_) => {}
            SliceOp::Bin(_) if at > 0 && matches!(ops[at - 1], SliceOp::Const(_)) && top > 0 => {}
            SliceOp::Bin(_) => top -= 1,
        }
        most = most.max(top);
    }
    most
}

/// One chunk of `ops`, `arg(i, to)` filling argument `i`'s `n` values. A
/// constant that is the right operand of the next step is applied as a scalar.
#[inline(always)]
fn chunk(
    ops: &[SliceOp],
    n: usize,
    stack: &mut [[f32; CHUNK]],
    mut arg: impl FnMut(u32, &mut [f32]),
) {
    let mut top = 0;
    let mut at = 0;
    while at < ops.len() {
        match (ops[at], ops.get(at + 1)) {
            (SliceOp::Const(c), Some(SliceOp::Bin(op))) if top > 0 => {
                binary_scalar(*op, &mut stack[top - 1][..n], c);
                at += 1;
            }
            (SliceOp::Arg(i), _) => {
                arg(i, &mut stack[top][..n]);
                top += 1;
            }
            (SliceOp::Const(c), _) => {
                stack[top][..n].fill(c);
                top += 1;
            }
            (SliceOp::Un(op), _) => unary(op, &mut stack[top - 1][..n]),
            (SliceOp::Bin(op), _) => {
                let (low, high) = stack.split_at_mut(top - 1);
                binary(op, &mut low[top - 2][..n], &high[0][..n]);
                top -= 1;
            }
        }
        at += 1;
    }
}

/// Run `body` with an operand stack for `ops`: two slots for the common
/// shallow program, the full stack otherwise.
#[inline(always)]
fn with_stack(ops: &[SliceOp], body: impl FnOnce(&mut [[f32; CHUNK]])) {
    // Three steps never hold more than two operands.
    if ops.len() <= 3 || depth(ops) <= 2 {
        body(&mut [[0f32; CHUNK]; 2]);
    } else {
        body(&mut [[0f32; CHUNK]; STACK]);
    }
}

/// Apply a one-argument transform to every element of `x`, in place.
pub(crate) fn transform(ops: &[SliceOp], x: &mut [f32]) {
    with_stack(ops, |stack| {
        for part in x.chunks_mut(CHUNK) {
            let n = part.len();
            chunk(ops, n, stack, |_, to| to.copy_from_slice(part));
            part.copy_from_slice(&stack[0][..n]);
        }
    });
}

/// `out[i] = ops(args[..][i])` for `len` elements; an empty program copies
/// argument 0. Each element's arguments are read before it is written, so an
/// argument may be the output.
///
/// # Safety
/// `out` and every slice argument must hold `len` elements, and a slice
/// argument that overlaps `out` must not start before it.
pub(crate) unsafe fn evaluate(ops: &[SliceOp], args: &[Operand], out: *mut f32, len: usize) {
    // One operator over two whole arguments needs no operand stack.
    if let ([SliceOp::Arg(i), SliceOp::Arg(j), SliceOp::Bin(op)], true) = (ops, args.len() >= 2)
        && let (Operand::Slice(a), Operand::Slice(b)) = (args[*i as usize], args[*j as usize])
    {
        // SAFETY: the caller's contract; element `i` is read before it is
        // written and later reads are at or past `i`.
        macro_rules! each {
            ($f:expr) => {
                for i in 0..len {
                    unsafe { *out.add(i) = $f(*a.add(i), *b.add(i)) };
                }
            };
        }
        match op {
            BinOp::Add => each!(|a: f32, b: f32| a + b),
            BinOp::Sub => each!(|a: f32, b: f32| a - b),
            BinOp::Mul => each!(|a: f32, b: f32| a * b),
            BinOp::Div => each!(|a: f32, b: f32| a / b),
            BinOp::Min => each!(|a: f32, b: f32| if b < a { b } else { a }),
            BinOp::Max => each!(|a: f32, b: f32| if b > a { b } else { a }),
            _ => each!(|a: f32, b: f32| f32::from_bits(apply_bin(
                *op,
                NumTy::F32,
                a.to_bits(),
                b.to_bits()
            ))),
        }
        return;
    }
    let identity = [SliceOp::Arg(0)];
    let ops = if ops.is_empty() { &identity[..] } else { ops };
    with_stack(ops, |stack| {
        let mut at = 0;
        while at < len {
            let n = (len - at).min(CHUNK);
            chunk(ops, n, stack, |i, to| match args[i as usize] {
                // SAFETY: the caller's contract; `to` is the stack, never an argument.
                Operand::Slice(values) => unsafe {
                    std::ptr::copy_nonoverlapping(values.add(at), to.as_mut_ptr(), n)
                },
                Operand::Scalar(value) => to.fill(value),
            });
            // SAFETY: the caller's contract.
            unsafe { std::ptr::copy_nonoverlapping(stack[0].as_ptr(), out.add(at), n) };
            at += n;
        }
    });
}

fn unary(op: UnOp, x: &mut [f32]) {
    match op {
        UnOp::Neg => x.iter_mut().for_each(|v| *v = -*v),
        UnOp::Abs => x.iter_mut().for_each(|v| *v = v.abs()),
        UnOp::Sqrt => x.iter_mut().for_each(|v| *v = v.sqrt()),
        _ => x
            .iter_mut()
            .for_each(|v| *v = f32::from_bits(apply_un(op, NumTy::F32, v.to_bits()))),
    }
}

fn binary_scalar(op: BinOp, a: &mut [f32], b: f32) {
    match op {
        BinOp::Add => a.iter_mut().for_each(|a| *a += b),
        BinOp::Sub => a.iter_mut().for_each(|a| *a -= b),
        BinOp::Mul => a.iter_mut().for_each(|a| *a *= b),
        BinOp::Div => a.iter_mut().for_each(|a| *a /= b),
        BinOp::Min => a.iter_mut().for_each(|a| *a = if b < *a { b } else { *a }),
        BinOp::Max => a.iter_mut().for_each(|a| *a = if b > *a { b } else { *a }),
        _ => a
            .iter_mut()
            .for_each(|a| *a = f32::from_bits(apply_bin(op, NumTy::F32, a.to_bits(), b.to_bits()))),
    }
}

fn binary(op: BinOp, a: &mut [f32], b: &[f32]) {
    let pairs = a.iter_mut().zip(b);
    match op {
        BinOp::Add => pairs.for_each(|(a, b)| *a += *b),
        BinOp::Sub => pairs.for_each(|(a, b)| *a -= *b),
        BinOp::Mul => pairs.for_each(|(a, b)| *a *= *b),
        BinOp::Div => pairs.for_each(|(a, b)| *a /= *b),
        // The comparison-select form the JIT emits, not `f32::min`.
        BinOp::Min => pairs.for_each(|(a, b)| *a = if *b < *a { *b } else { *a }),
        BinOp::Max => pairs.for_each(|(a, b)| *a = if *b > *a { *b } else { *a }),
        _ => pairs.for_each(|(a, b)| {
            *a = f32::from_bits(apply_bin(op, NumTy::F32, a.to_bits(), b.to_bits()))
        }),
    }
}

/// A map kernel whose operands are plain f32 buffers: each either as long as
/// the output and contiguous from an offset, or one element broadcast.
#[derive(Clone, Debug)]
pub struct MapSlices {
    pub out: usize,
    pub elements: usize,
    /// `(binding, offset, broadcast)` per argument.
    pub args: Vec<(usize, usize, bool)>,
    pub ops: Vec<SliceOp>,
}

impl MapSlices {
    pub(crate) fn name(
        out: usize,
        elements: usize,
        args: &[(usize, usize, bool)],
        body: &str,
    ) -> String {
        let args: Vec<String> = args
            .iter()
            .map(|(binding, offset, broadcast)| {
                format!("{binding}:{offset}:{}", u8::from(*broadcast))
            })
            .collect();
        format!("cpu_map_slices:{out},{elements};{};{body}", args.join(","))
    }

    pub fn parse(name: &str) -> Option<Self> {
        let mut parts = name.strip_prefix("cpu_map_slices:")?.split(';');
        let (out, elements) = parts.next()?.split_once(',')?;
        let args = parts
            .next()?
            .split(',')
            .filter(|arg| !arg.is_empty())
            .map(|arg| {
                let mut fields = arg.split(':');
                Some((
                    fields.next()?.parse().ok()?,
                    fields.next()?.parse().ok()?,
                    fields.next()? == "1",
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            out: out.parse().ok()?,
            elements: elements.parse().ok()?,
            args,
            ops: decode(parts.next()?)?,
        })
    }

    pub(crate) fn run(&self, bufs: &[RawBuf]) -> Result<()> {
        let exceeds = || Error::Device("a slice map exceeds one of its bindings".into());
        let out = bufs.get(self.out).ok_or_else(exceeds)?;
        if out.bytes / 4 < self.elements {
            return Err(exceeds());
        }
        let mut args = [Operand::Scalar(0.); STACK];
        if self.args.len() > STACK {
            return Err(Error::Device("a slice map has too many arguments".into()));
        }
        for (slot, &(binding, offset, broadcast)) in args.iter_mut().zip(&self.args) {
            let buf = bufs.get(binding).ok_or_else(exceeds)?;
            let needed = offset + if broadcast { 1 } else { self.elements };
            if buf.bytes / 4 < needed {
                return Err(exceeds());
            }
            // SAFETY: the range was checked against the binding just above.
            *slot = unsafe {
                let ptr = (buf.ptr as *const f32).add(offset);
                if broadcast {
                    Operand::Scalar(*ptr)
                } else {
                    Operand::Slice(ptr)
                }
            };
        }
        // SAFETY: the output and every argument hold `elements` (checked above).
        unsafe {
            evaluate(
                &self.ops,
                &args[..self.args.len()],
                out.ptr as *mut f32,
                self.elements,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relu() -> ScalarExpr {
        ScalarExpr::new(
            ScalarKind::Bin {
                op: BinOp::Max,
                a: ScalarExpr::arg(0, Dtype::F32),
                b: ScalarExpr::new(
                    ScalarKind::Lit(fusor_ir::scalar::Lit(Splat::F32(0.0))),
                    Dtype::F32,
                ),
            },
            Dtype::F32,
        )
    }

    #[test]
    #[ignore]
    fn time_a_map() {
        let map = MapSlices::parse("cpu_map_slices:0,258;1:0:0,2:0:0;x0.x1.b0").unwrap();
        let mut out = vec![0f32; 258];
        let (a, b): (Vec<f32>, Vec<f32>) = (0..258).map(|i| (i as f32, 1.)).unzip();
        let raw = |p: *const f32| RawBuf {
            ptr: p as *mut u8,
            bytes: 258 * 4,
        };
        let bufs = [raw(out.as_mut_ptr()), raw(a.as_ptr()), raw(b.as_ptr())];
        let start = std::time::Instant::now();
        for _ in 0..1_000_000 {
            map.run(&bufs).unwrap();
            std::hint::black_box(&out);
        }
        println!("map add 258: {:.0} ns", start.elapsed().as_secs_f64() * 1e3);
    }

    #[test]
    #[ignore]
    fn time_a_transform() {
        let ops = decode(&encode(&relu(), 1).unwrap()).unwrap();
        let mut values: Vec<f32> = (0..128).map(|i| i as f32 - 50.0).collect();
        let start = std::time::Instant::now();
        for _ in 0..1_000_000 {
            transform(&ops, &mut values);
            std::hint::black_box(&values);
        }
        println!(
            "transform relu 128: {:.0} ns",
            start.elapsed().as_secs_f64() * 1e3
        );
    }

    #[test]
    fn transforms_round_trip_through_their_text() {
        assert_eq!(
            encode(&ScalarExpr::arg(0, Dtype::F32), 1).as_deref(),
            Some("")
        );
        let ops = decode(&encode(&relu(), 1).unwrap()).unwrap();
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
        transform(&ops, &mut values);
        assert!(
            values
                .iter()
                .enumerate()
                .all(|(i, v)| *v == (i as f32 - 50.0).max(0.0))
        );
    }

    #[test]
    fn a_map_over_two_buffers_and_a_broadcast_runs_in_place() {
        let body = ScalarExpr::new(
            ScalarKind::Bin {
                op: BinOp::Add,
                a: ScalarExpr::new(
                    ScalarKind::Bin {
                        op: BinOp::Mul,
                        a: ScalarExpr::arg(0, Dtype::F32),
                        b: ScalarExpr::arg(2, Dtype::F32),
                    },
                    Dtype::F32,
                ),
                b: ScalarExpr::arg(1, Dtype::F32),
            },
            Dtype::F32,
        );
        let name = MapSlices::name(
            0,
            150,
            &[(0, 0, false), (1, 3, false), (2, 1, true)],
            &encode(&body, 3).unwrap(),
        );
        let map = MapSlices::parse(&name).unwrap();
        let mut x: Vec<f32> = (0..150).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..160).map(|i| 1000. + i as f32).collect();
        let scale = [9f32, 2.];
        let raw = |ptr: *const f32, len: usize| RawBuf {
            ptr: ptr as *mut u8,
            bytes: len * 4,
        };
        map.run(&[
            raw(x.as_mut_ptr(), 150),
            raw(y.as_ptr(), 160),
            raw(scale.as_ptr(), 2),
        ])
        .unwrap();
        assert!(
            x.iter()
                .enumerate()
                .all(|(i, v)| *v == i as f32 * 2. + 1003. + i as f32)
        );
    }
}
