//! The slow-path host functions every emitted kernel calls for what it does
//! not inline (odd element widths, transcendentals, casts). Shared by the
//! Cranelift and wasm emitters; on wasm32 they are exported so a generated
//! module can import them.

use fusor_ir::dtype::RoundMode;
use fusor_ir::ir::kernel::ScalarElement;
use fusor_ir::scalar::{BinOp, UnOp};

use crate::emit::expr::NumTy;

#[inline(never)]
pub(crate) extern "C" fn jit_read(
    ptr: *const u8,
    bytes: usize,
    index: u32,
    mask: u32,
    fill: u32,
    elem: u32,
) -> u32 {
    let elem = decode_elem(elem);
    if mask == 0 || index as usize >= bytes / elem.byte_size() as usize {
        fill
    } else {
        unsafe { crate::emit::expr::read_elem(elem, ptr, index as usize) }
    }
}

#[inline(never)]
pub(crate) extern "C" fn jit_write(
    ptr: *mut u8,
    bytes: usize,
    index: u32,
    value: u32,
    mask: u32,
    elem: u32,
) {
    let elem = decode_elem(elem);
    if mask != 0 && (index as usize) < bytes / elem.byte_size() as usize {
        unsafe { crate::emit::expr::write_elem(elem, ptr, index as usize, value) };
    }
}

#[inline(never)]
pub(crate) extern "C" fn jit_un(code: u32, ty: u32, bits: u32) -> u32 {
    let op = decode_un(code);
    let ty = decode_ty(ty);
    crate::emit::expr::apply_un(op, ty, bits)
}

#[inline(never)]
pub(crate) extern "C" fn jit_bin(code: u32, ty: u32, a: u32, b: u32) -> u32 {
    let op = decode_bin(code);
    let ty = decode_ty(ty);
    crate::emit::expr::apply_bin(op, ty, a, b)
}

#[inline(never)]
pub(crate) extern "C" fn jit_cast(from: u32, to: u32, bits: u32) -> u32 {
    crate::emit::expr::apply_cast(decode_ty(from), decode_ty(to), bits)
}

#[inline(never)]
pub(crate) extern "C" fn jit_round(mode: u32, bits: u32) -> u32 {
    crate::emit::expr::round_mode(decode_round(mode), f32::from_bits(bits)).to_bits()
}

#[inline(never)]
pub(crate) extern "C" fn jit_narrow(elem: u32, bits: u32) -> u32 {
    crate::emit::expr::apply_narrow(decode_elem(elem), bits)
}

#[inline(never)]
pub(crate) extern "C" fn jit_unpack(bits: u32, high: u32) -> u32 {
    let raw = if high == 0 {
        bits as u16
    } else {
        (bits >> 16) as u16
    };
    half::f16::from_bits(raw).to_f32().to_bits()
}

/// Host-call operand codes are declaration order (`x as u32`); these invert them.
macro_rules! decoders {
    ($($f:ident: $t:ident = [$($v:ident),*];)*) => {$(
        pub(crate) fn $f(v: u32) -> $t {
            [$($t::$v),*][v as usize]
        }
    )*};
}

decoders! {
    decode_ty: NumTy = [F32, U32, I32];
    decode_elem: ScalarElement = [F32, F16, BF16, U32, I32, Bool];
    decode_round: RoundMode = [HalfToEven, HalfAwayFromZero, Floor, Ceil, Trunc];
    decode_un: UnOp = [Exp, ApproximateExp, LessApproximateExp, Exp2, Log, Log2, Sqrt, InverseSqrt,
        Sin, Cos, Tan, Tanh, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Abs, Neg,
        Unpack2x16Float];
    decode_bin: BinOp = [Add, Sub, Mul, Div, Rem, Pow, Min, Max, BitAnd, BitOr, BitXor, Shr, Shl,
        LogicalAnd, LogicalOr];
}
