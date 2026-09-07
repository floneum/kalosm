//! Kernel expressions -> SIMD lane operations at a statically-known width.
//!
//! This is the SSA tape. One [`Instr`] per distinct `TileExprKind` node, one
//! register slot per node; flattening the hash-consed DAG in topological order
//! is the CSE.
//!
//! The register file is `[u32; W]` with `W` a const generic resolved once per
//! launch, so an `MxN` register accumulator tile is expressible.
//!
//! `ElementType::Scalar(F16|BF16)` loads widen to `F32` registers and stores
//! narrow: the emitter half of the `widen-compute` rule.

use fusor_ir::dtype::RoundMode;
use fusor_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor_ir::scalar::{BinOp, CmpOp, UnOp};

use super::access::AccessForm;

/// A tape register index.
pub(crate) type Slot = u32;

/// Compute type of a register. Storage-only narrow floats never appear: they
/// widen on load and narrow on store.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumTy {
    F32,
    U32,
    I32,
}

impl NumTy {
    /// The compute type a stored element widens to.
    pub const fn of(e: ScalarElement) -> Self {
        match e {
            ScalarElement::F32 | ScalarElement::F16 | ScalarElement::BF16 => Self::F32,
            ScalarElement::U32 | ScalarElement::Bool => Self::U32,
            ScalarElement::I32 => Self::I32,
        }
    }
}

/// A cross-lane reduction, already resolved to a CPU strategy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RKind {
    /// Horizontal reduce of the `W` lanes of one register by `log2(W)`
    /// shuffle-reduce steps, broadcast back across the register.
    Subgroup,
    /// Read a group result the preceding tree pass already broadcast into
    /// `tile`. `group` is the lane-group width.
    TileGroup { tile: u16, group: u32 },
}

/// One tape instruction. `out` is the base register slot; a vector-typed
/// result occupies `out .. out + lanes`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    /// Bit pattern splatted across the register.
    Const {
        out: Slot,
        bits: u32,
    },
    /// `chunk_base + [0, 1, ... W-1]`, the lane index inside the block.
    LaneId {
        out: Slot,
    },
    /// A workgroup-uniform u32 splat resolved at launch.
    Uniform {
        out: Slot,
        which: UniformSrc,
    },
    LoadLocal {
        out: Slot,
        local: u16,
    },
    /// Masked load. `index` is a per-lane element index; `form` is the access
    /// lowering chosen once at compile time from the layout.
    Load {
        out: Slot,
        buf: u16,
        elem: ScalarElement,
        index: Slot,
        mask: Slot,
        fill: Slot,
        form: AccessForm,
    },
    LoadTile {
        out: Slot,
        tile: u16,
        elem: ScalarElement,
        index: Slot,
    },
    Un {
        out: Slot,
        op: UnOp,
        x: Slot,
        ty: NumTy,
    },
    Bin {
        out: Slot,
        op: BinOp,
        a: Slot,
        b: Slot,
        ty: NumTy,
    },
    /// A contracted `a * b + c`. Minted only when the operand's
    /// `NumericContract::contract` is set; a strict value emits separate
    /// `Bin{Mul}` and `Bin{Add}` instructions.
    Fma {
        out: Slot,
        a: Slot,
        b: Slot,
        c: Slot,
    },
    Cmp {
        out: Slot,
        op: CmpOp,
        a: Slot,
        b: Slot,
        ty: NumTy,
    },
    /// A lane mask materialized to `1.0`/`0.0` (or `1`/`0`) in `ty`.
    MaskToValue {
        out: Slot,
        x: Slot,
        ty: NumTy,
    },
    /// A value tested against zero, producing a lane mask.
    ValueToMask {
        out: Slot,
        x: Slot,
        ty: NumTy,
    },
    Round {
        out: Slot,
        mode: RoundMode,
        x: Slot,
    },
    Cast {
        out: Slot,
        x: Slot,
        from: NumTy,
        to: NumTy,
    },
    /// Narrow to a storage element and widen straight back — what a `Cast` to
    /// `F16`/`BF16` means when the register file only holds f32.
    Narrow {
        out: Slot,
        x: Slot,
        to: ScalarElement,
    },
    Bitcast {
        out: Slot,
        x: Slot,
    },
    Select {
        out: Slot,
        c: Slot,
        t: Slot,
        f: Slot,
    },
    /// `parts[i]` copied into `out + i`.
    VecCompose {
        out: Slot,
        parts: Vec<Slot>,
    },
    VecComponent {
        out: Slot,
        base: Slot,
        component: u32,
    },
    Dot {
        out: Slot,
        a: Slot,
        b: Slot,
        lanes: u32,
    },
    Reduce {
        out: Slot,
        op: TileReduceOp,
        x: Slot,
        kind: RKind,
        /// Lane-group base index, for `RKind::TileGroup`.
        group_base: Slot,
    },
    /// Unpack a `u32` of two packed f16s into a 2-lane f32 vector.
    Unpack2x16 {
        out: Slot,
        x: Slot,
    },
    /// Run a rank-2 address through the declared divmod chain of
    /// `maps[map]`. Only the divmods the `MultiFlattenMap` declares are
    /// performed, because `divmod_ops()` is the cost term.
    Rc2Index {
        out: Slot,
        row: Slot,
        col: Slot,
        map: u16,
    },
    Copy {
        out: Slot,
        x: Slot,
    },
}

impl Instr {
    pub fn out(&self) -> Slot {
        match self {
            Instr::Const { out, .. }
            | Instr::LaneId { out }
            | Instr::Uniform { out, .. }
            | Instr::LoadLocal { out, .. }
            | Instr::Load { out, .. }
            | Instr::LoadTile { out, .. }
            | Instr::Un { out, .. }
            | Instr::Bin { out, .. }
            | Instr::Fma { out, .. }
            | Instr::Cmp { out, .. }
            | Instr::MaskToValue { out, .. }
            | Instr::ValueToMask { out, .. }
            | Instr::Round { out, .. }
            | Instr::Cast { out, .. }
            | Instr::Narrow { out, .. }
            | Instr::Bitcast { out, .. }
            | Instr::Select { out, .. }
            | Instr::VecCompose { out, .. }
            | Instr::VecComponent { out, .. }
            | Instr::Dot { out, .. }
            | Instr::Reduce { out, .. }
            | Instr::Unpack2x16 { out, .. }
            | Instr::Rc2Index { out, .. }
            | Instr::Copy { out, .. } => *out,
        }
    }

    /// Is this a fused multiply-add? `numeric_contract_blocks_contraction`
    /// inspects the tape with this.
    pub fn is_fma(&self) -> bool {
        matches!(self, Instr::Fma { .. })
    }
}

/// Workgroup-uniform u32 sources, resolved at launch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UniformSrc {
    ProgramX,
    ProgramY,
    ProgramZ,
    GridX,
    GridY,
    GridZ,
    SubgroupSize,
    NumSubgroups,
    /// Lane index divided by the register width.
    SubgroupId,
    /// Lane index modulo the register width.
    SubgroupLane,
}

const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;
const PI_2_HI: f32 = std::f32::consts::FRAC_PI_2;
const PI_2_MID: f32 = 7.549_789_4e-8;
const PI_2_LO: f32 = 5.390_302_6e-15;
const TWO_OVER_PI: f32 = std::f32::consts::FRAC_2_PI;

/// `x * 2^n` by exponent-bit assembly, saturating outside the normal range.
#[inline(always)]
fn ldexp(x: f32, n: i32) -> f32 {
    if n > 127 {
        return x * f32::from_bits(0x7F00_0000) * f32::from_bits(0x7F00_0000);
    }
    if n < -126 {
        let step = f32::from_bits(((-100 + 127) as u32) << 23);
        let rest = n + 100;
        if rest < -126 {
            return 0.0;
        }
        return x * step * f32::from_bits(((rest + 127) as u32) << 23);
    }
    x * f32::from_bits(((n + 127) as u32) << 23)
}

/// Round to nearest, ties to even, without a libm call.
#[inline(always)]
pub(crate) fn rint(x: f32) -> f32 {
    let t = trunc(x);
    let frac = x - t;
    let a = frac.abs();
    let bump = if a > 0.5 {
        1.0
    } else if a < 0.5 || (t as i64) % 2 == 0 {
        0.0
    } else {
        1.0
    };
    if x < 0.0 { t - bump } else { t + bump }
}

#[inline(always)]
pub(crate) fn trunc(x: f32) -> f32 {
    if x.is_nan() || x.abs() >= 8_388_608.0 {
        // |x| >= 2^23 (or NaN): already integral, or not representable.
        return x;
    }
    (x as i32) as f32
}

#[inline(always)]
pub(crate) fn floorf(x: f32) -> f32 {
    let t = trunc(x);
    if x < 0.0 && t != x { t - 1.0 } else { t }
}

#[inline(always)]
pub(crate) fn ceilf(x: f32) -> f32 {
    let t = trunc(x);
    if x > 0.0 && t != x { t + 1.0 } else { t }
}

#[inline(always)]
pub(crate) fn round_half_away(x: f32) -> f32 {
    let t = trunc(x);
    let a = (x - t).abs();
    if a >= 0.5 {
        if x < 0.0 { t - 1.0 } else { t + 1.0 }
    } else {
        t
    }
}

#[inline(always)]
pub(crate) fn round_mode(mode: RoundMode, x: f32) -> f32 {
    match mode {
        RoundMode::HalfToEven => rint(x),
        RoundMode::HalfAwayFromZero => round_half_away(x),
        RoundMode::Floor => floorf(x),
        RoundMode::Ceil => ceilf(x),
        RoundMode::Trunc => trunc(x),
    }
}

/// `exp(r)` for `|r| <= ln2/2`. Taylor to `r^8`; the `r^9` term is below 1e-10
/// over the reduced range.
#[inline(always)]
fn exp_poly(r: f32) -> f32 {
    let mut p = 2.480_158_7e-5; // 1/8!
    p = p * r + 1.984_127e-4; // 1/7!
    p = p * r + 1.388_888_9e-3; // 1/6!
    p = p * r + 8.333_333e-3; // 1/5!
    p = p * r + 4.166_666_6e-2; // 1/4!
    p = p * r + 0.166_666_67; // 1/3!
    p = p * r + 0.5;
    p = p * r + 1.0;
    p * r + 1.0
}

/// `e^x`: Cody–Waite reduction plus the degree-8 polynomial above.
#[inline(always)]
pub(crate) fn expf(x: f32) -> f32 {
    if x > 88.72 {
        return f32::INFINITY;
    }
    if x < -103.0 {
        return 0.0;
    }
    let n = rint(x * LOG2_E);
    let r = x - n * LN2_HI - n * LN2_LO;
    ldexp(exp_poly(r), n as i32)
}

/// `2^x`.
#[inline(always)]
pub(crate) fn exp2f(x: f32) -> f32 {
    if x > 127.9 {
        return f32::INFINITY;
    }
    if x < -149.0 {
        return 0.0;
    }
    let n = rint(x);
    let r = (x - n) * LN2;
    ldexp(exp_poly(r), n as i32)
}

/// `log(1 + u)`, accurate near zero.
#[inline(always)]
pub(crate) fn log1pf(u: f32) -> f32 {
    if u <= -1.0 {
        return if u == -1.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    if u.abs() < 0.414_213_57 {
        let s = u / (2.0 + u);
        let s2 = s * s;
        let mut p = 1.0 / 13.0;
        p = p * s2 + 1.0 / 11.0;
        p = p * s2 + 1.0 / 9.0;
        p = p * s2 + 1.0 / 7.0;
        p = p * s2 + 0.2;
        p = p * s2 + 1.0 / 3.0;
        p = p * s2 + 1.0;
        return 2.0 * s * p;
    }
    logf(1.0 + u)
}

/// Split `x` into `(mantissa in [sqrt(1/2), sqrt(2)), exponent)`.
#[inline(always)]
fn frexp_norm(x: f32) -> (f32, i32) {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    (m, e)
}

/// Natural log by mantissa/exponent split plus a degree-13 odd polynomial in
/// `s = (m-1)/(m+1)`.
#[inline(always)]
pub(crate) fn logf(x: f32) -> f32 {
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    // Scale denormals into the normal range before splitting.
    let (x, bias) = if x < f32::MIN_POSITIVE {
        (x * 16_777_216.0, -24.0f32)
    } else {
        (x, 0.0)
    };
    let (m, e) = frexp_norm(x);
    let e = e as f32 + bias;
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 0.2;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    // `e * LN2` in two pieces keeps the exponent contribution exact.
    2.0 * s * p + (e * LN2_HI + e * LN2_LO)
}

#[inline(always)]
pub(crate) fn log2f(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return logf(x) * LOG2_E;
    }
    let (x, bias) = if x < f32::MIN_POSITIVE {
        (x * 16_777_216.0, -24.0f32)
    } else {
        (x, 0.0)
    };
    let (m, e) = frexp_norm(x);
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 0.2;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    (e as f32 + bias) + 2.0 * s * p * LOG2_E
}

#[inline(always)]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    if y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f32::INFINITY };
    }
    if x < 0.0 {
        // Only integral exponents are defined; match WGSL's signed magnitude.
        let n = rint(y);
        if n != y {
            return f32::NAN;
        }
        let mag = expf(y * logf(-x));
        return if (n as i64) % 2 == 0 { mag } else { -mag };
    }
    expf(y * logf(x))
}

/// Cody–Waite reduction by `pi/2`, returning `(r, quadrant)`.
#[inline(always)]
fn trig_reduce(x: f32) -> (f32, i32) {
    let n = rint(x * TWO_OVER_PI);
    let r = x - n * PI_2_HI - n * PI_2_MID - n * PI_2_LO;
    (r, (n as i64 & 3) as i32)
}

/// `sin(r)` for `|r| <= pi/4`, degree 9.
#[inline(always)]
fn sin_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = 2.755_731_9e-6; // 1/9!
    p = p * z - 1.984_127e-4; // -1/7!
    p = p * z + 8.333_333e-3; // 1/5!
    p = p * z - 0.166_666_67; // -1/3!
    r + r * z * p
}

/// `cos(r)` for `|r| <= pi/4`, degree 10.
#[inline(always)]
fn cos_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = -2.755_732e-7; // -1/10!
    p = p * z + 2.480_158_7e-5; // 1/8!
    p = p * z - 1.388_888_9e-3; // -1/6!
    p = p * z + 4.166_666_6e-2; // 1/4!
    p = p * z - 0.5;
    1.0 + z * p
}

#[inline(always)]
pub(crate) fn sinf(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, q) = trig_reduce(x);
    match q {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        _ => -cos_poly(r),
    }
}

#[inline(always)]
pub(crate) fn cosf(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, q) = trig_reduce(x);
    match q {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    }
}

#[inline(always)]
pub(crate) fn tanf(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, q) = trig_reduce(x);
    let s = sin_poly(r);
    let c = cos_poly(r);
    if q & 1 == 0 { s / c } else { -c / s }
}

/// `tanh` by the [7/8] Padé rational near zero and the `exp` identity outside
/// it, so the relative error stays flat across the whole line.
#[inline(always)]
pub(crate) fn tanhf(x: f32) -> f32 {
    let a = x.abs();
    if a < 0.55 {
        let z = x * x;
        let num = x * (135_135.0 + z * (17_325.0 + z * (378.0 + z)));
        let den = 135_135.0 + z * (62_370.0 + z * (3_150.0 + z * 28.0));
        return num / den;
    }
    if a > 9.011 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let e = expf(2.0 * a);
    let t = 1.0 - 2.0 / (e + 1.0);
    if x < 0.0 { -t } else { t }
}

#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    // A single hardware instruction, not a libm call.
    x.sqrt()
}

#[inline(always)]
pub(crate) fn rsqrtf(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Cephes single-precision `asin`, with the exact argument reduction at 0.5.
#[inline(always)]
pub(crate) fn asinf(x: f32) -> f32 {
    let a = x.abs();
    if a > 1.0 {
        return f32::NAN;
    }
    let (z, base, flag) = if a > 0.5 {
        let z = 0.5 * (1.0 - a);
        (z, sqrtf(z), true)
    } else {
        (a * a, a, false)
    };
    let mut p = 4.216_32e-2;
    p = p * z + 2.418_131e-2;
    p = p * z + 4.547_002_6e-2;
    p = p * z + 7.495_300_3e-2;
    p = p * z + 0.166_667_52;
    let r = p * z * base + base;
    let r = if flag {
        PI_2_HI + PI_2_MID - 2.0 * r
    } else {
        r
    };
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub(crate) fn acosf(x: f32) -> f32 {
    if x > 0.5 {
        // Keeps relative accuracy as the result approaches zero.
        2.0 * asinf(sqrtf(0.5 - 0.5 * x))
    } else if x < -0.5 {
        std::f32::consts::PI - 2.0 * asinf(sqrtf(0.5 + 0.5 * x))
    } else {
        (PI_2_HI + PI_2_MID) - asinf(x)
    }
}

/// Cephes single-precision `atan`.
#[inline(always)]
pub(crate) fn atanf(x: f32) -> f32 {
    let a = x.abs();
    let (a, y) = if a > 2.414_213_6 {
        (-1.0 / a, PI_2_HI + PI_2_MID)
    } else if a > 0.414_213_57 {
        ((a - 1.0) / (a + 1.0), std::f32::consts::FRAC_PI_4)
    } else {
        (a, 0.0)
    };
    let z = a * a;
    let mut p = 8.053_744_5e-2;
    p = p * z - 0.138_776_86;
    p = p * z + 0.199_777_11;
    p = p * z - 0.333_329_5;
    let r = y + a * z * p + a;
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub(crate) fn sinhf(x: f32) -> f32 {
    let a = x.abs();
    if a < 1.0 {
        let z = x * x;
        // Taylor to x^11: the x^13 term is below 1.6e-10 on |x| <= 1.
        let mut p = 1.0 / 39_916_800.0;
        p = p * z + 1.0 / 362_880.0;
        p = p * z + 1.0 / 5_040.0;
        p = p * z + 1.0 / 120.0;
        p = p * z + 1.0 / 6.0;
        return x + x * z * p;
    }
    let e = expf(a);
    let r = 0.5 * (e - 1.0 / e);
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub(crate) fn coshf(x: f32) -> f32 {
    let e = expf(x.abs());
    0.5 * (e + 1.0 / e)
}

#[inline(always)]
pub(crate) fn asinhf(x: f32) -> f32 {
    let a = x.abs();
    let r = if a < 1.0 {
        log1pf(a + a * a / (1.0 + sqrtf(1.0 + a * a)))
    } else {
        logf(a + sqrtf(a * a + 1.0))
    };
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub(crate) fn acoshf(x: f32) -> f32 {
    if x < 1.0 {
        return f32::NAN;
    }
    let t = x - 1.0;
    if t < 0.5 {
        log1pf(t + sqrtf(2.0 * t + t * t))
    } else {
        logf(x + sqrtf(x * x - 1.0))
    }
}

#[inline(always)]
pub(crate) fn atanhf(x: f32) -> f32 {
    let a = x.abs();
    if a >= 1.0 {
        return if a == 1.0 {
            if x > 0.0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let r = 0.5 * log1pf(2.0 * a / (1.0 - a));
    if x < 0.0 { -r } else { r }
}

/// Scalar helpers used only for operations Cranelift cannot lower directly.
#[inline(always)]
pub(crate) fn apply_un(op: UnOp, ty: NumTy, bits: u32) -> u32 {
    match (op, ty) {
        (UnOp::Neg, NumTy::F32) => bits ^ 0x8000_0000,
        (UnOp::Abs, NumTy::F32) => bits & 0x7fff_ffff,
        (UnOp::Neg, NumTy::I32) => (bits as i32).wrapping_neg() as u32,
        (UnOp::Abs, NumTy::I32) => (bits as i32).wrapping_abs() as u32,
        (UnOp::Neg, NumTy::U32) => bits.wrapping_neg(),
        (UnOp::Abs, NumTy::U32) | (UnOp::Unpack2x16Float, _) => bits,
        (op, _) => {
            let value = f32::from_bits(bits);
            match op {
                UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => expf(value),
                UnOp::Exp2 => exp2f(value),
                UnOp::Log => logf(value),
                UnOp::Log2 => log2f(value),
                UnOp::Sqrt => sqrtf(value),
                UnOp::InverseSqrt => rsqrtf(value),
                UnOp::Sin => sinf(value),
                UnOp::Cos => cosf(value),
                UnOp::Tan => tanf(value),
                UnOp::Tanh => tanhf(value),
                UnOp::Asin => asinf(value),
                UnOp::Acos => acosf(value),
                UnOp::Atan => atanf(value),
                UnOp::Sinh => sinhf(value),
                UnOp::Cosh => coshf(value),
                UnOp::Asinh => asinhf(value),
                UnOp::Acosh => acoshf(value),
                UnOp::Atanh => atanhf(value),
                UnOp::Abs => value.abs(),
                UnOp::Neg => -value,
                UnOp::Unpack2x16Float => value,
            }
            .to_bits()
        }
    }
}

#[inline(always)]
pub(crate) fn apply_bin(op: BinOp, ty: NumTy, a: u32, b: u32) -> u32 {
    match ty {
        NumTy::F32 => {
            let (x, y) = (f32::from_bits(a), f32::from_bits(b));
            match op {
                BinOp::Add => (x + y).to_bits(),
                BinOp::Sub => (x - y).to_bits(),
                BinOp::Mul => (x * y).to_bits(),
                BinOp::Div => (x / y).to_bits(),
                BinOp::Rem => (x - trunc(x / y) * y).to_bits(),
                BinOp::Pow => powf(x, y).to_bits(),
                BinOp::Min => (if y < x { y } else { x }).to_bits(),
                BinOp::Max => (if y > x { y } else { x }).to_bits(),
                BinOp::BitAnd => a & b,
                BinOp::BitOr => a | b,
                BinOp::BitXor => a ^ b,
                BinOp::Shr | BinOp::Shl => a,
                BinOp::LogicalAnd => f32::from(x != 0.0 && y != 0.0).to_bits(),
                BinOp::LogicalOr => f32::from(x != 0.0 || y != 0.0).to_bits(),
            }
        }
        NumTy::U32 => match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a.checked_div(b).unwrap_or(u32::MAX),
            BinOp::Rem => {
                if b == 0 {
                    0
                } else {
                    a % b
                }
            }
            BinOp::Pow => a.wrapping_pow(b),
            BinOp::Min => a.min(b),
            BinOp::Max => a.max(b),
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            BinOp::Shr => a >> (b & 31),
            BinOp::Shl => a << (b & 31),
            BinOp::LogicalAnd => u32::from(a != 0 && b != 0),
            BinOp::LogicalOr => u32::from(a != 0 || b != 0),
        },
        NumTy::I32 => {
            let (x, y) = (a as i32, b as i32);
            match op {
                BinOp::Add => x.wrapping_add(y) as u32,
                BinOp::Sub => x.wrapping_sub(y) as u32,
                BinOp::Mul => x.wrapping_mul(y) as u32,
                BinOp::Div => {
                    if y == 0 {
                        u32::MAX
                    } else {
                        x.wrapping_div(y) as u32
                    }
                }
                BinOp::Rem => {
                    if y == 0 {
                        0
                    } else {
                        x.wrapping_rem(y) as u32
                    }
                }
                BinOp::Pow => x.wrapping_pow(y.max(0) as u32) as u32,
                BinOp::Min => x.min(y) as u32,
                BinOp::Max => x.max(y) as u32,
                BinOp::BitAnd => a & b,
                BinOp::BitOr => a | b,
                BinOp::BitXor => a ^ b,
                BinOp::Shr => (x >> (y & 31)) as u32,
                BinOp::Shl => (x << (y & 31)) as u32,
                BinOp::LogicalAnd => i32::from(x != 0 && y != 0) as u32,
                BinOp::LogicalOr => i32::from(x != 0 || y != 0) as u32,
            }
        }
    }
}

#[inline(always)]
pub(crate) fn apply_cast(from: NumTy, to: NumTy, bits: u32) -> u32 {
    if from == to {
        return bits;
    }
    match (from, to) {
        (NumTy::F32, NumTy::U32) => {
            let value = f32::from_bits(bits);
            if value <= 0.0 || value.is_nan() {
                0
            } else if value >= 4_294_967_296.0 {
                u32::MAX
            } else {
                value as u32
            }
        }
        (NumTy::F32, NumTy::I32) => f32::from_bits(bits) as i32 as u32,
        (NumTy::U32, NumTy::F32) => (bits as f32).to_bits(),
        (NumTy::I32, NumTy::F32) => (bits as i32 as f32).to_bits(),
        _ => bits,
    }
}

#[inline(always)]
pub(crate) fn apply_narrow(to: ScalarElement, bits: u32) -> u32 {
    let value = f32::from_bits(bits);
    match to {
        ScalarElement::F16 => half::f16::from_f32(value).to_f32().to_bits(),
        ScalarElement::BF16 => half::bf16::from_f32(value).to_f32().to_bits(),
        _ => bits,
    }
}

/// Read one element of a storage element type out of raw bytes, widened to a
/// compute register lane.
///
/// # Safety
/// `index` must be inside the buffer `base` points at.
#[inline(always)]
pub(crate) unsafe fn read_elem(elem: ScalarElement, base: *const u8, index: usize) -> u32 {
    unsafe {
        match elem {
            ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool => {
                (base as *const u32).add(index).read_unaligned()
            }
            ScalarElement::F16 => {
                let raw = (base as *const u16).add(index).read_unaligned();
                half::f16::from_bits(raw).to_f32().to_bits()
            }
            ScalarElement::BF16 => {
                let raw = (base as *const u16).add(index).read_unaligned();
                half::bf16::from_bits(raw).to_f32().to_bits()
            }
        }
    }
}

/// Write one compute register lane back into a storage element type.
///
/// # Safety
/// `index` must be inside the buffer `base` points at, and no other thread may
/// be writing the same element (`verify_launch` invariant 3).
#[inline(always)]
pub(crate) unsafe fn write_elem(elem: ScalarElement, base: *mut u8, index: usize, bits: u32) {
    unsafe {
        match elem {
            ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool => {
                (base as *mut u32).add(index).write_unaligned(bits);
            }
            ScalarElement::F16 => {
                let v = half::f16::from_f32(f32::from_bits(bits)).to_bits();
                (base as *mut u16).add(index).write_unaligned(v);
            }
            ScalarElement::BF16 => {
                let v = half::bf16::from_f32(f32::from_bits(bits)).to_bits();
                (base as *mut u16).add(index).write_unaligned(v);
            }
        }
    }
}
