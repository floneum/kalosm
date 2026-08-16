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

use fusor2_ir::dtype::RoundMode;
use fusor2_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor2_ir::scalar::{BinOp, CmpOp, UnOp};

use super::access::AccessForm;

/// A tape register index.
pub type Slot = u32;

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
    Const { out: Slot, bits: u32 },
    /// `chunk_base + [0, 1, ... W-1]`, the lane index inside the block.
    LaneId { out: Slot },
    /// A workgroup-uniform u32 splat resolved at launch.
    Uniform { out: Slot, which: UniformSrc },
    LoadLocal { out: Slot, local: u16 },
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
    Fma { out: Slot, a: Slot, b: Slot, c: Slot },
    Cmp {
        out: Slot,
        op: CmpOp,
        a: Slot,
        b: Slot,
        ty: NumTy,
    },
    /// A lane mask materialized to `1.0`/`0.0` (or `1`/`0`) in `ty`.
    MaskToValue { out: Slot, x: Slot, ty: NumTy },
    /// A value tested against zero, producing a lane mask.
    ValueToMask { out: Slot, x: Slot, ty: NumTy },
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
    Bitcast { out: Slot, x: Slot },
    Select { out: Slot, c: Slot, t: Slot, f: Slot },
    /// `parts[i]` copied into `out + i`.
    VecCompose { out: Slot, parts: Vec<Slot> },
    VecComponent { out: Slot, base: Slot, component: u32 },
    Dot { out: Slot, a: Slot, b: Slot, lanes: u32 },
    Reduce {
        out: Slot,
        op: TileReduceOp,
        x: Slot,
        kind: RKind,
        /// Lane-group base index, for `RKind::TileGroup`.
        group_base: Slot,
    },
    /// Unpack a `u32` of two packed f16s into a 2-lane f32 vector.
    Unpack2x16 { out: Slot, x: Slot },
    /// Run a rank-2 address through the declared divmod chain of
    /// `maps[map]`. Only the divmods the `MultiFlattenMap` declares are
    /// performed, because `divmod_ops()` is the cost term.
    Rc2Index {
        out: Slot,
        row: Slot,
        col: Slot,
        map: u16,
    },
    Copy { out: Slot, x: Slot },
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

/// A `W`-lane register. Bits, so `Bitcast` is free and a mask is `0`/`!0`.
///
/// `W` is a const generic: `[Reg<W>; M]` is a register accumulator tile, and
/// every operation below is a plain elementwise loop over a statically-sized
/// array, which lowers to whole vector instructions under the target features
/// `dispatch!` established.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(align(32))]
pub struct Reg<const W: usize>(pub [u32; W]);

impl<const W: usize> Default for Reg<W> {
    fn default() -> Self {
        Self([0; W])
    }
}

impl<const W: usize> Reg<W> {
    #[inline(always)]
    pub fn splat_bits(bits: u32) -> Self {
        Self([bits; W])
    }
    #[inline(always)]
    pub fn splat_f32(v: f32) -> Self {
        Self([v.to_bits(); W])
    }
    #[inline(always)]
    pub fn splat_u32(v: u32) -> Self {
        Self([v; W])
    }
    #[inline(always)]
    pub fn f(self, i: usize) -> f32 {
        f32::from_bits(self.0[i])
    }
    #[inline(always)]
    pub fn u(self, i: usize) -> u32 {
        self.0[i]
    }
    #[inline(always)]
    pub fn i(self, i: usize) -> i32 {
        self.0[i] as i32
    }
    #[inline(always)]
    pub fn from_f(v: [f32; W]) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = v[k].to_bits();
        }
        Self(o)
    }
    #[inline(always)]
    pub fn to_f(self) -> [f32; W] {
        let mut o = [0f32; W];
        for k in 0..W {
            o[k] = f32::from_bits(self.0[k]);
        }
        o
    }
    /// Elementwise f32 map.
    #[inline(always)]
    pub fn mapf(self, f: impl Fn(f32) -> f32) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = f(f32::from_bits(self.0[k])).to_bits();
        }
        Self(o)
    }
    /// Elementwise f32 zip.
    #[inline(always)]
    pub fn zipf(self, b: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = f(f32::from_bits(self.0[k]), f32::from_bits(b.0[k])).to_bits();
        }
        Self(o)
    }
    #[inline(always)]
    pub fn zipu(self, b: Self, f: impl Fn(u32, u32) -> u32) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = f(self.0[k], b.0[k]);
        }
        Self(o)
    }
    #[inline(always)]
    pub fn zipi(self, b: Self, f: impl Fn(i32, i32) -> i32) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = f(self.0[k] as i32, b.0[k] as i32) as u32;
        }
        Self(o)
    }
    /// Per-lane select on a `0`/`!0` mask.
    #[inline(always)]
    pub fn select(mask: Self, t: Self, f: Self) -> Self {
        let mut o = [0u32; W];
        for k in 0..W {
            o[k] = (t.0[k] & mask.0[k]) | (f.0[k] & !mask.0[k]);
        }
        Self(o)
    }
}

const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const LOG2_E: f32 = 1.442_695_04;
const LN2: f32 = 0.693_147_18;
const PI_2_HI: f32 = 1.570_796_3;
const PI_2_MID: f32 = 7.549_789_4e-8;
const PI_2_LO: f32 = 5.390_302_6e-15;
const TWO_OVER_PI: f32 = 0.636_619_78;

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
pub fn rint(x: f32) -> f32 {
    let t = trunc(x);
    let frac = x - t;
    let a = frac.abs();
    let bump = if a > 0.5 {
        1.0
    } else if a < 0.5 {
        0.0
    } else if (t as i64) % 2 == 0 {
        0.0
    } else {
        1.0
    };
    if x < 0.0 { t - bump } else { t + bump }
}

#[inline(always)]
pub fn trunc(x: f32) -> f32 {
    if !(x.abs() < 8_388_608.0) {
        // |x| >= 2^23 (or NaN): already integral, or not representable.
        return x;
    }
    (x as i32) as f32
}

#[inline(always)]
pub fn floorf(x: f32) -> f32 {
    let t = trunc(x);
    if x < 0.0 && t != x { t - 1.0 } else { t }
}

#[inline(always)]
pub fn ceilf(x: f32) -> f32 {
    let t = trunc(x);
    if x > 0.0 && t != x { t + 1.0 } else { t }
}

#[inline(always)]
pub fn round_half_away(x: f32) -> f32 {
    let t = trunc(x);
    let a = (x - t).abs();
    if a >= 0.5 {
        if x < 0.0 { t - 1.0 } else { t + 1.0 }
    } else {
        t
    }
}

#[inline(always)]
pub fn round_mode(mode: RoundMode, x: f32) -> f32 {
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
    p = p * r + 1.984_126_98e-4; // 1/7!
    p = p * r + 1.388_888_9e-3; // 1/6!
    p = p * r + 8.333_333_3e-3; // 1/5!
    p = p * r + 4.166_666_6e-2; // 1/4!
    p = p * r + 0.166_666_67; // 1/3!
    p = p * r + 0.5;
    p = p * r + 1.0;
    p * r + 1.0
}

/// `e^x`: Cody–Waite reduction plus the degree-8 polynomial above.
#[inline(always)]
pub fn expf(x: f32) -> f32 {
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
pub fn exp2f(x: f32) -> f32 {
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
pub fn log1pf(u: f32) -> f32 {
    if u <= -1.0 {
        return if u == -1.0 { f32::NEG_INFINITY } else { f32::NAN };
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
    if m > 1.414_213_6 {
        m *= 0.5;
        e += 1;
    }
    (m, e)
}

/// Natural log by mantissa/exponent split plus a degree-13 odd polynomial in
/// `s = (m-1)/(m+1)`.
#[inline(always)]
pub fn logf(x: f32) -> f32 {
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
pub fn log2f(x: f32) -> f32 {
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
pub fn powf(x: f32, y: f32) -> f32 {
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
    p = p * z - 1.984_126_98e-4; // -1/7!
    p = p * z + 8.333_333_3e-3; // 1/5!
    p = p * z - 0.166_666_67; // -1/3!
    r + r * z * p
}

/// `cos(r)` for `|r| <= pi/4`, degree 10.
#[inline(always)]
fn cos_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = -2.755_731_9e-7; // -1/10!
    p = p * z + 2.480_158_7e-5; // 1/8!
    p = p * z - 1.388_888_9e-3; // -1/6!
    p = p * z + 4.166_666_6e-2; // 1/4!
    p = p * z - 0.5;
    1.0 + z * p
}

#[inline(always)]
pub fn sinf(x: f32) -> f32 {
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
pub fn cosf(x: f32) -> f32 {
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
pub fn tanf(x: f32) -> f32 {
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
pub fn tanhf(x: f32) -> f32 {
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
pub fn sqrtf(x: f32) -> f32 {
    // A single hardware instruction, not a libm call.
    x.sqrt()
}

#[inline(always)]
pub fn rsqrtf(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Cephes single-precision `asin`, with the exact argument reduction at 0.5.
#[inline(always)]
pub fn asinf(x: f32) -> f32 {
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
    let mut p = 4.216_319_9e-2;
    p = p * z + 2.418_131_1e-2;
    p = p * z + 4.547_002_6e-2;
    p = p * z + 7.495_300_3e-2;
    p = p * z + 0.166_667_52;
    let r = p * z * base + base;
    let r = if flag { PI_2_HI + PI_2_MID - 2.0 * r } else { r };
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub fn acosf(x: f32) -> f32 {
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
pub fn atanf(x: f32) -> f32 {
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
pub fn sinhf(x: f32) -> f32 {
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
pub fn coshf(x: f32) -> f32 {
    let e = expf(x.abs());
    0.5 * (e + 1.0 / e)
}

#[inline(always)]
pub fn asinhf(x: f32) -> f32 {
    let a = x.abs();
    let r = if a < 1.0 {
        log1pf(a + a * a / (1.0 + sqrtf(1.0 + a * a)))
    } else {
        logf(a + sqrtf(a * a + 1.0))
    };
    if x < 0.0 { -r } else { r }
}

#[inline(always)]
pub fn acoshf(x: f32) -> f32 {
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
pub fn atanhf(x: f32) -> f32 {
    let a = x.abs();
    if a >= 1.0 {
        return if a == 1.0 {
            if x > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        };
    }
    let r = 0.5 * log1pf(2.0 * a / (1.0 - a));
    if x < 0.0 { -r } else { r }
}

/// Apply a unary op elementwise across a register.
#[inline(always)]
pub fn apply_un<const W: usize>(op: UnOp, ty: NumTy, x: Reg<W>) -> Reg<W> {
    match (op, ty) {
        (UnOp::Neg, NumTy::F32) => Reg(core::array::from_fn(|k| x.0[k] ^ 0x8000_0000)),
        (UnOp::Abs, NumTy::F32) => Reg(core::array::from_fn(|k| x.0[k] & 0x7FFF_FFFF)),
        (UnOp::Neg, NumTy::I32) => Reg(core::array::from_fn(|k| x.i(k).wrapping_neg() as u32)),
        (UnOp::Abs, NumTy::I32) => Reg(core::array::from_fn(|k| x.i(k).wrapping_abs() as u32)),
        (UnOp::Neg, NumTy::U32) => Reg(core::array::from_fn(|k| x.0[k].wrapping_neg())),
        (UnOp::Abs, NumTy::U32) => x,
        // `Unpack2x16Float` has its own instruction: its result is a 2-lane
        // vector, not a scalar register.
        (UnOp::Unpack2x16Float, _) => x,
        (op, _) => x.mapf(|v| match op {
            // Both approximate exponentials lower to the exact `exp`; the
            // relaxed contract permits the substitution.
            UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => expf(v),
            UnOp::Exp2 => exp2f(v),
            UnOp::Log => logf(v),
            UnOp::Log2 => log2f(v),
            UnOp::Sqrt => sqrtf(v),
            UnOp::InverseSqrt => rsqrtf(v),
            UnOp::Sin => sinf(v),
            UnOp::Cos => cosf(v),
            UnOp::Tan => tanf(v),
            UnOp::Tanh => tanhf(v),
            UnOp::Asin => asinf(v),
            UnOp::Acos => acosf(v),
            UnOp::Atan => atanf(v),
            UnOp::Sinh => sinhf(v),
            UnOp::Cosh => coshf(v),
            UnOp::Asinh => asinhf(v),
            UnOp::Acosh => acoshf(v),
            UnOp::Atanh => atanhf(v),
            UnOp::Abs => v.abs(),
            UnOp::Neg => -v,
            UnOp::Unpack2x16Float => v,
        }),
    }
}

/// Apply a binary op elementwise across two registers.
#[inline(always)]
pub fn apply_bin<const W: usize>(op: BinOp, ty: NumTy, a: Reg<W>, b: Reg<W>) -> Reg<W> {
    match ty {
        NumTy::F32 => match op {
            BinOp::Add => a.zipf(b, |x, y| x + y),
            BinOp::Sub => a.zipf(b, |x, y| x - y),
            BinOp::Mul => a.zipf(b, |x, y| x * y),
            BinOp::Div => a.zipf(b, |x, y| x / y),
            BinOp::Rem => a.zipf(b, |x, y| x - trunc(x / y) * y),
            BinOp::Pow => a.zipf(b, powf),
            BinOp::Min => a.zipf(b, |x, y| if y < x { y } else { x }),
            BinOp::Max => a.zipf(b, |x, y| if y > x { y } else { x }),
            BinOp::BitAnd => a.zipu(b, |x, y| x & y),
            BinOp::BitOr => a.zipu(b, |x, y| x | y),
            BinOp::BitXor => a.zipu(b, |x, y| x ^ y),
            BinOp::Shr | BinOp::Shl => a,
            BinOp::LogicalAnd => a.zipf(b, |x, y| f32::from(x != 0.0 && y != 0.0)),
            BinOp::LogicalOr => a.zipf(b, |x, y| f32::from(x != 0.0 || y != 0.0)),
        },
        NumTy::U32 => match op {
            BinOp::Add => a.zipu(b, u32::wrapping_add),
            BinOp::Sub => a.zipu(b, u32::wrapping_sub),
            BinOp::Mul => a.zipu(b, u32::wrapping_mul),
            BinOp::Div => a.zipu(b, |x, y| if y == 0 { u32::MAX } else { x / y }),
            BinOp::Rem => a.zipu(b, |x, y| if y == 0 { 0 } else { x % y }),
            BinOp::Pow => a.zipu(b, u32::wrapping_pow),
            BinOp::Min => a.zipu(b, u32::min),
            BinOp::Max => a.zipu(b, u32::max),
            BinOp::BitAnd => a.zipu(b, |x, y| x & y),
            BinOp::BitOr => a.zipu(b, |x, y| x | y),
            BinOp::BitXor => a.zipu(b, |x, y| x ^ y),
            BinOp::Shr => a.zipu(b, |x, y| x >> (y & 31)),
            BinOp::Shl => a.zipu(b, |x, y| x << (y & 31)),
            BinOp::LogicalAnd => a.zipu(b, |x, y| u32::from(x != 0 && y != 0)),
            BinOp::LogicalOr => a.zipu(b, |x, y| u32::from(x != 0 || y != 0)),
        },
        NumTy::I32 => match op {
            BinOp::Add => a.zipi(b, i32::wrapping_add),
            BinOp::Sub => a.zipi(b, i32::wrapping_sub),
            BinOp::Mul => a.zipi(b, i32::wrapping_mul),
            BinOp::Div => a.zipi(b, |x, y| if y == 0 { -1 } else { x.wrapping_div(y) }),
            BinOp::Rem => a.zipi(b, |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) }),
            BinOp::Pow => a.zipi(b, |x, y| x.wrapping_pow(y.max(0) as u32)),
            BinOp::Min => a.zipi(b, i32::min),
            BinOp::Max => a.zipi(b, i32::max),
            BinOp::BitAnd => a.zipu(b, |x, y| x & y),
            BinOp::BitOr => a.zipu(b, |x, y| x | y),
            BinOp::BitXor => a.zipu(b, |x, y| x ^ y),
            BinOp::Shr => a.zipi(b, |x, y| x >> (y & 31)),
            BinOp::Shl => a.zipi(b, |x, y| x << (y & 31)),
            BinOp::LogicalAnd => a.zipi(b, |x, y| i32::from(x != 0 && y != 0)),
            BinOp::LogicalOr => a.zipi(b, |x, y| i32::from(x != 0 || y != 0)),
        },
    }
}

/// All six comparisons, producing a lane mask. Materialization to 1.0/0.0 is a
/// separate instruction, minted only when the mask is consumed as a value.
#[inline(always)]
pub fn apply_cmp<const W: usize>(op: CmpOp, ty: NumTy, a: Reg<W>, b: Reg<W>) -> Reg<W> {
    let mut o = [0u32; W];
    for k in 0..W {
        let t = match ty {
            NumTy::F32 => {
                let (x, y) = (a.f(k), b.f(k));
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                }
            }
            NumTy::U32 => {
                let (x, y) = (a.u(k), b.u(k));
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                }
            }
            NumTy::I32 => {
                let (x, y) = (a.i(k), b.i(k));
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                }
            }
        };
        o[k] = if t { u32::MAX } else { 0 };
    }
    Reg(o)
}

/// Numeric conversion between compute types.
#[inline(always)]
pub fn apply_cast<const W: usize>(from: NumTy, to: NumTy, x: Reg<W>) -> Reg<W> {
    if from == to {
        return x;
    }
    let mut o = [0u32; W];
    for k in 0..W {
        o[k] = match (from, to) {
            (NumTy::F32, NumTy::U32) => {
                let v = x.f(k);
                if v <= 0.0 || v.is_nan() {
                    0
                } else if v >= 4_294_967_296.0 {
                    u32::MAX
                } else {
                    v as u32
                }
            }
            (NumTy::F32, NumTy::I32) => (x.f(k) as i32) as u32,
            (NumTy::U32, NumTy::F32) => (x.u(k) as f32).to_bits(),
            (NumTy::I32, NumTy::F32) => (x.i(k) as f32).to_bits(),
            _ => x.0[k],
        };
    }
    Reg(o)
}

/// Narrow to a storage element and widen back — a `Cast` to `F16`/`BF16` when
/// the register file only holds f32.
#[inline(always)]
pub fn apply_narrow<const W: usize>(to: ScalarElement, x: Reg<W>) -> Reg<W> {
    match to {
        ScalarElement::F16 => x.mapf(|v| half::f16::from_f32(v).to_f32()),
        ScalarElement::BF16 => x.mapf(|v| half::bf16::from_f32(v).to_f32()),
        _ => x,
    }
}

/// Read one element of a storage element type out of raw bytes, widened to a
/// compute register lane.
///
/// # Safety
/// `index` must be inside the buffer `base` points at.
#[inline(always)]
pub unsafe fn read_elem(elem: ScalarElement, base: *const u8, index: usize) -> u32 {
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
pub unsafe fn write_elem(elem: ScalarElement, base: *mut u8, index: usize, bits: u32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f32, b: f32) -> f32 {
        if a == b {
            return 0.0;
        }
        (a - b).abs() / b.abs().max(1.0)
    }

    #[test]
    fn transcendentals_match_libm() {
        let n = 4096usize;
        let mut worst: Vec<(&str, f32)> = Vec::new();
        macro_rules! sweep {
            ($name:literal, $lo:expr, $hi:expr, $ours:expr, $ref:expr) => {{
                let mut w = 0f32;
                for i in 0..n {
                    let x = $lo + ($hi - $lo) * (i as f32) / ((n - 1) as f32);
                    let got: f32 = ($ours)(x);
                    let want: f32 = ($ref)(x as f64) as f32;
                    w = w.max(rel(got, want));
                }
                worst.push(($name, w));
            }};
        }
        sweep!("exp", -20.0f32, 20.0f32, expf, |x: f64| x.exp());
        sweep!("exp2", -30.0f32, 30.0f32, exp2f, |x: f64| x.exp2());
        sweep!("log", 1e-6f32, 1e6f32, logf, |x: f64| x.ln());
        sweep!("log2", 1e-6f32, 1e6f32, log2f, |x: f64| x.log2());
        sweep!("sqrt", 0.0f32, 1e6f32, sqrtf, |x: f64| x.sqrt());
        sweep!("inverse_sqrt", 1e-3f32, 1e3f32, rsqrtf, |x: f64| 1.0 / x.sqrt());
        sweep!("sin", -12.0f32, 12.0f32, sinf, |x: f64| x.sin());
        sweep!("cos", -12.0f32, 12.0f32, cosf, |x: f64| x.cos());
        sweep!("tan", -1.4f32, 1.4f32, tanf, |x: f64| x.tan());
        sweep!("tanh", -12.0f32, 12.0f32, tanhf, |x: f64| x.tanh());
        sweep!("asin", -0.999f32, 0.999f32, asinf, |x: f64| x.asin());
        sweep!("acos", -0.999f32, 0.999f32, acosf, |x: f64| x.acos());
        sweep!("atan", -50.0f32, 50.0f32, atanf, |x: f64| x.atan());
        sweep!("sinh", -10.0f32, 10.0f32, sinhf, |x: f64| x.sinh());
        sweep!("cosh", -10.0f32, 10.0f32, coshf, |x: f64| x.cosh());
        sweep!("asinh", -50.0f32, 50.0f32, asinhf, |x: f64| x.asinh());
        sweep!("acosh", 1.0001f32, 50.0f32, acoshf, |x: f64| x.acosh());
        sweep!("atanh", -0.995f32, 0.995f32, atanhf, |x: f64| x.atanh());
        sweep!("abs", -10.0f32, 10.0f32, f32::abs, |x: f64| x.abs());
        sweep!("neg", -10.0f32, 10.0f32, |x: f32| -x, |x: f64| -x);
        sweep!("pow", 0.01f32, 20.0f32, |x: f32| powf(x, 1.7), |x: f64| x
            .powf(1.7));

        for (name, e) in &worst {
            assert!(*e <= 1e-6, "{name}: max relative error {e:e} exceeds 1e-6");
        }
        assert_eq!(worst.len(), 21);
    }

    #[test]
    fn round_modes_are_exact() {
        assert_eq!(round_half_away(0.5), 1.0);
        assert_eq!(round_half_away(-0.5), -1.0);
        assert_eq!(round_half_away(2.5), 3.0);
        assert_eq!(round_half_away(-2.5), -3.0);
        assert_eq!(rint(0.5), 0.0);
        assert_eq!(rint(1.5), 2.0);
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(floorf(-1.2), -2.0);
        assert_eq!(ceilf(-1.2), -1.0);
        assert_eq!(trunc(-1.9), -1.0);
        // Ties away from zero at every half-integer in [-8, 8]: what MSQ1
        // export idempotence depends on.
        for i in -16i32..=16 {
            if i % 2 == 0 {
                continue;
            }
            let x = i as f32 * 0.5;
            let r = round_half_away(x);
            assert_eq!(r.abs(), x.abs() + 0.5, "round({x}) = {r}");
            assert_eq!(r.signum(), x.signum());
        }
    }

    #[test]
    fn comparisons_vectorize() {
        let a: Reg<8> = Reg::from_f([1.0, -2.0, 3.5, 0.0, -0.0, 7.0, 1e9, -1e9]);
        let b: Reg<8> = Reg::from_f([1.0, 2.0, -3.5, 0.0, 0.0, 6.0, 1e9, 1e9]);
        let m = apply_cmp(CmpOp::Lt, NumTy::F32, a, b);
        let want = [false, true, false, false, false, false, false, true];
        for k in 0..8 {
            assert_eq!(m.0[k] != 0, want[k], "lane {k}");
        }
    }

    #[test]
    fn f16_bf16_widen_to_f32_registers() {
        let x: Reg<4> = Reg::from_f([1.5, -0.25, 3.0, 100.0]);
        let n = apply_narrow(ScalarElement::F16, x);
        for k in 0..4 {
            assert_eq!(n.f(k), half::f16::from_f32(x.f(k)).to_f32());
        }
        let n = apply_narrow(ScalarElement::BF16, x);
        for k in 0..4 {
            assert_eq!(n.f(k), half::bf16::from_f32(x.f(k)).to_f32());
        }
        assert_eq!(NumTy::of(ScalarElement::F16), NumTy::F32);
        assert_eq!(NumTy::of(ScalarElement::BF16), NumTy::F32);
    }
}
