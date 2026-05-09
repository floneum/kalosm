use std::{
    fmt,
    ops::{Index, Range, RangeInclusive},
    sync::Arc,
};

use crate::Device;

const DEFAULT_GENERATED_DIM_MAX: usize = 8192;
const DIM_SAMPLE_ATTEMPTS: usize = 128;
const SHAPE_SAMPLE_ATTEMPTS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis<const I: usize>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelShape<const DIMS: usize> {
    dims: [usize; DIMS],
}

impl<const DIMS: usize> KernelShape<DIMS> {
    pub const fn new(dims: [usize; DIMS]) -> Self {
        Self { dims }
    }

    pub const fn dims(&self) -> [usize; DIMS] {
        self.dims
    }

    pub fn get<const I: usize>(&self, _axis: Axis<I>) -> usize {
        assert!(I < DIMS, "axis index {I} is outside KernelShape<{DIMS}>");
        self.dims[I]
    }
}

impl<const DIMS: usize, const I: usize> Index<Axis<I>> for KernelShape<DIMS> {
    type Output = usize;

    fn index(&self, axis: Axis<I>) -> &Self::Output {
        assert!(I < DIMS, "axis index {I} is outside KernelShape<{DIMS}>");
        &self.dims[axis_index(axis)]
    }
}

const fn axis_index<const I: usize>(_axis: Axis<I>) -> usize {
    I
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelDeviceCaps {
    pub subgroups_supported: bool,
    pub cooperative_matrix_supported: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl KernelDeviceCaps {
    pub fn from_device(device: &Device) -> Self {
        let limits = device.limits();
        Self {
            subgroups_supported: device.subgroups_supported(),
            cooperative_matrix_supported: device.cooperative_matrix_supported(),
            min_subgroup_size: device.min_subgroup_size(),
            max_subgroup_size: device.max_subgroup_size(),
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

pub trait ShapeRng {
    fn next_u64(&mut self) -> u64;

    fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            0
        } else {
            (self.next_u64() % upper_exclusive as u64) as usize
        }
    }

    fn next_inclusive(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            min
        } else if max - min == usize::MAX {
            self.next_u64() as usize
        } else {
            min + self.next_usize(max - min + 1)
        }
    }
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct DeterministicShapeRng(u64);

#[cfg(test)]
impl ShapeRng for DeterministicShapeRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
}

#[cfg(test)]
pub(crate) fn assert_selector_generates<const DIMS: usize, Ctx, Variant>(
    selector: &ShapeSelector<DIMS, Ctx, Variant>,
    cases: impl IntoIterator<Item = (Variant, Ctx, KernelDeviceCaps)>,
) where
    Ctx: Copy,
    Variant: Copy + PartialEq + fmt::Debug,
{
    let mut rng = DeterministicShapeRng::default();
    for (variant, ctx, caps) in cases {
        let shape = selector
            .generate_for(variant, &ctx, caps, &mut rng)
            .expect("variant should generate");
        assert_eq!(selector.select(shape, &ctx, caps), Some(variant));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimConstraint {
    Any,
    Eq(usize),
    Range { min: usize, max: usize },
    Choices(Vec<usize>),
    MultipleOf(usize),
    And(Vec<DimConstraint>),
    Or(Vec<DimConstraint>),
}

impl DimConstraint {
    pub fn matches(&self, value: usize) -> bool {
        match self {
            Self::Any => true,
            Self::Eq(expected) => value == *expected,
            Self::Range { min, max } => (*min..=*max).contains(&value),
            Self::Choices(choices) => choices.contains(&value),
            Self::MultipleOf(divisor) => value.is_multiple_of(*divisor),
            Self::And(parts) => parts.iter().all(|part| part.matches(value)),
            Self::Or(parts) => parts.iter().any(|part| part.matches(value)),
        }
    }

    pub fn and(self, other: DimConstraint) -> DimConstraint {
        match (self, other) {
            (DimConstraint::And(mut left), DimConstraint::And(right)) => {
                left.extend(right);
                DimConstraint::And(left)
            }
            (DimConstraint::And(mut left), right) => {
                left.push(right);
                DimConstraint::And(left)
            }
            (left, DimConstraint::And(mut right)) => {
                right.insert(0, left);
                DimConstraint::And(right)
            }
            (left, right) => DimConstraint::And(vec![left, right]),
        }
    }

    pub fn or(self, other: DimConstraint) -> DimConstraint {
        match (self, other) {
            (DimConstraint::Or(mut left), DimConstraint::Or(right)) => {
                left.extend(right);
                DimConstraint::Or(left)
            }
            (DimConstraint::Or(mut left), right) => {
                left.push(right);
                DimConstraint::Or(left)
            }
            (left, DimConstraint::Or(mut right)) => {
                right.insert(0, left);
                DimConstraint::Or(right)
            }
            (left, right) => DimConstraint::Or(vec![left, right]),
        }
    }

    fn sample(&self, rng: &mut impl ShapeRng) -> Option<usize> {
        match self {
            Self::Or(parts) => sample_or(parts, rng),
            _ => {
                let hints = DimHints::from_constraint(self)?;
                sample_from_hints(self, hints, rng)
            }
        }
    }
}

pub fn any() -> DimConstraint {
    DimConstraint::Any
}

pub fn eq(value: usize) -> DimConstraint {
    DimConstraint::Eq(value)
}

pub fn range(bounds: impl Into<DimRange>) -> DimConstraint {
    let range = bounds.into();
    DimConstraint::Range {
        min: range.min,
        max: range.max,
    }
}

pub fn choices(values: impl Into<Vec<usize>>) -> DimConstraint {
    DimConstraint::Choices(values.into())
}

pub fn multiple_of(divisor: usize) -> DimConstraint {
    assert!(divisor > 0, "multiple_of requires a non-zero divisor");
    DimConstraint::MultipleOf(divisor)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimRange {
    min: usize,
    max: usize,
}

impl From<RangeInclusive<usize>> for DimRange {
    fn from(value: RangeInclusive<usize>) -> Self {
        let (min, max) = value.into_inner();
        Self { min, max }
    }
}

impl From<Range<usize>> for DimRange {
    fn from(value: Range<usize>) -> Self {
        if value.start >= value.end {
            return Self { min: 1, max: 0 };
        }
        Self {
            min: value.start,
            max: value.end - 1,
        }
    }
}

type RulePredicate<const DIMS: usize, Ctx> =
    Arc<dyn Fn(KernelShape<DIMS>, &Ctx, KernelDeviceCaps) -> bool + Send + Sync + 'static>;

pub struct ShapeRule<const DIMS: usize, Ctx> {
    axes: [Option<DimConstraint>; DIMS],
    predicates: Vec<RulePredicate<DIMS, Ctx>>,
}

impl<const DIMS: usize, Ctx> fmt::Debug for ShapeRule<DIMS, Ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShapeRule")
            .field("axes", &self.axes)
            .field("predicates", &self.predicates.len())
            .finish()
    }
}

impl<const DIMS: usize, Ctx> Default for ShapeRule<DIMS, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const DIMS: usize, Ctx> ShapeRule<DIMS, Ctx> {
    pub fn new() -> Self {
        Self {
            axes: std::array::from_fn(|_| None),
            predicates: Vec::new(),
        }
    }

    pub fn axis<const I: usize>(mut self, axis: Axis<I>, constraint: DimConstraint) -> Self {
        assert!(I < DIMS, "axis index {I} is outside ShapeRule<{DIMS}>");
        self.axes[axis_index(axis)] = Some(match self.axes[axis_index(axis)].take() {
            Some(existing) => existing.and(constraint),
            None => constraint,
        });
        self
    }

    pub fn when(
        mut self,
        predicate: impl Fn(KernelShape<DIMS>, &Ctx, KernelDeviceCaps) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.predicates.push(Arc::new(predicate));
        self
    }

    pub fn when_ctx(self, predicate: impl Fn(&Ctx) -> bool + Send + Sync + 'static) -> Self {
        self.when(move |_shape, ctx, _caps| predicate(ctx))
    }

    pub fn when_caps(
        self,
        predicate: impl Fn(KernelDeviceCaps) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.when(move |_shape, _ctx, caps| predicate(caps))
    }

    pub fn matches(&self, shape: KernelShape<DIMS>, ctx: &Ctx, caps: KernelDeviceCaps) -> bool {
        self.axes.iter().zip(shape.dims).all(|(constraint, dim)| {
            constraint
                .as_ref()
                .is_none_or(|constraint| constraint.matches(dim))
        }) && self
            .predicates
            .iter()
            .all(|predicate| predicate(shape, ctx, caps))
    }

    fn sample(&self, rng: &mut impl ShapeRng) -> Option<KernelShape<DIMS>> {
        let dims = self
            .axes
            .iter()
            .map(|constraint| {
                constraint
                    .as_ref()
                    .unwrap_or(&DimConstraint::Any)
                    .sample(rng)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(KernelShape::new(dims.try_into().ok()?))
    }
}

#[derive(Clone)]
struct SelectorRule<const DIMS: usize, Ctx, Variant> {
    variant: Variant,
    rule: Arc<ShapeRule<DIMS, Ctx>>,
}

pub struct ShapeSelector<const DIMS: usize, Ctx, Variant> {
    rules: Vec<SelectorRule<DIMS, Ctx, Variant>>,
}

impl<const DIMS: usize, Ctx, Variant> Default for ShapeSelector<DIMS, Ctx, Variant> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const DIMS: usize, Ctx, Variant> ShapeSelector<DIMS, Ctx, Variant> {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn rule(mut self, variant: Variant, rule: ShapeRule<DIMS, Ctx>) -> Self {
        self.rules.push(SelectorRule {
            variant,
            rule: Arc::new(rule),
        });
        self
    }

    pub fn variants(&self) -> Vec<Variant>
    where
        Variant: Copy + PartialEq,
    {
        let mut variants = Vec::new();
        for rule in &self.rules {
            if !variants.contains(&rule.variant) {
                variants.push(rule.variant);
            }
        }
        variants
    }

    pub fn select(
        &self,
        shape: KernelShape<DIMS>,
        ctx: &Ctx,
        caps: KernelDeviceCaps,
    ) -> Option<Variant>
    where
        Variant: Copy,
    {
        self.rules
            .iter()
            .find(|rule| rule.rule.matches(shape, ctx, caps))
            .map(|rule| rule.variant)
    }

    pub fn generate_for(
        &self,
        variant: Variant,
        ctx: &Ctx,
        caps: KernelDeviceCaps,
        rng: &mut impl ShapeRng,
    ) -> Option<KernelShape<DIMS>>
    where
        Variant: Copy + PartialEq,
    {
        for rule in self.rules.iter().filter(|rule| rule.variant == variant) {
            for _ in 0..SHAPE_SAMPLE_ATTEMPTS {
                let Some(shape) = rule.rule.sample(rng) else {
                    break;
                };
                if self.select(shape, ctx, caps) == Some(variant) {
                    return Some(shape);
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
struct DimHints {
    min: Option<usize>,
    max: Option<usize>,
    exact: Option<usize>,
    choices: Option<Vec<usize>>,
    multiple_of: Option<usize>,
}

impl DimHints {
    fn new() -> Self {
        Self {
            min: None,
            max: None,
            exact: None,
            choices: None,
            multiple_of: None,
        }
    }

    fn from_constraint(constraint: &DimConstraint) -> Option<Self> {
        let mut hints = Self::new();
        hints.apply(constraint)?;
        Some(hints)
    }

    fn apply(&mut self, constraint: &DimConstraint) -> Option<()> {
        match constraint {
            DimConstraint::Any => {}
            DimConstraint::Eq(value) => {
                self.restrict_exact(*value)?;
            }
            DimConstraint::Range { min, max } => {
                self.min = Some(self.min.map_or(*min, |existing| existing.max(*min)));
                self.max = Some(self.max.map_or(*max, |existing| existing.min(*max)));
                if self.min? > self.max? {
                    return None;
                }
            }
            DimConstraint::Choices(choices) => {
                if choices.is_empty() {
                    return None;
                }
                let mut choices = choices.clone();
                choices.sort_unstable();
                choices.dedup();
                if let Some(existing) = self.choices.as_ref() {
                    choices.retain(|choice| existing.contains(choice));
                }
                if choices.is_empty() {
                    return None;
                }
                self.choices = Some(choices);
            }
            DimConstraint::MultipleOf(divisor) => {
                self.multiple_of = Some(match self.multiple_of {
                    Some(existing) => lcm(existing, *divisor),
                    None => *divisor,
                });
            }
            DimConstraint::And(parts) => {
                for part in parts {
                    self.apply(part)?;
                }
            }
            DimConstraint::Or(_) => return None,
        }
        Some(())
    }

    fn restrict_exact(&mut self, value: usize) -> Option<()> {
        if self.exact.is_some_and(|exact| exact != value) {
            return None;
        }
        self.exact = Some(value);
        self.min = Some(self.min.map_or(value, |existing| existing.max(value)));
        self.max = Some(self.max.map_or(value, |existing| existing.min(value)));
        Some(())
    }
}

fn sample_or(parts: &[DimConstraint], rng: &mut impl ShapeRng) -> Option<usize> {
    if parts.is_empty() {
        return None;
    }
    let start = rng.next_usize(parts.len());
    for offset in 0..parts.len() {
        let part = &parts[(start + offset) % parts.len()];
        if let Some(value) = part.sample(rng)
            && part.matches(value)
        {
            return Some(value);
        }
    }
    None
}

fn sample_from_hints(
    constraint: &DimConstraint,
    mut hints: DimHints,
    rng: &mut impl ShapeRng,
) -> Option<usize> {
    if let Some(choices) = hints.choices.take() {
        let start = rng.next_usize(choices.len());
        for offset in 0..choices.len() {
            let value = choices[(start + offset) % choices.len()];
            if constraint.matches(value) {
                return Some(value);
            }
        }
        return None;
    }

    if let Some(value) = hints.exact
        && constraint.matches(value)
    {
        return Some(value);
    }

    let min = hints.min.unwrap_or(1);
    let max = hints.max.unwrap_or(DEFAULT_GENERATED_DIM_MAX);
    if min > max {
        return None;
    }

    if let Some(divisor) = hints.multiple_of {
        let first = round_up_to_multiple(min, divisor)?;
        if first > max {
            return None;
        }
        let count = (max - first) / divisor + 1;
        for _ in 0..DIM_SAMPLE_ATTEMPTS {
            let value = first + rng.next_usize(count) * divisor;
            if constraint.matches(value) {
                return Some(value);
            }
        }
        for offset in 0..count.min(DIM_SAMPLE_ATTEMPTS) {
            let value = first + offset * divisor;
            if constraint.matches(value) {
                return Some(value);
            }
        }
        return None;
    }

    for _ in 0..DIM_SAMPLE_ATTEMPTS {
        let value = rng.next_inclusive(min, max);
        if constraint.matches(value) {
            return Some(value);
        }
    }
    for value in min..=max.min(min.saturating_add(DIM_SAMPLE_ATTEMPTS - 1)) {
        if constraint.matches(value) {
            return Some(value);
        }
    }
    None
}

fn round_up_to_multiple(value: usize, divisor: usize) -> Option<usize> {
    let remainder = value % divisor;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(divisor - remainder)
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis<0> = Axis;
    const B: Axis<1> = Axis;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Variant {
        First,
        Second,
        Predicate,
        Impossible,
    }

    #[derive(Clone, Copy)]
    struct Ctx {
        enabled: bool,
    }

    fn caps() -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: true,
            cooperative_matrix_supported: true,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_storage_size: 64 * 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }

    #[test]
    fn selection_is_first_match() {
        let selector = ShapeSelector::<2, (), Variant>::new()
            .rule(Variant::First, ShapeRule::new().axis(A, range(1..=8)))
            .rule(Variant::Second, ShapeRule::new().axis(A, eq(4)));

        assert_eq!(
            selector.select(KernelShape::new([4, 1]), &(), caps()),
            Some(Variant::First)
        );
    }

    #[test]
    fn typed_predicates_participate_in_selection() {
        let selector = ShapeSelector::<2, Ctx, Variant>::new()
            .rule(
                Variant::Predicate,
                ShapeRule::new()
                    .axis(A, multiple_of(4))
                    .axis(B, choices([16, 32, 64]))
                    .when_ctx(|ctx: &Ctx| ctx.enabled)
                    .when_caps(|caps| caps.max_compute_invocations_per_workgroup >= 256),
            )
            .rule(Variant::Second, ShapeRule::new());

        assert_eq!(
            selector.select(KernelShape::new([8, 32]), &Ctx { enabled: true }, caps()),
            Some(Variant::Predicate)
        );
        assert_eq!(
            selector.select(KernelShape::new([8, 32]), &Ctx { enabled: false }, caps()),
            Some(Variant::Second)
        );
    }

    #[test]
    fn impossible_generation_returns_none() {
        let selector = ShapeSelector::<2, (), Variant>::new().rule(
            Variant::Impossible,
            ShapeRule::new().axis(A, eq(3).and(eq(4))),
        );

        assert!(
            selector
                .generate_for(
                    Variant::Impossible,
                    &(),
                    caps(),
                    &mut DeterministicShapeRng::default()
                )
                .is_none()
        );
    }

    #[test]
    fn generated_shape_revalidates_against_ordered_selector() {
        let selector = ShapeSelector::<2, (), Variant>::new()
            .rule(Variant::First, ShapeRule::new().axis(A, eq(1)))
            .rule(
                Variant::Second,
                ShapeRule::new()
                    .axis(A, multiple_of(8).and(range(8..=128)))
                    .axis(B, range(1..=16)),
            );

        let mut rng = DeterministicShapeRng::default();
        let shape = selector
            .generate_for(Variant::Second, &(), caps(), &mut rng)
            .unwrap();
        assert_eq!(selector.select(shape, &(), caps()), Some(Variant::Second));
        assert!(shape[A].is_multiple_of(8));
    }

    #[test]
    fn generated_shape_that_would_match_earlier_rule_is_rejected() {
        let selector = ShapeSelector::<2, (), Variant>::new()
            .rule(Variant::First, ShapeRule::new())
            .rule(Variant::Second, ShapeRule::new().axis(A, eq(4)));

        assert!(
            selector
                .generate_for(
                    Variant::Second,
                    &(),
                    caps(),
                    &mut DeterministicShapeRng::default()
                )
                .is_none()
        );
    }

    #[test]
    fn variants_are_unique_in_rule_order() {
        let selector = ShapeSelector::<2, (), Variant>::new()
            .rule(Variant::First, ShapeRule::new().axis(A, eq(1)))
            .rule(Variant::Second, ShapeRule::new().axis(A, eq(2)))
            .rule(Variant::First, ShapeRule::new().axis(A, eq(3)));

        assert_eq!(selector.variants(), vec![Variant::First, Variant::Second]);
    }
}
