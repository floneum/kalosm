//! `FORM_SLAB`: a chain of launches that all keep a leading prefix of their
//! output rows independent becomes one dispatch, one workgroup per slab of
//! that prefix, with a barrier between stages. Additive: the extractor picks
//! by cost; this rule only proves locality ([`slab_local`]).

use crate::egraph::{Builder, ClassId, Facts, Id, RuleTag};
use crate::ir::launch::{IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

rule!(
    FORM_SLAB,
    level = Level::Launch,
    heads = [OpTag::LaunchMap, OpTag::LaunchFold],
    tag = RuleTag::Additive,
    apply = form_slab,
);

/// Most members one slab carries; every launch in a chain mints a slab of
/// everything before it, so this bounds quadratic growth.
const MAX_MEMBERS: usize = 512;

/// Fewest slabs worth a dispatch of their own.
const MIN_SLABS: u64 = 32;

pub fn form_slab(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    // The CPU pays nothing per dispatch; a slab is a GPU shape.
    if b.caps().kind != crate::device::DeviceKind::Gpu {
        return None;
    }
    let Op::Launch(op) = &node.op else {
        return None;
    };
    let ops = stage(op)?.ops;
    if is_contraction(b, id) {
        return None;
    }

    // Producers first: the launch spelling of each operand, or the longest
    // slab already ending there.
    let mut producers = Producers::default();
    let mut members: Vec<Id> = Vec::new();
    for o in ops {
        // A value every slab reads the same way cannot be a stage: each
        // workgroup writes only its own slab.
        if !varies(o, op) {
            continue;
        }
        if let Some(p) = producers.get(b, o.src) {
            match &b.node(p).op {
                Op::Launch(Launch::Slab { members: ms, .. }) => members.extend(ms.iter().copied()),
                _ => members.push(p),
            }
        }
    }
    if members.is_empty() {
        return None;
    }
    members.push(id);

    // One member per class, first spelling wins.
    let mut seen: FxHashSet<ClassId> = FxHashSet::default();
    members.retain(|m| seen.insert(b.class_of(*m)));

    // Two root chains sharing a stage (an optimizer's moments and its
    // parameter) become one kernel. Roots only: merged forward chains grow
    // past what one kernel should run.
    let root_classes: FxHashSet<ClassId> = b.roots().iter().map(|r| b.class_of(*r)).collect();
    let head_is_root = root_classes.contains(&b.class_of(id));
    for _ in 0..4 {
        if !head_is_root {
            break;
        }
        let mut added: Vec<Id> = Vec::new();
        for m in members.clone() {
            for r in b.readers_of(m) {
                if r == id || b.class_of(r) == b.class_of(id) {
                    continue;
                }
                if let Op::Launch(Launch::Slab { members: ms, .. }) = &b.node(r).op
                    && ms
                        .last()
                        .is_some_and(|l| root_classes.contains(&b.class_of(*l)))
                    && ms.contains(&m)
                    && !ms.iter().any(|x| b.class_of(*x) == b.class_of(id))
                {
                    added.extend(ms.iter().copied());
                }
            }
        }
        added.retain(|m| seen.insert(b.class_of(*m)));
        if added.is_empty() {
            break;
        }
        members.extend(added);
        if members.len() > MAX_MEMBERS {
            return None;
        }
    }
    // The head goes last again.
    members.retain(|m| *m != id);
    members.push(id);

    // Close over the inputs: an outside value computed from a member joins
    // as a stage; one with no stage (a contraction of a member) instead
    // evicts every member it is computed from.
    let mut deps = Deps::new();
    let mut converged = false;
    for _ in 0..64 {
        let mut added: Vec<Id> = Vec::new();
        let mut drop: FxHashSet<ClassId> = FxHashSet::default();
        deps.reset(b, &seen);
        for m in &members {
            for c in b.node(*m).children.iter().copied() {
                let class = b.class_of(c);
                if seen.contains(&class) {
                    continue;
                }
                if !deps.depends(b, c, &seen) {
                    continue;
                }
                match producers.get(b, c).map(|p| (p, b.node(p).op.clone())) {
                    Some((_, Op::Launch(Launch::Slab { members: ms, .. }))) => {
                        added.extend(ms.iter().copied())
                    }
                    Some((p, _)) => added.push(p),
                    None => deps.members_under(b, c, &seen, &mut drop),
                }
            }
        }
        if !drop.is_empty() {
            members.retain(|m| !drop.contains(&b.class_of(*m)));
            for c in &drop {
                seen.remove(c);
            }
            if !members.contains(&id) {
                return None;
            }
            continue;
        }
        added.retain(|m| seen.insert(b.class_of(*m)));
        if added.is_empty() {
            converged = true;
            break;
        }
        members.extend(added);
        if members.len() > MAX_MEMBERS {
            return None;
        }
    }
    if !converged {
        return None;
    }
    if members.len() < 2 {
        return None;
    }
    // The head is the chain's sink and goes last.
    let rest: Vec<Id> = members.iter().copied().filter(|m| *m != id).collect();
    let mut members = order_members(b, rest)?;
    members.push(id);

    // The longest tail that partitions and fits the bindings; a dropped
    // prefix member is produced before the kernel and read as an input.
    let budget = b.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let first = members.len().saturating_sub(MAX_MEMBERS);
    // Each member's finest partition against the whole chain; a tail's is a
    // multiple of it, so a count dividing this divides that.
    let all: FxHashSet<ClassId> = members.iter().map(|m| b.class_of(*m)).collect();
    let finests: Vec<Option<u64>> = members.iter().map(|m| finest(b, *m, &all)).collect();

    let widths: Vec<Option<u64>> = members
        .iter()
        .map(|m| match &b.node(*m).op {
            Op::Launch(op) => stage(op)?.space.iterations(),
            _ => None,
        })
        .collect();
    let mut chosen: Option<(Vec<Id>, u64)> = None;
    for cut in first..members.len() - 1 {
        let tail = &members[cut..];
        let mut g = 0u64;
        let mut widest = 0u64;
        for i in cut..members.len() {
            let (Some(f), Some(w)) = (finests[i], widths[i]) else {
                g = 0;
                break;
            };
            g = gcd(g, f);
            widest = widest.max(w);
        }
        if g == 0 {
            continue;
        }
        let Some(slabs) = coarsen(g, widest, b.caps()) else {
            continue;
        };
        let mut inputs: FxHashSet<ClassId> = FxHashSet::default();
        outside_inputs(b, tail, &mut inputs);
        // Uniform block, output, outside inputs and root members bind; other
        // externally read members are the extractor's check.
        let root_members = tail[..tail.len() - 1]
            .iter()
            .filter(|m| root_classes.contains(&b.class_of(**m)))
            .count();
        let inputs = inputs
            .iter()
            .filter(|c| own_buffer(b, **c, &root_classes))
            .count();
        if 3 + inputs + root_members > budget {
            continue;
        }
        chosen = Some((tail.to_vec(), slabs));
        break;
    }
    let (members, slabs) = chosen?;

    let slab = b
        .add_launch(Launch::Slab {
            slabs: u32::try_from(slabs).ok()?,
            members: SmallVec::from_vec(members),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, slab).ok()
}

/// A launch as a slab stage: a `Map`, or a `Fold` with one scalar slot per
/// post and no promoted axis. A chain stops at anything else.
struct Stage<'a> {
    space: &'a IndexSpace,
    /// A slab may partition `space.dims[start..end]` for `end` in
    /// `start + 1..=end_max`: the stage's output rows.
    start: usize,
    end_max: usize,
    ops: &'a [Operand],
}

fn stage(op: &Launch) -> Option<Stage<'_>> {
    match op {
        Launch::Map { space, ops, .. } => {
            space.iterations()?;
            Some(Stage {
                space,
                start: 0,
                end_max: space.rank(),
                ops,
            })
        }
        Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            post,
            ops,
            ..
        } => {
            if !vec_axes.is_empty()
                || carrier.width() == 0
                || post.is_empty()
                || carrier.lanes() != Some(carrier.width() as u64)
            {
                return None;
            }
            space.iterations()?;
            let axis = *axis as usize;
            let (start, end_max) = if axis == 0 {
                (1, space.rank())
            } else {
                (0, axis)
            };
            if start >= end_max {
                return None;
            }
            Some(Stage {
                space,
                start,
                end_max,
                ops,
            })
        }
        _ => None,
    }
}

/// Whether `o` is read differently across the stage's space (a gather or a
/// symbolic map is taken to vary).
fn varies(o: &Operand, op: &Launch) -> bool {
    let Some(total) = stage(op).and_then(|st| st.space.iterations()) else {
        return true;
    };
    let Some(map) = o.address_map() else {
        return true;
    };
    map.terms
        .iter()
        .enumerate()
        .any(|(i, t)| t.stride != 0 && (u64::from(t.divisor) < total || map.needs_modulo(i, total)))
}

/// Every class `members` read that none of them computes, into `out`.
pub(crate) fn outside_inputs(b: &Builder<'_>, members: &[Id], out: &mut FxHashSet<ClassId>) {
    let own: FxHashSet<ClassId> = members.iter().map(|m| b.class_of(*m)).collect();
    for m in members {
        for c in b.node(*m).children.iter() {
            let class = b.class_of(*c);
            if !own.contains(&class) {
                out.insert(class);
            }
        }
    }
}

/// Whether a class binds its own buffer (a leaf or a root); everything else
/// shares the step arena's binding.
pub(crate) fn own_buffer(b: &Builder<'_>, class: ClassId, roots: &FxHashSet<ClassId>) -> bool {
    roots.contains(&class)
        || b.class_members(class.0).iter().any(|m| {
            matches!(
                b.node(*m).op,
                Op::Logical(crate::ir::logical::Logical::Leaf(_))
            )
        })
}

/// Whether `id` is a contraction whose tiled kernel should stay its own
/// dispatch rather than run as a one-lane-per-output fold stage.
pub(crate) fn is_contraction(b: &Builder<'_>, id: Id) -> bool {
    has_contract_spelling(b, id) && !small_fold(b, id)
}

/// Whether `id`'s class has a matrix-shaped `Contract` spelling; a batched
/// dot (`m = n = 1`) gains nothing from tiling.
pub(crate) fn has_contract_spelling(b: &Builder<'_>, id: Id) -> bool {
    b.class_members(id).iter().any(|m| match &b.node(*m).op {
        Op::Launch(Launch::Contract { m, n, .. }) => {
            m.as_const() != Some(1) || n.as_const() != Some(1)
        }
        _ => false,
    })
}

/// A plain sum over a short axis (split-K partials): a fine stage whatever
/// its class also spells.
pub(crate) fn small_fold(b: &Builder<'_>, id: Id) -> bool {
    const MAX_K: u64 = 64;
    match &b.node(id).op {
        Op::Launch(Launch::Fold {
            space,
            axis,
            ops,
            carrier,
            ..
        }) => {
            ops.len() == 1
                && carrier.lift.len() == 1
                && matches!(carrier.lift[0].kind(), crate::scalar::ScalarKind::Arg(0))
                && space
                    .dims
                    .get(*axis as usize)
                    .and_then(|d| d.as_const())
                    .is_some_and(|k| k <= MAX_K)
        }
        _ => false,
    }
}

/// How much a stage spelling is worth as a member: fewest copy operands,
/// then most operands (producers inlined), then most split-K splits, then
/// the earliest id.
pub(crate) fn stage_rank(
    b: &Builder<'_>,
    m: Id,
    is_copy: impl FnMut(Id) -> bool,
) -> (isize, usize, u64, isize) {
    let Op::Launch(op) = &b.node(m).op else {
        return (isize::MIN, 0, 0, 0);
    };
    let Some(Stage { ops, .. }) = stage(op) else {
        return (isize::MIN, 0, 0, 0);
    };
    let copies = count_copies(ops, is_copy) as isize;
    let splits = match op {
        Launch::Fold { space, axis, .. } if small_fold(b, m) => space
            .dims
            .get(*axis as usize)
            .and_then(|d| d.as_const())
            .unwrap_or(0),
        _ => 0,
    };
    (-copies, ops.len(), splits, -(m.0 as isize))
}

fn count_copies<'a>(
    ops: impl IntoIterator<Item = &'a Operand>,
    mut is_copy: impl FnMut(Id) -> bool,
) -> usize {
    ops.into_iter()
        .filter(|o| o.layout.strides().iter().any(|s| s.as_const() != Some(0)))
        .filter(|o| is_copy(o.src))
        .count()
}

/// Whether `id`'s class is spelled only by identity maps: a copy of a view.
pub(crate) fn is_copy_class(b: &Builder<'_>, id: Id) -> bool {
    let mut copy = false;
    for m in b.class_members(id) {
        match &b.node(m).op {
            Op::Launch(Launch::Map { body, ops, .. })
                if ops.len() == 1 && matches!(body.kind(), crate::scalar::ScalarKind::Arg(0)) =>
            {
                copy = true;
            }
            Op::Logical(crate::ir::logical::Logical::Leaf(_)) | Op::Launch(_) => return false,
            _ => {}
        }
    }
    copy
}

/// Per-class producer choices; class metadata is fixed until `form_slab`
/// adds its launch.
#[derive(Default)]
struct Producers {
    copies: FxHashMap<ClassId, bool>,
    chosen: FxHashMap<ClassId, Option<Id>>,
}

impl Producers {
    /// The launch spelling that can be a stage, or the longest slab ending
    /// in this class.
    fn get(&mut self, b: &Builder<'_>, src: Id) -> Option<Id> {
        let class = b.class_of(src);
        if let Some(chosen) = self.chosen.get(&class) {
            return *chosen;
        }
        let contraction = has_contract_spelling(b, src);
        let mut best_slab: Option<(usize, Id)> = None;
        let mut best_stage = None;
        for m in b.class_members(src) {
            // In a contraction's class only a sum of partials (or a slab
            // ending in one) is a stage.
            if contraction {
                let last = match &b.node(m).op {
                    Op::Launch(Launch::Slab { members, .. }) => {
                        members.last().copied().unwrap_or(m)
                    }
                    _ => m,
                };
                if !small_fold(b, last) {
                    continue;
                }
            }
            match &b.node(m).op {
                Op::Launch(Launch::Slab { members, .. }) => {
                    if best_slab.is_none_or(|(n, _)| members.len() > n) {
                        best_slab = Some((members.len(), m));
                    }
                }
                Op::Launch(op) if stage(op).is_some() => {
                    let rank = stage_rank(b, m, |id| {
                        *self
                            .copies
                            .entry(b.class_of(id))
                            .or_insert_with(|| is_copy_class(b, id))
                    });
                    if best_stage.is_none_or(|(best, _)| rank > best) {
                        best_stage = Some((rank, m));
                    }
                }
                _ => {}
            }
        }
        // A stage every workgroup computes identically is shared by every
        // chain, not a member: two slabs may not own one member.
        let stage = best_stage.map(|(_, s)| s);
        let chosen = if stage.is_some_and(|s| invariant_stage(b, s)) {
            None
        } else {
            best_slab.map(|(_, s)| s).or(stage)
        };
        self.chosen.insert(class, chosen);
        chosen
    }
}

/// Whether no operand of the stage varies over its space.
fn invariant_stage(b: &Builder<'_>, m: Id) -> bool {
    let Op::Launch(op) = &b.node(m).op else {
        return false;
    };
    let Some(st) = stage(op) else {
        return false;
    };
    st.ops.iter().all(|o| !varies(o, op))
}

/// `members` topologically ordered by operand classes, or `None` on a cycle.
fn order_members(b: &Builder<'_>, members: Vec<Id>) -> Option<Vec<Id>> {
    let index: FxHashMap<ClassId, usize> = members
        .iter()
        .enumerate()
        .map(|(i, m)| (b.class_of(*m), i))
        .collect();
    let n = members.len();
    let mut indegree = vec![0usize; n];
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, m) in members.iter().enumerate() {
        let mut seen: SmallVec<[usize; 8]> = SmallVec::new();
        for c in b.node(*m).children.iter() {
            let Some(&i) = index.get(&b.class_of(*c)) else {
                continue;
            };
            if i == j || seen.contains(&i) {
                continue;
            }
            seen.push(i);
            out[i].push(j);
            indegree[j] += 1;
        }
    }
    // Kahn's algorithm, ties by position, so the order is deterministic.
    let mut ready: Vec<usize> = (0..n).filter(|i| indegree[*i] == 0).collect();
    ready.reverse();
    let mut sorted = Vec::with_capacity(n);
    while let Some(i) = ready.pop() {
        sorted.push(members[i]);
        for &j in &out[i] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                ready.push(j);
                ready.sort_unstable_by(|a, c| c.cmp(a));
            }
        }
    }
    (sorted.len() == n).then_some(sorted)
}

/// The finest partition member `m` admits: the extent of the longest
/// leading output-row prefix at which every member operand is slab-local.
fn finest(b: &Builder<'_>, m: Id, classes: &FxHashSet<ClassId>) -> Option<u64> {
    let Op::Launch(op) = &b.node(m).op else {
        return None;
    };
    let st = stage(op)?;
    let dims: Vec<u64> = st
        .space
        .dims
        .iter()
        .map(|d| d.as_const())
        .collect::<Option<_>>()?;
    (st.start + 1..=st.end_max)
        .rev()
        .find(|end| slab_local(b, m, &dims, st.start, *end, classes))
        .map(|end| dims[st.start..end].iter().product())
}

/// The slab count from the finest common partition `g`, coarsened until the
/// widest stage gives each slab a full block of work.
fn coarsen(g: u64, widest: u64, caps: &crate::device::Caps) -> Option<u64> {
    let block = u64::from(crate::ir::launch::emitted_block(1, caps));
    let slabs = largest_divisor_at_most(g, widest / block);
    // A chain too small to fill `MIN_SLABS` blocks runs as one dispatch.
    let floor = if widest < MIN_SLABS * block {
        1
    } else {
        MIN_SLABS
    };
    (slabs >= floor).then_some(slabs)
}

fn largest_divisor_at_most(n: u64, limit: u64) -> u64 {
    let mut best = 1;
    let mut divisor = 1;
    while divisor <= limit && divisor <= n / divisor {
        if n.is_multiple_of(divisor) {
            let paired = n / divisor;
            // Paired divisors descend, so the first within the limit wins.
            if paired <= limit {
                return paired;
            }
            best = divisor;
        }
        divisor += 1;
    }
    best
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Whether member `m` reads only its own slab of every member operand, for
/// slabs over `dims[start..end]`: the slab coordinate's stride `S` satisfies
/// `S * L == elements(operand)` and every other term stays under `S`.
fn slab_local(
    b: &Builder<'_>,
    m: Id,
    dims: &[u64],
    start: usize,
    end: usize,
    classes: &FxHashSet<ClassId>,
) -> bool {
    let Op::Launch(op) = &b.node(m).op else {
        return false;
    };
    let Some(st) = stage(op) else {
        return false;
    };
    let total: u64 = dims.iter().product();
    let leading: u64 = dims[start..end].iter().product();
    if leading == 0 {
        return false;
    }
    let inner: u64 = dims[end..].iter().product();
    let outer = inner * leading;
    let above = total / outer;
    for o in st.ops {
        // Outside operands are complete before the dispatch.
        if !classes.contains(&b.class_of(o.src)) {
            continue;
        }
        let Some(elements) = b
            .facts_of(o.src)
            .shape
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
        else {
            return false;
        };
        if elements == 0 {
            return false;
        }
        let Some(map) = o.address_map() else {
            return false;
        };
        let Some((lead, rest)) = address_span(&map, total, inner, leading, above) else {
            return false;
        };
        if lead == 0 {
            return false;
        }
        if lead.checked_mul(leading) != Some(elements) || rest >= lead {
            return false;
        }
    }
    true
}

/// Split an address map over `i = c_above * outer + c * inner + r` into
/// `(lead, rest)`: the stride per step of `c`, and the largest offset
/// everything else can add.
fn address_span(
    map: &crate::ir::launch::AddressMap,
    total: u64,
    inner: u64,
    leading: u64,
    above: u64,
) -> Option<(u64, u64)> {
    let outer = inner * leading;
    let mut lead: u64 = 0;
    let mut rest: u64 = u64::from(map.offset);
    for (i, t) in map.terms.iter().enumerate() {
        let (d, n, s) = (
            u64::from(t.divisor),
            u64::from(t.modulus),
            u64::from(t.stride),
        );
        if d == 0 || n == 0 {
            return None;
        }
        let wraps = map.needs_modulo(i, total);
        if (!wraps && d >= total) || s == 0 {
            continue;
        }
        if d >= outer {
            // Wholly above the slab prefix.
            let span = if wraps { n - 1 } else { (total - 1) / d };
            rest = span.checked_mul(s).and_then(|v| rest.checked_add(v))?;
            continue;
        }
        if !inner.is_multiple_of(d) {
            return None;
        }
        let q = inner / d;
        if wraps {
            // `(c_above * q * leading + c * q + r / d) % n`.
            if q.is_multiple_of(n) {
                rest = (n - 1).checked_mul(s).and_then(|v| rest.checked_add(v))?;
                continue;
            }
            if !n.is_multiple_of(q) {
                return None;
            }
            let span = n / q;
            if span < leading {
                return None;
            }
            if span > leading {
                if !span.is_multiple_of(leading) {
                    return None;
                }
                let above_span = (span / leading).min(above);
                rest = (above_span - 1)
                    .checked_mul(q * leading * s)
                    .and_then(|v| rest.checked_add(v))?;
            }
        } else if above > 1 {
            rest = (above - 1)
                .checked_mul(q * leading * s)
                .and_then(|v| rest.checked_add(v))?;
        }
        lead = q.checked_mul(s).and_then(|v| lead.checked_add(v))?;
        rest = (q - 1).checked_mul(s).and_then(|v| rest.checked_add(v))?;
    }
    Some((lead, rest))
}

/// Dependence on the member classes, memoized across one rule application.
/// Nothing below the earliest id of any member class can reach one.
pub(crate) struct Deps {
    memo: FxHashMap<Id, bool>,
    floor: Id,
}

impl Deps {
    pub(crate) fn new() -> Self {
        Self {
            memo: FxHashMap::default(),
            floor: Id(0),
        }
    }

    /// Retarget at `classes`: the floor is the smallest id of any of them.
    pub(crate) fn reset(&mut self, b: &Builder<'_>, classes: &FxHashSet<ClassId>) {
        self.memo.clear();
        self.floor = classes
            .iter()
            .flat_map(|c| b.class_ids(c.0))
            .min()
            .unwrap_or(Id(0));
    }

    /// Whether the value at `start` is computed from any class in `classes`.
    pub(crate) fn depends(
        &mut self,
        b: &Builder<'_>,
        start: Id,
        classes: &FxHashSet<ClassId>,
    ) -> bool {
        let floor = self.floor;
        let mut stack: Vec<(Id, bool)> = vec![(start, false)];
        while let Some((x, expanded)) = stack.pop() {
            if expanded {
                let v = b
                    .node(x)
                    .children
                    .iter()
                    .any(|c| self.memo.get(c).copied().unwrap_or(false));
                self.memo.insert(x, v);
                continue;
            }
            if self.memo.contains_key(&x) {
                continue;
            }
            if classes.contains(&b.class_of(x)) {
                self.memo.insert(x, true);
                continue;
            }
            if x < floor {
                self.memo.insert(x, false);
                continue;
            }
            stack.push((x, true));
            for c in b.node(x).children.iter() {
                if !self.memo.contains_key(c) {
                    stack.push((*c, false));
                }
            }
        }
        self.memo.get(&start).copied().unwrap_or(false)
    }

    /// The member classes the value at `start` is computed from, into `out`.
    fn members_under(
        &mut self,
        b: &Builder<'_>,
        start: Id,
        classes: &FxHashSet<ClassId>,
        out: &mut FxHashSet<ClassId>,
    ) {
        let floor = self.floor;
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        let mut stack = vec![start];
        while let Some(x) = stack.pop() {
            if x < floor || !seen.insert(x) {
                continue;
            }
            let class = b.class_of(x);
            if classes.contains(&class) {
                out.insert(class);
                continue;
            }
            stack.extend(b.node(x).children.iter().copied());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::largest_divisor_at_most;

    #[test]
    fn slab_divisor_matches_exhaustive_selection() {
        for n in 0u64..=256 {
            for limit in 0..=n + 1 {
                let expected = (1..=n)
                    .filter(|d| n.is_multiple_of(*d) && *d <= limit)
                    .max()
                    .unwrap_or(1);
                assert_eq!(largest_divisor_at_most(n, limit), expected);
            }
        }
        for (n, limit, expected) in [
            (1 << 30, 1000, 512),
            (1_000_000_007, 1_000_000, 1),
            (1_000_000_000_000, 1_000_000, 1_000_000),
            (u64::MAX, 3, 3),
        ] {
            assert_eq!(largest_divisor_at_most(n, limit), expected);
        }
    }
}
