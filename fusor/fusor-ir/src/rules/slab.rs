//! `FORM_SLAB`: a chain of launches that all keep a leading prefix of their
//! output rows independent becomes one dispatch, one workgroup per slab of
//! that prefix.
//!
//! A small model's step is hundreds of launches that each do microseconds of
//! work, and the dispatch is most of what each one costs. When every launch
//! in a chain partitions the same way — sequence `s` of a batch reads only
//! sequence `s` of what came before it — the chain can run as stages of one
//! kernel with a barrier between them, and the dispatch is paid once.
//!
//! The rule is additive: the slab joins the last member's class beside the
//! plain launch, and the extractor picks by cost. Nothing here decides that a
//! slab is faster; it decides that one is *correct*, which is the locality
//! proof in [`slab_local`].

use crate::egraph::{Builder, ClassId, Facts, Id, RuleTag};
use crate::ir::launch::{IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

rule!(
    FORM_SLAB_MAP,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = form_slab,
);

rule!(
    FORM_SLAB_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = form_slab,
);

/// Most members one slab carries. Every launch in a chain mints a slab of
/// everything before it, so member lists grow quadratically with chain
/// length; this keeps that bounded.
const MAX_MEMBERS: usize = 512;

/// Fewest slabs worth a dispatch of their own.
const MIN_SLABS: u64 = 32;

pub fn form_slab(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let r = form_slab_inner(b, id, node);
    if std::env::var_os("FUSOR_SLAB_LOG").is_some() {
        eprintln!("SLAB {id} -> {r:?} ({})", LAST_REASON.with(|c| c.get()));
    }
    r
}

thread_local! { static LAST_REASON: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") }; }
fn why(r: &'static str) {
    LAST_REASON.with(|c| c.set(r));
}

fn form_slab_inner(b: &mut Builder<'_>, id: Id, node: &Node) -> Option<Id> {
    why("ok");
    // The CPU target runs one lane count per dispatch and pays nothing per
    // dispatch that a slab would save; a slab is a GPU shape.
    if b.caps().kind != crate::device::DeviceKind::Gpu {
        why("cpu");
        return None;
    }
    // Bisection aids: `FUSOR_NO_SLAB` disables the rule, `FUSOR_SLAB_MAX_MEMBERS`
    // caps a slab's length.
    if std::env::var_os("FUSOR_NO_SLAB").is_some() {
        why("disabled");
        return None;
    }
    // `FUSOR_SLAB_HEAD_MAX=<id>`: only heads up to that id form slabs, for
    // bisecting a wrong plan down to one slab.
    if let Some(max) = std::env::var("FUSOR_SLAB_HEAD_MAX")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        && id.0 > max
    {
        why("disabled");
        return None;
    }
    let Op::Launch(op) = &node.op else {
        return None;
    };
    let Some((_, ops)) = stage_parts(op) else {
        why("not a stage");
        return None;
    };
    if is_contraction(b, id) {
        why("not a stage");
        return None;
    }

    // Producers first: the launch spelling of each operand, or the longest
    // slab already ending there.
    let mut members: Vec<Id> = Vec::new();
    for o in ops {
        // A value every slab reads the same way — a broadcast scalar, a
        // per-row table — cannot be a stage, since each workgroup writes
        // only its own slab; it stays an input computed before the kernel.
        if !varies(o, op) {
            continue;
        }
        if let Some(p) = producer_stage(b, o.src) {
            match &b.node(p).op {
                Op::Launch(Launch::Slab { members: ms, .. }) => members.extend(ms.iter().copied()),
                _ => members.push(p),
            }
        }
    }
    // A root's update chain — an optimizer state, a parameter — is one of
    // many independent chains the step ends in. Every earlier root stage
    // joins this head's slab: independent members run as stages with
    // nothing between them but a barrier, and one dispatch replaces one per
    // root. The tail that fits the bindings is what gets minted.
    // Opt-in (`FUSOR_BATCH_ROOTS`): measured a net loss on the transformer
    // step — the batches it wins are outweighed by the unfused spellings
    // they carry and the extraction time the trials cost.
    if std::env::var_os("FUSOR_BATCH_ROOTS").is_some()
        && b.roots().iter().any(|r| b.class_of(*r) == b.class_of(id))
    {
        if std::env::var_os("FUSOR_SLAB_LOG").is_some() {
            eprintln!("  ROOTHEAD {id}: {} roots", b.roots().len());
        }
        let mut seen_root: FxHashSet<ClassId> = FxHashSet::default();
        seen_root.insert(b.class_of(id));
        // Roots before this one in the caller's order, whichever spelling
        // of each is best — the fused ones are minted late and have ids
        // past the head's.
        let head_root = b
            .roots()
            .iter()
            .copied()
            .filter(|r| b.class_of(*r) == b.class_of(id))
            .min()
            .unwrap_or(id);
        for r in b.roots().to_vec() {
            let class = b.class_of(r);
            if r >= head_root || !seen_root.insert(class) {
                continue;
            }
            let pick = b
                .class_members(r)
                .into_iter()
                .filter(|m| !is_contraction(b, *m))
                .filter(|m| matches!(&b.node(*m).op, Op::Launch(op) if stage_parts(op).is_some()))
                .max_by_key(|m| stage_rank(b, *m));
            if let Some(m) = pick {
                if std::env::var_os("FUSOR_SLAB_LOG").is_some() {
                    let ranks: Vec<String> = b
                        .class_members(r)
                        .into_iter()
                        .filter(|m| !is_contraction(b, *m))
                        .filter(|m| matches!(&b.node(*m).op, Op::Launch(op) if stage_parts(op).is_some()))
                        .map(|m| format!("{m}:{:?}", stage_rank(b, m)))
                        .collect();
                    eprintln!(
                        "  ROOTPICK {id}: root class {} -> {m} from {ranks:?}",
                        class.0.index()
                    );
                }
                members.push(m);
            }
        }
    }
    let trace = std::env::var("FUSOR_SLAB_TRACE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        == Some(id.0);
    if trace {
        let ops_s: Vec<String> = ops
            .iter()
            .map(|o| {
                format!(
                    "{}:c{}:varies={}:prod={:?}",
                    o.src,
                    b.class_of(o.src).0.index(),
                    varies(o, op),
                    producer_stage(b, o.src)
                )
            })
            .collect();
        eprintln!("TRACE {id}: ops {ops_s:?} members {members:?}");
        // Each member's own operands: where a chain stops and why.
        let mut seen_m: FxHashSet<Id> = FxHashSet::default();
        let mut stack: Vec<Id> = members.clone();
        while let Some(m) = stack.pop() {
            if !seen_m.insert(m) {
                continue;
            }
            let Op::Launch(mop) = &b.node(m).op else {
                continue;
            };
            let Some((_, mops)) = stage_parts(mop) else {
                continue;
            };
            for o in mops {
                let c = b.class_of(o.src);
                let kinds: Vec<String> = b
                    .class_members(o.src)
                    .iter()
                    .map(|x| format!("{x}:{:?}:sf={}", b.node(*x).op.tag(), small_fold(b, *x)))
                    .collect();
                eprintln!(
                    "TRACE {id}: member {m} reads {}:c{} varies={} contraction={} prod={:?} {kinds:?}",
                    o.src,
                    c.0.index(),
                    varies(o, mop),
                    has_contract_spelling(b, o.src),
                    producer_stage(b, o.src)
                );
                if let Some(p) = producer_stage(b, o.src) {
                    stack.push(p);
                }
            }
        }
    }
    if members.is_empty() {
        why("no producer stage");
        return None;
    }
    members.push(id);

    // One member per class, first spelling wins.
    let mut seen: FxHashSet<ClassId> = FxHashSet::default();
    members.retain(|m| seen.insert(b.class_of(*m)));

    // Two root chains sharing a stage — an optimizer's moments and its
    // parameter — become one kernel: the other root's slab joins, its head
    // a middle member with a buffer. Roots only: a forward chain merged
    // this way grows past what one kernel should run. Bounded, since each
    // round adds whole slabs.
    let root_classes_all: FxHashSet<ClassId> = b.roots().iter().map(|r| b.class_of(*r)).collect();
    let head_is_root = root_classes_all.contains(&b.class_of(id));
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
                        .is_some_and(|l| root_classes_all.contains(&b.class_of(*l)))
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
            why("member count");
            return None;
        }
    }
    // The head goes last again.
    members.retain(|m| *m != id);
    members.push(id);

    // Close over the inputs: a value read from outside that is computed from
    // a member would have to run between two stages, so its stage joins the
    // slab — and a merge of two chains admits everything between them. A
    // dependent input with no stage — a contraction of a member — splits
    // the chain instead: every member it is computed from leaves.
    let mut deps = Deps::new();
    let mut converged = false;
    for _ in 0..64 {
        let mut added: Vec<Id> = Vec::new();
        let mut drop: FxHashSet<ClassId> = FxHashSet::default();
        deps.reset(b, &members);
        for m in &members {
            for c in b.node(*m).children.iter().copied() {
                let class = b.class_of(c);
                if seen.contains(&class) {
                    continue;
                }
                // An input computed from no member stays outside — except a
                // sum of split-K partials read by nothing else, which is one
                // stage cheaper here than as its own dispatch.
                if !deps.depends(b, c, &seen) {
                    continue;
                }
                match producer_stage(b, c).map(|p| (p, b.node(p).op.clone())) {
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
                why("a dependent input has no stage");
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
            why("member count");
            return None;
        }
    }
    if !converged {
        if std::env::var_os("FUSOR_SLAB_LOG").is_some() {
            eprintln!("  NOCONVERGE {id}: {} members", members.len());
        }
        why("closure did not converge");
        return None;
    }
    let cap = std::env::var("FUSOR_SLAB_MAX_MEMBERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(MAX_MEMBERS);
    if members.len() < 2 {
        why("member count");
        return None;
    }
    // The head is the chain's sink — nothing added has a larger id, so
    // nothing reads it — and goes last whatever position the closure's
    // additions took.
    let rest: Vec<Id> = members.iter().copied().filter(|m| *m != id).collect();
    let Some(mut members) = order_members(b, rest) else {
        why("cycle");
        return None;
    };
    members.push(id);
    if trace {
        eprintln!("TRACE {id}: closed {members:?}");
    }

    // The longest tail of the chain that partitions and fits the bindings.
    // Dropping a prefix of the topological order is always sound: a dropped
    // member is produced before the kernel and read as an input. The kernel
    // binds the uniform block, its own output, every distinct value read
    // from outside, and every middle member something outside the slab
    // reads — one nothing else reads stays in workgroup memory.
    let budget = b.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let root_classes: FxHashSet<ClassId> = b.roots().iter().map(|r| b.class_of(*r)).collect();
    let first = members.len().saturating_sub(cap);
    // Each member's finest partition against the whole chain. A tail has
    // fewer members, so fewer member operands to be local to, so its finest
    // is a multiple of this one; a count that divides this divides that.
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
    let mut reason = "no common slab count";
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
            if std::env::var_os("FUSOR_SLAB_LOG").is_some() {
                eprintln!(
                    "  NOSLAB {id} cut {cut}: gcd {g} widest {widest} finests {:?}",
                    &finests[cut..]
                );
            }
            continue;
        };
        let classes: FxHashSet<ClassId> = tail.iter().map(|m| b.class_of(*m)).collect();
        let mut inputs: FxHashSet<ClassId> = FxHashSet::default();
        for m in tail {
            for c in b.node(*m).children.iter() {
                let class = b.class_of(*c);
                if !classes.contains(&class) {
                    inputs.insert(class);
                }
            }
        }
        // Members something outside reads bind too; which those are is the
        // extractor's to know (`slab_bindings_fit`), and asking the graph
        // here costs a reader scan per member per chain. A root member is
        // read back by the caller, so it always binds; the rest is the
        // optimistic budget.
        let root_members = tail[..tail.len() - 1]
            .iter()
            .filter(|m| root_classes.contains(&b.class_of(**m)))
            .count();
        // Only leaves and roots bind their own buffers; the rest share the
        // step arena's one binding.
        let owns = |c: &ClassId| {
            root_classes.contains(c)
                || b.class_members(c.0).iter().any(|m| {
                    matches!(
                        b.node(*m).op,
                        Op::Logical(crate::ir::logical::Logical::Leaf(_))
                    )
                })
        };
        let inputs = inputs.iter().filter(|c| owns(c)).count();
        if trace {
            eprintln!(
                "TRACE {id}: cut {cut} slabs {slabs} inputs {inputs} root_members {root_members}"
            );
        }
        if 3 + inputs + root_members > budget {
            reason = "over the binding budget";
            continue;
        }
        chosen = Some((tail.to_vec(), slabs));
        break;
    }
    let Some((members, slabs)) = chosen else {
        why(reason);
        return None;
    };

    let slab = b
        .add_launch(Launch::Slab {
            slabs: u32::try_from(slabs).ok()?,
            members: SmallVec::from_vec(members),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, slab).ok()
}

/// The space and operands of a launch that can be a stage: a `Map`, or a
/// `Fold` with one scalar slot per post and no promoted axis. Everything
/// else — contractions in their tiled families, gathers — is not a stage,
/// and a chain stops at it.
fn stage_parts(op: &Launch) -> Option<(&IndexSpace, &[Operand])> {
    stage(op).map(|st| (st.space, st.ops))
}

/// A launch as a slab stage.
struct Stage<'a> {
    space: &'a IndexSpace,
    /// The axes a slab may partition are `space.dims[start..end]` for any
    /// `end` in `start + 1..=end_max`: the stage's output rows, in the order
    /// the space walks them. A fold that reduces its leading axis starts at
    /// the second; one that reduces a later axis stops there.
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
            // Every slot scalar: the stage writes one value per post the
            // way the fold's own lowering does.
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

/// Whether `o` is read differently at different points of the stage's
/// space: some term of its address map has a stride and a divisor inside
/// the space. A gather or a symbolic map is taken to vary.
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

/// Whether `id`'s class also has a `Contract` spelling: the value is a
/// contraction, and its fold spelling is the fallback for devices without
/// a tiled family. A slab stage runs it as one lane per output walking the
/// reduced axis, which is what tiling exists to avoid; the tiled kernel
/// stays its own dispatch and the stages fuse around it.
pub(crate) fn is_contraction(b: &Builder<'_>, id: Id) -> bool {
    has_contract_spelling(b, id) && !small_fold(b, id)
}

/// Whether `id`'s class has a `Contract` spelling.
pub(crate) fn has_contract_spelling(b: &Builder<'_>, id: Id) -> bool {
    b.class_members(id)
        .iter()
        .any(|m| matches!(b.node(*m).op, Op::Launch(Launch::Contract { .. })))
}

/// A fold over a short axis — the sum of split-K partials — is one lane
/// walking a few elements, and a fine stage whatever its class also
/// spells.
pub(crate) fn small_fold(b: &Builder<'_>, id: Id) -> bool {
    const MAX_K: u64 = 64;
    match &b.node(id).op {
        // One operand summed as it is: a contraction spelled as a fold
        // multiplies two.
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

/// How much a stage spelling is worth as a member: fewest operands that
/// are identity copies of something else (each such operand is a copy
/// stage or a copy dispatch the chain would carry), then the most operands
/// (the most producers inlined), then the earliest id.
pub(crate) fn stage_rank(b: &Builder<'_>, m: Id) -> (isize, usize, u64, isize) {
    let Op::Launch(op) = &b.node(m).op else {
        return (isize::MIN, 0, 0, 0);
    };
    let Some((_, ops)) = stage_parts(op) else {
        return (isize::MIN, 0, 0, 0);
    };
    let copies = copy_operands(b, ops) as isize;
    // Among sums of split-K partials, the most split: the contraction
    // feeding it runs with the most workgroups.
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

/// Operands that are copies of some other value and are read varyingly: a
/// broadcast read of a copy class is a scalar, not a copy dispatch.
pub(crate) fn copy_operands(b: &Builder<'_>, ops: &[Operand]) -> usize {
    ops.iter()
        .filter(|o| is_copy_class(b, o.src))
        .filter(|o| o.layout.strides().iter().any(|s| s.as_const() != Some(0)))
        .count()
}

/// Whether `id`'s class is spelled by an identity map — a copy of a view
/// of some other value.
pub(crate) fn is_copy_class(b: &Builder<'_>, id: Id) -> bool {
    let mut copy = false;
    for m in b.class_members(id) {
        match &b.node(m).op {
            Op::Launch(Launch::Map { body, ops, .. })
                if ops.len() == 1 && matches!(body.kind(), crate::scalar::ScalarKind::Arg(0)) =>
            {
                copy = true;
            }
            // A leaf, or any spelling that computes something: the value
            // is not merely a copy of another.
            Op::Logical(crate::ir::logical::Logical::Leaf(_)) | Op::Launch(_) => return false,
            _ => {}
        }
    }
    copy
}

/// The launch spelling of `src`'s value that can be a stage, or the slab
/// with the most members already ending in that class.
fn producer_stage(b: &Builder<'_>, src: Id) -> Option<Id> {
    let contraction = has_contract_spelling(b, src);
    let mut best_slab: Option<(usize, Id)> = None;
    let mut stage: Option<Id> = None;
    for m in b.class_members(src) {
        // In a contraction's class only a plain sum of partials, or a
        // slab ending in one, is a stage; the tiled kernel stays its own
        // dispatch.
        if contraction {
            let last = match &b.node(m).op {
                Op::Launch(Launch::Slab { members, .. }) => members.last().copied().unwrap_or(m),
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
            Op::Launch(op)
                if stage_parts(op).is_some()
                    && stage.is_none_or(|s| stage_rank(b, m) > stage_rank(b, s)) =>
            {
                stage = Some(m);
            }
            _ => {}
        }
    }
    // A stage every workgroup would compute identically — a scalar of the
    // step count, a broadcast — is one launch every chain shares, not a
    // member: two slabs may not own one member, and it would make every
    // chain's slab overlap every other's.
    if stage.is_some_and(|s| invariant_stage(b, s)) {
        return None;
    }
    best_slab.map(|(_, s)| s).or(stage)
}

/// Whether no operand of the stage varies over its space.
fn invariant_stage(b: &Builder<'_>, m: Id) -> bool {
    let Op::Launch(op) = &b.node(m).op else {
        return false;
    };
    let Some((_, ops)) = stage_parts(op) else {
        return false;
    };
    ops.iter().all(|o| !varies(o, op))
}

/// `members` in an order every member's producers precede it, or `None` on
/// a cycle. Edges are operand classes: a member reading a value another
/// member's class carries reads that member.
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
/// leading prefix of its output rows at which every operand another member
/// produces is slab-local.
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

/// The slab count from the members' finest common partition `g`, coarsened
/// until the widest stage gives each slab a full block of work: a workgroup
/// runs every stage over its slab, and a slab of a few elements is a block
/// of idle lanes stepping through barriers.
fn coarsen(g: u64, widest: u64, caps: &crate::device::Caps) -> Option<u64> {
    let block = u64::from(crate::ir::launch::emitted_block(1, caps));
    let slabs = (1..=g)
        .filter(|d| g.is_multiple_of(*d) && widest / d >= block)
        .max()
        .unwrap_or(1);
    // A chain too small to fill `MIN_SLABS` blocks is a few workgroups
    // whichever way it is cut; one dispatch for it beats several.
    let floor = if widest < MIN_SLABS * block {
        1
    } else {
        MIN_SLABS
    };
    (slabs >= floor).then_some(slabs)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Whether member `m` reads only its own slab of every operand another
/// member produces, for slabs over `dims[start..end]` of its space.
///
/// A stage's flat index `i` addresses an operand through its
/// [`crate::ir::launch::AddressMap`]: `offset + Σ ((i / d) % n) * stride`.
/// With the prefix's extent `L` over an inner block of `I` elements, the
/// slab coordinate is `(i / I) % L`. A term is *leading* when its value
/// changes with that coordinate, and the map is slab-local iff exactly one
/// term is leading, it is the coordinate itself (`d` divides `I`, no wrap
/// below `L`), and its stride `S` satisfies `S * L == elements(operand)`, so
/// slab `s` of the space maps onto slab `s` of the operand — and every other
/// term together with the offset stays under `S`, so a row never reaches
/// into the next. A member read the same way from every slab is not local:
/// each workgroup wrote only its own slab of it.
///
/// Operands from outside the slab are read whole from buffers complete
/// before the dispatch, so they are read however the stage likes.
///
/// Neither condition mentions the slab count: a partition that is local at
/// one count is local at every divisor of it.
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
    let log = std::env::var_os("FUSOR_SLAB_LOG").is_some();
    for (oi, o) in st.ops.iter().enumerate() {
        // An outside operand is read whole from a buffer complete before
        // the dispatch; the closure in `form_slab` saw to it that none is
        // computed from a member.
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
            if log {
                eprintln!("  LOCAL {m} op{oi}: no address map ({:?})", o.access);
            }
            return false;
        };
        let fail = |what: &str| {
            if log {
                eprintln!(
                    "  LOCAL {m} op{oi}: {what}; space {dims:?} prefix {start}..{end} leading \
                     {leading} inner {inner} elements {elements} offset {} terms {:?}",
                    map.offset,
                    map.terms
                        .iter()
                        .map(|t| (t.divisor, t.modulus, t.stride))
                        .collect::<Vec<_>>()
                );
            }
        };
        let (lead, rest) = match address_span(&map, total, inner, leading, above) {
            Ok(v) => v,
            Err(what) => {
                fail(what);
                return false;
            }
        };
        if lead == 0 {
            fail("a member is read the same way from every slab");
            return false;
        }
        if lead.checked_mul(leading) != Some(elements) || rest >= lead {
            fail("leading stride does not partition the operand");
            return false;
        }
    }
    true
}

/// Split an operand's address map over `i = c_above * outer + c * inner + r`
/// into `(lead, rest)`: the stride per step of the slab coordinate `c`, and
/// the largest everything else — the offset and every bounded term — can
/// add. Each term `((i / d) % n) * s` splits so provided `d` divides `inner`
/// or lies wholly above the slab prefix.
fn address_span(
    map: &crate::ir::launch::AddressMap,
    total: u64,
    inner: u64,
    leading: u64,
    above: u64,
) -> Result<(u64, u64), &'static str> {
    const OVERFLOW: &str = "address arithmetic overflows";
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
            return Err("a degenerate term");
        }
        let wraps = map.needs_modulo(i, total);
        if (!wraps && d >= total) || s == 0 {
            continue;
        }
        if d >= outer {
            // Wholly above the slab prefix: bounded, and paid whatever the
            // slab.
            let span = if wraps { n - 1 } else { (total - 1) / d };
            rest = span
                .checked_mul(s)
                .and_then(|v| rest.checked_add(v))
                .ok_or(OVERFLOW)?;
            continue;
        }
        if !inner.is_multiple_of(d) {
            return Err("term divisor does not divide the inner block");
        }
        let q = inner / d;
        if wraps {
            // `(c_above * q * leading + c * q + r / d) % n`.
            if q.is_multiple_of(n) {
                rest = (n - 1)
                    .checked_mul(s)
                    .and_then(|v| rest.checked_add(v))
                    .ok_or(OVERFLOW)?;
                continue;
            }
            if !n.is_multiple_of(q) {
                return Err("wrap straddles the inner block");
            }
            let span = n / q;
            if span < leading {
                return Err("wrap folds slabs together");
            }
            if span > leading {
                if !span.is_multiple_of(leading) {
                    return Err("wrap straddles the slab prefix");
                }
                let above_span = (span / leading).min(above);
                rest = (above_span - 1)
                    .checked_mul(q * leading * s)
                    .and_then(|v| rest.checked_add(v))
                    .ok_or(OVERFLOW)?;
            }
        } else if above > 1 {
            rest = (above - 1)
                .checked_mul(q * leading * s)
                .and_then(|v| rest.checked_add(v))
                .ok_or(OVERFLOW)?;
        }
        lead = q
            .checked_mul(s)
            .and_then(|v| lead.checked_add(v))
            .ok_or(OVERFLOW)?;
        rest = (q - 1)
            .checked_mul(s)
            .and_then(|v| rest.checked_add(v))
            .ok_or(OVERFLOW)?;
    }
    Ok((lead, rest))
}

/// Dependence on the member classes, memoized across one rule application.
/// Children have smaller ids than their parents, so nothing below the
/// earliest member can reach one: the walk prunes there.
pub(crate) struct Deps {
    memo: FxHashMap<Id, bool>,
    /// The earliest id in any member's class; see [`Self::reset`].
    floor: Id,
}

impl Deps {
    pub(crate) fn new() -> Self {
        Self {
            memo: FxHashMap::default(),
            floor: Id(0),
        }
    }

    /// [`Self::reset`] for a set of classes.
    pub(crate) fn reset_classes(&mut self, b: &Builder<'_>, classes: &FxHashSet<ClassId>) {
        self.memo.clear();
        self.floor = classes
            .iter()
            .flat_map(|c| b.class_ids(c.0))
            .min()
            .unwrap_or(Id(0));
    }

    /// A node reads a class by whichever of its ids the author wrote, so
    /// the floor is the smallest id in any member's class: nothing below it
    /// can reach a member.
    fn reset(&mut self, b: &Builder<'_>, members: &[Id]) {
        self.memo.clear();
        self.floor = members
            .iter()
            .flat_map(|m| b.class_ids(*m))
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
