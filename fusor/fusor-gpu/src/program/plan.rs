use fusor_ir::{
    Result,
    dtype::Dtype,
    egraph::{EGraph, Id},
    error::Error,
    ir::{
        Op,
        logical::{LeafKind, Logical},
    },
    semantics::children::children_logical,
    shape::SymId,
};
use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) const BLOCK: u32 = 256;
/// Linear-job workgroups when the caller does not choose. At 64 a GPU with
/// a few dozen cores holds under two linear workgroups per core; 128 fills
/// them (the TINY transformer step: 1.57 -> 1.44 ms). A power of two keeps
/// row-aligned ownership shares for power-of-two shapes.
const DEFAULT_GROUPS: u32 = 128;

/// Scheduling controls for fixed-shape programs.
#[derive(Clone, Copy, Debug)]
pub struct ProgramOptions {
    /// Linear-job workgroups (1..=256). One forces a single-workgroup schedule;
    /// None selects by workload size (one for tiny programs, else 128).
    /// Parallel matrix and indexed-reduction jobs use their own bounded grids,
    /// sized to the work they actually perform.
    pub workgroups: Option<u32>,
    /// Use f32 matrix instructions when the backend and device support them.
    pub matrix_acceleration: bool,
    /// Use subgroup collectives independently of matrix instructions.
    pub subgroup_acceleration: bool,
    /// Upper bound on stages per kernel; bounds shader size and register pressure.
    pub max_region_stages: usize,
}
impl Default for ProgramOptions {
    fn default() -> Self {
        Self {
            workgroups: None,
            matrix_acceleration: true,
            subgroup_acceleration: true,
            max_region_stages: 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Uniform {
    pub sym: SymId,
    pub dtype: Dtype,
    pub(crate) offset: u32,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub id: Id,
    pub dtype: Dtype,
    pub elements: u32,
    pub(crate) offset: u32,
    pub uniform: Option<SymId>,
}

#[derive(Clone, Debug)]
pub struct ProgramStats {
    pub stages: usize,
    pub kernels: usize,
    /// Default linear-job grid; a packed dispatch may contain several job grids.
    pub workgroups: u32,
    pub arena_bytes: u64,
    pub dedicated_bytes: u64,
    pub state_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct Value {
    pub id: Id,
    pub shape: Vec<u32>,
    pub dtype: Dtype,
    pub op: Logical,
    pub offset: Option<u32>,
    pub forwarded: bool,
}
impl Value {
    pub(crate) fn len(&self) -> u32 {
        self.shape.iter().product()
    }
    pub(crate) fn materialized(&self) -> bool {
        !self.forwarded
            && !matches!(
                self.op,
                Logical::Restride { .. } | Logical::Leaf(LeafKind::Const { .. })
            )
    }
}

pub(super) fn visit_sources(
    values: &[Value],
    by_id: &FxHashMap<Id, usize>,
    id: Id,
    visit: &mut impl FnMut(&Value),
) {
    let value = &values[by_id[&id]];
    match value.op {
        Logical::Restride { x, .. } => visit_sources(values, by_id, x, visit),
        Logical::Leaf(LeafKind::Const { .. }) => {}
        _ => {
            visit(value);
            if value.forwarded {
                for dep in children_logical(&value.op) {
                    visit_sources(values, by_id, dep, visit);
                }
            }
        }
    }
}

/// A checked, device-independent program. Unsupported logical operations or
/// symbolic dimensions fail compilation before allocation or shader creation.
pub struct Plan {
    pub(crate) values: Vec<Value>,
    pub(crate) by_id: FxHashMap<Id, usize>,
    pub(crate) feedback: Vec<(Id, Id)>,
    pub(crate) roots: Vec<Id>,
    outputs: FxHashMap<Id, Id>,
    pub(crate) regions: Vec<super::regions::Region>,
    inputs: Vec<Input>,
    pub(crate) uniforms: Vec<Uniform>,
    stats: ProgramStats,
    pub(crate) options: ProgramOptions,
}
impl Plan {
    pub fn compile(
        graph: &EGraph,
        roots: &[Id],
        feedback: &[(Id, Id)],
        max_bytes: u64,
    ) -> Result<Self> {
        Self::compile_with_options(graph, roots, feedback, max_bytes, ProgramOptions::default())
    }
    pub fn compile_with_options(
        graph: &EGraph,
        roots: &[Id],
        feedback: &[(Id, Id)],
        max_bytes: u64,
        options: ProgramOptions,
    ) -> Result<Self> {
        let groups = options.workgroups.unwrap_or(DEFAULT_GROUPS);
        if groups == 0 || groups > 256 || options.max_region_stages == 0 {
            return Err(Error::Plan(
                "program workgroups must be between 1 and 256".into(),
            ));
        }
        if roots.is_empty() {
            return Err(Error::Plan("a program needs at least one output".into()));
        }
        let mut values = Vec::new();
        let mut by_id = FxHashMap::default();
        let mut visiting = FxHashSet::default();
        fn visit(
            g: &EGraph,
            id: Id,
            values: &mut Vec<Value>,
            by_id: &mut FxHashMap<Id, usize>,
            visiting: &mut FxHashSet<Id>,
            max_bytes: u64,
        ) -> Result<()> {
            if id.index() >= g.len() {
                return Err(Error::Plan("program value is outside the graph".into()));
            }
            if by_id.contains_key(&id) {
                return Ok(());
            }
            if !visiting.insert(id) {
                return Err(Error::Plan(format!("cyclic logical program at {id}")));
            }
            let selected = g
                .members(g.class_of(id))
                .into_iter()
                .filter(|m| matches!(g.node(*m).op, Op::Logical(_)))
                .min_by_key(|m| m.index())
                .ok_or_else(|| Error::Plan(format!("no logical definition for {id}")))?;
            if let Some(index) = by_id.get(&selected).copied() {
                by_id.insert(id, index);
                visiting.remove(&id);
                return Ok(());
            }
            let Op::Logical(op) = &g.node(selected).op else {
                unreachable!()
            };
            let op = op.clone();
            if matches!(
                op,
                Logical::Window { .. } | Logical::Dequant { .. } | Logical::Project { .. }
            ) {
                return Err(Error::Plan(format!(
                    "fixed program does not support {:?}",
                    op.tag()
                )));
            }
            let facts = g.facts(selected);
            if !matches!(facts.dtype, Dtype::F32 | Dtype::U32 | Dtype::I32) {
                return Err(Error::Dtype(format!(
                    "fixed program needs f32/u32/i32, got {:?}",
                    facts.dtype
                )));
            }
            let shape: Vec<u32> = facts
                .shape
                .iter()
                .map(|d| {
                    d.as_const()
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|n| *n > 0)
                        .ok_or_else(|| {
                            Error::Shape("fixed program needs nonempty constant shapes".into())
                        })
                })
                .collect::<Result<_>>()?;
            let elements = shape
                .iter()
                .try_fold(1u32, |a, b| a.checked_mul(*b))
                .filter(|n| *n <= u32::MAX - 2 * BLOCK)
                .ok_or_else(|| {
                    Error::Shape("fixed program index/loop extent exceeds u32".into())
                })?;
            if u64::from(elements) * 4 > max_bytes {
                return Err(Error::Plan(
                    "a program value exceeds the device buffer budget".into(),
                ));
            }
            for dep in children_logical(&op) {
                visit(g, dep, values, by_id, visiting, max_bytes)?;
            }
            if let Logical::Restride { x, specs, .. } = &op {
                let source = &values[by_id[x]];
                if specs.len() != shape.len() {
                    return Err(Error::Shape(
                        "view rank differs from its index recipe".into(),
                    ));
                }
                let mut last = 0u64;
                for (axis, spec) in specs.iter().enumerate() {
                    let offset = spec.offset.as_const().ok_or_else(|| {
                        Error::Shape("program view offsets must be constant".into())
                    })?;
                    if spec.multiplier == 0 {
                        if offset != 0 {
                            return Err(Error::Shape("broadcast view offsets must be zero".into()));
                        }
                        continue;
                    }
                    if spec.input_dim as usize >= source.shape.len() {
                        return Err(Error::Shape("view source axis is out of bounds".into()));
                    }
                    let stride = u64::from(
                        source.shape[spec.input_dim as usize + 1..]
                            .iter()
                            .product::<u32>(),
                    );
                    let term = u64::from(shape[axis] - 1)
                        .checked_mul(u64::from(spec.multiplier))
                        .and_then(|n| n.checked_add(offset))
                        .and_then(|n| n.checked_mul(stride))
                        .ok_or_else(|| Error::Shape("view address overflows".into()))?;
                    last = last
                        .checked_add(term)
                        .ok_or_else(|| Error::Shape("view address overflows".into()))?;
                }
                if last >= u64::from(source.len()) {
                    return Err(Error::Shape(
                        "view address exceeds its logical source".into(),
                    ));
                }
            }
            if let Logical::Fold {
                axis, ins, carrier, ..
            } = &op
                && (ins.is_empty()
                    || *axis as usize >= values[by_id[&ins[0]]].shape.len()
                    || carrier.width() != 1
                    || carrier.lanes() != Some(1))
            {
                return Err(Error::Plan(
                    "program reductions require a valid axis and scalar carrier".into(),
                ));
            }
            by_id.insert(id, values.len());
            by_id.insert(selected, values.len());
            values.push(Value {
                id: selected,
                shape,
                dtype: facts.dtype,
                op,
                offset: None,
                forwarded: false,
            });
            visiting.remove(&id);
            Ok(())
        }
        for id in roots
            .iter()
            .copied()
            .chain(feedback.iter().flat_map(|(a, b)| [*a, *b]))
        {
            visit(graph, id, &mut values, &mut by_id, &mut visiting, max_bytes)?;
        }
        let mut states = FxHashSet::default();
        for (input, output) in feedback {
            let a = &values[by_id[input]];
            let b = &values[by_id[output]];
            if !states.insert(a.id)
                || !matches!(
                    a.op,
                    Logical::Leaf(LeafKind::Buffer { .. } | LeafKind::Param { .. })
                )
            {
                return Err(Error::Plan(
                    "feedback destinations must be distinct external leaves".into(),
                ));
            }
            if a.dtype != b.dtype || a.shape != b.shape {
                return Err(Error::Shape(
                    "feedback dtype and shape must match exactly".into(),
                ));
            }
        }
        // Retained views/constants and aliased feedback get an explicit
        // snapshot. This makes simultaneous swaps and strided state updates
        // correct, including when reads and writes have different ownership.
        let mut outputs = FxHashMap::default();
        for id in roots
            .iter()
            .copied()
            .chain(feedback.iter().map(|(_, out)| *out))
        {
            let v = &values[by_id[&id]];
            if matches!(v.op, Logical::Restride { .. } | Logical::Leaf(_)) {
                if outputs.contains_key(&id) {
                    continue;
                }
                let snapshot = Id(u32::try_from(graph.len() + outputs.len())
                    .map_err(|_| Error::Plan("program snapshot id overflow".into()))?);
                let value = Value {
                    id: snapshot,
                    shape: v.shape.clone(),
                    dtype: v.dtype,
                    op: Logical::Map {
                        expr: fusor_ir::scalar::ScalarExpr::arg(0, v.dtype),
                        ins: vec![id].into(),
                        outs: 1,
                    },
                    offset: None,
                    forwarded: false,
                };
                by_id.insert(snapshot, values.len());
                values.push(value);
                outputs.insert(id, snapshot);
            }
        }
        let root_ids: Vec<_> = roots
            .iter()
            .map(|id| outputs.get(id).copied().unwrap_or(*id))
            .collect();
        let feedback: Vec<_> = feedback
            .iter()
            .map(|(a, b)| (*a, outputs.get(b).copied().unwrap_or(*b)))
            .collect();
        let roots = root_ids.as_slice();
        let feedback = feedback.as_slice();
        fn view_base(values: &[Value], map: &FxHashMap<Id, usize>, mut id: Id) -> Id {
            while let Logical::Restride { x, .. } = values[map[&id]].op {
                id = x;
            }
            values[map[&id]].id
        }
        let retained: FxHashSet<_> = roots
            .iter()
            .copied()
            .chain(feedback.iter().flat_map(|(a, b)| [*a, *b]))
            .map(|id| view_base(&values, &by_id, id))
            .collect();
        if options.workgroups.map_or_else(
            || values.iter().map(Value::len).max().unwrap_or(0) > 4096,
            |groups| groups > 1,
        ) {
            super::workloads::split_contractions(
                &mut values,
                &mut by_id,
                max_bytes,
                u32::try_from(graph.len())
                    .map_err(|_| Error::Plan("program workload id overflow".into()))?,
            )?;
        }
        let mut uses: FxHashMap<Id, usize> = FxHashMap::default();
        for v in &values {
            if matches!(v.op, Logical::Restride { .. }) {
                continue;
            }
            for dep in children_logical(&v.op) {
                *uses.entry(view_base(&values, &by_id, dep)).or_default() += 1;
            }
        }
        fn expr_cost(e: &fusor_ir::scalar::ScalarExpr, args: &[usize]) -> usize {
            use fusor_ir::scalar::ScalarKind as S;
            let cost = |x| expr_cost(x, args);
            1 + match e.kind() {
                S::Arg(i) => args.get(*i as usize).copied().unwrap_or(1),
                S::Un { x, .. }
                | S::Cast { x, .. }
                | S::Bitcast { x, .. }
                | S::Round { x, .. }
                | S::Splat { x, .. } => cost(x),
                S::Bin { a, b, .. } | S::Cmp { a, b, .. } | S::Dot { a, b } => cost(a) + cost(b),
                S::Select { c, t, f } => cost(c) + cost(t) + cost(f),
                _ => 0,
            }
        }
        let mut costs = FxHashMap::default();
        for i in 0..values.len() {
            let v = &values[i];
            if let Logical::Map { expr, ins, outs: 1 } = &v.op {
                let args = ins
                    .iter()
                    .map(|id| {
                        costs
                            .get(&view_base(&values, &by_id, *id))
                            .copied()
                            .unwrap_or(1)
                    })
                    .collect::<Vec<_>>();
                let cost = expr_cost(expr, &args);
                if (uses.get(&v.id) == Some(&1) || ins.is_empty())
                    && !retained.contains(&v.id)
                    && cost <= 48
                {
                    costs.insert(v.id, cost);
                    values[i].forwarded = true;
                }
            }
        }
        let stages: Vec<_> = values
            .iter()
            .filter(|v| {
                !v.forwarded && !matches!(v.op, Logical::Leaf(_) | Logical::Restride { .. })
            })
            .map(|v| v.id)
            .collect();
        if stages.is_empty() {
            return Err(Error::Plan("a program needs a computation".into()));
        }
        let groups = if options.workgroups.is_none()
            && values.iter().map(Value::len).max().unwrap_or(0) <= 4096
        {
            1
        } else {
            groups
        };
        let regions =
            super::regions::schedule(&values, &by_id, &stages, groups, options.max_region_stages);
        let stages: Vec<_> = regions.iter().flat_map(|r| r.stages()).collect();
        // Workgroups do not advance through a region in lockstep. Global
        // storage may be reused only after a dispatch boundary.
        let times: FxHashMap<_, _> = if groups == 1 {
            stages
                .iter()
                .enumerate()
                .map(|(i, id)| (*id, i + 1))
                .collect()
        } else {
            regions
                .iter()
                .enumerate()
                .flat_map(|(i, r)| r.stages().map(move |id| (id, i + 1)))
                .collect()
        };
        let mut first = FxHashMap::default();
        let mut last = FxHashMap::default();
        for id in &stages {
            first.insert(*id, times[id]);
            last.insert(*id, times[id]);
        }
        for id in &stages {
            for dep in children_logical(&values[by_id[id]].op) {
                visit_sources(&values, &by_id, dep, &mut |v| {
                    if v.materialized() {
                        last.entry(v.id)
                            .and_modify(|n| *n = (*n).max(times[id]))
                            .or_insert(times[id]);
                    }
                });
            }
        }
        let end = stages.len() + 2;
        for id in roots
            .iter()
            .copied()
            .chain(feedback.iter().flat_map(|(a, b)| [*a, *b]))
        {
            visit_sources(&values, &by_id, id, &mut |v| {
                if v.materialized() {
                    last.insert(v.id, end);
                }
            });
        }
        // Inputs survive across steps, including those not updated by feedback.
        for v in &values {
            if matches!(v.op, Logical::Leaf(_)) {
                first.insert(v.id, 0);
                last.insert(v.id, end);
            }
        }
        let mut order: Vec<_> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.materialized())
            .map(|(i, _)| i)
            .collect();
        order.sort_by_key(|i| (std::cmp::Reverse(values[*i].len()), *i));
        let dedicated: u64 = order.iter().map(|i| u64::from(values[*i].len()) * 4).sum();
        let requests: Vec<_> = order
            .iter()
            .map(|i| (u64::from(values[*i].len()) * 4, 4))
            .collect();
        let (size, offsets) = fusor_ir::packing::pack_interference(
            &requests,
            fusor_ir::packing::Fit::First,
            |a, b| {
                let (a, b) = (values[order[a]].id, values[order[b]].id);
                first[&a] <= last[&b] && first[&b] <= last[&a]
            },
        )?;
        if size > max_bytes {
            return Err(Error::Plan(
                "program arena exceeds device buffer budget".into(),
            ));
        }
        let mut size =
            u32::try_from(size / 4).map_err(|_| Error::Plan("program arena exceeds u32".into()))?;
        for (i, offset) in order.into_iter().zip(offsets) {
            values[i].offset = Some((offset / 4) as u32);
        }
        // Independent pairwise check: no live values may share bytes.
        #[cfg(any(test, feature = "compiler-tests"))]
        for (i, a) in values.iter().enumerate().filter(|(_, v)| v.materialized()) {
            for b in values[..i].iter().filter(|v| v.materialized()) {
                if first[&a.id] <= last[&b.id] && first[&b.id] <= last[&a.id] {
                    let (ao, bo) = (a.offset.unwrap(), b.offset.unwrap());
                    if ao + a.len() > bo && bo + b.len() > ao {
                        return Err(Error::Plan("live program allocations overlap".into()));
                    }
                }
            }
        }
        let inputs = values
            .iter()
            .filter_map(|v| match v.op {
                Logical::Leaf(LeafKind::Buffer { .. } | LeafKind::Param { .. }) => Some(Input {
                    id: v.id,
                    dtype: v.dtype,
                    elements: v.len(),
                    offset: v.offset.unwrap(),
                    uniform: None,
                }),
                Logical::Leaf(LeafKind::Uniform { sym, .. }) => Some(Input {
                    id: v.id,
                    dtype: v.dtype,
                    elements: 1,
                    offset: v.offset.unwrap(),
                    uniform: Some(sym),
                }),
                _ => None,
            })
            .collect();
        let mut symbols = std::collections::BTreeMap::new();
        fn scan(
            e: &fusor_ir::scalar::ScalarExpr,
            out: &mut std::collections::BTreeMap<SymId, Dtype>,
        ) {
            use fusor_ir::scalar::ScalarKind as S;
            match e.kind() {
                S::Uniform(sym) => {
                    // Graph scalar bindings are f32. Each use converts from that
                    // shared word, so one symbol can also serve integer expressions.
                    out.insert(*sym, Dtype::F32);
                }
                S::Un { x, .. }
                | S::Cast { x, .. }
                | S::Bitcast { x, .. }
                | S::Round { x, .. }
                | S::Splat { x, .. } => scan(x, out),
                S::Bin { a, b, .. } | S::Cmp { a, b, .. } | S::Dot { a, b } => {
                    scan(a, out);
                    scan(b, out);
                }
                S::Select { c, t, f } => {
                    scan(c, out);
                    scan(t, out);
                    scan(f, out);
                }
                _ => {}
            }
        }
        for v in &values {
            match &v.op {
                Logical::Map { expr, .. } => scan(expr, &mut symbols),
                Logical::Fold { carrier, .. } => {
                    for e in carrier.lift.iter().chain(&carrier.merge) {
                        scan(e, &mut symbols);
                    }
                }
                _ => {}
            }
        }
        let mut uniforms = Vec::new();
        for (sym, dtype) in symbols {
            let offset = size;
            size = size
                .checked_add(1)
                .ok_or_else(|| Error::Plan("program uniforms overflow arena".into()))?;
            uniforms.push(Uniform { sym, dtype, offset });
        }
        if u64::from(size) * 4 > max_bytes {
            return Err(Error::Plan(
                "program uniforms exceed device buffer budget".into(),
            ));
        }
        let stats = ProgramStats {
            stages: stages.len(),
            kernels: regions.len() + usize::from(groups > 1 && !feedback.is_empty()),
            workgroups: groups,
            arena_bytes: u64::from(size) * 4,
            dedicated_bytes: dedicated + uniforms.len() as u64 * 4,
            state_bytes: feedback
                .iter()
                .map(|(id, _)| u64::from(values[by_id[id]].len()) * 4)
                .sum(),
        };
        Ok(Self {
            values,
            by_id,
            feedback: feedback.to_vec(),
            roots: roots.to_vec(),
            outputs,
            regions,
            inputs,
            uniforms,
            stats,
            options,
        })
    }
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }
    pub fn uniforms(&self) -> &[Uniform] {
        &self.uniforms
    }
    pub fn stats(&self) -> &ProgramStats {
        &self.stats
    }
    /// One line per stage of `region`: op kind, shape and whether it is stored.
    pub fn describe_region(&self, region: usize) -> String {
        let mut out = String::new();
        for job in &self.regions[region].jobs {
            out.push_str(&format!(" [job groups={} tiled={}]", job.groups, job.tiled));
            for id in &job.stages {
                let v = self.value(*id);
                let tag = format!("{:?}", v.op);
                let tag: String = tag.chars().take_while(|c| c.is_alphanumeric()).collect();
                out.push_str(&format!(
                    " {tag}{:?}{}",
                    v.shape,
                    if v.materialized() { "" } else { "*" }
                ));
            }
        }
        out
    }
    pub(crate) fn job_groups(&self, job: &super::regions::Job, cooperative: bool) -> u32 {
        if job.tiled {
            super::regions::matrix_groups(self.value(job.stages[0]), cooperative)
        } else {
            job.groups
        }
    }
    pub(crate) fn groups(&self, region: usize, cooperative: bool) -> u32 {
        self.regions.get(region).map_or(self.stats.workgroups, |r| {
            r.jobs.iter().map(|j| self.job_groups(j, cooperative)).sum()
        })
    }
    pub fn shaders(&self, cooperative: bool) -> Result<Vec<String>> {
        self.shaders_with(super::ProgramAcceleration {
            subgroups: cooperative,
            subgroup_scatter: cooperative,
            matrices: if cooperative {
                super::MatrixInstructions::Native
            } else {
                super::MatrixInstructions::Portable
            },
        })
    }
    pub(crate) fn shaders_with(
        &self,
        acceleration: super::ProgramAcceleration,
    ) -> Result<Vec<String>> {
        (0..self.stats.kernels)
            .map(|region| super::emit::shader(self, region, acceleration))
            .collect()
    }
    pub(crate) fn value(&self, id: Id) -> &Value {
        &self.values[self.by_id[&id]]
    }
    pub(crate) fn output(&self, id: Id) -> Result<&Value> {
        let id = if self.feedback.iter().any(|(input, _)| *input == id) {
            id
        } else {
            self.outputs.get(&id).copied().unwrap_or(id)
        };
        if !self.roots.contains(&id) && !self.feedback.iter().any(|(input, _)| *input == id) {
            return Err(Error::Plan(
                "value is not a retained program output or state".into(),
            ));
        }
        let value = self.value(id);
        if value.offset.is_none() {
            return Err(Error::Plan(
                "program readback requires a materialized output".into(),
            ));
        }
        Ok(value)
    }
}
