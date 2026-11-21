use crate::trace::{ScheduleMode, ScheduleTrace, Scheduler, TraceError, stable_digest};
use fusor_ir::dtype::{NumericContract, RoundMode};
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, MemoryLevel, ReduceKind, ScalarElement,
    Source, Stmt, Tile, TileExpr, TileExprKind, TileLiteral, WorkgroupAxis,
};
use fusor_ir::scalar::{BinOp, CmpOp, UnOp};
use fusor_ir::shape::MultiFlattenMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryValue {
    F32(f32),
    U32(u32),
    I32(i32),
    Bool(bool),
    Vector(Vec<MemoryValue>),
}

impl MemoryValue {
    fn f32(&self) -> Result<f32, SimError> {
        match self {
            Self::F32(value) => Ok(*value),
            _ => Err(SimError::Type(format!("expected f32, got {self:?}"))),
        }
    }

    fn u32(&self) -> Result<u32, SimError> {
        match self {
            Self::U32(value) => Ok(*value),
            Self::I32(value) if *value >= 0 => Ok(*value as u32),
            _ => Err(SimError::Type(format!("expected u32 index, got {self:?}"))),
        }
    }

    fn bool(&self) -> Result<bool, SimError> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(SimError::Type(format!("expected bool, got {self:?}"))),
        }
    }

    fn components(&self) -> Vec<MemoryValue> {
        match self {
            Self::Vector(values) => values.clone(),
            value => vec![value.clone()],
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(value) => [vec![1], value.to_bits().to_le_bytes().to_vec()].concat(),
            Self::U32(value) => [vec![2], value.to_le_bytes().to_vec()].concat(),
            Self::I32(value) => [vec![3], value.to_le_bytes().to_vec()].concat(),
            Self::Bool(value) => vec![4, u8::from(*value)],
            Self::Vector(values) => {
                let mut bytes = vec![5];
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    bytes.extend(value.bytes());
                }
                bytes
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindingMemory {
    pub id: u64,
    pub cells: Vec<Option<MemoryValue>>,
}

impl BindingMemory {
    pub fn uninitialized(id: u64, elements: usize) -> Self {
        Self {
            id,
            cells: vec![None; elements],
        }
    }

    pub fn f32(id: u64, values: impl IntoIterator<Item = f32>) -> Self {
        Self {
            id,
            cells: values
                .into_iter()
                .map(|v| Some(MemoryValue::F32(v)))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Limits {
    pub max_steps: u64,
    pub max_memory_bytes: u64,
    pub max_workgroups: u64,
    pub max_invocations: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            max_memory_bytes: 16 << 20,
            max_workgroups: 4096,
            max_invocations: 65_536,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    Unsupported(String),
    Bounds {
        space: String,
        index: usize,
        length: usize,
    },
    Uninitialized {
        space: String,
        index: usize,
    },
    Race {
        space: String,
        index: usize,
        first: String,
        second: String,
    },
    DivergentBarrier {
        workgroup: u32,
        reason: String,
    },
    StepLimit {
        limit: u64,
    },
    MemoryLimit {
        required: u64,
        limit: u64,
    },
    InvocationLimit {
        required: u64,
        limit: u64,
    },
    Type(String),
    Trace(TraceError),
    Execution {
        source: Box<SimError>,
        trace: ScheduleTrace,
    },
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(op) => write!(f, "unsupported KernelIr: {op}"),
            Self::Bounds {
                space,
                index,
                length,
            } => {
                write!(f, "{space} index {index} is outside length {length}")
            }
            Self::Uninitialized { space, index } => {
                write!(f, "uninitialized {space} read at index {index}")
            }
            Self::Race {
                space,
                index,
                first,
                second,
            } => {
                write!(
                    f,
                    "unordered {space} conflict at {index}: {first} vs {second}"
                )
            }
            Self::DivergentBarrier { workgroup, reason } => {
                write!(
                    f,
                    "workgroup {workgroup} barrier divergence/deadlock: {reason}"
                )
            }
            Self::StepLimit { limit } => write!(f, "simulator step limit {limit} exceeded"),
            Self::MemoryLimit { required, limit } => {
                write!(f, "simulator memory {required} exceeds limit {limit}")
            }
            Self::InvocationLimit { required, limit } => {
                write!(f, "simulator invocations {required} exceed limit {limit}")
            }
            Self::Type(message) => f.write_str(message),
            Self::Trace(error) => write!(f, "trace: {error}"),
            Self::Execution { source, .. } => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for SimError {}

impl SimError {
    pub fn schedule_trace(&self) -> Option<&ScheduleTrace> {
        match self {
            Self::Execution { trace, .. } => Some(trace),
            _ => None,
        }
    }
}

fn validate_unique_bindings(buffers: &BTreeMap<u32, BindingMemory>) -> Result<(), SimError> {
    let mut ids = HashMap::new();
    for (&binding, memory) in buffers {
        if let Some(previous) = ids.insert(memory.id, binding) {
            return Err(SimError::Unsupported(format!(
                "storage aliasing between bindings {previous} and {binding} (memory id {})",
                memory.id
            )));
        }
    }
    Ok(())
}

fn validate_kernel_storage(ir: &KernelIr) -> Result<(), SimError> {
    fn validate_element(element: ElementType) -> Result<(), SimError> {
        match element {
            ElementType::Scalar(
                ScalarElement::F32 | ScalarElement::U32 | ScalarElement::I32 | ScalarElement::Bool,
            ) => Ok(()),
            _ => Err(SimError::Unsupported(format!(
                "storage/tile element type {element:?}"
            ))),
        }
    }

    fn validate_tile_expr(
        expr: &TileExpr,
        reductions: &mut HashMap<usize, usize>,
    ) -> Result<(), SimError> {
        if let TileExprKind::Reduce { .. } = expr.kind() {
            *reductions.entry(expr.node_ptr()).or_default() += 1;
        }
        match expr.kind() {
            TileExprKind::Load {
                src: Source::Storage(view),
                ..
            } => validate_element(view.buffer.element)?,
            TileExprKind::LoadTile { tile, .. } => validate_element(tile.element)?,
            TileExprKind::Reduce { kind, .. } => {
                if let ReduceKind::Workgroup { scratch, .. } = &**kind {
                    validate_element(scratch.element)?;
                }
            }
            _ => {}
        }
        let mut error = None;
        expr.kind().visit_children(&mut |child| {
            if error.is_none() {
                error = validate_tile_expr(child, reductions).err();
            }
        });
        error.map_or(Ok(()), Err)
    }

    fn validate_statements(statements: &[Stmt]) -> Result<(), SimError> {
        for stmt in statements {
            let mut reductions = HashMap::new();
            let mut error = None;
            stmt.for_each_expr(&mut |expr| {
                if error.is_none() {
                    error = validate_tile_expr(expr, &mut reductions).err();
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
            if reductions.len() > 1 || reductions.values().any(|count| *count > 1) {
                return Err(SimError::Unsupported(
                    "multiple or repeated expression reductions in one statement".into(),
                ));
            }
            match stmt {
                Stmt::StoreTile { dst, .. } | Stmt::FillTile { dst, .. } => {
                    validate_element(dst.element)?
                }
                Stmt::Reduce { kind, scratch, .. } => {
                    for tile in scratch {
                        validate_element(tile.element)?;
                    }
                    if let ReduceKind::Workgroup { scratch, .. } = &**kind {
                        validate_element(scratch.element)?;
                    }
                }
                Stmt::If { accept, reject, .. } => {
                    validate_statements(accept)?;
                    validate_statements(reject)?;
                }
                Stmt::Loop { body, .. } => validate_statements(body)?,
                _ => {}
            }
        }
        Ok(())
    }

    for buffer in &ir.buffers {
        validate_element(buffer.element)?;
    }
    validate_statements(&ir.body)
}

impl From<TraceError> for SimError {
    fn from(value: TraceError) -> Self {
        Self::Trace(value)
    }
}

#[derive(Clone, Debug)]
pub struct Simulator {
    limits: Limits,
}

impl Simulator {
    pub fn new(limits: Limits) -> Self {
        Self { limits }
    }

    pub fn run(
        &self,
        ir: &fusor_ir::ir::kernel::KernelIr,
        buffers: &mut BTreeMap<u32, BindingMemory>,
        seed: u64,
    ) -> Result<RunResult, SimError> {
        self.run_with_mode(ir, buffers, ScheduleMode::Seed(seed))
    }

    pub fn replay(
        &self,
        ir: &fusor_ir::ir::kernel::KernelIr,
        buffers: &mut BTreeMap<u32, BindingMemory>,
        trace: &ScheduleTrace,
    ) -> Result<RunResult, SimError> {
        self.run_with_mode(ir, buffers, ScheduleMode::Replay(trace))
    }

    fn run_with_mode(
        &self,
        ir: &fusor_ir::ir::kernel::KernelIr,
        buffers: &mut BTreeMap<u32, BindingMemory>,
        mode: ScheduleMode<'_>,
    ) -> Result<RunResult, SimError> {
        let invocations_per_group = u64::from(ir.block);
        let workgroups = u64::from(ir.grid[0])
            .saturating_mul(u64::from(ir.grid[1]))
            .saturating_mul(u64::from(ir.grid[2]));
        let invocations = workgroups.saturating_mul(invocations_per_group);
        let program_digest = kernel_digest(ir);
        let input_digest = execution_config_digest(buffers, self.limits);
        let mut scheduler = Scheduler::new(mode, program_digest, input_digest)?;
        let preflight = (|| {
            validate_kernel_storage(ir)?;
            validate_unique_bindings(buffers)?;
            if workgroups > self.limits.max_workgroups {
                return Err(SimError::InvocationLimit {
                    required: workgroups,
                    limit: self.limits.max_workgroups,
                });
            }
            if invocations > self.limits.max_invocations {
                return Err(SimError::InvocationLimit {
                    required: invocations,
                    limit: self.limits.max_invocations,
                });
            }
            if ir.block == 0 {
                return Err(SimError::Type("KernelIr block width is zero".into()));
            }
            let global_bytes = buffers
                .values()
                .map(|buffer| (buffer.cells.len() as u64).saturating_mul(4))
                .sum::<u64>();
            if global_bytes > self.limits.max_memory_bytes {
                return Err(SimError::MemoryLimit {
                    required: global_bytes,
                    limit: self.limits.max_memory_bytes,
                });
            }
            Ok(global_bytes)
        })();
        let global_bytes = match preflight {
            Ok(bytes) => bytes,
            Err(source) => {
                return Err(SimError::Execution {
                    source: Box::new(source),
                    trace: scheduler.finish()?,
                });
            }
        };
        let mut simulator = Execution {
            ir,
            buffers,
            limits: self.limits,
            global_bytes,
            steps: 0,
            accesses: HashMap::new(),
        };
        for workgroup in 0..workgroups as u32 {
            let xyz = workgroup_coordinates(workgroup, ir.grid);
            if let Err(error) = simulator.run_workgroup(&mut scheduler, workgroup, xyz) {
                if matches!(error, SimError::Trace(_)) {
                    return Err(error);
                }
                let trace = scheduler.finish()?;
                return Err(SimError::Execution {
                    source: Box::new(error),
                    trace,
                });
            }
        }
        let trace = scheduler.finish()?;
        Ok(RunResult {
            trace,
            steps: simulator.steps,
        })
    }
}

fn kernel_digest(ir: &KernelIr) -> u64 {
    // KernelIr's Debug/derived hashes include cached expression hashes built
    // from allocation addresses. Use the compiler's declaration-renaming,
    // pointer-free identity; unlike pipeline identity, replay includes the grid.
    let mut bytes = fusor_tile::planner::kernel_identity(ir)
        .to_le_bytes()
        .to_vec();
    for extent in ir.grid {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    stable_digest(&bytes)
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub trace: ScheduleTrace,
    pub steps: u64,
}

pub fn execute(
    ir: &fusor_ir::ir::kernel::KernelIr,
    buffers: &mut BTreeMap<u32, BindingMemory>,
    seed: u64,
    limits: Limits,
) -> Result<RunResult, SimError> {
    Simulator::new(limits).run(ir, buffers, seed)
}

pub fn replay(
    ir: &fusor_ir::ir::kernel::KernelIr,
    buffers: &mut BTreeMap<u32, BindingMemory>,
    trace: &ScheduleTrace,
    limits: Limits,
) -> Result<RunResult, SimError> {
    Simulator::new(limits).replay(ir, buffers, trace)
}

struct Execution<'a> {
    ir: &'a fusor_ir::ir::kernel::KernelIr,
    buffers: &'a mut BTreeMap<u32, BindingMemory>,
    limits: Limits,
    global_bytes: u64,
    steps: u64,
    accesses: HashMap<(Space, u32, u64, usize), Vec<Access>>,
}

impl Execution<'_> {
    fn run_workgroup(
        &mut self,
        scheduler: &mut Scheduler<'_>,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<(), SimError> {
        let mut group = GroupState::default();
        let mut lanes: Vec<Lane<'_>> = (0..self.ir.block)
            .map(|lane| Lane::new(&self.ir.body, lane))
            .collect();
        loop {
            for lane in &mut lanes {
                if lane.waiting.is_none() {
                    self.normalize_lane(lane, &mut group, workgroup, xyz)?;
                }
            }
            let runnable: Vec<u32> = lanes
                .iter()
                .enumerate()
                .filter_map(|(index, lane)| {
                    (lane.active && lane.waiting.is_none()).then_some(index as u32)
                })
                .collect();
            if runnable.is_empty() {
                if lanes.iter().all(|lane| !lane.active) {
                    return Ok(());
                }
                if self.resolve_waiters(&mut lanes, &mut group, workgroup, xyz)? {
                    continue;
                }
                return Err(SimError::DivergentBarrier {
                    workgroup,
                    reason: "no runnable lane can reach the outstanding synchronization".into(),
                });
            }
            let lane_id = scheduler.choose(workgroup, "statement", &runnable)?;
            self.steps += 1;
            if self.steps > self.limits.max_steps {
                return Err(SimError::StepLimit {
                    limit: self.limits.max_steps,
                });
            }
            let lane_index = lane_id as usize;
            let pending = peek_stmt(&lanes[lane_index]).and_then(first_reduce_in_stmt);
            if let Some(reduce) = pending {
                let pointer = reduce.node_ptr();
                let visit = group
                    .reduction_visits
                    .get(&(lane_id, pointer))
                    .copied()
                    .unwrap_or(0);
                if !group.reductions.contains_key(&(pointer, visit)) {
                    lanes[lane_index].waiting = Some(Waiting::ExprReduce {
                        expr: reduce,
                        visit,
                    });
                    continue;
                }
            }
            let stmt = next_stmt(&mut lanes[lane_index]).ok_or_else(|| {
                SimError::Type("scheduler selected a lane without a statement".into())
            })?;
            self.step_statement(
                stmt,
                &mut lanes[lane_index],
                &mut group,
                lane_id,
                workgroup,
                xyz,
            )?;
        }
    }

    fn normalize_lane(
        &mut self,
        lane: &mut Lane<'_>,
        group: &mut GroupState,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<(), SimError> {
        loop {
            let Some(frame) = lane.frames.last() else {
                lane.active = false;
                return Ok(());
            };
            match frame {
                Frame::Sequence { body, pc } if *pc >= body.len() => {
                    lane.frames.pop();
                }
                Frame::Sequence { .. } => return Ok(()),
                Frame::Loop { .. } => {
                    self.steps = self.steps.saturating_add(1);
                    if self.steps > self.limits.max_steps {
                        return Err(SimError::StepLimit {
                            limit: self.limits.max_steps,
                        });
                    }
                    let (count, iteration, index, accumulators, body) = match lane.frames.last() {
                        Some(Frame::Loop {
                            count,
                            iteration,
                            index,
                            accumulators,
                            body,
                        }) => (*count, *iteration, *index, *accumulators, *body),
                        _ => unreachable!(),
                    };
                    let mut updates = Vec::with_capacity(accumulators.len());
                    for Accumulator { local, update, .. } in accumulators {
                        updates.push((
                            local.clone(),
                            self.eval(update, lane, group, lane.lane, workgroup, xyz)?,
                        ));
                    }
                    for (local, value) in updates {
                        lane.locals.insert(local.id(), value);
                    }
                    let next_iteration = iteration.saturating_add(1);
                    if count.is_none_or(|count| next_iteration < count) {
                        if let Some(index) = index {
                            lane.locals
                                .insert(index.id(), MemoryValue::U32(next_iteration));
                        }
                        if let Some(Frame::Loop { iteration, .. }) = lane.frames.last_mut() {
                            *iteration = next_iteration;
                        }
                        lane.frames.push(Frame::Sequence { body, pc: 0 });
                    } else {
                        lane.frames.pop();
                    }
                }
            }
        }
    }

    fn step_statement<'a>(
        &mut self,
        stmt: &'a Stmt,
        lane: &mut Lane<'a>,
        group: &mut GroupState,
        lane_id: u32,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<(), SimError> {
        match stmt {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            } => {
                if self
                    .eval(mask, lane, group, lane_id, workgroup, xyz)?
                    .bool()?
                {
                    let index = self.eval_addr(addr, dst, lane, group, lane_id, workgroup, xyz)?;
                    let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                    self.store_storage(dst, index, value, lane_id, workgroup, group)?;
                }
            }
            Stmt::AtomicAdd {
                dst,
                addr,
                value,
                mask,
            } => {
                if self
                    .eval(mask, lane, group, lane_id, workgroup, xyz)?
                    .bool()?
                {
                    let index = self.eval_addr(addr, dst, lane, group, lane_id, workgroup, xyz)?;
                    let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                    let old = self.load_storage(dst, index, lane_id, workgroup, group, true)?;
                    let sum = binary(BinOp::Add, old, value)?;
                    self.store_storage_inner(dst, index, sum, lane_id, workgroup, group, true)?;
                }
            }
            Stmt::StoreLocal { dst, value } => {
                let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                lane.locals.insert(dst.id(), value);
            }
            Stmt::StoreTile { dst, index, value } => {
                let index = self
                    .eval(index, lane, group, lane_id, workgroup, xyz)?
                    .u32()? as usize;
                let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                self.store_tile(dst, index, value, lane_id, workgroup, group)?;
            }
            Stmt::FillTile { dst, value, bounds } => {
                if bounds.iter().any(Option::is_some) {
                    return Err(SimError::Unsupported(
                        "bounded FillTile (including 2D bounds)".into(),
                    ));
                }
                let (source, row, col, mask, fill) = match value.kind() {
                    TileExprKind::Load {
                        src: Source::Storage(source),
                        addr,
                        mask,
                        fill,
                    } => {
                        let Addr::Rc2 { row, col } = &**addr else {
                            return Err(SimError::Unsupported(
                                "FillTile source must use a rank-2 address".into(),
                            ));
                        };
                        (source, row, col, mask, fill)
                    }
                    _ => {
                        return Err(SimError::Unsupported(
                            "FillTile source must be a dense storage Load".into(),
                        ));
                    }
                };
                let [rows, cols] = dst.layout.extents.as_slice() else {
                    return Err(SimError::Unsupported(
                        "FillTile requires a rank-2 destination tile".into(),
                    ));
                };
                let row_base = self
                    .eval(row, lane, group, lane_id, workgroup, xyz)?
                    .u32()?;
                let col_base = self
                    .eval(col, lane, group, lane_id, workgroup, xyz)?
                    .u32()?;
                let count = usize::try_from(dst.layout.element_count()).map_err(|_| {
                    SimError::MemoryLimit {
                        required: u64::MAX,
                        limit: self.limits.max_memory_bytes,
                    }
                })?;
                let mut index = lane_id as usize;
                while index < count {
                    let local_row = index / *cols as usize;
                    let local_col = index % *cols as usize;
                    if local_row >= *rows as usize {
                        return Err(SimError::Bounds {
                            space: "FillTile destination".into(),
                            index,
                            length: count,
                        });
                    }
                    let value = if self
                        .eval(mask, lane, group, lane_id, workgroup, xyz)?
                        .bool()?
                    {
                        let source_index = map_coordinates(
                            &source.layout.indexing,
                            &[
                                row_base.wrapping_add(local_row as u32) as usize,
                                col_base.wrapping_add(local_col as u32) as usize,
                            ],
                        )?;
                        self.load_storage(
                            source,
                            source.offset as usize + source_index,
                            lane_id,
                            workgroup,
                            group,
                            false,
                        )?
                    } else {
                        self.eval(fill, lane, group, lane_id, workgroup, xyz)?
                    };
                    self.store_tile(dst, index, value, lane_id, workgroup, group)?;
                    index = index.saturating_add(self.ir.block as usize);
                }
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                let condition = self
                    .eval(condition, lane, group, lane_id, workgroup, xyz)?
                    .bool()?;
                lane.frames.push(Frame::Sequence {
                    body: if condition { accept } else { reject },
                    pc: 0,
                });
            }
            Stmt::Loop {
                count,
                index,
                accumulators,
                body,
            } => {
                let count = count
                    .as_ref()
                    .map(|count| {
                        self.eval(count, lane, group, lane_id, workgroup, xyz)?
                            .u32()
                    })
                    .transpose()?;
                for Accumulator { local, init, .. } in accumulators {
                    let value = self.eval(init, lane, group, lane_id, workgroup, xyz)?;
                    lane.locals.insert(local.id(), value);
                }
                if let Some(index) = index {
                    lane.locals.insert(index.id(), MemoryValue::U32(0));
                }
                if count == Some(0) {
                    return Ok(());
                }
                lane.frames.push(Frame::Loop {
                    count,
                    iteration: 0,
                    index: index.as_ref(),
                    accumulators,
                    body,
                });
                lane.frames.push(Frame::Sequence { body, pc: 0 });
            }
            Stmt::Break => {
                if let Some(index) = lane
                    .frames
                    .iter()
                    .rposition(|frame| matches!(frame, Frame::Loop { .. }))
                {
                    lane.frames.truncate(index);
                } else {
                    return Err(SimError::Type("break outside loop".into()));
                }
            }
            Stmt::Return => {
                lane.active = false;
                lane.frames.clear();
            }
            Stmt::Barrier | Stmt::StorageBarrier => {
                let statement = stmt as *const Stmt as usize;
                let instance = lane.barrier_visits.entry(statement).or_default();
                let current = *instance;
                *instance = instance.saturating_add(1);
                let kind = if matches!(stmt, Stmt::Barrier) {
                    BarrierKind::Workgroup
                } else {
                    BarrierKind::Storage
                };
                lane.waiting = Some(Waiting::Barrier {
                    kind,
                    statement,
                    instance: current,
                });
            }
            Stmt::Reduce { .. } => {
                lane.waiting = Some(Waiting::Reduce(stmt as *const Stmt as usize));
            }
            Stmt::CoopStore { .. } | Stmt::CoopStoreTile { .. } => {
                return Err(SimError::Unsupported(format!("{stmt:?}")));
            }
        }
        Ok(())
    }

    fn resolve_waiters(
        &mut self,
        lanes: &mut [Lane<'_>],
        group: &mut GroupState,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<bool, SimError> {
        let waiting: Vec<_> = lanes
            .iter()
            .filter(|lane| lane.active)
            .map(|lane| lane.waiting.clone())
            .collect();
        if waiting.iter().any(Option::is_none)
            || (waiting.iter().any(Option::is_some) && lanes.iter().any(|lane| !lane.active))
        {
            return Err(SimError::DivergentBarrier {
                workgroup,
                reason: "some invocation exited or skipped a barrier".into(),
            });
        }
        if waiting.is_empty() {
            return Ok(false);
        }
        let first = waiting[0].clone().unwrap();
        if waiting.iter().any(|entry| entry.as_ref() != Some(&first)) {
            return Err(SimError::DivergentBarrier {
                workgroup,
                reason: "invocations reached different synchronization points".into(),
            });
        }
        match first {
            Waiting::Barrier { kind, .. } => {
                for lane in lanes.iter_mut().filter(|lane| lane.active) {
                    lane.waiting = None;
                }
                group.workgroup_generation = group.workgroup_generation.wrapping_add(1);
                if kind == BarrierKind::Storage {
                    group.storage_generation = group.storage_generation.wrapping_add(1);
                }
                Ok(true)
            }
            Waiting::Reduce(pointer) => {
                let stmt = find_stmt(&self.ir.body, pointer).ok_or_else(|| {
                    SimError::Type("collective statement was not found in KernelIr".into())
                })?;
                self.resolve_reduce(stmt, lanes, group, workgroup, xyz)?;
                for lane in lanes.iter_mut().filter(|lane| lane.active) {
                    lane.waiting = None;
                }
                group.workgroup_generation = group.workgroup_generation.wrapping_add(1);
                Ok(true)
            }
            Waiting::ExprReduce { expr, visit } => {
                self.resolve_expr_reduce(&expr, visit, lanes, group, workgroup, xyz)?;
                for lane in lanes.iter_mut().filter(|lane| lane.active) {
                    lane.waiting = None;
                }
                group.workgroup_generation = group.workgroup_generation.wrapping_add(1);
                Ok(true)
            }
        }
    }

    fn resolve_expr_reduce(
        &mut self,
        expr: &TileExpr,
        visit: u64,
        lanes: &[Lane<'_>],
        group: &mut GroupState,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<(), SimError> {
        let TileExprKind::Reduce { op, kind, value } = expr.kind() else {
            return Err(SimError::Type(
                "expression reduction rendezvous is invalid".into(),
            ));
        };
        let group_size = match kind.as_ref() {
            ReduceKind::Workgroup { group_size, .. } => *group_size,
            ReduceKind::Subgroup => return Err(SimError::Unsupported("subgroup reduction".into())),
        };
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > lanes.len() as u32
            || !(lanes.len() as u32).is_multiple_of(group_size)
        {
            return Err(SimError::Unsupported(format!(
                "workgroup expression reduction group size {group_size} does not divide block {}",
                lanes.len()
            )));
        }
        let mut partials = Vec::with_capacity(lanes.len());
        for (lane_id, lane) in lanes.iter().enumerate() {
            partials.push(self.eval(value, lane, group, lane_id as u32, workgroup, xyz)?);
        }
        let mut results = Vec::with_capacity(lanes.len());
        for start in (0..lanes.len()).step_by(group_size as usize) {
            let mut value = partials[start].clone();
            for partial in partials
                .iter()
                .skip(start + 1)
                .take(group_size as usize - 1)
            {
                value = binary(op.binary(), value, partial.clone())?;
            }
            results.extend(std::iter::repeat_n(value, group_size as usize));
        }
        group
            .reductions
            .insert((expr.node_ptr(), visit), (results, lanes.len()));
        Ok(())
    }

    fn resolve_reduce(
        &mut self,
        stmt: &Stmt,
        lanes: &mut [Lane<'_>],
        group: &mut GroupState,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<(), SimError> {
        let Stmt::Reduce {
            kind,
            values,
            merge,
            fast,
            outs,
            ..
        } = stmt
        else {
            return Err(SimError::Type(
                "reduction rendezvous points at a non-reduction".into(),
            ));
        };
        let group_size = match kind.as_ref() {
            ReduceKind::Workgroup { group_size, .. } => *group_size,
            ReduceKind::Subgroup => return Err(SimError::Unsupported("subgroup reduction".into())),
        };
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > lanes.len() as u32
            || !(lanes.len() as u32).is_multiple_of(group_size)
        {
            return Err(SimError::Unsupported(format!(
                "workgroup reduction group size {group_size} does not divide block {}",
                lanes.len()
            )));
        }
        let mut partials = Vec::with_capacity(lanes.len());
        for (lane_id, lane) in lanes.iter().enumerate() {
            let mut row = Vec::with_capacity(values.len());
            for value in values {
                row.push(self.eval(value, lane, group, lane_id as u32, workgroup, xyz)?);
            }
            partials.push(row);
        }
        let mut group_results = Vec::new();
        for start in (0..lanes.len()).step_by(group_size as usize) {
            let mut tree = partials[start..start + group_size as usize].to_vec();
            if values.len() == 1 && fast.is_some() {
                let op = fast.unwrap().binary();
                let mut value = tree[0][0].clone();
                for partial in tree.iter().skip(1) {
                    value = binary(op, value, partial[0].clone())?;
                }
                group_results.push(vec![value]);
                continue;
            }
            let mut width = group_size as usize;
            while width > 1 {
                let half = width / 2;
                for offset in 0..half {
                    let left = tree[offset].clone();
                    let right = tree[offset + half].clone();
                    let mut temporary = Lane::new(&[], 0);
                    for (local, value) in merge.lhs.iter().zip(left) {
                        temporary.locals.insert(local.id(), value);
                    }
                    for (local, value) in merge.rhs.iter().zip(right) {
                        temporary.locals.insert(local.id(), value);
                    }
                    let merged = merge
                        .body
                        .iter()
                        .map(|expr| self.eval(expr, &temporary, group, 0, workgroup, xyz))
                        .collect::<Result<Vec<_>, _>>()?;
                    tree[offset] = merged;
                }
                width = half;
            }
            group_results.push(tree[0].clone());
        }
        for (group_index, start) in (0..lanes.len()).step_by(group_size as usize).enumerate() {
            for lane in lanes.iter_mut().skip(start).take(group_size as usize) {
                for (local, value) in outs.iter().zip(group_results[group_index].iter()) {
                    lane.locals.insert(local.id(), value.clone());
                }
            }
        }
        Ok(())
    }

    fn eval(
        &mut self,
        expr: &TileExpr,
        lane: &Lane<'_>,
        group: &mut GroupState,
        lane_id: u32,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<MemoryValue, SimError> {
        match expr.kind() {
            TileExprKind::Literal(literal) => literal_value(*literal),
            TileExprKind::Builtin(builtin) => match builtin {
                Builtin::Lane => Ok(MemoryValue::U32(lane_id)),
                Builtin::ProgramId(axis) => Ok(MemoryValue::U32(axis_value(*axis, xyz))),
                Builtin::NumWorkgroups(axis) => {
                    Ok(MemoryValue::U32(axis_value(*axis, self.ir.grid)))
                }
                Builtin::SubgroupId
                | Builtin::SubgroupLane
                | Builtin::SubgroupSize
                | Builtin::NumSubgroups => {
                    Err(SimError::Unsupported(format!("builtin {builtin:?}")))
                }
            },
            TileExprKind::LoadLocal(local) => {
                lane.locals
                    .get(&local.id())
                    .cloned()
                    .ok_or_else(|| SimError::Uninitialized {
                        space: "local".into(),
                        index: local.id() as usize,
                    })
            }
            TileExprKind::Load {
                src,
                addr,
                mask,
                fill,
            } => {
                if !self
                    .eval(mask, lane, group, lane_id, workgroup, xyz)?
                    .bool()?
                {
                    return self.eval(fill, lane, group, lane_id, workgroup, xyz);
                }
                let Source::Storage(view) = src else {
                    return Err(SimError::Unsupported("quantized storage load".into()));
                };
                let index = self.eval_addr(addr, view, lane, group, lane_id, workgroup, xyz)?;
                self.load_storage(view, index, lane_id, workgroup, group, false)
            }
            TileExprKind::LoadTile { tile, index } => {
                let index = self
                    .eval(index, lane, group, lane_id, workgroup, xyz)?
                    .u32()? as usize;
                self.load_tile(tile, index, lane_id, workgroup, group)
            }
            TileExprKind::Unary { op, value, .. } => {
                let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                unary(*op, value)
            }
            TileExprKind::Binary {
                op, left, right, ..
            } => {
                let left = self.eval(left, lane, group, lane_id, workgroup, xyz)?;
                let right = self.eval(right, lane, group, lane_id, workgroup, xyz)?;
                binary(*op, left, right)
            }
            TileExprKind::Compare { op, left, right } => {
                let left = self.eval(left, lane, group, lane_id, workgroup, xyz)?;
                let right = self.eval(right, lane, group, lane_id, workgroup, xyz)?;
                compare(*op, left, right)
            }
            TileExprKind::Round { mode, value } => {
                let value = self
                    .eval(value, lane, group, lane_id, workgroup, xyz)?
                    .f32()?;
                Ok(MemoryValue::F32(round(*mode, value)))
            }
            TileExprKind::Cast { value, to } => {
                let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                cast(value, *to)
            }
            TileExprKind::Bitcast { value, to } => {
                let value = self.eval(value, lane, group, lane_id, workgroup, xyz)?;
                bitcast(value, *to)
            }
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                if self
                    .eval(condition, lane, group, lane_id, workgroup, xyz)?
                    .bool()?
                {
                    self.eval(accept, lane, group, lane_id, workgroup, xyz)
                } else {
                    self.eval(reject, lane, group, lane_id, workgroup, xyz)
                }
            }
            TileExprKind::Vec { parts, .. } => {
                let mut values = Vec::with_capacity(parts.len());
                for part in parts {
                    values.push(self.eval(part, lane, group, lane_id, workgroup, xyz)?);
                }
                Ok(MemoryValue::Vector(values))
            }
            TileExprKind::VecComponent { vector, component } => {
                let value = self.eval(vector, lane, group, lane_id, workgroup, xyz)?;
                match value {
                    MemoryValue::Vector(values) => values
                        .get(*component as usize)
                        .cloned()
                        .ok_or_else(|| SimError::Bounds {
                            space: "vector component".into(),
                            index: *component as usize,
                            length: values.len(),
                        }),
                    _ => Err(SimError::Type(
                        "VecComponent operand is not a vector".into(),
                    )),
                }
            }
            TileExprKind::Dot { left, right } => {
                let left = self
                    .eval(left, lane, group, lane_id, workgroup, xyz)?
                    .components();
                let right = self
                    .eval(right, lane, group, lane_id, workgroup, xyz)?
                    .components();
                if left.len() != right.len() {
                    return Err(SimError::Type("dot vector widths differ".into()));
                }
                let mut total = 0.0f32;
                for (a, b) in left.into_iter().zip(right) {
                    total += a.f32()? * b.f32()?;
                }
                Ok(MemoryValue::F32(total))
            }
            TileExprKind::Reduce { .. } => {
                let pointer = expr.node_ptr();
                let visit = group
                    .reduction_visits
                    .get(&(lane_id, pointer))
                    .copied()
                    .unwrap_or(0);
                let key = (pointer, visit);
                let (value, empty) = {
                    let (values, remaining) = group.reductions.get_mut(&key).ok_or_else(|| {
                        SimError::Type(
                            "reduction expression evaluated outside its rendezvous".into(),
                        )
                    })?;
                    let value =
                        values
                            .get(lane_id as usize)
                            .cloned()
                            .ok_or_else(|| SimError::Bounds {
                                space: "reduction lane".into(),
                                index: lane_id as usize,
                                length: values.len(),
                            })?;
                    *remaining = remaining.saturating_sub(1);
                    (value, *remaining == 0)
                };
                if empty {
                    group.reductions.remove(&key);
                }
                group
                    .reduction_visits
                    .insert((lane_id, pointer), visit.wrapping_add(1));
                Ok(value)
            }
            TileExprKind::CoopZero { .. }
            | TileExprKind::CoopLoad { .. }
            | TileExprKind::CoopMma { .. } => {
                Err(SimError::Unsupported(format!("{:?}", expr.kind())))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_addr(
        &mut self,
        addr: &Addr,
        view: &fusor_ir::ir::kernel::StorageView,
        lane: &Lane<'_>,
        group: &mut GroupState,
        lane_id: u32,
        workgroup: u32,
        xyz: [u32; 3],
    ) -> Result<usize, SimError> {
        let base = match addr {
            Addr::Linear(index) => {
                let index = self
                    .eval(index, lane, group, lane_id, workgroup, xyz)?
                    .u32()?;
                map_flat_index(&view.layout.indexing, &view.layout.extents, index as usize)?
            }
            Addr::Rc2 { row, col } => {
                let row = self
                    .eval(row, lane, group, lane_id, workgroup, xyz)?
                    .u32()? as usize;
                let col = self
                    .eval(col, lane, group, lane_id, workgroup, xyz)?
                    .u32()? as usize;
                map_coordinates(&view.layout.indexing, &[row, col])?
            }
        };
        Ok(view.offset as usize + base)
    }

    fn load_storage(
        &mut self,
        view: &fusor_ir::ir::kernel::StorageView,
        index: usize,
        lane: u32,
        workgroup: u32,
        group: &GroupState,
        atomic: bool,
    ) -> Result<MemoryValue, SimError> {
        let binding = view.buffer.binding;
        let memory = self
            .buffers
            .get(&binding)
            .ok_or_else(|| SimError::Type(format!("storage binding {binding} is not supplied")))?;
        let length = memory.cells.len();
        let memory_id = memory.id;
        let value = memory
            .cells
            .get(index)
            .ok_or_else(|| SimError::Bounds {
                space: format!("storage binding {binding}"),
                index,
                length,
            })?
            .clone();
        self.record_access(
            Space::Storage,
            memory_id,
            index,
            false,
            atomic,
            lane,
            workgroup,
            group,
        )?;
        value.ok_or_else(|| SimError::Uninitialized {
            space: format!("storage binding {binding}"),
            index,
        })
    }

    fn store_storage(
        &mut self,
        view: &fusor_ir::ir::kernel::StorageView,
        index: usize,
        value: MemoryValue,
        lane: u32,
        workgroup: u32,
        group: &GroupState,
    ) -> Result<(), SimError> {
        self.store_storage_inner(view, index, value, lane, workgroup, group, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn store_storage_inner(
        &mut self,
        view: &fusor_ir::ir::kernel::StorageView,
        index: usize,
        value: MemoryValue,
        lane: u32,
        workgroup: u32,
        group: &GroupState,
        atomic: bool,
    ) -> Result<(), SimError> {
        let binding = view.buffer.binding;
        let memory = self
            .buffers
            .get(&binding)
            .ok_or_else(|| SimError::Type(format!("storage binding {binding} is not supplied")))?;
        let length = memory.cells.len();
        let memory_id = memory.id;
        if index >= length {
            return Err(SimError::Bounds {
                space: format!("storage binding {binding}"),
                index,
                length,
            });
        }
        self.record_access(
            Space::Storage,
            memory_id,
            index,
            true,
            atomic,
            lane,
            workgroup,
            group,
        )?;
        self.buffers.get_mut(&binding).unwrap().cells[index] = Some(value);
        Ok(())
    }

    fn load_tile(
        &mut self,
        tile: &Tile,
        index: usize,
        lane: u32,
        workgroup: u32,
        group: &mut GroupState,
    ) -> Result<MemoryValue, SimError> {
        self.ensure_tile(tile, group)?;
        let cells = group.tiles.get(&tile.id()).unwrap();
        let value = cells.get(index).ok_or_else(|| SimError::Bounds {
            space: format!("workgroup tile {}", tile.name),
            index,
            length: cells.len(),
        })?;
        self.record_access(
            Space::Workgroup,
            tile.id(),
            index,
            false,
            false,
            lane,
            workgroup,
            group,
        )?;
        value.clone().ok_or_else(|| SimError::Uninitialized {
            space: format!("workgroup tile {}", tile.name),
            index,
        })
    }

    fn store_tile(
        &mut self,
        tile: &Tile,
        index: usize,
        value: MemoryValue,
        lane: u32,
        workgroup: u32,
        group: &mut GroupState,
    ) -> Result<(), SimError> {
        self.ensure_tile(tile, group)?;
        let length = group.tiles.get(&tile.id()).unwrap().len();
        if index >= length {
            return Err(SimError::Bounds {
                space: format!("workgroup tile {}", tile.name),
                index,
                length,
            });
        }
        self.record_access(
            Space::Workgroup,
            tile.id(),
            index,
            true,
            false,
            lane,
            workgroup,
            group,
        )?;
        group.tiles.get_mut(&tile.id()).unwrap()[index] = Some(value);
        Ok(())
    }

    fn ensure_tile(&self, tile: &Tile, group: &mut GroupState) -> Result<(), SimError> {
        if !group.tiles.contains_key(&tile.id()) {
            let elements = usize::try_from(tile.layout.element_count()).map_err(|_| {
                SimError::MemoryLimit {
                    required: u64::MAX,
                    limit: self.limits.max_memory_bytes,
                }
            })?;
            let tile_bytes = (elements as u64).saturating_mul(4);
            let group_bytes = group
                .tiles
                .values()
                .map(|cells| (cells.len() as u64).saturating_mul(4))
                .sum::<u64>();
            let required = self
                .global_bytes
                .saturating_add(group_bytes)
                .saturating_add(tile_bytes);
            if required > self.limits.max_memory_bytes {
                return Err(SimError::MemoryLimit {
                    required,
                    limit: self.limits.max_memory_bytes,
                });
            }
            group.tiles.insert(tile.id(), vec![None; elements]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_access(
        &mut self,
        space: Space,
        memory: u64,
        index: usize,
        write: bool,
        atomic: bool,
        lane: u32,
        workgroup: u32,
        group: &GroupState,
    ) -> Result<(), SimError> {
        let scope = if space == Space::Workgroup {
            workgroup
        } else {
            0
        };
        let entry = self
            .accesses
            .entry((space, scope, memory, index))
            .or_default();
        for previous in entry.iter() {
            if !(write || previous.write) || (atomic && previous.atomic) {
                continue;
            }
            let same_lane = previous.workgroup == workgroup && previous.lane == lane;
            let synchronized = previous.workgroup == workgroup
                && match space {
                    Space::Workgroup => previous.workgroup_generation != group.workgroup_generation,
                    Space::Storage => previous.storage_generation != group.storage_generation,
                };
            if !same_lane && !synchronized {
                return Err(SimError::Race {
                    space: match space {
                        Space::Storage => "storage".into(),
                        Space::Workgroup => "workgroup".into(),
                    },
                    index,
                    first: format!("wg{}:lane{}", previous.workgroup, previous.lane),
                    second: format!("wg{workgroup}:lane{lane}"),
                });
            }
        }
        entry.push(Access {
            workgroup,
            lane,
            workgroup_generation: group.workgroup_generation,
            storage_generation: group.storage_generation,
            write,
            atomic,
        });
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Space {
    Storage,
    Workgroup,
}

#[derive(Clone, Debug)]
struct Access {
    workgroup: u32,
    lane: u32,
    workgroup_generation: u64,
    storage_generation: u64,
    write: bool,
    atomic: bool,
}

#[derive(Default)]
struct GroupState {
    tiles: HashMap<u64, Vec<Option<MemoryValue>>>,
    reductions: HashMap<(usize, u64), (Vec<MemoryValue>, usize)>,
    reduction_visits: HashMap<(u32, usize), u64>,
    workgroup_generation: u64,
    storage_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BarrierKind {
    Workgroup,
    Storage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Waiting {
    Barrier {
        kind: BarrierKind,
        statement: usize,
        instance: u64,
    },
    Reduce(usize),
    ExprReduce {
        expr: TileExpr,
        visit: u64,
    },
}

enum Frame<'a> {
    Sequence {
        body: &'a [Stmt],
        pc: usize,
    },
    Loop {
        count: Option<u32>,
        iteration: u32,
        index: Option<&'a fusor_ir::ir::kernel::Local>,
        accumulators: &'a [Accumulator],
        body: &'a [Stmt],
    },
}

struct Lane<'a> {
    lane: u32,
    active: bool,
    waiting: Option<Waiting>,
    barrier_visits: HashMap<usize, u64>,
    frames: Vec<Frame<'a>>,
    locals: HashMap<u64, MemoryValue>,
}

impl<'a> Lane<'a> {
    fn new(body: &'a [Stmt], lane: u32) -> Self {
        Self {
            lane,
            active: true,
            waiting: None,
            barrier_visits: HashMap::new(),
            frames: vec![Frame::Sequence { body, pc: 0 }],
            locals: HashMap::new(),
        }
    }
}

fn next_stmt<'a>(lane: &mut Lane<'a>) -> Option<&'a Stmt> {
    let Some(Frame::Sequence { body, pc }) = lane.frames.last_mut() else {
        return None;
    };
    let stmt = body.get(*pc)?;
    *pc += 1;
    Some(stmt)
}

fn peek_stmt<'a>(lane: &Lane<'a>) -> Option<&'a Stmt> {
    match lane.frames.last()? {
        Frame::Sequence { body, pc } => body.get(*pc),
        Frame::Loop { .. } => None,
    }
}

fn first_reduce_in_stmt(stmt: &Stmt) -> Option<TileExpr> {
    let mut found = None;
    stmt.for_each_expr(&mut |expr| {
        if found.is_none() {
            found = first_reduce_in_expr(expr);
        }
    });
    found
}

fn first_reduce_in_expr(expr: &TileExpr) -> Option<TileExpr> {
    if matches!(expr.kind(), TileExprKind::Reduce { .. }) {
        return Some(expr.clone());
    }
    let mut found = None;
    expr.kind().visit_children(&mut |child| {
        if found.is_none() {
            found = first_reduce_in_expr(child);
        }
    });
    found
}

fn buffers_digest(buffers: &BTreeMap<u32, BindingMemory>) -> u64 {
    let mut bytes = Vec::new();
    for (binding, memory) in buffers {
        bytes.extend_from_slice(&binding.to_le_bytes());
        bytes.extend_from_slice(&memory.id.to_le_bytes());
        bytes.extend_from_slice(&(memory.cells.len() as u64).to_le_bytes());
        for cell in &memory.cells {
            if let Some(value) = cell {
                bytes.push(1);
                bytes.extend(value.bytes());
            } else {
                bytes.push(0);
            }
        }
    }
    stable_digest(&bytes)
}

fn execution_config_digest(buffers: &BTreeMap<u32, BindingMemory>, limits: Limits) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&limits.max_steps.to_le_bytes());
    bytes.extend_from_slice(&limits.max_memory_bytes.to_le_bytes());
    bytes.extend_from_slice(&limits.max_workgroups.to_le_bytes());
    bytes.extend_from_slice(&limits.max_invocations.to_le_bytes());
    bytes.extend_from_slice(&buffers_digest(buffers).to_le_bytes());
    stable_digest(&bytes)
}

fn workgroup_coordinates(index: u32, grid: [u32; 3]) -> [u32; 3] {
    let x = if grid[0] == 0 { 0 } else { index % grid[0] };
    let rest = index.checked_div(grid[0]).unwrap_or(0);
    let y = if grid[1] == 0 { 0 } else { rest % grid[1] };
    let z = rest.checked_div(grid[1]).unwrap_or(0);
    [x, y, z]
}

fn axis_value(axis: WorkgroupAxis, xyz: [u32; 3]) -> u32 {
    match axis {
        WorkgroupAxis::X => xyz[0],
        WorkgroupAxis::Y => xyz[1],
        WorkgroupAxis::Z => xyz[2],
    }
}

fn literal_value(literal: TileLiteral) -> Result<MemoryValue, SimError> {
    match literal {
        TileLiteral::F32(bits) => Ok(MemoryValue::F32(f32::from_bits(bits))),
        TileLiteral::U32(value) => Ok(MemoryValue::U32(value)),
        TileLiteral::I32(value) => Ok(MemoryValue::I32(value)),
        TileLiteral::Bool(value) => Ok(MemoryValue::Bool(value)),
        TileLiteral::F16(_) | TileLiteral::BF16(_) => {
            Err(SimError::Unsupported(format!("literal {literal:?}")))
        }
    }
}

fn unary(op: UnOp, value: MemoryValue) -> Result<MemoryValue, SimError> {
    let result = match op {
        UnOp::Abs => MemoryValue::F32(value.f32()?.abs()),
        UnOp::Neg => MemoryValue::F32(-value.f32()?),
        UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => {
            MemoryValue::F32(value.f32()?.exp())
        }
        UnOp::Exp2 => MemoryValue::F32(value.f32()?.exp2()),
        UnOp::Log => MemoryValue::F32(value.f32()?.ln()),
        UnOp::Log2 => MemoryValue::F32(value.f32()?.log2()),
        UnOp::Sqrt => MemoryValue::F32(value.f32()?.sqrt()),
        UnOp::InverseSqrt => MemoryValue::F32(value.f32()?.sqrt().recip()),
        UnOp::Sin => MemoryValue::F32(value.f32()?.sin()),
        UnOp::Cos => MemoryValue::F32(value.f32()?.cos()),
        UnOp::Tan => MemoryValue::F32(value.f32()?.tan()),
        UnOp::Tanh => MemoryValue::F32(value.f32()?.tanh()),
        UnOp::Asin => MemoryValue::F32(value.f32()?.asin()),
        UnOp::Acos => MemoryValue::F32(value.f32()?.acos()),
        UnOp::Atan => MemoryValue::F32(value.f32()?.atan()),
        UnOp::Sinh => MemoryValue::F32(value.f32()?.sinh()),
        UnOp::Cosh => MemoryValue::F32(value.f32()?.cosh()),
        UnOp::Asinh => MemoryValue::F32(value.f32()?.asinh()),
        UnOp::Acosh => MemoryValue::F32(value.f32()?.acosh()),
        UnOp::Atanh => MemoryValue::F32(value.f32()?.atanh()),
        UnOp::Unpack2x16Float => return Err(SimError::Unsupported("Unpack2x16Float".into())),
    };
    Ok(result)
}

fn binary(op: BinOp, left: MemoryValue, right: MemoryValue) -> Result<MemoryValue, SimError> {
    if let (MemoryValue::Vector(left), MemoryValue::Vector(right)) = (&left, &right) {
        if left.len() != right.len() {
            return Err(SimError::Type("vector widths differ".into()));
        }
        return Ok(MemoryValue::Vector(
            left.iter()
                .cloned()
                .zip(right.iter().cloned())
                .map(|(a, b)| binary(op, a, b))
                .collect::<Result<_, _>>()?,
        ));
    }
    if let MemoryValue::Vector(values) = left {
        return Ok(MemoryValue::Vector(
            values
                .into_iter()
                .map(|a| binary(op, a, right.clone()))
                .collect::<Result<_, _>>()?,
        ));
    }
    if let MemoryValue::Vector(values) = right {
        return Ok(MemoryValue::Vector(
            values
                .into_iter()
                .map(|b| binary(op, left.clone(), b))
                .collect::<Result<_, _>>()?,
        ));
    }
    match (left, right) {
        (MemoryValue::F32(a), MemoryValue::F32(b)) => Ok(MemoryValue::F32(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Rem => a % b,
            BinOp::Pow => a.powf(b),
            BinOp::Min => a.min(b),
            BinOp::Max => a.max(b),
            _ => return Err(SimError::Type(format!("{op:?} is not an f32 op"))),
        })),
        (MemoryValue::U32(a), MemoryValue::U32(b)) => Ok(MemoryValue::U32(match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a
                .checked_div(b)
                .ok_or_else(|| SimError::Type("u32 division by zero".into()))?,
            BinOp::Rem => a
                .checked_rem(b)
                .ok_or_else(|| SimError::Type("u32 remainder by zero".into()))?,
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            BinOp::Min => a.min(b),
            BinOp::Max => a.max(b),
            BinOp::Shr => a.wrapping_shr(b),
            BinOp::Shl => a.wrapping_shl(b),
            _ => return Err(SimError::Type(format!("{op:?} is not a u32 op"))),
        })),
        (MemoryValue::I32(a), MemoryValue::I32(b)) => Ok(MemoryValue::I32(match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => a
                .checked_div(b)
                .ok_or_else(|| SimError::Type("i32 division overflow or zero".into()))?,
            BinOp::Rem => a
                .checked_rem(b)
                .ok_or_else(|| SimError::Type("i32 remainder overflow or zero".into()))?,
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            BinOp::Min => a.min(b),
            BinOp::Max => a.max(b),
            BinOp::Shr => a.wrapping_shr(b as u32),
            BinOp::Shl => a.wrapping_shl(b as u32),
            _ => return Err(SimError::Type(format!("{op:?} is not an i32 op"))),
        })),
        (MemoryValue::Bool(a), MemoryValue::Bool(b)) => Ok(MemoryValue::Bool(match op {
            BinOp::LogicalAnd => a && b,
            BinOp::LogicalOr => a || b,
            BinOp::BitAnd => a & b,
            BinOp::BitOr => a | b,
            BinOp::BitXor => a ^ b,
            _ => return Err(SimError::Type(format!("{op:?} is not a bool op"))),
        })),
        (a, b) => Err(SimError::Type(format!(
            "binary operands mismatch: {a:?}, {b:?}"
        ))),
    }
}

fn compare(op: CmpOp, left: MemoryValue, right: MemoryValue) -> Result<MemoryValue, SimError> {
    if let (MemoryValue::Vector(left), MemoryValue::Vector(right)) = (&left, &right) {
        if left.len() != right.len() {
            return Err(SimError::Type("comparison vector widths differ".into()));
        }
        return Ok(MemoryValue::Vector(
            left.iter()
                .cloned()
                .zip(right.iter().cloned())
                .map(|(a, b)| compare(op, a, b))
                .collect::<Result<_, _>>()?,
        ));
    }
    match (left, right) {
        (MemoryValue::F32(a), MemoryValue::F32(b)) => Ok(MemoryValue::Bool(compare_ord(op, a, b))),
        (MemoryValue::U32(a), MemoryValue::U32(b)) => Ok(MemoryValue::Bool(compare_ord(op, a, b))),
        (MemoryValue::I32(a), MemoryValue::I32(b)) => Ok(MemoryValue::Bool(compare_ord(op, a, b))),
        (MemoryValue::Bool(a), MemoryValue::Bool(b)) => Ok(MemoryValue::Bool(match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            _ => return Err(SimError::Type("ordered boolean comparison".into())),
        })),
        (a, b) => Err(SimError::Type(format!(
            "comparison operands mismatch: {a:?}, {b:?}"
        ))),
    }
}

fn compare_ord<T: PartialOrd + PartialEq>(op: CmpOp, a: T, b: T) -> bool {
    match op {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
    }
}

fn round(mode: RoundMode, value: f32) -> f32 {
    match mode {
        RoundMode::HalfToEven => value.round_ties_even(),
        RoundMode::HalfAwayFromZero => value.round(),
        RoundMode::Floor => value.floor(),
        RoundMode::Ceil => value.ceil(),
        RoundMode::Trunc => value.trunc(),
    }
}

fn cast(value: MemoryValue, to: ElementType) -> Result<MemoryValue, SimError> {
    if let (MemoryValue::Vector(values), ElementType::Vector { .. }) = (&value, to) {
        return Ok(MemoryValue::Vector(
            values
                .iter()
                .cloned()
                .map(|item| {
                    cast(
                        item,
                        ElementType::Scalar(match to {
                            ElementType::Vector { scalar, .. } => scalar,
                            _ => unreachable!(),
                        }),
                    )
                })
                .collect::<Result<_, _>>()?,
        ));
    }
    let ElementType::Scalar(scalar) = to else {
        return Err(SimError::Unsupported(format!("cast to {to:?}")));
    };
    match scalar {
        ScalarElement::F32 => match value {
            MemoryValue::F32(v) => Ok(MemoryValue::F32(v)),
            MemoryValue::U32(v) => Ok(MemoryValue::F32(v as f32)),
            MemoryValue::I32(v) => Ok(MemoryValue::F32(v as f32)),
            MemoryValue::Bool(v) => Ok(MemoryValue::F32(if v { 1.0 } else { 0.0 })),
            _ => Err(SimError::Unsupported("vector-to-f32 cast".into())),
        },
        ScalarElement::U32 => match value {
            MemoryValue::F32(v) => Ok(MemoryValue::U32(v as u32)),
            MemoryValue::U32(v) => Ok(MemoryValue::U32(v)),
            MemoryValue::I32(v) => Ok(MemoryValue::U32(v as u32)),
            MemoryValue::Bool(v) => Ok(MemoryValue::U32(u32::from(v))),
            _ => Err(SimError::Unsupported("vector-to-u32 cast".into())),
        },
        ScalarElement::I32 => match value {
            MemoryValue::F32(v) => Ok(MemoryValue::I32(v as i32)),
            MemoryValue::U32(v) => Ok(MemoryValue::I32(v as i32)),
            MemoryValue::I32(v) => Ok(MemoryValue::I32(v)),
            MemoryValue::Bool(v) => Ok(MemoryValue::I32(i32::from(v))),
            _ => Err(SimError::Unsupported("vector-to-i32 cast".into())),
        },
        ScalarElement::Bool => match value {
            MemoryValue::Bool(v) => Ok(MemoryValue::Bool(v)),
            MemoryValue::F32(v) => Ok(MemoryValue::Bool(v != 0.0)),
            MemoryValue::U32(v) => Ok(MemoryValue::Bool(v != 0)),
            MemoryValue::I32(v) => Ok(MemoryValue::Bool(v != 0)),
            _ => Err(SimError::Unsupported("vector-to-bool cast".into())),
        },
        ScalarElement::F16 | ScalarElement::BF16 => {
            Err(SimError::Unsupported(format!("cast to {scalar:?}")))
        }
    }
}

fn bitcast(value: MemoryValue, to: ElementType) -> Result<MemoryValue, SimError> {
    let ElementType::Scalar(to) = to else {
        return Err(SimError::Unsupported("vector bitcast".into()));
    };
    match (value, to) {
        (MemoryValue::F32(v), ScalarElement::U32) => Ok(MemoryValue::U32(v.to_bits())),
        (MemoryValue::U32(v), ScalarElement::F32) => Ok(MemoryValue::F32(f32::from_bits(v))),
        (MemoryValue::I32(v), ScalarElement::U32) => Ok(MemoryValue::U32(v as u32)),
        (MemoryValue::U32(v), ScalarElement::I32) => Ok(MemoryValue::I32(v as i32)),
        (MemoryValue::F32(v), ScalarElement::I32) => Ok(MemoryValue::I32(v.to_bits() as i32)),
        (MemoryValue::I32(v), ScalarElement::F32) => Ok(MemoryValue::F32(f32::from_bits(v as u32))),
        (MemoryValue::F32(v), ScalarElement::F32) => Ok(MemoryValue::F32(v)),
        (MemoryValue::U32(v), ScalarElement::U32) => Ok(MemoryValue::U32(v)),
        (MemoryValue::I32(v), ScalarElement::I32) => Ok(MemoryValue::I32(v)),
        (_, ScalarElement::F16 | ScalarElement::BF16) => {
            Err(SimError::Unsupported("16-bit bitcast".into()))
        }
        (_, ScalarElement::Bool) => Err(SimError::Unsupported("bitcast to or from bool".into())),
        _ => Err(SimError::Unsupported("bitcast element pair".into())),
    }
}

fn map_flat_index(
    indexing: &MultiFlattenMap,
    extents: &[u32],
    flat: usize,
) -> Result<usize, SimError> {
    if indexing.groups.len() != extents.len() {
        return Err(SimError::Type(
            "KernelIr layout rank does not match its index map".into(),
        ));
    }
    let total = extents
        .iter()
        .fold(1usize, |acc, extent| acc.saturating_mul(*extent as usize));
    if flat >= total {
        return Err(SimError::Bounds {
            space: "logical address".into(),
            index: flat,
            length: total,
        });
    }
    let mut coords = vec![0usize; extents.len()];
    for axis in 0..extents.len() {
        let stride = extents[axis + 1..]
            .iter()
            .fold(1usize, |acc, extent| acc.saturating_mul(*extent as usize));
        coords[axis] = flat.checked_div(stride).unwrap_or(0) % extents[axis].max(1) as usize;
    }
    map_coordinates(indexing, &coords)
}

fn map_coordinates(indexing: &MultiFlattenMap, coords: &[usize]) -> Result<usize, SimError> {
    if indexing.groups.len() != coords.len() {
        return Err(SimError::Type(format!(
            "address rank {} differs from layout rank {}",
            coords.len(),
            indexing.groups.len()
        )));
    }
    let mut linear = 0usize;
    for (coordinate, group) in coords.iter().zip(&indexing.groups) {
        let extent = group
            .sub_axes
            .iter()
            .fold(1usize, |acc, axis| acc.saturating_mul(axis.extent as usize));
        if *coordinate >= extent {
            return Err(SimError::Bounds {
                space: "logical address".into(),
                index: *coordinate,
                length: extent,
            });
        }
        let mut divisor = extent;
        let mut rest = *coordinate;
        for sub_axis in &group.sub_axes {
            divisor /= sub_axis.extent.max(1) as usize;
            let digit = rest.checked_div(divisor).unwrap_or(0);
            rest = if divisor == 0 { 0 } else { rest % divisor };
            linear = linear.saturating_add(digit.saturating_mul(sub_axis.stride as usize));
        }
    }
    Ok(linear)
}

fn find_stmt(body: &[Stmt], pointer: usize) -> Option<&Stmt> {
    for stmt in body {
        if std::ptr::eq(stmt, pointer as *const Stmt) {
            return Some(stmt);
        }
        let nested = match stmt {
            Stmt::If { accept, reject, .. } => {
                find_stmt(accept, pointer).or_else(|| find_stmt(reject, pointer))
            }
            Stmt::Loop { body, .. } => find_stmt(body, pointer),
            _ => None,
        };
        if nested.is_some() {
            return nested;
        }
    }
    None
}

#[allow(dead_code)]
fn _type_contract(_: NumericContract, _: MemoryLevel) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::Choice;
    use fusor_ir::ir::kernel::{BufferAccess, BufferDecl, KernelIr, TileLayout};
    use std::sync::Arc;

    fn scalar(scalar: ScalarElement) -> ElementType {
        ElementType::Scalar(scalar)
    }

    fn literal_u32(value: u32) -> TileExpr {
        TileExpr::new(
            TileExprKind::Literal(TileLiteral::U32(value)),
            scalar(ScalarElement::U32),
        )
    }

    fn lane() -> TileExpr {
        TileExpr::new(
            TileExprKind::Builtin(Builtin::Lane),
            scalar(ScalarElement::U32),
        )
    }

    fn f32_literal(value: f32) -> TileExpr {
        TileExpr::new(
            TileExprKind::Literal(TileLiteral::F32(value.to_bits())),
            scalar(ScalarElement::F32),
        )
    }

    fn bool_literal(value: bool) -> TileExpr {
        TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(value)),
            scalar(ScalarElement::Bool),
        )
    }

    fn f32_load(local: &fusor_ir::ir::kernel::Local) -> TileExpr {
        TileExpr::new(
            TileExprKind::LoadLocal(local.clone()),
            scalar(ScalarElement::F32),
        )
    }

    fn add_f32(left: TileExpr, right: TileExpr) -> TileExpr {
        TileExpr::new(
            TileExprKind::Binary {
                op: BinOp::Add,
                left,
                right,
                numeric: NumericContract::RELAXED,
            },
            scalar(ScalarElement::F32),
        )
    }

    fn buffer(
        binding: u32,
        elements: u32,
        access: BufferAccess,
    ) -> Arc<fusor_ir::ir::kernel::BufferDecl> {
        Arc::new(BufferDecl {
            binding,
            element: scalar(ScalarElement::F32),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[elements]),
            access,
        })
    }

    fn view(buffer: Arc<fusor_ir::ir::kernel::BufferDecl>) -> fusor_ir::ir::kernel::StorageView {
        fusor_ir::ir::kernel::StorageView {
            layout: buffer.layout.clone(),
            buffer,
            offset: 0,
        }
    }

    fn store_kernel(
        elements: u32,
        grid: [u32; 3],
        block: u32,
    ) -> (KernelIr, BTreeMap<u32, BindingMemory>) {
        let output = view(buffer(1, elements, BufferAccess::ReadWrite));
        let body = vec![Stmt::Store {
            dst: output,
            addr: Addr::Linear(lane()),
            value: TileExpr::new(
                TileExprKind::Literal(TileLiteral::F32(1.0f32.to_bits())),
                scalar(ScalarElement::F32),
            ),
            mask: TileExpr::new(
                TileExprKind::Literal(TileLiteral::Bool(true)),
                scalar(ScalarElement::Bool),
            ),
        }];
        let ir = KernelIr {
            buffers: vec![
                buffer(0, 1, BufferAccess::Read),
                buffer(1, elements, BufferAccess::ReadWrite),
            ],
            grid,
            block,
            body,
            byte_arena: None,
            name: "store_test",
        };
        let memories = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, elements as usize)),
        ]);
        (ir, memories)
    }

    #[test]
    fn schedule_trace_roundtrips_and_rejects_corruption() {
        let (ir, mut memory) = store_kernel(2, [1, 1, 1], 2);
        let result = Simulator::new(Limits::default())
            .run(&ir, &mut memory, 77)
            .unwrap();
        assert_eq!(result.trace.choices.len(), 2);
        let json = serde_json::to_vec(&result.trace).unwrap();
        let trace: ScheduleTrace = serde_json::from_slice(&json).unwrap();
        let mut replay_memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        assert_eq!(
            Simulator::new(Limits::default())
                .replay(&ir, &mut replay_memory, &trace)
                .unwrap()
                .trace,
            trace
        );

        let mut truncated = trace.clone();
        truncated.choices.pop();
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut memory, &truncated),
            Err(SimError::Trace(TraceError::Truncated { .. }))
        ));

        let mut extra = trace.clone();
        extra.choices.push(extra.choices[0].clone());
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut memory, &extra),
            Err(SimError::Trace(TraceError::Extra { .. }))
        ));

        let mut impossible = trace.clone();
        impossible.choices[0].lane = 7;
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut memory, &impossible),
            Err(SimError::Trace(TraceError::ImpossibleChoice { .. }))
        ));

        let mut changed = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::f32(1, [1.0, 2.0])),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut changed, &trace),
            Err(SimError::Trace(TraceError::InputMismatch))
        ));

        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        let changed_limits = Limits {
            max_steps: 999,
            ..Limits::default()
        };
        assert!(matches!(
            Simulator::new(changed_limits).replay(&ir, &mut memory, &trace),
            Err(SimError::Trace(TraceError::InputMismatch))
        ));
    }

    #[test]
    fn zero_trip_loop_initializes_accumulators() {
        let local = Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
            ScalarElement::F32,
        )));
        let output = view(buffer(1, 1, BufferAccess::ReadWrite));
        let ir = KernelIr {
            buffers: vec![
                buffer(0, 1, BufferAccess::Read),
                buffer(1, 1, BufferAccess::ReadWrite),
            ],
            grid: [1, 1, 1],
            block: 1,
            body: vec![
                Stmt::Loop {
                    count: Some(literal_u32(0)),
                    index: None,
                    accumulators: vec![Accumulator {
                        local: local.clone(),
                        init: f32_literal(3.25),
                        update: f32_literal(99.0),
                    }],
                    body: Vec::new(),
                },
                Stmt::Store {
                    dst: output,
                    addr: Addr::Linear(literal_u32(0)),
                    value: f32_load(&local),
                    mask: bool_literal(true),
                },
            ],
            byte_arena: None,
            name: "zero_trip_accumulator_test",
        };
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 1)),
        ]);
        Simulator::new(Limits::default())
            .run(&ir, &mut memory, 3)
            .unwrap();
        assert_eq!(memory[&1].cells[0], Some(MemoryValue::F32(3.25)));
    }

    #[test]
    fn f32_i32_bitcasts_preserve_raw_bits() {
        let bits = 0xbf80_0001u32;
        let i32_value = bitcast(
            MemoryValue::F32(f32::from_bits(bits)),
            ElementType::Scalar(ScalarElement::I32),
        )
        .unwrap();
        assert_eq!(i32_value, MemoryValue::I32(bits as i32));
        assert_eq!(
            bitcast(i32_value, ElementType::Scalar(ScalarElement::F32)).unwrap(),
            MemoryValue::F32(f32::from_bits(bits))
        );
    }

    #[test]
    fn duplicate_memory_ids_and_vector_storage_are_rejected_with_replayable_preflight() {
        let (ir, mut memory) = store_kernel(1, [1, 1, 1], 1);
        memory.insert(1, BindingMemory::uninitialized(0, 1));
        let error = Simulator::new(Limits::default())
            .run(&ir, &mut memory, 1)
            .unwrap_err();
        assert!(matches!(
            &error,
            SimError::Execution { source, trace }
                if matches!(source.as_ref(), SimError::Unsupported(_)) && trace.choices.is_empty()
        ));

        let mut appended = error.schedule_trace().unwrap().clone();
        appended.choices.push(Choice {
            workgroup: 0,
            boundary: "statement".into(),
            runnable: vec![0],
            runnable_digest: 0,
            lane: 0,
        });
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut memory, &appended),
            Err(SimError::Trace(TraceError::Extra { .. }))
        ));

        let vector = Arc::new(BufferDecl {
            binding: 1,
            element: ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: 2,
            },
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[1]),
            access: BufferAccess::ReadWrite,
        });
        let mut vector_ir = ir;
        vector_ir.buffers[1] = vector;
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 1)),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).run(&vector_ir, &mut memory, 2),
            Err(SimError::Execution { source, .. }) if matches!(*source, SimError::Unsupported(_))
        ));
    }

    #[test]
    fn typed_input_digest_rejects_same_bits_with_different_element_type() {
        let (ir, _) = store_kernel(1, [1, 1, 1], 1);
        let mut f32_memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::f32(1, [1.0])),
        ]);
        let trace = Simulator::new(Limits::default())
            .run(&ir, &mut f32_memory, 8)
            .unwrap()
            .trace;
        let mut u32_memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (
                1,
                BindingMemory {
                    id: 1,
                    cells: vec![Some(MemoryValue::U32(1.0f32.to_bits()))],
                },
            ),
        ]);
        assert!(matches!(
            Simulator::new(Limits::default()).replay(&ir, &mut u32_memory, &trace),
            Err(SimError::Trace(TraceError::InputMismatch))
        ));
    }

    #[test]
    fn invocation_limit_failure_has_empty_replayable_dispatch_trace() {
        let (ir, mut memory) = store_kernel(2, [1, 1, 1], 2);
        let limits = Limits {
            max_invocations: 1,
            ..Limits::default()
        };
        let error = Simulator::new(limits).run(&ir, &mut memory, 9).unwrap_err();
        let trace = error.schedule_trace().unwrap().clone();
        assert!(trace.choices.is_empty());
        let mut replay_memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 2)),
        ]);
        assert!(matches!(
            Simulator::new(limits).replay(&ir, &mut replay_memory, &trace),
            Err(SimError::Execution { source, .. })
                if matches!(*source, SimError::InvocationLimit { .. })
        ));
        let mut appended = trace;
        appended.choices.push(Choice {
            workgroup: 0,
            boundary: "statement".into(),
            runnable: vec![0],
            runnable_digest: 0,
            lane: 0,
        });
        assert!(matches!(
            Simulator::new(limits).replay(&ir, &mut replay_memory, &appended),
            Err(SimError::Trace(TraceError::Extra { .. }))
        ));
    }

    #[test]
    fn bounded_fill_tile_and_repeated_expression_reductions_are_unsupported() {
        let tile = Arc::new(fusor_ir::ir::kernel::TileDecl::new(
            scalar(ScalarElement::F32),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[1]),
            "bounded_fill",
        ));
        let fill_ir = KernelIr {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 1,
            body: vec![Stmt::FillTile {
                dst: tile,
                value: f32_literal(0.0),
                bounds: [Some(literal_u32(1)), Some(literal_u32(1))],
            }],
            byte_arena: None,
            name: "bounded_fill_test",
        };
        let mut memory = BTreeMap::new();
        assert!(matches!(
            Simulator::new(Limits::default()).run(&fill_ir, &mut memory, 1),
            Err(SimError::Execution { source, .. }) if matches!(*source, SimError::Unsupported(_))
        ));

        let scratch = Arc::new(fusor_ir::ir::kernel::TileDecl::new(
            scalar(ScalarElement::F32),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[1]),
            "reduce_scratch",
        ));
        let reduce = TileExpr::new(
            TileExprKind::Reduce {
                op: fusor_ir::ir::kernel::TileReduceOp::Sum,
                kind: Box::new(ReduceKind::Workgroup {
                    scratch,
                    group_size: 1,
                }),
                value: f32_literal(1.0),
            },
            scalar(ScalarElement::F32),
        );
        let repeated = add_f32(reduce.clone(), reduce);
        let output = view(buffer(1, 1, BufferAccess::ReadWrite));
        let reduce_ir = KernelIr {
            buffers: vec![
                buffer(0, 1, BufferAccess::Read),
                buffer(1, 1, BufferAccess::ReadWrite),
            ],
            grid: [1, 1, 1],
            block: 1,
            body: vec![Stmt::Store {
                dst: output,
                addr: Addr::Linear(literal_u32(0)),
                value: repeated,
                mask: bool_literal(true),
            }],
            byte_arena: None,
            name: "repeated_reduce_test",
        };
        assert!(matches!(
            validate_kernel_storage(&reduce_ir),
            Err(SimError::Unsupported(message)) if message.contains("multiple or repeated")
        ));
    }

    #[test]
    fn injected_out_of_bounds_and_uninitialized_reads_fail() {
        let (ir, mut memory) = store_kernel(2, [1, 1, 1], 3);
        let error = Simulator::new(Limits::default())
            .run(&ir, &mut memory, 1)
            .unwrap_err();
        assert!(error.to_string().contains("logical address index 2"));

        let input = view(buffer(1, 1, BufferAccess::Read));
        let output = view(buffer(2, 1, BufferAccess::ReadWrite));
        let load = TileExpr::new(
            TileExprKind::Load {
                src: Source::Storage(input),
                addr: Box::new(Addr::Linear(literal_u32(0))),
                mask: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::Bool(true)),
                    scalar(ScalarElement::Bool),
                ),
                fill: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::F32(0)),
                    scalar(ScalarElement::F32),
                ),
            },
            scalar(ScalarElement::F32),
        );
        let ir = KernelIr {
            buffers: vec![
                buffer(0, 1, BufferAccess::Read),
                buffer(1, 1, BufferAccess::Read),
                buffer(2, 1, BufferAccess::ReadWrite),
            ],
            grid: [1, 1, 1],
            block: 1,
            body: vec![Stmt::Store {
                dst: output,
                addr: Addr::Linear(literal_u32(0)),
                value: load,
                mask: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::Bool(true)),
                    scalar(ScalarElement::Bool),
                ),
            }],
            byte_arena: None,
            name: "uninitialized_test",
        };
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 1)),
            (2, BindingMemory::uninitialized(2, 1)),
        ]);
        let error = Simulator::new(Limits::default())
            .run(&ir, &mut memory, 2)
            .unwrap_err();
        assert!(error.to_string().contains("uninitialized storage"));
    }

    #[test]
    fn workgroup_races_and_divergent_barriers_fail() {
        let tile = Arc::new(fusor_ir::ir::kernel::TileDecl::new(
            scalar(ScalarElement::F32),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[1]),
            "racy",
        ));
        let tile_store = Stmt::StoreTile {
            dst: tile,
            index: literal_u32(0),
            value: TileExpr::new(
                TileExprKind::Literal(TileLiteral::F32(0)),
                scalar(ScalarElement::F32),
            ),
        };
        let ir = KernelIr {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 2,
            body: vec![tile_store],
            byte_arena: None,
            name: "race_test",
        };
        let mut memory = BTreeMap::new();
        assert!(
            matches!(Simulator::new(Limits::default()).run(&ir, &mut memory, 3), Err(SimError::Execution { source, .. }) if matches!(*source, SimError::Race { .. }))
        );

        let condition = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Eq,
                left: lane(),
                right: literal_u32(0),
            },
            scalar(ScalarElement::Bool),
        );
        let ir = KernelIr {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 2,
            body: vec![Stmt::If {
                condition,
                accept: vec![Stmt::Barrier],
                reject: Vec::new(),
            }],
            byte_arena: None,
            name: "divergent_barrier_test",
        };
        let mut memory = BTreeMap::new();
        assert!(
            matches!(Simulator::new(Limits::default()).run(&ir, &mut memory, 4), Err(SimError::Execution { source, .. }) if matches!(*source, SimError::DivergentBarrier { .. }))
        );

        let condition = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Eq,
                left: lane(),
                right: literal_u32(0),
            },
            scalar(ScalarElement::Bool),
        );
        let ir = KernelIr {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 2,
            body: vec![Stmt::If {
                condition,
                accept: vec![Stmt::Barrier],
                reject: vec![Stmt::Barrier],
            }],
            byte_arena: None,
            name: "distinct_barriers_test",
        };
        let mut memory = BTreeMap::new();
        assert!(matches!(
            Simulator::new(Limits::default()).run(&ir, &mut memory, 5),
            Err(SimError::Execution { source, .. }) if matches!(*source, SimError::DivergentBarrier { .. })
        ));

        let (ir, mut memory) = store_kernel(1, [2, 1, 1], 1);
        assert!(matches!(
            Simulator::new(Limits::default()).run(&ir, &mut memory, 6),
            Err(SimError::Execution { source, .. }) if matches!(&*source, SimError::Race { space, .. } if space == "storage")
        ));

        let tile = Arc::new(fusor_ir::ir::kernel::TileDecl::new(
            scalar(ScalarElement::F32),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[1]),
            "per_group",
        ));
        let ir = KernelIr {
            buffers: Vec::new(),
            grid: [2, 1, 1],
            block: 1,
            body: vec![Stmt::StoreTile {
                dst: tile,
                index: literal_u32(0),
                value: f32_literal(1.0),
            }],
            byte_arena: None,
            name: "per_workgroup_tile_test",
        };
        let mut memory = BTreeMap::new();
        Simulator::new(Limits::default())
            .run(&ir, &mut memory, 7)
            .unwrap();
    }

    #[test]
    fn empty_unbounded_loop_hits_step_limit() {
        let ir = KernelIr {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 1,
            body: vec![Stmt::Loop {
                count: None,
                index: None,
                accumulators: Vec::new(),
                body: Vec::new(),
            }],
            byte_arena: None,
            name: "empty_loop_test",
        };
        let mut memory = BTreeMap::new();
        let limits = Limits {
            max_steps: 4,
            ..Limits::default()
        };
        assert!(matches!(
            Simulator::new(limits).run(&ir, &mut memory, 8),
            Err(SimError::Execution { source, .. }) if matches!(*source, SimError::StepLimit { limit: 4 })
        ));
    }

    #[test]
    fn nary_reduction_merges_all_slots_and_groups() {
        let lhs = [
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
        ];
        let rhs = [
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
        ];
        let outputs = [
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
            Arc::new(fusor_ir::ir::kernel::LocalDecl::new(scalar(
                ScalarElement::F32,
            ))),
        ];
        let scratches = [
            Arc::new(fusor_ir::ir::kernel::TileDecl::new(
                scalar(ScalarElement::F32),
                TileLayout::contiguous(MemoryLevel::Workgroup, &[8]),
                "merge_a",
            )),
            Arc::new(fusor_ir::ir::kernel::TileDecl::new(
                scalar(ScalarElement::F32),
                TileLayout::contiguous(MemoryLevel::Workgroup, &[8]),
                "merge_b",
            )),
        ];
        let lane_f = TileExpr::new(
            TileExprKind::Cast {
                value: lane(),
                to: scalar(ScalarElement::F32),
            },
            scalar(ScalarElement::F32),
        );
        let ten = literal_u32(10);
        let scaled = TileExpr::new(
            TileExprKind::Binary {
                op: BinOp::Mul,
                left: lane(),
                right: ten,
                numeric: NumericContract::RELAXED,
            },
            scalar(ScalarElement::U32),
        );
        let scaled = TileExpr::new(
            TileExprKind::Cast {
                value: scaled,
                to: scalar(ScalarElement::F32),
            },
            scalar(ScalarElement::F32),
        );
        let merge = fusor_ir::ir::kernel::MergeBody {
            lhs: lhs.iter().cloned().collect(),
            rhs: rhs.iter().cloned().collect(),
            body: vec![
                add_f32(f32_load(&lhs[0]), f32_load(&rhs[0])),
                add_f32(f32_load(&lhs[1]), f32_load(&rhs[1])),
            ]
            .into(),
        };
        let output_a = view(buffer(1, 8, BufferAccess::ReadWrite));
        let output_b = view(buffer(2, 8, BufferAccess::ReadWrite));
        let ir = KernelIr {
            buffers: vec![
                buffer(0, 1, BufferAccess::Read),
                buffer(1, 8, BufferAccess::ReadWrite),
                buffer(2, 8, BufferAccess::ReadWrite),
            ],
            grid: [1, 1, 1],
            block: 8,
            body: vec![
                Stmt::Reduce {
                    kind: Box::new(ReduceKind::Workgroup {
                        scratch: scratches[0].clone(),
                        group_size: 4,
                    }),
                    values: vec![lane_f, scaled].into(),
                    merge: Box::new(merge),
                    fast: None,
                    outs: outputs.iter().cloned().collect(),
                    scratch: scratches.iter().cloned().collect(),
                },
                Stmt::Store {
                    dst: output_a,
                    addr: Addr::Linear(lane()),
                    value: f32_load(&outputs[0]),
                    mask: TileExpr::new(
                        TileExprKind::Literal(TileLiteral::Bool(true)),
                        scalar(ScalarElement::Bool),
                    ),
                },
                Stmt::Store {
                    dst: output_b,
                    addr: Addr::Linear(lane()),
                    value: f32_load(&outputs[1]),
                    mask: TileExpr::new(
                        TileExprKind::Literal(TileLiteral::Bool(true)),
                        scalar(ScalarElement::Bool),
                    ),
                },
            ],
            byte_arena: None,
            name: "nary_reduce_test",
        };
        let mut memory = BTreeMap::from([
            (0, BindingMemory::uninitialized(0, 1)),
            (1, BindingMemory::uninitialized(1, 8)),
            (2, BindingMemory::uninitialized(2, 8)),
        ]);
        Simulator::new(Limits::default())
            .run(&ir, &mut memory, 9)
            .unwrap();
        let first: Vec<f32> = memory[&1]
            .cells
            .iter()
            .map(|cell| cell.as_ref().unwrap().f32().unwrap())
            .collect();
        let second: Vec<f32> = memory[&2]
            .cells
            .iter()
            .map(|cell| cell.as_ref().unwrap().f32().unwrap())
            .collect();
        assert_eq!(first, [6.0, 6.0, 6.0, 6.0, 22.0, 22.0, 22.0, 22.0]);
        assert_eq!(second, [60.0, 60.0, 60.0, 60.0, 220.0, 220.0, 220.0, 220.0]);
    }

    #[test]
    fn mixed_radix_coordinates_use_all_subaxis_extents() {
        use fusor_ir::shape::{AxisGroup, MultiFlattenMap, SubAxis};
        let indexing = MultiFlattenMap {
            groups: vec![AxisGroup {
                sub_axes: vec![
                    SubAxis {
                        extent: 2,
                        stride: 12,
                    },
                    SubAxis {
                        extent: 3,
                        stride: 4,
                    },
                    SubAxis {
                        extent: 4,
                        stride: 1,
                    },
                ]
                .into(),
            }]
            .into(),
        };
        for index in 0..24 {
            assert_eq!(map_coordinates(&indexing, &[index]).unwrap(), index);
        }
    }
}
