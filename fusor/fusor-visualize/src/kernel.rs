//! Capture address/control behavior from KernelIr without a GPU.
//! Data loads remain unknown. Unknown values can flow through arithmetic/stores,
//! but are rejected if needed for an address, predicate or loop count. This keeps
//! diagnostics honest for data-dependent gathers and branching.
use std::collections::BTreeMap;

use fusor_ir::ir::kernel::*;
use fusor_ir::scalar::{BinOp, CmpOp};

use crate::{
    AccessKind, Action, Error, Event, LaneAccess, Resource, Result, Space, Trace, require,
};

#[derive(Clone, Debug)]
pub struct CaptureOptions {
    pub subgroup_size: u32,
    pub workgroups: Vec<[u32; 3]>,
    /// Bounds both expression evaluations and control expansion, including loops
    /// with no memory events. Exceeding the budget returns an error, never a prefix.
    pub max_actions: usize,
    /// Concrete input bytes, usually binding 0's packed runtime dimensions/scalars.
    /// A write invalidates the whole binding; all unspecified data stays unknown.
    pub known_bindings: BTreeMap<u32, Vec<u8>>,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            subgroup_size: 32,
            workgroups: vec![[0; 3]],
            max_actions: 100_000,
            known_bindings: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Value {
    U(u32),
    I(i32),
    F(f32),
    B(bool),
    Unknown,
}

impl Value {
    fn index(self, source: &str) -> Result<u32> {
        match self {
            Self::U(v) => Ok(v),
            Self::I(v) if v >= 0 => Ok(v as u32),
            _ => Err(Error(format!(
                "{source}: address/count depends on unknown or non-integer data; supply an instrumented lane-address trace"
            ))),
        }
    }
    fn predicate(self, source: &str) -> Result<bool> {
        match self {
            Self::U(v) => Ok(v != 0),
            Self::I(v) => Ok(v != 0),
            Self::F(v) => Ok(v != 0.0),
            Self::B(v) => Ok(v),
            Self::Unknown => Err(Error(format!(
                "{source}: predicate depends on loaded data; supply an instrumented lane-address trace"
            ))),
        }
    }
}

/// Record selected workgroups in caller order. Subgroups are contiguous slices
/// of local invocation indices; workgroups execute serially in the virtual SM.
/// Each explicit IR expression use is evaluated (before backend CSE/vectorization).
pub fn capture(ir: &KernelIr, options: CaptureOptions) -> Result<Trace> {
    require(
        (1..=1024).contains(&ir.block),
        "capture requires an explicit workgroup size in 1..=1024",
    )?;
    require(
        (1..=1024).contains(&options.subgroup_size),
        "subgroup size must be 1..=1024",
    )?;
    require(
        !options.workgroups.is_empty(),
        "select at least one workgroup",
    )?;
    require(
        options.max_actions > 0,
        "capture action budget must be positive",
    )?;
    require(
        ir.byte_arena.is_none(),
        "byte-arena aliasing requires a post-emission lane-address trace",
    )?;
    validate_body(&ir.body)?;
    let mut recorder = Recorder {
        ir, known_bindings: options.known_bindings, remaining: options.max_actions, subgroup_size: options.subgroup_size,
        workgroup: [0; 3], locals: BTreeMap::new(), tiles: BTreeMap::new(), decoded: BTreeMap::new(),
        trace: Trace { version: 1, name: ir.name.into(), workgroup_size: ir.block,
            recorded_subgroup_size: Some(options.subgroup_size), resources: vec![], events: vec![],
            notes: vec!["Logical KernelIr accesses, with the actual quantized decoder expanded, before backend CSE/vectorization; loaded values are unknown unless binding bytes are supplied.".into(),
                "Load masks omit logical inactive reads. The GPU emitter may instead load clamped addresses on those lanes and select the fill afterward; inspect generated shaders for physical access behavior.".into(),
                "Serial sampled workgroups; contiguous local-lane subgroup assignment; not a hardware timing or race proof.".into()] },
    };
    for buffer in &ir.buffers {
        let id = u64::from(buffer.binding);
        let bytes = layout_bytes(&buffer.layout, scalar_bytes(buffer.element)?)?;
        if let Some(resource) = recorder.trace.resources.iter_mut().find(|r| r.id == id) {
            resource.bytes = resource.bytes.max(bytes);
        } else {
            recorder.trace.resources.push(Resource {
                id,
                name: format!("binding {}", buffer.binding),
                space: Space::Storage,
                bytes,
            });
        }
    }
    let mut visited = std::collections::BTreeSet::new();
    for workgroup in options.workgroups {
        require(
            workgroup.iter().zip(ir.grid).all(|(i, n)| *i < n),
            "selected workgroup is outside the dispatch grid",
        )?;
        require(
            visited.insert(workgroup),
            "workgroup selection contains duplicates",
        )?;
        recorder.workgroup = workgroup;
        recorder.locals.clear();
        recorder.block(&ir.body, vec![true; ir.block as usize], "body")?;
    }
    Ok(recorder.trace)
}

fn scalar_bytes(element: ElementType) -> Result<u32> {
    match element {
        ElementType::Scalar(s) => Ok(s.byte_size() as u32),
        _ => Err(Error(
            "vector/cooperative values require a post-emission lane-address trace".into(),
        )),
    }
}

fn layout_bytes(layout: &TileLayout, bytes: u32) -> Result<u64> {
    require(
        layout.extents.len() == layout.indexing.groups.len(),
        "invalid layout rank",
    )?;
    require(
        layout.extents.iter().all(|n| *n > 0),
        "zero-sized resource layout",
    )?;
    let mut max_index = 0u64;
    for group in &layout.indexing.groups {
        require(!group.sub_axes.is_empty(), "empty layout axis")?;
        for sub in &group.sub_axes {
            require(sub.extent > 0, "zero layout sub-axis extent")?;
            max_index = max_index
                .checked_add(u64::from(sub.extent - 1) * u64::from(sub.stride))
                .ok_or_else(|| Error("layout size overflow".into()))?;
        }
    }
    max_index
        .checked_add(1)
        .and_then(|n| n.checked_mul(u64::from(bytes)))
        .ok_or_else(|| Error("layout byte size overflow".into()))
}

/// Restate the exact decoder used by GPU emission. Address flattening follows
/// the quantized view's element strides; its storage offset remains word-based.
fn expand_quantized(expr: &TileExpr) -> Result<TileExpr> {
    let TileExprKind::Load {
        src: Source::Quantized(q),
        addr,
        mask,
        fill,
    } = expr.kind()
    else {
        return Err(Error("expected a quantized load".into()));
    };
    fusor_gguf::blocks::decode_load(q, addr, mask.clone(), fill.clone())
        .map_err(|e| Error(format!("quantized decoder: {e}")))
}

fn validate_expr(expr: &TileExpr) -> Result<()> {
    require(
        !matches!(expr.element(), ElementType::CoopMatrix { .. }),
        "cooperative values require a post-emission lane-address trace",
    )?;
    match expr.kind() {
        TileExprKind::Load {
            src: Source::Quantized(_),
            ..
        } => {
            return validate_expr(&expand_quantized(expr)?);
        }
        TileExprKind::Reduce { kind, .. } if !matches!(**kind, ReduceKind::Subgroup) => {
            return Err(Error(
                "workgroup reduction scratch lowering requires a post-emission lane-address trace"
                    .into(),
            ));
        }
        TileExprKind::CoopZero { .. }
        | TileExprKind::CoopLoad { .. }
        | TileExprKind::CoopMma { .. } => {
            return Err(Error(
                "vector/cooperative expressions require a post-emission lane-address trace".into(),
            ));
        }
        _ => {}
    }
    let mut result = Ok(());
    expr.kind().visit_children(&mut |child| {
        if result.is_ok() {
            result = validate_expr(child);
        }
    });
    result
}

fn validate_body(body: &[Stmt]) -> Result<()> {
    for stmt in body {
        match stmt {
            Stmt::FillTile { .. } | Stmt::CoopStore { .. } | Stmt::CoopStoreTile { .. } | Stmt::Reduce { .. } =>
                return Err(Error("collective fill/store/reduction has backend-specific memory instructions; capture a post-emission lane-address trace or use explicit scalar staging".into())),
            Stmt::Loop { count: None, .. } | Stmt::Break => return Err(Error("unbounded loops/break are not supported by the scalar IR recorder".into())),
            Stmt::If { accept, reject, .. } => { validate_body(accept)?; validate_body(reject)?; }
            Stmt::Loop { body, .. } => validate_body(body)?,
            _ => {}
        }
        let mut result = Ok(());
        stmt.for_each_expr(&mut |expr| {
            if result.is_ok() {
                result = validate_expr(expr);
            }
        });
        result?;
    }
    Ok(())
}

struct Recorder<'a> {
    ir: &'a KernelIr,
    known_bindings: BTreeMap<u32, Vec<u8>>,
    remaining: usize,
    subgroup_size: u32,
    workgroup: [u32; 3],
    locals: BTreeMap<u64, Vec<Value>>,
    tiles: BTreeMap<u64, u64>,
    decoded: BTreeMap<usize, TileExpr>,
    trace: Trace,
}

impl Recorder<'_> {
    fn budget(&mut self, source: &str) -> Result<()> {
        self.remaining = self.remaining.checked_sub(1).ok_or_else(|| Error(format!("{source}: capture action budget exceeded; select fewer workgroups or increase max_actions")))?;
        Ok(())
    }
    fn push(&mut self, source: &str, action: Action) {
        self.trace.events.push(Event {
            workgroup: self.workgroup,
            source: source.into(),
            action,
        });
    }
    fn tile_id(&mut self, tile: &Tile) -> Result<u64> {
        if let Some(id) = self.tiles.get(&tile.id()) {
            return Ok(*id);
        }
        let id = (1u64 << 32) + self.tiles.len() as u64;
        let bytes = layout_bytes(&tile.layout, scalar_bytes(tile.element)?)?;
        self.trace.resources.push(Resource {
            id,
            name: tile.name.into(),
            space: Space::Workgroup,
            bytes,
        });
        self.tiles.insert(tile.id(), id);
        Ok(id)
    }
    fn control(&mut self, source: &str, label: String, active: &[bool]) {
        self.push(
            source,
            Action::Control {
                label,
                lanes: active
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| a.then_some(i as u32))
                    .collect(),
            },
        );
    }
    fn store_local(&mut self, local: &Local, values: &[Value], active: &[bool]) {
        let slots = self
            .locals
            .entry(local.id())
            .or_insert(vec![Value::Unknown; active.len()]);
        for (i, on) in active.iter().enumerate() {
            if *on {
                slots[i] = values[i];
            }
        }
    }
    fn block(&mut self, body: &[Stmt], mut active: Vec<bool>, path: &str) -> Result<Vec<bool>> {
        for (i, stmt) in body.iter().enumerate() {
            if !active.iter().any(|a| *a) {
                break;
            }
            let source = format!("{path}[{i}]");
            self.budget(&source)?;
            match stmt {
                Stmt::Store {
                    dst,
                    addr,
                    value,
                    mask,
                }
                | Stmt::AtomicAdd {
                    dst,
                    addr,
                    value,
                    mask,
                } => {
                    self.eval(value, &active, &format!("{source}.value"))?;
                    let selected = self.mask(mask, &active, &format!("{source}.mask"))?;
                    let indices = self.addr(dst, addr, &selected, &source)?;
                    let lanes = self.accesses(
                        &indices,
                        &selected,
                        dst.offset,
                        scalar_bytes(dst.buffer.element)?,
                        &source,
                    )?;
                    let access = if matches!(stmt, Stmt::AtomicAdd { .. }) {
                        AccessKind::Atomic
                    } else {
                        AccessKind::Write
                    };
                    self.push(
                        &source,
                        Action::Memory {
                            resource: u64::from(dst.buffer.binding),
                            access,
                            lanes,
                        },
                    );
                    self.known_bindings.remove(&dst.buffer.binding);
                }
                Stmt::StoreTile { dst, index, value } => {
                    self.eval(value, &active, &format!("{source}.value"))?;
                    let indices = self.eval(index, &active, &format!("{source}.index"))?;
                    let resource = self.tile_id(dst)?;
                    let lanes =
                        self.accesses(&indices, &active, 0, scalar_bytes(dst.element)?, &source)?;
                    self.push(
                        &source,
                        Action::Memory {
                            resource,
                            access: AccessKind::Write,
                            lanes,
                        },
                    );
                }
                Stmt::StoreLocal { dst, value } => {
                    let values = self.eval(value, &active, &format!("{source}.value"))?;
                    self.store_local(dst, &values, &active);
                    self.control(&source, format!("local {} = value", dst.id()), &active);
                }
                Stmt::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let yes = self.mask(condition, &active, &format!("{source}.condition"))?;
                    let no = active.iter().zip(&yes).map(|(a, b)| *a && !b).collect();
                    self.control(&source, "if: accept lanes".into(), &yes);
                    let yes = self.block(accept, yes, &format!("{source}.then"))?;
                    let no = self.block(reject, no, &format!("{source}.else"))?;
                    active = yes.iter().zip(no).map(|(a, b)| *a || b).collect();
                }
                Stmt::Loop {
                    count: Some(count),
                    index,
                    accumulators,
                    body,
                } => {
                    let inits = accumulators
                        .iter()
                        .map(|acc| self.eval(&acc.init, &active, &format!("{source}.init")))
                        .collect::<Result<Vec<_>>>()?;
                    for (acc, value) in accumulators.iter().zip(inits) {
                        self.store_local(&acc.local, &value, &active);
                    }
                    let counts = self.eval(count, &active, &format!("{source}.count"))?;
                    let counts: Vec<u32> = counts
                        .iter()
                        .zip(&active)
                        .map(|(v, on)| if *on { v.index(&source) } else { Ok(0) })
                        .collect::<Result<_>>()?;
                    for iteration in 0..counts.iter().copied().max().unwrap_or(0) {
                        self.budget(&source)?;
                        let running: Vec<bool> = active
                            .iter()
                            .zip(&counts)
                            .map(|(on, n)| *on && iteration < *n)
                            .collect();
                        if let Some(index) = index {
                            self.store_local(
                                index,
                                &vec![Value::U(iteration); active.len()],
                                &running,
                            );
                        }
                        self.control(&source, format!("loop iteration {iteration}"), &running);
                        let survived =
                            self.block(body, running.clone(), &format!("{source}.loop"))?;
                        for (lane, on) in running.iter().enumerate() {
                            if *on && !survived[lane] {
                                active[lane] = false;
                            }
                        }
                        // Updates are simultaneous, matching the emitter's accumulator semantics.
                        let values = accumulators
                            .iter()
                            .map(|acc| {
                                self.eval(&acc.update, &survived, &format!("{source}.update"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        for (acc, value) in accumulators.iter().zip(values) {
                            self.store_local(&acc.local, &value, &survived);
                        }
                    }
                }
                Stmt::Barrier | Stmt::StorageBarrier => {
                    require(
                        active.iter().all(|a| *a),
                        format!("{source}: divergent workgroup barrier"),
                    )?;
                    self.push(
                        &source,
                        Action::Barrier {
                            storage: matches!(stmt, Stmt::StorageBarrier),
                        },
                    );
                }
                Stmt::Return => {
                    self.control(&source, "return".into(), &active);
                    active.fill(false);
                }
                _ => unreachable!("validated before recording"),
            }
        }
        Ok(active)
    }
    fn mask(&mut self, expr: &TileExpr, active: &[bool], source: &str) -> Result<Vec<bool>> {
        self.eval(expr, active, source)?
            .into_iter()
            .zip(active)
            .map(|(v, on)| if *on { v.predicate(source) } else { Ok(false) })
            .collect()
    }
    fn accesses(
        &self,
        indices: &[Value],
        active: &[bool],
        offset: u32,
        bytes: u32,
        source: &str,
    ) -> Result<Vec<LaneAccess>> {
        indices
            .iter()
            .zip(active)
            .enumerate()
            .filter(|(_, (_, on))| **on)
            .map(|(lane, (index, _))| {
                // StorageView::offset is in elements, just like the emitter.
                let address =
                    u64::from(index.index(source)?.wrapping_add(offset)) * u64::from(bytes);
                Ok(LaneAccess {
                    lane: lane as u32,
                    address,
                    bytes,
                })
            })
            .collect()
    }
    fn addr(
        &mut self,
        view: &StorageView,
        addr: &Addr,
        active: &[bool],
        source: &str,
    ) -> Result<Vec<Value>> {
        match addr {
            Addr::Linear(expr) => self.eval(expr, active, &format!("{source}.address")),
            Addr::Rc2 { row, col } => {
                require(
                    view.layout.indexing.rank() == 2,
                    "rank-2 address requires a rank-2 layout",
                )?;
                let row = self.eval(row, active, &format!("{source}.row"))?;
                let col = self.eval(col, active, &format!("{source}.col"))?;
                active
                    .iter()
                    .enumerate()
                    .map(|(lane, on)| {
                        if !on {
                            return Ok(Value::Unknown);
                        }
                        let mut index = 0u32;
                        for (coord, group) in [row[lane], col[lane]]
                            .into_iter()
                            .zip(&view.layout.indexing.groups)
                        {
                            require(!group.sub_axes.is_empty(), "empty index map group")?;
                            let mut coord = coord.index(source)?;
                            for (i, sub) in group.sub_axes.iter().enumerate().rev() {
                                require(sub.extent > 0, "zero index map extent")?;
                                let part = if i == 0 {
                                    coord
                                } else {
                                    let p = coord % sub.extent;
                                    coord /= sub.extent;
                                    p
                                };
                                index = index.wrapping_add(part.wrapping_mul(sub.stride));
                            }
                        }
                        Ok(Value::U(index))
                    })
                    .collect()
            }
        }
    }
    fn eval(&mut self, expr: &TileExpr, active: &[bool], source: &str) -> Result<Vec<Value>> {
        self.budget(source)?;
        let n = active.len();
        let values = match expr.kind() {
            TileExprKind::Literal(lit) => vec![
                match *lit {
                    TileLiteral::U32(v) => Value::U(v),
                    TileLiteral::I32(v) => Value::I(v),
                    TileLiteral::Bool(v) => Value::B(v),
                    TileLiteral::F32(v) => Value::F(f32::from_bits(v)),
                    TileLiteral::F16(_) | TileLiteral::BF16(_) => Value::Unknown,
                };
                n
            ],
            TileExprKind::Builtin(builtin) => (0..n as u32)
                .map(|lane| {
                    Value::U(match builtin {
                        Builtin::Lane => lane,
                        Builtin::SubgroupLane => lane % self.subgroup_size,
                        Builtin::SubgroupId => lane / self.subgroup_size,
                        Builtin::SubgroupSize => self.subgroup_size,
                        Builtin::NumSubgroups => self.ir.block.div_ceil(self.subgroup_size),
                        Builtin::ProgramId(axis) => self.workgroup[axis_index(*axis)],
                        Builtin::NumWorkgroups(axis) => self.ir.grid[axis_index(*axis)],
                    })
                })
                .collect(),
            TileExprKind::LoadLocal(local) => self
                .locals
                .get(&local.id())
                .cloned()
                .unwrap_or(vec![Value::Unknown; n]),
            TileExprKind::Load {
                src: Source::Storage(view),
                addr,
                mask,
                fill,
            } => {
                let mut values = self.eval(fill, active, &format!("{source}.fill"))?;
                let selected = self.mask(mask, active, &format!("{source}.mask"))?;
                let indices = self.addr(view, addr, &selected, source)?;
                let lanes = self.accesses(
                    &indices,
                    &selected,
                    view.offset,
                    scalar_bytes(view.buffer.element)?,
                    source,
                )?;
                self.push(
                    source,
                    Action::Memory {
                        resource: u64::from(view.buffer.binding),
                        access: AccessKind::Read,
                        lanes,
                    },
                );
                for (i, on) in selected.iter().enumerate() {
                    if *on {
                        let byte = u64::from(indices[i].index(source)?.wrapping_add(view.offset))
                            * u64::from(scalar_bytes(view.buffer.element)?);
                        values[i] = self
                            .known_bindings
                            .get(&view.buffer.binding)
                            .and_then(|data| {
                                usize::try_from(byte)
                                    .ok()
                                    .and_then(|b| data.get(b..b.checked_add(4)?))
                            })
                            .and_then(|data| <[u8; 4]>::try_from(data).ok())
                            .map(|bytes| match view.buffer.element {
                                ElementType::Scalar(ScalarElement::U32) => {
                                    Value::U(u32::from_le_bytes(bytes))
                                }
                                ElementType::Scalar(ScalarElement::I32) => {
                                    Value::I(i32::from_le_bytes(bytes))
                                }
                                ElementType::Scalar(ScalarElement::F32) => {
                                    Value::F(f32::from_le_bytes(bytes))
                                }
                                _ => Value::Unknown,
                            })
                            .unwrap_or(Value::Unknown);
                    }
                }
                values
            }
            TileExprKind::Load {
                src: Source::Quantized(_),
                ..
            } => {
                let id = expr.node_ptr();
                if let std::collections::btree_map::Entry::Vacant(entry) = self.decoded.entry(id) {
                    entry.insert(expand_quantized(expr)?);
                }
                let decoded = self.decoded[&id].clone();
                self.eval(&decoded, active, &format!("{source}.quantized"))?
            }
            TileExprKind::Vec { parts, .. } => {
                for (i, part) in parts.iter().enumerate() {
                    self.eval(part, active, &format!("{source}.component[{i}]"))?;
                }
                vec![Value::Unknown; n]
            }
            TileExprKind::VecComponent { vector, .. } => {
                self.eval(vector, active, &format!("{source}.vector"))?;
                vec![Value::Unknown; n]
            }
            TileExprKind::Dot { left, right } => {
                self.eval(left, active, &format!("{source}.left"))?;
                self.eval(right, active, &format!("{source}.right"))?;
                vec![Value::Unknown; n]
            }
            TileExprKind::LoadTile { tile, index } => {
                let indices = self.eval(index, active, &format!("{source}.index"))?;
                let resource = self.tile_id(tile)?;
                let lanes =
                    self.accesses(&indices, active, 0, scalar_bytes(tile.element)?, source)?;
                self.push(
                    source,
                    Action::Memory {
                        resource,
                        access: AccessKind::Read,
                        lanes,
                    },
                );
                vec![Value::Unknown; n]
            }
            TileExprKind::Binary {
                op, left, right, ..
            } => {
                let left = self.eval(left, active, &format!("{source}.left"))?;
                let right = self.eval(right, active, &format!("{source}.right"))?;
                (0..n)
                    .map(|i| {
                        if active[i] {
                            binary(*op, left[i], right[i])
                        } else {
                            Ok(Value::Unknown)
                        }
                    })
                    .collect::<Result<_>>()?
            }
            TileExprKind::Compare { op, left, right } => {
                let left = self.eval(left, active, &format!("{source}.left"))?;
                let right = self.eval(right, active, &format!("{source}.right"))?;
                left.into_iter()
                    .zip(right)
                    .map(|(a, b)| compare(*op, a, b))
                    .collect()
            }
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                let condition = self.eval(condition, active, &format!("{source}.condition"))?;
                // WGSL select is eager: both operands can contain memory reads.
                let accept = self.eval(accept, active, &format!("{source}.accept"))?;
                let reject = self.eval(reject, active, &format!("{source}.reject"))?;
                (0..n)
                    .map(|i| match condition[i].predicate(source) {
                        Ok(true) => accept[i],
                        Ok(false) => reject[i],
                        Err(_) => Value::Unknown,
                    })
                    .collect()
            }
            TileExprKind::Cast { value, to } => self
                .eval(value, active, source)?
                .into_iter()
                .map(|v| cast(v, *to, false))
                .collect(),
            TileExprKind::Bitcast { value, to } => self
                .eval(value, active, source)?
                .into_iter()
                .map(|v| cast(v, *to, true))
                .collect(),
            TileExprKind::Unary { value, .. } | TileExprKind::Round { value, .. } => {
                // Preserve reads, but do not invent arithmetic relevant to an address.
                self.eval(value, active, source)?;
                vec![Value::Unknown; n]
            }
            TileExprKind::Reduce { value, .. } => {
                self.eval(value, active, source)?;
                self.control(source, "subgroup reduction (value unknown)".into(), active);
                vec![Value::Unknown; n]
            }
            _ => unreachable!("validated before recording"),
        };
        Ok(values)
    }
}

fn axis_index(axis: WorkgroupAxis) -> usize {
    match axis {
        WorkgroupAxis::X => 0,
        WorkgroupAxis::Y => 1,
        WorkgroupAxis::Z => 2,
    }
}

fn cast(v: Value, to: ElementType, bitcast: bool) -> Value {
    use ScalarElement::*;
    match (v, to) {
        (Value::U(v), ElementType::Scalar(U32)) => Value::U(v),
        (Value::I(v), ElementType::Scalar(I32)) => Value::I(v),
        (Value::F(v), ElementType::Scalar(F32)) => Value::F(v),
        (Value::U(v), ElementType::Scalar(I32)) => Value::I(v as i32),
        (Value::I(v), ElementType::Scalar(U32)) => Value::U(v as u32),
        (Value::F(v), ElementType::Scalar(U32)) => {
            Value::U(if bitcast { v.to_bits() } else { v as u32 })
        }
        (Value::F(v), ElementType::Scalar(I32)) => Value::I(if bitcast {
            v.to_bits() as i32
        } else {
            v as i32
        }),
        (Value::U(v), ElementType::Scalar(F32)) => {
            Value::F(if bitcast { f32::from_bits(v) } else { v as f32 })
        }
        (Value::I(v), ElementType::Scalar(F32)) => Value::F(if bitcast {
            f32::from_bits(v as u32)
        } else {
            v as f32
        }),
        (v, ElementType::Scalar(Bool)) if !bitcast => {
            v.predicate("cast").map(Value::B).unwrap_or(Value::Unknown)
        }
        _ => Value::Unknown,
    }
}

fn binary(op: BinOp, a: Value, b: Value) -> Result<Value> {
    use BinOp::*;
    let result = match (a, b) {
        (Value::U(a), Value::U(b)) => Value::U(match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Mul => a.wrapping_mul(b),
            Div | Rem if b == 0 => {
                return Err(Error("division by zero in address arithmetic".into()));
            }
            Div => a / b,
            Rem => a % b,
            Min => a.min(b),
            Max => a.max(b),
            BitAnd => a & b,
            BitOr => a | b,
            BitXor => a ^ b,
            Shl => a.wrapping_shl(b),
            Shr => a.wrapping_shr(b),
            LogicalAnd => u32::from(a != 0 && b != 0),
            LogicalOr => u32::from(a != 0 || b != 0),
            Pow => return Ok(Value::Unknown),
        }),
        (Value::I(a), Value::I(b)) => Value::I(match op {
            Add => a.wrapping_add(b),
            Sub => a.wrapping_sub(b),
            Mul => a.wrapping_mul(b),
            Div | Rem if b == 0 => {
                return Err(Error("division by zero in address arithmetic".into()));
            }
            Div => a.wrapping_div(b),
            Rem => a.wrapping_rem(b),
            Min => a.min(b),
            Max => a.max(b),
            BitAnd => a & b,
            BitOr => a | b,
            BitXor => a ^ b,
            Shl => a.wrapping_shl(b as u32),
            Shr => a.wrapping_shr(b as u32),
            LogicalAnd => i32::from(a != 0 && b != 0),
            LogicalOr => i32::from(a != 0 || b != 0),
            Pow => return Ok(Value::Unknown),
        }),
        (Value::F(a), Value::F(b)) => Value::F(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
            Div => a / b,
            Rem => a % b,
            Pow => a.powf(b),
            Min => a.min(b),
            Max => a.max(b),
            _ => return Ok(Value::Unknown),
        }),
        (Value::B(a), Value::B(b)) => Value::B(match op {
            LogicalAnd => a && b,
            LogicalOr => a || b,
            _ => return Ok(Value::Unknown),
        }),
        _ => Value::Unknown,
    };
    Ok(result)
}

fn compare(op: CmpOp, a: Value, b: Value) -> Value {
    macro_rules! cmp {
        ($a:expr, $b:expr) => {
            Value::B(match op {
                CmpOp::Lt => $a < $b,
                CmpOp::Le => $a <= $b,
                CmpOp::Gt => $a > $b,
                CmpOp::Ge => $a >= $b,
                CmpOp::Eq => $a == $b,
                CmpOp::Ne => $a != $b,
            })
        };
    }
    match (a, b) {
        (Value::U(a), Value::U(b)) => cmp!(a, b),
        (Value::I(a), Value::I(b)) => cmp!(a, b),
        (Value::F(a), Value::F(b)) => cmp!(a, b),
        (Value::B(a), Value::B(b)) => cmp!(a, b),
        _ => Value::Unknown,
    }
}
