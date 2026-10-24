//! Compile ordinary affine copies and contiguous Scatter{Set} ranges to WebGPU
//! transfers. This is part of every GPU resolve, independent of execution policy.
use super::*;
use fusor_ir::dtype::Dtype;
use fusor_ir::extract::BindKind;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{AccessPlan, Launch, Operand};
use fusor_ir::ir::logical::{LeafKind, Logical, ScatterCombine};
use fusor_ir::scalar::ScalarKind;
use fusor_ir::shape::{Dim, Layout};
use fusor_tile::index_range::index_start;

#[derive(Clone)]
struct Region {
    src: Option<u32>,
    dst: u32,
    src_offset: u64,
    dst_offset: u64,
    bytes: u64,
}

pub(super) struct Transfers(Vec<Region>);

fn slot(cx: &LowerCtx<'_>, value: Id) -> Option<u32> {
    let class = cx.graph.class_of(value);
    cx.launch
        .bindings
        .iter()
        .find(|b| cx.graph.class_of(b.value) == class && !b.arena)
        .map(|b| b.binding)
}

fn zero(cx: &LowerCtx<'_>, id: Id) -> bool {
    matches!(&cx.graph.node(cx.selected(id)).op,
        Op::Logical(Logical::Leaf(LeafKind::Const { value, .. })) if value.bits() == 0)
}

impl Transfers {
    pub(super) fn plan(cx: &LowerCtx<'_>, binds: &BindingEnv) -> Option<Self> {
        if !matches!(&cx.graph.node(cx.launch.root).op,
            Op::Launch(Launch::Map { body, .. }) if matches!(body.kind(), ScalarKind::Arg(_)))
            && !matches!(
                &cx.graph.node(cx.launch.root).op,
                Op::Launch(Launch::Scatter {
                    combine: ScatterCombine::Set,
                    ..
                })
            )
        {
            return None;
        }
        let output = cx
            .launch
            .bindings
            .iter()
            .find(|b| b.value == cx.launch.root && b.kind != BindKind::Read)
            .or_else(|| cx.launch.bindings.iter().find(|b| b.kind != BindKind::Read))?;
        if output.arena {
            return None;
        }
        let (layout, dtype) = crate::lower::bound_layout(cx, output.value);
        if dtype.is_quantized() || layout != Layout::contiguous(&cx.graph.facts(output.value).shape)
        {
            return None;
        }
        let mut result = Self(Vec::new());
        match &cx.graph.node(cx.launch.root).op {
            Op::Launch(Launch::Map { body, ops, .. }) => {
                let ScalarKind::Arg(arg) = body.kind() else {
                    return None;
                };
                result.copy(
                    cx,
                    binds,
                    ops.get(*arg as usize)?,
                    &layout,
                    output.binding,
                    dtype,
                )?;
            }
            Op::Launch(Launch::Scatter {
                axis,
                combine: ScatterCombine::Set,
                ops,
                ..
            }) => {
                let [base, idx, upd] = ops.as_slice() else {
                    return None;
                };
                let axis = *axis as usize;
                let start = index_start(cx, idx)?;
                let count = binds.dim(*upd.layout.shape().get(axis)?)?;
                let end = start.checked_add(count)?;
                if end > binds.dim(*layout.shape().get(axis)?)? || end > u64::from(u32::MAX) + 1 {
                    return None;
                }
                if zero(cx, base.src) {
                    let bytes = layout
                        .shape()
                        .iter()
                        .try_fold(dtype.byte_size(), |n, d| n.checked_mul(binds.dim(*d)?))?;
                    if bytes == 0 || !bytes.is_multiple_of(4) {
                        return None;
                    }
                    result.0.push(Region {
                        src: None,
                        dst: output.binding,
                        src_offset: 0,
                        dst_offset: 0,
                        bytes,
                    });
                } else if slot(cx, base.src)? != output.binding {
                    result.copy(cx, binds, base, &layout, output.binding, dtype)?;
                }
                let mut shape = layout.shape().to_vec();
                shape[axis] = upd.layout.shape()[axis];
                let offset = start.checked_mul(binds.dim(layout.strides()[axis])?)?;
                let dst = Layout::from_parts(Dim::Const(offset), &shape, layout.strides()).ok()?;
                result.copy(cx, binds, upd, &dst, output.binding, dtype)?;
            }
            _ => return None,
        }
        // Supplied buffers may alias despite distinct graph values. Never select
        // a copy program that could overwrite one of its own sources.
        for region in &result.0 {
            if let Some(src) = region.src {
                let value = |binding| {
                    cx.launch
                        .bindings
                        .iter()
                        .find(|b| b.binding == binding)
                        .map(|b| b.value)
                };
                if src == region.dst {
                    return None;
                }
                if let (Some(src), Some(dst)) = (
                    value(src).and_then(|v| binds.buffers.get(&v)),
                    value(region.dst).and_then(|v| binds.buffers.get(&v)),
                ) && src.addr() == dst.addr()
                {
                    return None;
                }
            }
        }
        (!result.0.is_empty()).then_some(result)
    }

    pub(super) fn uses_binding(&self, binding: u32) -> bool {
        self.0
            .iter()
            .any(|r| r.src == Some(binding) || r.dst == binding)
    }

    pub(super) fn prepare(plan: &Plan, graph: &EGraph, binds: &BindingEnv) -> Vec<Option<Self>> {
        let mut transfers: Vec<_> = plan
            .launches
            .iter()
            .map(|launch| {
                Self::plan(
                    &LowerCtx {
                        plan,
                        launch,
                        graph,
                        symbols: &plan.symbols,
                        dim_bindings: &[],
                    },
                    binds,
                )
            })
            .collect();
        // Copy lowering removes index reads. Eliminate producers whose only
        // consumers vanished, while preserving every caller-owned output.
        let mut needed: FxHashSet<Id> = binds.buffers.keys().copied().collect();
        for (launch, transfer) in plan.launches.iter().zip(&mut transfers).rev() {
            if launch
                .bindings
                .iter()
                .filter(|b| b.kind != BindKind::Read)
                .all(|b| !needed.contains(&b.value))
            {
                *transfer = Some(Self(Vec::new()));
            }
            for binding in &launch.bindings {
                if transfer
                    .as_ref()
                    .is_none_or(|t| t.uses_binding(binding.binding))
                {
                    needed.insert(binding.value);
                }
            }
        }
        transfers
    }

    fn copy(
        &mut self,
        cx: &LowerCtx<'_>,
        binds: &BindingEnv,
        src: &Operand,
        dst: &Layout,
        dst_slot: u32,
        dtype: Dtype,
    ) -> Option<()> {
        if src.access != AccessPlan::Alias || src.layout.shape() != dst.shape() {
            return None;
        }
        let src_slot = slot(cx, src.src)?;
        let (storage, source_dtype) = crate::lower::bound_layout(cx, src.src);
        if source_dtype != dtype
            || storage != Layout::contiguous(&cx.graph.facts(cx.selected(src.src)).shape)
        {
            return None;
        }
        let dims = src
            .layout
            .shape()
            .iter()
            .map(|d| binds.dim(*d))
            .collect::<Option<Vec<_>>>()?;
        let ss = src
            .layout
            .strides()
            .iter()
            .map(|d| binds.dim(*d))
            .collect::<Option<Vec<_>>>()?;
        let ds = dst
            .strides()
            .iter()
            .map(|d| binds.dim(*d))
            .collect::<Option<Vec<_>>>()?;
        let mut split = dims.len();
        let mut run = 1u64;
        while split > 0 && (dims[split - 1] == 1 || (ss[split - 1] == run && ds[split - 1] == run))
        {
            split -= 1;
            run = run.checked_mul(dims[split])?;
        }
        let rows = dims[..split]
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))?;
        if rows == 0 || rows > 4096 || self.0.len() + usize::try_from(rows).ok()? > 4096 {
            return None;
        }
        let bytes = run.checked_mul(dtype.byte_size())?;
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return None;
        }
        for row in 0..rows {
            let (mut so, mut to) = (binds.dim(src.layout.offset())?, binds.dim(dst.offset())?);
            let mut rem = row;
            for axis in (0..split).rev() {
                let at = rem % dims[axis];
                rem /= dims[axis];
                so = so.checked_add(at.checked_mul(ss[axis])?)?;
                to = to.checked_add(at.checked_mul(ds[axis])?)?;
            }
            let src_offset = so.checked_mul(dtype.byte_size())?;
            let dst_offset = to.checked_mul(dtype.byte_size())?;
            if !src_offset.is_multiple_of(4) || !dst_offset.is_multiple_of(4) {
                return None;
            }
            self.0.push(Region {
                src: Some(src_slot),
                dst: dst_slot,
                src_offset,
                dst_offset,
                bytes,
            });
        }
        Some(())
    }

    pub(super) fn record(&self, buffers: &[Buf]) -> Result<CommandRecord> {
        use crate::launch::BufferTransfer;
        let mut copies = Vec::with_capacity(self.0.len());
        for r in &self.0 {
            let dst = buffers
                .get(r.dst as usize)
                .ok_or_else(|| Error::Plan("copy destination binding missing".into()))?;
            let d = crate::pool::GpuBuffer::of(dst, "copy destination")?;
            if r.dst_offset
                .checked_add(r.bytes)
                .is_none_or(|end| end > d.size)
            {
                return Err(Error::Plan("copy destination range exceeds storage".into()));
            }
            if let Some(src) = r.src {
                let src = buffers
                    .get(src as usize)
                    .ok_or_else(|| Error::Plan("copy source binding missing".into()))?;
                let s = crate::pool::GpuBuffer::of(src, "copy source")?;
                if src.addr() == dst.addr()
                    || r.src_offset
                        .checked_add(r.bytes)
                        .is_none_or(|end| end > s.size)
                {
                    return Err(Error::Plan("invalid copy source range or alias".into()));
                }
                copies.push(BufferTransfer::Copy {
                    src: src.clone(),
                    dst: dst.clone(),
                    src_offset: r.src_offset,
                    dst_offset: r.dst_offset,
                    bytes: r.bytes,
                });
                crate::pool::COPY_BYTES.fetch_add(r.bytes, std::sync::atomic::Ordering::Relaxed);
            } else {
                copies.push(BufferTransfer::Clear {
                    dst: dst.clone(),
                    offset: r.dst_offset,
                    bytes: r.bytes,
                });
            }
        }
        Ok(CommandRecord::Transfers { copies })
    }
}
