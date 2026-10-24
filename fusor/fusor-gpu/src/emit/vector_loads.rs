//! Pack complete adjacent scalar read groups into physical vector loads.
//! This changes storage access width only; scalar arithmetic remains untouched.
use fusor_ir::ir::kernel::{
    Addr, Buffer, BufferAccess, ElementType, ScalarElement, Source, StorageView, TileExpr,
    TileExprKind, TileLiteral, simplify_index,
};
use fusor_tile::build::Kernel;
use rustc_hash::{FxHashMap, FxHashSet};

pub(super) fn analyze(buffers: &[Buffer], expressions: &FxHashSet<TileExpr>) -> FxHashSet<u32> {
    let mut rejected = FxHashSet::default();
    for buffer in buffers {
        // A vector must never hide a partial final storage element. Symbolic
        // declarations have zero extent and stay scalar, as do mixed views.
        let count = buffer.layout.element_count();
        if buffer.access != BufferAccess::Read
            || count == 0
            || !count.is_multiple_of(4)
            || !matches!(
                buffer.element,
                ElementType::Scalar(ScalarElement::F32 | ScalarElement::I32 | ScalarElement::U32)
            )
            || buffers
                .iter()
                .any(|other| other.binding == buffer.binding && other.element != buffer.element)
        {
            rejected.insert(buffer.binding);
        }
    }
    let b = Kernel::new();
    let mut groups: FxHashMap<u32, FxHashMap<TileExpr, u8>> = FxHashMap::default();
    for expr in expressions {
        let TileExprKind::Load {
            src: Source::Storage(view),
            addr,
            mask,
            ..
        } = expr.kind()
        else {
            continue;
        };
        let binding = view.buffer.binding;
        if rejected.contains(&binding) {
            continue;
        }
        let group = (|| {
            // Masking and tails retain their existing scalar access path.
            if !mask.is_constant_true() {
                return None;
            }
            let index = physical_index(&b, view, addr)?;
            let component = simplify_index(&b.rem(index.clone(), b.u32(4)));
            let TileExprKind::Literal(TileLiteral::U32(component)) = component.kind() else {
                return None;
            };
            let group = simplify_index(&b.div(index, b.u32(4)));
            Some((group, *component))
        })();
        if let Some((group, component)) = group {
            *groups.entry(binding).or_default().entry(group).or_default() |= 1 << component;
        } else {
            rejected.insert(binding);
        }
    }
    groups
        .into_iter()
        .filter_map(|(binding, groups)| {
            (!rejected.contains(&binding) && groups.values().all(|&components| components == 15))
                .then_some(binding)
        })
        .collect()
}

fn physical_index(b: &Kernel, view: &StorageView, addr: &Addr) -> Option<TileExpr> {
    let index = match addr {
        Addr::Linear(index) => index.clone(),
        Addr::Rc2 { row, col } => {
            let [rows, cols] = &view.layout.indexing.groups[..] else {
                return None;
            };
            let [r] = &rows.sub_axes[..] else { return None };
            let [c] = &cols.sub_axes[..] else { return None };
            b.add(
                b.mul(row.clone(), b.u32(r.stride)),
                b.mul(col.clone(), b.u32(c.stride)),
            )
        }
    };
    Some(simplify_index(&b.add(index, b.u32(view.offset))))
}
