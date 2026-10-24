//! Pack complete adjacent scalar read groups into physical vector loads.
//! This changes storage access width only; scalar arithmetic remains untouched.
use fusor_ir::ir::kernel::{
    Addr, Buffer, BufferAccess, ElementType, ScalarElement, Source, StorageView, TileExpr,
    TileExprKind, TileLiteral, simplify_index,
};
use fusor_tile::build::Kernel;
use rustc_hash::{FxHashMap, FxHashSet};

pub(super) fn analyze(
    buffers: &[Buffer],
    expressions: &FxHashSet<TileExpr>,
) -> FxHashMap<u32, u32> {
    let mut widths = FxHashMap::default();
    for width in [2, 4] {
        for binding in analyze_width(buffers, expressions, width) {
            widths.insert(binding, width);
        }
    }
    widths
}

fn analyze_width(
    buffers: &[Buffer],
    expressions: &FxHashSet<TileExpr>,
    width: u32,
) -> FxHashSet<u32> {
    let mut rejected = FxHashSet::default();
    for buffer in buffers {
        // The symbolic declaration extent is only a lower-bound placeholder;
        // its divisibility cannot prove that physical storage has full vectors.
        let count = buffer.layout.element_count();
        if !buffer.element_count_is_static
            || buffer.access != BufferAccess::Read
            || count == 0
            || !count.is_multiple_of(u64::from(width))
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
            let component = simplify_index(&b.rem(index.clone(), b.u32(width)));
            let TileExprKind::Literal(TileLiteral::U32(component)) = component.kind() else {
                return None;
            };
            let group = simplify_index(&b.div(index, b.u32(width)));
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
            (!rejected.contains(&binding)
                && groups
                    .values()
                    .all(|&components| components == (1 << width) - 1))
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
            let stride = |index: &TileExpr, stride| match stride {
                0 => b.u32(0),
                1 => index.clone(),
                _ => b.mul(index.clone(), b.u32(stride)),
            };
            b.add(stride(row, r.stride), stride(col, c.stride))
        }
    };
    Some(simplify_index(&b.add(index, b.u32(view.offset))))
}
