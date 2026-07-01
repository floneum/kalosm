use crate::Layout;
use fusor_tile_ir as tile_ir;

#[derive(Clone)]
pub(crate) struct DirectMatrixLayout {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) offset: u32,
    pub(crate) layout: tile_ir::Layout,
}

pub(crate) fn flatten_matrix_layout(layout: &Layout) -> Option<DirectMatrixLayout> {
    flatten_matrix_layout_split(layout, layout.shape().len().checked_sub(1)?)
}

/// Flatten a strided layout into a 2-D matrix view: `shape[..row_dims]`
/// flattens to rows, `shape[row_dims..]` to columns. Sides whose dims merge
/// affinely use a plain strided layout; anything else (a conv im2col window,
/// a non-affine batch prefix) becomes a `MultiFlattenMap` whose sub-axes
/// divmod the flat coordinate back apart per load.
pub(crate) fn flatten_matrix_layout_split(
    layout: &Layout,
    row_dims: usize,
) -> Option<DirectMatrixLayout> {
    let shape = layout.shape();
    let strides = layout.strides();
    if row_dims == 0 || row_dims >= shape.len() || shape.contains(&0) {
        return None;
    }

    let rows = shape[..row_dims]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    let cols = shape[row_dims..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    let rows_u32 = rows.try_into().ok()?;
    let cols_u32 = cols.try_into().ok()?;
    let offset = layout.offset().try_into().ok()?;
    let side_is_affine = |range: std::ops::Range<usize>| {
        range
            .clone()
            .zip(range.skip(1))
            .all(|(axis, next)| strides[axis] == strides[next].saturating_mul(shape[next]))
    };

    let layout = if side_is_affine(0..row_dims) && side_is_affine(row_dims..shape.len()) {
        let row_stride: u32 = strides[row_dims - 1].try_into().ok()?;
        let col_stride: u32 = strides[shape.len() - 1].try_into().ok()?;
        tile_ir::Layout::strided(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([rows_u32, cols_u32]),
            &[row_stride, col_stride],
        )
    } else {
        let axis_group = |range: std::ops::Range<usize>| -> Option<tile_ir::AxisGroup> {
            let mut sub_axes = Vec::with_capacity(range.len());
            for axis in range {
                // Extent-1 axes contribute nothing to the flat coordinate
                // decomposition; dropping them saves a divmod per load.
                if shape[axis] == 1 {
                    continue;
                }
                sub_axes.push(tile_ir::SubAxis {
                    extent: shape[axis].try_into().ok()?,
                    stride: strides[axis].try_into().ok()?,
                });
            }
            if sub_axes.is_empty() {
                sub_axes.push(tile_ir::SubAxis {
                    extent: 1,
                    stride: 0,
                });
            }
            Some(tile_ir::AxisGroup { sub_axes })
        };
        tile_ir::Layout::with_indexing(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([rows_u32, cols_u32]),
            tile_ir::MultiFlattenMap {
                groups: vec![axis_group(0..row_dims)?, axis_group(row_dims..shape.len())?],
            },
        )
    };

    Some(DirectMatrixLayout {
        rows: rows_u32,
        cols: cols_u32,
        offset,
        layout,
    })
}

pub(crate) fn tile_storage_read_with_direct_layout_typed(
    phase: &mut tile_ir::tile::Program,
    element: tile_ir::ElementType,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage {
    phase.storage_read_with_layout_offset(element, view.layout, view.offset)
}

pub(crate) fn tile_storage_write_with_direct_layout_typed(
    phase: &mut tile_ir::tile::Program,
    element: tile_ir::ElementType,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage {
    phase.storage_write_with_layout_offset(element, view.layout, view.offset)
}

pub(crate) fn tile_storage_read_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage {
    tile_storage_read_with_direct_layout_typed(phase, tile_ir::ElementType::F32, view)
}

pub(crate) fn tile_storage_write_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage {
    tile_storage_write_with_direct_layout_typed(phase, tile_ir::ElementType::F32, view)
}
