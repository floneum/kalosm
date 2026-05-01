use crate::Layout;
use phase_token_prototype as tile_ir;

pub(crate) struct DirectMatrixLayout {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) offset: u32,
    pub(crate) layout: tile_ir::Layout,
    pub(crate) index_map: Option<tile_ir::StorageIndexMap>,
}

pub(crate) fn flatten_matrix_layout(layout: &Layout) -> Option<DirectMatrixLayout> {
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() < 2 || shape.contains(&0) {
        return None;
    }

    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    let cols = *shape.last()?;
    let rows_u32 = rows.try_into().ok()?;
    let cols_u32 = cols.try_into().ok()?;
    let offset = layout.offset().try_into().ok()?;
    let prefix_is_affine = (0..shape.len().saturating_sub(2))
        .all(|axis| strides[axis] == strides[axis + 1].saturating_mul(shape[axis + 1]));

    let (layout, index_map) = if prefix_is_affine {
        let row_stride = strides[shape.len() - 2];
        let col_stride = strides[shape.len() - 1];
        (
            tile_ir::Layout::strided(
                tile_ir::MemoryLevel::Storage,
                tile_ir::Shape::new([rows_u32, cols_u32]),
                tile_ir::Strides::new([row_stride.try_into().ok()?, col_stride.try_into().ok()?]),
            ),
            None,
        )
    } else {
        let prefix_shape = shape[..shape.len() - 1]
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let prefix_strides = strides[..strides.len() - 1]
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        (
            tile_ir::Layout::contiguous(
                tile_ir::MemoryLevel::Storage,
                tile_ir::Shape::new([rows_u32, cols_u32]),
            ),
            Some(tile_ir::StorageIndexMap::FlattenedMatrix(
                tile_ir::FlattenedMatrixMap {
                    prefix_shape,
                    prefix_strides,
                    column_stride: strides[strides.len() - 1].try_into().ok()?,
                },
            )),
        )
    };

    Some(DirectMatrixLayout {
        rows: rows_u32,
        cols: cols_u32,
        offset,
        layout,
        index_map,
    })
}

pub(crate) fn tile_storage_read_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage<tile_ir::F32, 2> {
    if let Some(index_map) = view.index_map {
        phase.storage_read_with_layout_offset_and_index_map::<tile_ir::F32, 2>(
            view.layout,
            view.offset,
            index_map,
        )
    } else {
        phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(view.layout, view.offset)
    }
}

pub(crate) fn tile_storage_write_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage<tile_ir::F32, 2> {
    if let Some(index_map) = view.index_map {
        phase.storage_write_with_layout_offset_and_index_map::<tile_ir::F32, 2>(
            view.layout,
            view.offset,
            index_map,
        )
    } else {
        phase.storage_write_with_layout_offset::<tile_ir::F32, 2>(view.layout, view.offset)
    }
}
