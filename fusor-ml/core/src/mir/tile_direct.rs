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

    let layout = if prefix_is_affine {
        let row_stride: u32 = strides[shape.len() - 2].try_into().ok()?;
        let col_stride: u32 = strides[shape.len() - 1].try_into().ok()?;
        tile_ir::Layout::strided(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([rows_u32, cols_u32]),
            &[row_stride, col_stride],
        )
    } else {
        let prefix_shape: Vec<u32> = shape[..shape.len() - 1]
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let prefix_strides: Vec<u32> = strides[..strides.len() - 1]
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let column_stride: u32 = strides[strides.len() - 1].try_into().ok()?;
        let m_group = tile_ir::AxisGroup {
            sub_axes: prefix_shape
                .iter()
                .zip(prefix_strides.iter())
                .map(|(&extent, &stride)| tile_ir::SubAxis { extent, stride })
                .collect(),
        };
        let k_group = tile_ir::AxisGroup {
            sub_axes: vec![tile_ir::SubAxis {
                extent: cols_u32,
                stride: column_stride,
            }],
        };
        tile_ir::Layout::with_indexing(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([rows_u32, cols_u32]),
            tile_ir::MultiFlattenMap {
                groups: vec![m_group, k_group],
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

pub(crate) fn tile_storage_read_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage<tile_ir::F32, 2> {
    phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(view.layout, view.offset)
}

pub(crate) fn tile_storage_write_with_direct_layout(
    phase: &mut tile_ir::tile::Program,
    view: DirectMatrixLayout,
) -> tile_ir::tile::Storage<tile_ir::F32, 2> {
    phase.storage_write_with_layout_offset::<tile_ir::F32, 2>(view.layout, view.offset)
}
