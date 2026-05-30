use fusor::{arange, DataType, Device, SimdElement, Tensor};

fn pad_to<const R: usize, T: DataType + SimdElement + std::fmt::Debug>(
    tensor: &Tensor<R, T>,
    shape: [usize; R],
    value: T,
) -> Tensor<R, T> {
    let mut current = tensor.to_concrete();
    for (d, &target_dim) in shape.iter().enumerate() {
        let current_dim = current.shape()[d];
        if current_dim < target_dim {
            let pad_size = target_dim - current_dim;
            let mut pad_shape = current.shape();
            pad_shape[d] = pad_size;
            let padding = Tensor::full(&current.device(), pad_shape, value);
            current = Tensor::cat(vec![current, padding], d);
        }
    }
    current
}

#[cfg(test)]
#[tokio::test]
async fn test_pad() {
    let device = Device::new().await.unwrap();
    let tensor = arange(&device, 0u32, 12).reshape([3, 4]).to_concrete();
    let value = u32::MAX;

    let padded_tensor = pad_to(&tensor, [4, 5], value);

    println!("Original Tensor shape: {:?}", tensor.shape());
    println!("Padded Tensor shape: {:?}", padded_tensor.shape());
    let slice = padded_tensor.as_slice().await.unwrap();
    let data: Vec<u32> = slice.as_slice().to_vec();
    // Reshape to 2D for comparison
    let expected = vec![
        0,
        1,
        2,
        3,
        u32::MAX,
        4,
        5,
        6,
        7,
        u32::MAX,
        8,
        9,
        10,
        11,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
    ];
    assert_eq!(data, expected);
}

pub(crate) fn get_window_index(
    grid_thw: impl IntoIterator<Item = (usize, usize, usize)>,
    window_size: usize,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    patch_size: usize,
    device: &Device,
) -> fusor::Result<(Tensor<1, u32>, Vec<u32>)> {
    let mut window_index = vec![];
    let mut cu_window_seqlens = vec![0];
    let mut window_index_id = 0;
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;

    for (grid_t, grid_h, grid_w) in grid_thw {
        let llm_grid_h = grid_h / spatial_merge_size;
        let llm_grid_w = grid_w / spatial_merge_size;
        let index = arange(device, 0u32, (grid_t * llm_grid_h * llm_grid_w) as u32)
            .reshape([grid_t, llm_grid_h, llm_grid_w])
            .to_concrete();
        let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
        let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
        let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
        let index_padded = {
            pad_to(
                &index,
                [grid_t, llm_grid_h + pad_h, llm_grid_w + pad_w],
                u32::MAX,
            )
            .reshape([
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            ])
            .permute([0, 1, 3, 2, 4])
            .reshape([
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            ])
            .to_concrete()
        };

        let index_padded_flat = index_padded
            .reshape([index_padded.shape().iter().product::<usize>()])
            .to_concrete();
        let slice = pollster::block_on(index_padded_flat.as_slice())?;
        let index_padded_slice: Vec<u32> = slice.as_slice().to_vec();

        // index_padded shape: [grid_t, num_windows, window_h, window_w]
        // Flattened in slice.
        // We need to calculate seqlens per window.
        // Dimension 2 and 3 are window dimensions.
        // Sum over dim 2 and 3 means sum over last two dimensions.
        let window_area = vit_merger_window_size * vit_merger_window_size;
        let num_windows = grid_t * num_windows_h * num_windows_w;

        let mut seqlens_vec = Vec::with_capacity(num_windows);

        for window_idx in 0..num_windows {
            let start = window_idx * window_area;
            let end = start + window_area;
            let window = &index_padded_slice[start..end];
            let count = window.iter().filter(|&&x| x != u32::MAX).count() as u32;
            seqlens_vec.push(count);
        }

        // index_new is all non-MAX elements
        let index_new_vec = index_padded_slice
            .iter()
            .filter(|&&x| x != u32::MAX)
            .copied()
            .collect::<Vec<_>>();

        let index_new = Tensor::new(device, &index_new_vec);
        // window_index_id is u32
        window_index.push(index_new + window_index_id);

        // Calculate cu_seqlens_tmp
        // seqlens.cumsum(0) * spatial_merge_unit
        let mut cumsum = 0;
        let mut cu_seqlens_tmp = Vec::with_capacity(seqlens_vec.len());
        for len in seqlens_vec {
            cumsum += len;
            cu_seqlens_tmp.push(cumsum * spatial_merge_unit as u32);
        }

        let last_val = *cu_window_seqlens.last().unwrap();
        cu_window_seqlens.extend(cu_seqlens_tmp.into_iter().map(|x| x + last_val));

        window_index_id += (grid_t * llm_grid_h * llm_grid_w) as u32;
    }

    let window_index = Tensor::cat(window_index, 0);

    Ok((window_index, cu_window_seqlens))
}

#[cfg(test)]
#[tokio::test]
async fn test_get_window_index() {
    let window_size = 2;
    let spatial_merge_size = 1;
    let spatial_merge_unit = 2;
    let patch_size = 1;
    let grid_thw = vec![(1, 8, 4)];
    let device = Device::new().await.unwrap();

    let (window_index, cu_window_seqlens) = get_window_index(
        grid_thw,
        window_size,
        spatial_merge_size,
        spatial_merge_unit,
        patch_size,
        &device,
    )
    .unwrap();

    let slice = window_index.as_slice().await.unwrap();
    let window_index_vec: Vec<u32> = slice.as_slice().to_vec();
    println!("Window Index: {window_index_vec:?}");
    assert_eq!(
        window_index_vec,
        vec![
            0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22, 23,
            24, 25, 28, 29, 26, 27, 30, 31
        ]
    );
    println!("CU Window Seqlens: {cu_window_seqlens:?}");
    assert_eq!(
        cu_window_seqlens,
        vec![0, 8, 16, 16, 24, 32, 32, 40, 48, 48, 56, 64, 64, 64, 64, 64]
    );
}
