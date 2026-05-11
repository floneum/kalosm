use fusor_tile_ir::{tile, GgmlQuantFormat, KernelIr, Shape, WorkgroupAxis, F32};
use fusor_tile_ir_kernels as tile_ir_kernels;

fn softmax_ir(rows: u32, cols: u32) -> KernelIr {
    const BLOCK: usize = 128;
    assert!(cols <= BLOCK as u32, "example emits one tile per row");

    tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let y = phase.storage_write::<F32, 2>(Shape::new([rows, cols]));

        phase.program_grid::<BLOCK>([1, rows, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.arange();
            let mask = col.lt(cols);
            let values = program.load(x.at(&row, &col), mask.clone(), -3.4028235e38);
            let max = program.reduce_max(values.clone());
            let shifted = values - max;
            let exp = shifted.exp();
            let sum = program.reduce_sum(exp.clone());

            program.store(y.at(row, col), exp / sum, mask);
        });
    })
}

fn qmatmul_ir(format: GgmlQuantFormat, m: u32, n: u32, k: u32) -> KernelIr {
    tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([m, k]));
        let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
        let y = phase.storage_write::<F32, 2>(Shape::new([m, n]));

        tile_ir_kernels::qmatmul::<8, 4, 8>(phase, &a, &b, &y, 4);
    })
}

fn qgemv_ir(format: GgmlQuantFormat, n: u32, k: u32) -> KernelIr {
    tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, k]));
        let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, n]));

        tile_ir_kernels::qgemv::<4, 64>(phase, &a, &b, &y, 4, 1);
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = GgmlQuantFormat::Q4_0;
    let k = format.block_elements();

    softmax_ir(3, 100).lower_to_naga()?;
    qmatmul_ir(format, 32, 64, k).lower_to_naga()?;
    qgemv_ir(format, 64, k).lower_to_naga()?;

    println!("typed tile IR examples lowered successfully");
    Ok(())
}
