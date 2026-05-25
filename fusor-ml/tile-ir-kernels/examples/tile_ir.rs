use fusor_tile_ir::{tile, GgmlQuantFormat, KernelIr, Shape, TileLiteral, WorkgroupAxis, F32};
use fusor_tile_ir_kernels as tile_ir_kernels;

fn fused_bias_gelu_residual_ir(rows: u32, cols: u32) -> KernelIr {
    const BLOCK: usize = 128;
    assert!(cols <= BLOCK as u32, "example emits one tile per row");

    tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let bias = phase.storage_read::<F32, 1>(Shape::new([cols]));
        let residual = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let y = phase.storage_write::<F32, 2>(Shape::new([rows, cols]));

        phase.program_grid::<BLOCK>([1, rows, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.lane();
            let mask = col.lt(cols);

            let x = program.load(x.at((&row, &col)), mask.clone(), 0.0);
            let bias = program.load(bias.at(&col), mask.clone(), TileLiteral::f32(0.0));
            let residual = program.load(residual.at((&row, &col)), mask.clone(), 0.0);

            // One tile expression becomes one fused kernel: bias add, GELU,
            // residual add, and the final write happen without intermediate
            // storage.
            let activated = (x + bias).gelu();
            program.store(y.at((row, col)), activated + residual, mask);
        });
    })
}

fn fused_rms_norm_silu_ir(rows: u32, cols: u32, eps: f32) -> KernelIr {
    const BLOCK: usize = 128;
    assert!(cols <= BLOCK as u32, "example emits one tile per row");

    tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let weight = phase.storage_read::<F32, 1>(Shape::new([cols]));
        let y = phase.storage_write::<F32, 2>(Shape::new([rows, cols]));

        phase.program_grid::<BLOCK>([1, rows, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.lane();
            let mask = col.lt(cols);

            let x = program.load(x.at((&row, &col)), mask.clone(), 0.0);
            let weight = program.load(weight.at(&col), mask.clone(), TileLiteral::f32(0.0));

            let square = x.clone() * x.clone();
            let sum_square = program.reduce_sum(square);
            let mean_square = sum_square / tile::Tile::literal(cols as f32);
            let inv_rms = (mean_square + tile::Tile::literal(eps)).inverse_sqrt();

            // The normalization reduction feeds directly into the per-element
            // scale and SiLU activation, so callers can fuse the whole block
            // as a single Tile IR program.
            let normalized = x * inv_rms * weight;
            program.store(y.at((row, col)), normalized.silu(), mask);
        });
    })
}

fn softmax_ir(rows: u32, cols: u32) -> KernelIr {
    const BLOCK: usize = 128;
    assert!(cols <= BLOCK as u32, "example emits one tile per row");

    tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let y = phase.storage_write::<F32, 2>(Shape::new([rows, cols]));

        phase.program_grid::<BLOCK>([1, rows, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.lane();
            let mask = col.lt(cols);
            let values = program.load(x.at((&row, &col)), mask.clone(), -3.4028235e38);
            let max = program.reduce_max(values.clone());
            let shifted = values - max;
            let exp = shifted.exp();
            let sum = program.reduce_sum(exp.clone());

            program.store(y.at((row, col)), exp / sum, mask);
        });
    })
}

fn qmatmul_ir(format: GgmlQuantFormat, m: u32, n: u32, k: u32) -> KernelIr {
    tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([m, k]));
        let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
        let y = phase.storage_write::<F32, 2>(Shape::new([m, n]));

        tile_ir_kernels::qmatmul_with_epilogue(
            phase,
            &a,
            &b,
            &y,
            4,
            &tile_ir_kernels::QmatmulEpilogues::empty(),
            8,
            4,
        );
    })
}

fn qgemv_ir(format: GgmlQuantFormat, n: u32, k: u32) -> KernelIr {
    tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, k]));
        let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, n]));

        tile_ir_kernels::qgemv_with_epilogue(
            phase,
            &a,
            &b,
            &y,
            1,
            Option::<&tile_ir_kernels::UnaryEpilogue>::None,
        );
    })
}

fn qgemv_with_silu_epilogue_ir(format: GgmlQuantFormat, n: u32, k: u32) -> KernelIr {
    tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, k]));
        let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, n]));

        // Kernel helpers can accept Tile IR closures too. The qgemv body owns
        // the quantized dot product, then injects this epilogue before the
        // store so activation fusion does not need a second dispatch.
        let silu = tile_ir_kernels::UnaryEpilogue::new("silu", |value| value.silu());
        tile_ir_kernels::qgemv_with_epilogue(phase, &a, &b, &y, 1, Some(&silu));
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = GgmlQuantFormat::Q4_0;
    let k = format.block_elements();

    fused_bias_gelu_residual_ir(3, 100).lower_to_naga()?;
    fused_rms_norm_silu_ir(3, 100, 1e-5).lower_to_naga()?;
    softmax_ir(3, 100).lower_to_naga()?;
    qmatmul_ir(format, 32, 64, k).lower_to_naga()?;
    qgemv_ir(format, 64, k).lower_to_naga()?;
    qgemv_with_silu_epilogue_ir(format, 64, k).lower_to_naga()?;

    println!("typed tile IR examples lowered successfully");
    Ok(())
}
