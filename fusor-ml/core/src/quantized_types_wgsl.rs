use std::fmt::Write;

use fusor_gguf::{BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType};

fn byte_array_array_u32<W: Write>(f: &mut W, name: &str, byte_size: usize) -> std::fmt::Result {
    assert!(byte_size.is_multiple_of(4));
    let size = byte_size / 4;
    writeln!(f, "    {name}: array<u32, {size}>,")
}

fn float_type(use_f16: bool) -> &'static str {
    if use_f16 { "f16" } else { "f32" }
}

pub(crate) fn write_q4_0_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q4_0 = GgmlType::Q4_0;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q4_0} {{")?;
    writeln!(f, "    scale: {float_type},")?;
    byte_array_array_u32(f, "data", BlockQ4_0::WEIGHTS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q5_0_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q5_0 = GgmlType::Q5_0;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q5_0} {{")?;
    writeln!(f, "    scale: {float_type},")?;
    byte_array_array_u32(f, "data_high_bits", BlockQ5_0::WEIGHTS_HIGH_BITS_SIZE)?;
    byte_array_array_u32(f, "data_low_bits", BlockQ5_0::WEIGHTS_LOW_BITS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q8_0_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q8_0 = GgmlType::Q8_0;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q8_0} {{")?;
    writeln!(f, "    scale: {float_type},")?;
    byte_array_array_u32(f, "data", BlockQ8_0::WEIGHTS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q4_k_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q4_k = GgmlType::Q4K;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q4_k} {{")?;
    writeln!(f, "    scale: {float_type},")?;
    writeln!(f, "    min: {float_type},")?;
    byte_array_array_u32(f, "scales", BlockQ4K::SCALES_SIZE)?;
    byte_array_array_u32(f, "data", BlockQ4K::WEIGHTS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q5_k_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q5_k = GgmlType::Q5K;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q5_k} {{")?;
    writeln!(f, "    scale: {float_type},")?;
    writeln!(f, "    min: {float_type},")?;
    byte_array_array_u32(f, "scales", BlockQ5K::SCALES_SIZE)?;
    byte_array_array_u32(f, "qh", BlockQ5K::QH_SIZE)?;
    byte_array_array_u32(f, "qs", BlockQ5K::QS_SIZE)?;
    writeln!(f, "}};")?;

    Ok(())
}

pub(crate) fn write_q6_k_type<W: Write>(f: &mut W, use_f16: bool) -> std::fmt::Result {
    let q6_k = GgmlType::Q6K;
    let float_type = float_type(use_f16);
    writeln!(f, "struct {q6_k} {{")?;
    byte_array_array_u32(f, "data_low_bits", BlockQ6K::WEIGHTS_LOW_BITS_SIZE)?;
    byte_array_array_u32(f, "data_high_bits", BlockQ6K::WEIGHTS_HIGH_BITS_SIZE)?;
    byte_array_array_u32(f, "scales", BlockQ6K::SCALES_SIZE)?;
    writeln!(f, "    scale: {float_type},")?;
    writeln!(f, "}};")?;

    Ok(())
}
