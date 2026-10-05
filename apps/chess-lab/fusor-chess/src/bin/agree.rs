//! Native diagnostic: a checkpoint's agreement with reference scores.
//! Usage: agree <evalset.json> <checkpoint.bin> [count scale]
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::cpu;
    let args: Vec<String> = std::env::args().collect();
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let rows: Vec<(Vec<u32>, i32)> = rows.as_array().unwrap().iter().map(|r| {
        (r["state"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(), r["cp"].as_i64().unwrap() as i32)
    }).collect();
    let data: Vec<f32> = std::fs::read(&args[2]).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let net = cpu::Net::from_checkpoint(&data).expect("a Rookie checkpoint");
    let scale = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(rookie_fusor::count_scale);
    let (all, balanced) = cpu::agreement(&net.with_count_scale(scale), &rows);
    println!("r = {all:.3} overall, {balanced:.3} on balanced positions");
}
