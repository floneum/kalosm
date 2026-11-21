//! Native diagnostic: where the CPU evaluator's time goes, per kernel launch.
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::cpu::{Net, Position, start_state};
    let net = Net::new(rookie_fusor::Config::default());
    let pos = Position::from_state(&start_state());
    for _ in 0..3 {
        net.evaluate(&pos);
    }
    let launches = net.profile();
    let total: f64 = launches.iter().map(|l| l.2).sum();
    for (name, grid, us) in &launches {
        println!("{us:8.2} us  {grid:?}  {name}");
    }
    println!("{} launches, {total:.1} us", launches.len());
    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        net.evaluate(&pos);
    }
    println!("evaluate: {:.2} us each", start.elapsed().as_secs_f64() * 100.);
}
