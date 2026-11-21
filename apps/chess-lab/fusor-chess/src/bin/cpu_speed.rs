//! Native diagnostic: CPU search speed and depth per time limit.
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{clock, cpu::{Net, Search}};
    pollster::block_on(async {
        let args: Vec<String> = std::env::args().collect();
        let openings: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
        let data: Vec<f32> = std::fs::read(&args[2]).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let net = Net::from_checkpoint(&data).expect("a Rookie checkpoint");
        let clock = clock::millis;
        for millis in [150., 500., 1500.] {
            let (mut nodes, mut depth, mut time) = (0u64, 0u32, 0.);
            for s in openings.iter().take(8) {
                let start = std::time::Instant::now();
                let found = Search::new(&net, s, &clock).think(millis).unwrap();
                time += start.elapsed().as_secs_f64();
                nodes += found.nodes;
                depth += found.depth;
            }
            println!("{millis} ms: {:.0} nodes/s, mean depth {:.1}, mean time {:.0} ms", nodes as f64 / time, depth as f64 / 8., time * 1000. / 8.);
        }
    });
}
