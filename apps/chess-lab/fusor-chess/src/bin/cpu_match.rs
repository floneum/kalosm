//! Native diagnostic: two CPU engines play every opening twice, colors swapped,
//! each with a node budget per move.
//! Usage: cpu_match <openings.json> <games> <a.bin> <nodes|ms> <options> <b.bin> <nodes|ms> <options> [a count scale] [b count scale]
//! A time budget with 10 games in parallel: keep the machine otherwise idle.
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{clock, cpu::{Net, Position, Search}};
    let args: Vec<String> = std::env::args().collect();
    let openings: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let games: usize = args[2].parse().unwrap();
    #[derive(Clone, Copy)]
    enum Budget {
        Nodes(u64),
        Millis(f64),
    }
    let engine = |at: usize| -> (Net, Budget, String) {
        let data: Vec<f32> = std::fs::read(&args[at]).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let net = Net::from_checkpoint(&data).expect("a Rookie checkpoint");
        // The budget is nodes per move, or milliseconds with an "ms" suffix. The
        // third field names the search options: "base" (no pruning), "lmr",
        // "null" or "lmr+null".
        let budget = &args[at + 1];
        let budget = match budget.strip_suffix("ms") {
            Some(ms) => Budget::Millis(ms.parse().unwrap()),
            None => Budget::Nodes(budget.parse().unwrap()),
        };
        (net, budget, args[at + 2].clone())
    };
    // Optional trailing arguments: each engine's piece-count input scale.
    let scaled = |(net, nodes, options): (Net, Budget, String), at: usize| {
        let scale = args.get(at).and_then(|s| s.parse().ok()).unwrap_or_else(rookie_fusor::count_scale);
        (net.with_count_scale(scale), nodes, options)
    };
    let engines = [scaled(engine(3), 9), scaled(engine(6), 10)];
    let workers = 10;
    let start = std::time::Instant::now();
    let results: Vec<i32> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers).map(|w| {
            let (engines, openings) = (engines.clone(), &openings);
            scope.spawn(move || {
                let mut scores = Vec::new();
                for g in (w..games).step_by(workers) {
                    let a_white = g % 2 == 0;
                    let mut state = openings[(g / 2) % openings.len()].clone();
                    let mut pos = Position::from_state(&state);
                    let mut keys: Vec<u64> = (0..=pos.ply as usize).map(|i| (state[225 + i * 2] as u64) << 32 | state[224 + i * 2] as u64).collect();
                    let score = loop {
                        if !pos.has_move() {
                            break if pos.in_check() { if (pos.side == 1) == a_white { -1 } else { 1 } } else { 0 };
                        }
                        let key = pos.key();
                        if pos.halfmove >= 100 || pos.material_draw() || keys.iter().filter(|&&k| k == key).count() >= 3 || pos.ply >= 240 {
                            break 0;
                        }
                        for (i, k) in keys.iter().enumerate().take(241) {
                            state[224 + i * 2] = *k as u32;
                            state[225 + i * 2] = (*k >> 32) as u32;
                        }
                        for (i, b) in pos.board.iter().enumerate() {
                            state[i] = *b as i32 as u32;
                        }
                        state[128] = pos.side as i32 as u32;
                        state[129] = pos.rights;
                        state[130] = pos.ep as u32;
                        state[131] = pos.halfmove;
                        state[132] = pos.ply;
                        state[133] = pos.kings[0] as u32;
                        state[134] = pos.kings[1] as u32;
                        let (net, nodes, options) = &engines[usize::from((pos.side == 1) != a_white)];
                        let clock = clock::millis;
                        // "q<margin>" extends leaves along moves the model's linear
                        // values rate above the margin.
                        let margin = options
                            .split('+')
                            .find_map(|o| o.strip_prefix('q'))
                            .and_then(|m| m.parse().ok())
                            .unwrap_or(f32::MAX);
                        // "d<plies>" caps the extension depth.
                        let plies = options
                            .split('+')
                            .find_map(|o| o.strip_prefix('d'))
                            .and_then(|m| m.parse().ok())
                            .unwrap_or(8);
                        // "noearly" searches until time runs out; "aspire" opens each
                        // depth with a narrow window.
                        let mut search = Search::new(net, &state, &clock)
                            .pruning(options.contains("lmr"), options.contains("null"))
                            .quiescence(margin)
                            .quiescence_depth(plies)
                            .root_options(options.contains("early"), options.contains("aspire"))
                            .gain_ordering(!options.contains("nogain"))
                            .pruning_extras(options.contains("ext"), !options.contains("nofutile"));
                        let m = match *nodes {
                            Budget::Nodes(n) => search.think_nodes(n),
                            Budget::Millis(ms) => search.think(ms),
                        }
                        .unwrap()
                        .movement;
                        let _ = pos.make(m);
                        keys.push(pos.key());
                    };
                    scores.push(score);
                }
                scores
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (w, l) = (results.iter().filter(|&&s| s == 1).count(), results.iter().filter(|&&s| s == -1).count());
    let d = results.len() - w - l;
    let score = (w as f64 + d as f64 / 2.) / results.len() as f64;
    let elo = -400. * (1. / score.clamp(1e-3, 1. - 1e-3) - 1.).log10();
    println!("A vs B: +{w} -{l} ={d} of {}; score {score:.3}; Elo {elo:+.0}; {:.0}s", results.len(), start.elapsed().as_secs_f64());
}
