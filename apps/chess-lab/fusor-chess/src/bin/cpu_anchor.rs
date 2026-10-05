//! Native diagnostic: the CPU search against Stockfish limited by UCI_Elo.
//! Usage: cpu_anchor <openings.json> <checkpoint.bin> <elo> [games] [ms per move]
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{clock, cpu::{Net, Position, Search}};
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    let args: Vec<String> = std::env::args().collect();
    let openings: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let elo: u32 = args[3].parse().unwrap();
    let games: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let millis: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(300.);
    // Milliseconds our side spends searching the opponent's position before each
    // of its own moves, as the app does while a human thinks (ROOKIE_PONDER_MS).
    let ponder_ms: f64 = std::env::var("ROOKIE_PONDER_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.);
    let data: Vec<f32> = std::fs::read(&args[2]).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let net = Net::from_checkpoint(&data).expect("a Rookie checkpoint");
    fn fen(p: &Position) -> String {
        let mut out = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let v = p.board[rank * 16 + file];
                if v == 0 { empty += 1; continue; }
                if empty > 0 { out.push_str(&empty.to_string()); empty = 0; }
                let c = b" pnbrqk"[v.unsigned_abs() as usize] as char;
                out.push(if v > 0 { c.to_ascii_uppercase() } else { c });
            }
            if empty > 0 { out.push_str(&empty.to_string()); }
            if rank > 0 { out.push('/'); }
        }
        let rights: String = ["K", "Q", "k", "q"].iter().enumerate().filter(|(i, _)| p.rights >> i & 1 == 1).map(|(_, c)| *c).collect();
        let ep = if p.ep < 0 { "-".into() } else { format!("{}{}", (b'a' + (p.ep & 7) as u8) as char, (p.ep >> 4) + 1) };
        format!("{out} {} {} {ep} {} {}", if p.side == 1 { "w" } else { "b" }, if rights.is_empty() { "-".into() } else { rights }, p.halfmove, p.ply / 2 + 1)
    }
    fn uci(m: u32) -> String {
        let sq = |s: i32| format!("{}{}", (b'a' + (s & 7) as u8) as char, (s >> 4) + 1);
        let promo = ["", "", "n", "b", "r", "q"][((m >> 14) & 7) as usize];
        format!("{}{}{promo}", sq((m & 127) as i32), sq(((m >> 7) & 127) as i32))
    }
    let workers = 10;
    let results: Vec<i32> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers).map(|w| {
            let (net, openings) = (net.clone(), &openings);
            scope.spawn(move || {
                let mut child = Command::new("stockfish").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
                let mut input = child.stdin.take().unwrap();
                let mut output = BufReader::new(child.stdout.take().unwrap());
                let mut until = |prefix: &str| -> String {
                    let mut line = String::new();
                    loop { line.clear(); output.read_line(&mut line).unwrap(); if line.starts_with(prefix) { return line.trim().to_string(); } }
                };
                for cmd in ["uci", "setoption name Threads value 1", "setoption name UCI_LimitStrength value true", &format!("setoption name UCI_Elo value {elo}"), "isready"] {
                    writeln!(input, "{cmd}").unwrap();
                }
                until("readyok");
                let mut scores = Vec::new();
                for g in (w..games).step_by(workers) {
                    let engine_white = g % 2 == 0;
                    let mut state = openings[(g / 2) % openings.len()].clone();
                    let mut pos = Position::from_state(&state);
                    let mut keys: Vec<u64> = (0..=pos.ply as usize).map(|i| (state[225 + i * 2] as u64) << 32 | state[224 + i * 2] as u64).collect();
                    let mut table = rookie_fusor::cpu::Table::new();
                    let score = loop {
                        let mut legal = pos.moves();
                        if legal.is_empty() { break if pos.in_check() { if (pos.side == 1) == engine_white { -1 } else { 1 } } else { 0 }; }
                        let key = pos.key();
                        if pos.halfmove >= 100 || pos.material_draw() || keys.iter().filter(|&&k| k == key).count() >= 3 || pos.ply >= 240 { break 0; }
                        let m = if (pos.side == 1) == engine_white {
                            // The search reads the game's key history from the state words.
                            for (i, k) in keys.iter().enumerate().take(241) {
                                state[224 + i * 2] = *k as u32;
                                state[225 + i * 2] = (*k >> 32) as u32;
                            }
                            for (i, b) in pos.board.iter().enumerate() { state[i] = *b as i32 as u32; }
                            state[128] = pos.side as i32 as u32; state[129] = pos.rights; state[130] = pos.ep as u32;
                            state[131] = pos.halfmove; state[132] = pos.ply; state[133] = pos.kings[0] as u32; state[134] = pos.kings[1] as u32;
                            let clock = clock::millis;
                            let mut search = Search::with_table(&net, &state, &clock, std::mem::take(&mut table));
                            let found = search.think(millis).unwrap();
                            table = search.into_table();
                            found.movement
                        } else {
                            if ponder_ms > 0. {
                                // Ponder the opponent's position while they "think".
                                for (i, b) in pos.board.iter().enumerate() { state[i] = *b as i32 as u32; }
                                state[128] = pos.side as i32 as u32; state[129] = pos.rights; state[130] = pos.ep as u32;
                                state[131] = pos.halfmove; state[132] = pos.ply; state[133] = pos.kings[0] as u32; state[134] = pos.kings[1] as u32;
                                for (i, k) in keys.iter().enumerate().take(241) {
                                    state[224 + i * 2] = *k as u32;
                                    state[225 + i * 2] = (*k >> 32) as u32;
                                }
                                let clock = clock::millis;
                                let mut search = Search::with_table(&net, &state, &clock, std::mem::take(&mut table));
                                let _ = search.think(ponder_ms);
                                table = search.into_table();
                            }
                            writeln!(input, "position fen {}", fen(&pos)).unwrap();
                            writeln!(input, "go movetime {}", millis as u32).unwrap();
                            let best = until("bestmove");
                            let text = best.split_whitespace().nth(1).unwrap().to_string();
                            legal.sort();
                            *legal.iter().find(|&&m| uci(m) == text).expect("Stockfish move is legal")
                        };
                        let _ = pos.make(m);
                        keys.push(pos.key());
                    };
                    scores.push(score);
                }
                writeln!(input, "quit").unwrap();
                let _ = child.wait();
                scores
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let (w, l) = (results.iter().filter(|&&s| s == 1).count(), results.iter().filter(|&&s| s == -1).count());
    let d = results.len() - w - l;
    let score = (w as f64 + d as f64 / 2.) / results.len() as f64;
    let diff = -400. * (1. / score.clamp(1e-3, 1. - 1e-3) - 1.).log10();
    println!("CPU search vs Stockfish UCI_Elo {elo} ({millis} ms each): +{w} -{l} ={d} of {}; score {score:.3}; about {:.0} Elo", results.len(), elo as f64 + diff);
}
