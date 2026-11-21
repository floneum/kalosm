//! Native diagnostic: build a measurement set. CPU self-play with some random
//! moves gives varied positions; full-strength Stockfish scores each one. Used
//! only to measure trained models (see `learn`), never to train.
//! Usage: evalset <openings.json> <checkpoint.bin> <out.json> [games]
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{Config, Model, clock, cpu::{Position, Search}};
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    let args: Vec<String> = std::env::args().collect();
    let openings: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let games: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(300);
    let net = pollster::block_on(async {
        let mut model = Model::new(Config::default()).await.unwrap();
        let data: Vec<f32> = std::fs::read(&args[2]).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        model.load(&data).unwrap();
        model.cpu_net().await.unwrap().with_count_scale(0.)
    });
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
    let workers = 10;
    let rows: Vec<(Vec<u32>, i32)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers).map(|w| {
            let (net, openings) = (net.clone(), &openings);
            scope.spawn(move || {
                let mut child = Command::new("stockfish").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
                let mut input = child.stdin.take().unwrap();
                let mut output = BufReader::new(child.stdout.take().unwrap());
                writeln!(input, "setoption name Threads value 1\nisready").unwrap();
                let mut line = String::new();
                loop { line.clear(); output.read_line(&mut line).unwrap(); if line.starts_with("readyok") { break; } }
                let mut seed = 0x9e3779b9u32.wrapping_mul(w as u32 + 1);
                let mut random = move || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed };
                let mut rows = Vec::new();
                for g in (w..games).step_by(workers) {
                    let mut state = openings[g % openings.len()].clone();
                    let mut pos = Position::from_state(&state);
                    loop {
                        let legal = pos.moves();
                        if legal.is_empty() || pos.halfmove >= 100 || pos.material_draw() || pos.ply >= 200 { break; }
                        for (i, b) in pos.board.iter().enumerate() { state[i] = *b as i32 as u32; }
                        state[128] = pos.side as i32 as u32; state[129] = pos.rights; state[130] = pos.ep as u32;
                        state[131] = pos.halfmove; state[132] = pos.ply; state[133] = pos.kings[0] as u32; state[134] = pos.kings[1] as u32;
                        if random() % 6 == 0 && !pos.in_check() {
                            writeln!(input, "position fen {}\ngo depth 10", fen(&pos)).unwrap();
                            let mut score = None;
                            loop {
                                line.clear();
                                output.read_line(&mut line).unwrap();
                                if line.starts_with("bestmove") { break; }
                                if let Some(at) = line.find(" score ") {
                                    let mut words = line[at + 7..].split_whitespace();
                                    score = match (words.next(), words.next().and_then(|v| v.parse::<i32>().ok())) {
                                        (Some("cp"), Some(v)) => Some(v.clamp(-2000, 2000)),
                                        (Some("mate"), Some(v)) => Some(if v > 0 { 2000 } else { -2000 }),
                                        _ => score,
                                    };
                                }
                            }
                            if let Some(cp) = score {
                                rows.push((state[..135].to_vec(), cp));
                            }
                        }
                        // One move in five is random, for variety of material and structure.
                        let m = if random() % 5 == 0 {
                            legal[random() as usize % legal.len()]
                        } else {
                            let clock = clock::millis;
                            Search::new(&net, &state, &clock).think_nodes(2000).unwrap().movement
                        };
                        let _ = pos.make(m);
                    }
                }
                writeln!(input, "quit").unwrap();
                let _ = child.wait();
                rows
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let json: Vec<serde_json::Value> = rows.iter().map(|(s, cp)| serde_json::json!({"state": s, "cp": cp})).collect();
    std::fs::write(&args[3], serde_json::to_vec(&json).unwrap()).unwrap();
    let decisive = rows.iter().filter(|(_, cp)| cp.abs() > 300).count();
    println!("{} positions scored; {decisive} with |score| > 3 pawns", rows.len());
}
