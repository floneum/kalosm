//! Native diagnostic: the CPU rules and evaluator against the GPU. Legal moves,
//! positions after each move and position keys against the rules fixtures; the
//! CPU value against the compiled model; keys against the JS encoder's.
//! Usage: cpu_check fixtures.json [checkpoint.bin] [openings.json]
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{Config, INPUT, Model, cpu::Position};
    pollster::block_on(async {
        let args: Vec<String> = std::env::args().collect();
        let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
        let rows = rows.as_array().unwrap();
        let state = |row: &serde_json::Value, key: &str| -> Vec<u32> {
            let mut s = vec![0u32; 768];
            for (i, v) in row[key].as_array().unwrap().iter().enumerate() {
                s[i] = v.as_i64().unwrap() as u32;
            }
            s
        };
        let (mut move_errors, mut after_errors) = (0, 0);
        for row in rows {
            let s = state(row, "state");
            let mut pos = Position::from_state(&s);
            let mut ours = pos.moves();
            let mut expected: Vec<u32> = row["moves"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            ours.sort();
            expected.sort();
            if ours != expected {
                move_errors += 1;
                continue;
            }
            for m in ours {
                let before = pos.clone();
                let undo = pos.make(m);
                let after = &row["after"][m.to_string()];
                let words: Vec<i64> = after.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
                let got: Vec<i64> = pos.board.iter().map(|&b| b as i64)
                    .chain([pos.side as i64, pos.rights as i64, pos.ep as i64, pos.halfmove as i64, pos.ply as i64, pos.kings[0] as i64, pos.kings[1] as i64])
                    .collect();
                if got != words {
                    after_errors += 1;
                }
                pos.unmake(m, &undo);
                if pos.board != before.board || pos.rights != before.rights || pos.ep != before.ep {
                    after_errors += 1;
                }
            }
        }
        println!("rules: {} positions, {move_errors} legal-move mismatches, {after_errors} make/unmake mismatches", rows.len());

        // Values: CPU network against the compiled model on the same positions.
        let mut model = Model::new(Config::default()).await.unwrap();
        if let Some(path) = args.get(2) {
            let data: Vec<f32> = std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            model.load(&data).unwrap();
        }
        let net = model.cpu_net().await.unwrap();
        let batch = model.batch;
        let mut x = vec![0f32; batch * INPUT];
        let positions: Vec<Position> = rows.iter().take(batch).map(|r| Position::from_state(&state(r, "state"))).collect();
        for (b, pos) in positions.iter().enumerate() {
            for (i, v) in rookie_fusor::cpu::features(pos).into_iter().enumerate() {
                x[b * INPUT + i] = v;
            }
        }
        let gpu = model.predict(&x).await.unwrap();
        let mut worst = 0f32;
        for (b, pos) in positions.iter().enumerate() {
            let cpu = net.evaluate_position(pos);
            worst = worst.max((cpu - gpu[b]).abs());
        }
        let spread = gpu.iter().take(positions.len()).fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        println!("values: {} positions in [{:.3}, {:.3}], largest CPU/GPU difference {worst:.2e}", positions.len(), spread.0, spread.1);
        let mut incremental = 0f32;
        for row in rows {
            let mut pos = Position::from_state(&state(row, "state"));
            incremental = incremental.max(net.incremental_error(&mut pos));
        }
        println!("incremental: largest difference {incremental:.2e}");
        // Keys: openings carry the JS encoder's key for the current position.
        if let Some(path) = args.get(3) {
            let openings: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let mut key_errors = 0;
            for s in &openings {
                let mut pos = Position::from_state(s);
                let at = 224 + pos.ply as usize * 2;
                if pos.key() != ((s[at + 1] as u64) << 32 | s[at] as u64) {
                    key_errors += 1;
                }
            }
            println!("position keys: {} openings, {key_errors} mismatches with the JS encoder", openings.len());
        }
    });
}
