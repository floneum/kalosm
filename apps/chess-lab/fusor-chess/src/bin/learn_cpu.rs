//! Native diagnostic: self-play on CPU threads (the CPU alpha-beta with
//! incremental evaluation), training on the GPU. Workers play games with the
//! latest weights and send (features, target) rows; the trainer batches them.
//! Environment: ROOKIE_NODES (search nodes per move), ROOKIE_SECONDS,
//! ROOKIE_STEPS (batches per 1024 new rows), ROOKIE_WINDOW (rows sampled from),
//! ROOKIE_LAMBDA (λ-return targets; 0 uses ROOKIE_RESULT blending),
//! ROOKIE_RESULT (result weight), ROOKIE_RANDOM (random opening plies),
//! ROOKIE_EVALSET, ROOKIE_SAVE, ROOKIE_LR, ROOKIE_REPORT_EVERY (seconds), ROOKIE_WIDTH,
//! ROOKIE_DEPTH, ROOKIE_HIDDEN.
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rookie_fusor::{Config, INPUT, Model, clock, cpu::{self, Net}};
    use std::sync::{Arc, RwLock, atomic::{AtomicBool, AtomicU64, Ordering}, mpsc};
    let var = |name: &str, default: f64| std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(default);
    let nodes = var("ROOKIE_NODES", 400.) as u64;
    let seconds = var("ROOKIE_SECONDS", 60.);
    let steps = var("ROOKIE_STEPS", 4.) as usize;
    let window = var("ROOKIE_WINDOW", 65536.) as usize;
    let result_weight = var("ROOKIE_RESULT", 0.05) as f32;
    let lambda = var("ROOKIE_LAMBDA", 0.) as f32;
    let random_plies = var("ROOKIE_RANDOM", 6.) as u32;
    pollster::block_on(async {
        let width = var("ROOKIE_WIDTH", 128.) as usize;
        let config = Config {
            width,
            depth: var("ROOKIE_DEPTH", 2.) as usize,
            hidden: var("ROOKIE_HIDDEN", 32.) as usize,
        };
        let mut model = Model::new(config).await.unwrap();
        model.learning_rate = var("ROOKIE_LR", model.learning_rate as f64) as f32;
        let batch = model.batch;
        // The latest weights, versioned; every worker owns a network it refreshes from them.
        let shared: Arc<RwLock<(u64, Vec<Vec<f32>>)>> = Arc::new(RwLock::new((1, model.value_parameters().await.unwrap())));
        let stop = Arc::new(AtomicBool::new(false));
        let plies = Arc::new(AtomicU64::new(0));
        let games = Arc::new(AtomicU64::new(0));
        let (tx, rx) = mpsc::channel::<(Vec<f32>, f32)>();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).saturating_sub(2).max(1);
        let mut handles = Vec::new();
        for w in 0..workers {
            let (shared, stop, plies, games, tx) = (shared.clone(), stop.clone(), plies.clone(), games.clone(), tx.clone());
            handles.push(std::thread::spawn(move || {
                let mut seed = 0x9e3779b9u32.wrapping_mul(w as u32 + 1) ^ 0x5bd1e995;
                let mut random = move || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed };
                let clock = clock::millis;
                let mut table = cpu::Table::new();
                let mut net = Net::new(config);
                let mut have = 0;
                while !stop.load(Ordering::Relaxed) {
                    {
                        let latest = shared.read().unwrap();
                        if latest.0 != have {
                            net.set_parameters(&latest.1);
                            have = latest.0;
                        }
                    }
                    let game = cpu::self_play_game(&net, nodes, result_weight, lambda, random_plies, &mut random, &clock, &mut table);
                    let rows = game.rows;
                    let result = game.result;
                    plies.fetch_add(u64::from(game.plies), Ordering::Relaxed);
                    games.fetch_add(1, Ordering::Relaxed);
                    let _ = result;
                    for row in rows {
                        if tx.send(row).is_err() {
                            return;
                        }
                    }
                }
            }));
        }
        drop(tx);
        let start = std::time::Instant::now();
        let mut replay: std::collections::VecDeque<(Vec<f32>, f32)> = std::collections::VecDeque::new();
        let mut fresh = 0usize;
        let mut seed = 12345u32;
        let mut random = move || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed };
        let mut x = vec![0f32; batch * INPUT];
        let mut values = vec![0f32; batch];
        let mut last_publish = std::time::Instant::now();
        let mut published = 0u32;
        let report_every = var("ROOKIE_REPORT_EVERY", 0.);
        let mut next_report = report_every;
        let mut reporter = Net::new(config);
        let evalset: Option<Vec<(Vec<u32>, i32)>> = std::env::var("ROOKIE_EVALSET").ok().map(|path| {
            let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            rows.as_array().unwrap().iter().map(|r| {
                (r["state"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(), r["cp"].as_i64().unwrap() as i32)
            }).collect()
        });
        while start.elapsed().as_secs_f64() < seconds {
            if report_every > 0. && start.elapsed().as_secs_f64() >= next_report {
                next_report += report_every;
                if let Some(rows) = &evalset {
                    reporter.set_parameters(&shared.read().unwrap().1);
                    let (all, balanced) = cpu::agreement(&reporter, rows);
                    println!("  {:.0}s: {} games, {} plies, {} updates; r = {all:.3} overall, {balanced:.3} balanced",
                        start.elapsed().as_secs_f64(), games.load(Ordering::Relaxed), plies.load(Ordering::Relaxed), model.step);
                }
                if let Ok(path) = std::env::var("ROOKIE_SAVE") {
                    let state = model.save().await.unwrap();
                    std::fs::write(format!("{path}.{:.0}s", start.elapsed().as_secs_f64()), bytemuck::cast_slice::<f32, u8>(&state)).unwrap();
                }
            }
            while let Ok(row) = rx.try_recv() {
                replay.push_back(row);
                fresh += 1;
                if replay.len() > window {
                    replay.pop_front();
                }
            }
            if fresh >= batch && replay.len() >= batch {
                fresh -= batch;
                for _ in 0..steps {
                    for b in 0..batch {
                        let (features, target) = &replay[random() as usize % replay.len()];
                        x[b * INPUT..(b + 1) * INPUT].copy_from_slice(features);
                        values[b] = *target;
                    }
                    model.train(&x, &values, 1).await.unwrap();
                }
                if last_publish.elapsed().as_secs_f64() > 0.25 {
                    let parameters = model.value_parameters().await.unwrap();
                    let mut latest = shared.write().unwrap();
                    *latest = (latest.0 + 1, parameters);
                    last_publish = std::time::Instant::now();
                    published += 1;
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        stop.store(true, Ordering::Relaxed);
        drop(rx);
        for h in handles {
            let _ = h.join();
        }
        let net = model.cpu_net().await.unwrap();
        let mut line = format!(
            "cpu self-play: {workers} threads, {nodes} nodes/move: {} games, {} plies, {} updates, {published} weight publishes in {:.0}s",
            games.load(Ordering::Relaxed), plies.load(Ordering::Relaxed), model.step, start.elapsed().as_secs_f64()
        );
        if let Some(rows) = &evalset {
            let (all, balanced) = cpu::agreement(&net, rows);
            line += &format!("; r = {all:.3} overall, {balanced:.3} balanced");
        }
        if let Ok(path) = std::env::var("ROOKIE_SAVE") {
            let state = model.save().await.unwrap();
            std::fs::write(path, bytemuck::cast_slice::<f32, u8>(&state)).unwrap();
        }
        println!("{line}");
    });
}
