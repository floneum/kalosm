//! Transcribe the JFK sample with every `WhisperSource` preset and report
//! load time, transcription time and the text.

use kalosm::sound::*;
use rodio::Decoder;
use std::time::Instant;

fn presets() -> Vec<(&'static str, fn() -> WhisperSource)> {
    vec![
        ("tiny_en", WhisperSource::tiny_en),
        ("tiny", WhisperSource::tiny),
        ("base", WhisperSource::base),
        ("base_en", WhisperSource::base_en),
        ("medium", WhisperSource::medium),
        ("medium_en", WhisperSource::medium_en),
        ("large_v3", WhisperSource::large_v3),
        ("distil_medium_en", WhisperSource::distil_medium_en),
        ("distil_large_v3_5", WhisperSource::distil_large_v3_5),
        ("distil_large_v3", WhisperSource::distil_large_v3),
        ("large_v3_turbo", WhisperSource::large_v3_turbo),
    ]
}

fn main() -> Result<(), anyhow::Error> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let contents = std::fs::read("./models/rwhisper/examples/samples_jfk.wav")?;
    pollster::block_on(async {
        let mut rows = Vec::new();
        for (name, make) in presets() {
            if !args.is_empty() && !args.iter().any(|a| a == name) {
                continue;
            }
            println!("===== {name} =====");
            let t = Instant::now();
            let model = match WhisperBuilder::default().with_source(make()).build().await {
                Ok(m) => m,
                Err(e) => {
                    rows.push((name, 0.0, 0.0, format!("load: {e}")));
                    continue;
                }
            };
            let load = t.elapsed().as_secs_f64();
            let audio = Decoder::new(std::io::Cursor::new(contents.clone()))?;
            let t = Instant::now();
            let mut text = model.transcribe(audio);
            let mut out = String::new();
            while let Some(segment) = text.next().await {
                out.push_str(segment.text());
            }
            let secs = t.elapsed().as_secs_f64();
            let out = out.trim().to_string();
            println!("{out}");
            let lower = out.to_lowercase();
            let note = if lower.contains("ask not what your country can do for you") {
                "ok".to_string()
            } else {
                format!("SUSPECT: {out}")
            };
            rows.push((name, load, secs, note));
        }
        println!("\n{:<20} {:>7} {:>9}  note", "preset", "load s", "transc s");
        for (name, load, secs, note) in rows {
            println!("{name:<20} {load:>7.1} {secs:>9.2}  {note}");
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}
