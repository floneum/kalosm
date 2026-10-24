use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute, queue,
    style::Print,
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use fusor_visualize::{
    GpuConfig, Simulator, Step, Trace, demo,
    kernel::{self, CaptureOptions},
    render,
};

const HELP: &str = "fusor-visualize — step through a virtual GPU's memory actions

Usage: cargo run -p fusor-visualize -- [OPTIONS]

  --demo NAME              coalesced, strided, broadcast, reuse, bank-conflict, padded
  --trace FILE             replay a version-1 JSON lane-address trace instead
  --format text|tui|json   output mode (default: text); json emits one step per line
  --subgroup-size N        virtual subgroup width (default: 32)
  --workgroup-size N       demo workgroup width (default: 64)
  --workgroups N           capture first N demo workgroups (1 or 2; default: 1)
  --transaction-bytes N    aligned storage transaction size (default: 32)
  --cache-line-bytes N     virtual cache line size (default: 128)
  --cache-bytes N          LRU capacity, 0 disables cache (default: 16384)
  --banks N               shared-memory banks (default: 32)
  --bank-width-bytes N     shared-memory bank word size (default: 4)
  --max-actions N          IR capture budget (default: 100000)
  --max-steps N            stop output after N steps (default: all)
  --export-trace FILE      write the captured/loaded trace for replay
  --help                  show this help

TUI: n/right/space next, p/left previous, r/home reset, e/end run to end,
     j/k or up/down scroll, PgUp/PgDn page, q/escape quit.
All figures are model estimates. No GPU is required.";

struct Options {
    config: GpuConfig,
    demo: String,
    trace: Option<PathBuf>,
    format: String,
    block: u32,
    workgroups: u32,
    max_actions: usize,
    max_steps: Option<usize>,
    export: Option<PathBuf>,
}

fn parse() -> Result<Option<Options>, String> {
    let mut opts = Options {
        config: GpuConfig::default(),
        demo: "strided".into(),
        trace: None,
        format: "text".into(),
        block: 64,
        workgroups: 1,
        max_actions: 100_000,
        max_steps: None,
        export: None,
    };
    let mut args = std::env::args().skip(1);
    let mut demo_set = false;
    let mut geometry_set = false;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        let number = || {
            value
                .parse::<u32>()
                .map_err(|_| format!("{arg} requires a non-negative 32-bit integer"))
        };
        match arg.as_str() {
            "--demo" => {
                opts.demo = value;
                demo_set = true;
            }
            "--trace" => opts.trace = Some(value.into()),
            "--format" => opts.format = value,
            "--subgroup-size" => opts.config.subgroup_size = number()?,
            "--workgroup-size" => {
                opts.block = number()?;
                geometry_set = true;
            }
            "--workgroups" => {
                opts.workgroups = number()?;
                geometry_set = true;
            }
            "--transaction-bytes" => opts.config.transaction_bytes = number()?,
            "--cache-line-bytes" => opts.config.cache_line_bytes = number()?,
            "--cache-bytes" => opts.config.cache_bytes = number()?,
            "--banks" => opts.config.banks = number()?,
            "--bank-width-bytes" => opts.config.bank_width_bytes = number()?,
            "--max-actions" => opts.max_actions = number()? as usize,
            "--max-steps" => opts.max_steps = Some(number()? as usize),
            "--export-trace" => opts.export = Some(value.into()),
            _ => return Err(format!("unknown option {arg}; see --help")),
        }
    }
    if !["text", "tui", "json"].contains(&opts.format.as_str()) {
        return Err("format must be text, tui or json".into());
    }
    if opts.trace.is_some() && (demo_set || geometry_set) {
        return Err(
            "--trace cannot be combined with --demo, --workgroup-size or --workgroups".into(),
        );
    }
    if opts.format == "tui" && opts.max_steps.is_some() {
        return Err("--max-steps applies to text/json output; the TUI steps interactively".into());
    }
    if !(1..=2).contains(&opts.workgroups) {
        return Err("demo workgroups must be 1 or 2".into());
    }
    opts.config.validate().map_err(|e| e.to_string())?;
    Ok(Some(opts))
}

fn main() {
    if let Err(error) = run() {
        if error
            .downcast_ref::<io::Error>()
            .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
        {
            return;
        }
        eprintln!("fusor-visualize: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(opts) = parse().map_err(io::Error::other)? else {
        return Ok(());
    };
    let trace: Trace = if let Some(path) = opts.trace {
        if std::fs::metadata(&path)?.len() > 32 * 1024 * 1024 {
            return Err("trace exceeds the CLI's 32 MiB input limit".into());
        }
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?
    } else {
        kernel::capture(
            &demo::kernel(&opts.demo, opts.block)?,
            CaptureOptions {
                subgroup_size: opts.config.subgroup_size,
                workgroups: (0..opts.workgroups).map(|x| [x, 0, 0]).collect(),
                max_actions: opts.max_actions,
                ..CaptureOptions::default()
            },
        )?
    };
    let mut simulator = Simulator::new(trace, opts.config)?;
    if let Some(path) = opts.export {
        // Create-new prevents an accidental overwrite of a kernel or existing trace.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(&mut file, simulator.trace())?;
        writeln!(file)?;
    }
    if opts.format == "tui" {
        return tui(&mut simulator).map_err(Into::into);
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    if opts.format == "text" {
        writeln!(
            out,
            "{} | virtual GPU | subgroup {} | workgroup {}",
            render::clean(&simulator.trace().name),
            simulator.config().subgroup_size,
            simulator.trace().workgroup_size
        )?;
        writeln!(
            out,
            "sector {} B, cache {} B / line {} B, banks {} x {} B",
            simulator.config().transaction_bytes,
            simulator.config().cache_bytes,
            simulator.config().cache_line_bytes,
            simulator.config().banks,
            simulator.config().bank_width_bytes
        )?;
        for note in &simulator.trace().notes {
            writeln!(out, "Model: {}", render::clean(note))?;
        }
    } else {
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"type": "config", "config": simulator.config(), "name": simulator.trace().name, "notes": simulator.trace().notes}),
        )?;
        writeln!(out)?;
    }
    for _ in 0..opts.max_steps.unwrap_or(usize::MAX) {
        let Some(step) = simulator.step() else {
            break;
        };
        if opts.format == "json" {
            serde_json::to_writer(&mut out, &serde_json::json!({"type": "step", "step": step}))?;
            writeln!(out)?;
        } else {
            write!(
                out,
                "{}",
                render::step(&step, simulator.trace(), simulator.config())
            )?;
        }
    }
    let exhausted = simulator.is_finished();
    if opts.format == "json" {
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"type": "summary", "complete": exhausted, "totals": simulator.totals()}),
        )?;
        writeln!(out)?;
    } else {
        writeln!(
            out,
            "\n{}\n{}",
            if exhausted {
                "COMPLETE"
            } else {
                "STOPPED AT STEP LIMIT (totals are a prefix)"
            },
            render::summary(simulator.totals())
        )?;
    }
    out.flush()?;
    Ok(())
}

struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
    }
}

fn tui(sim: &mut Simulator) -> io::Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(io::Error::other(
            "TUI requires a terminal; use --format text or --format json",
        ));
    }
    terminal::enable_raw_mode()?;
    let _guard = TerminalGuard;
    let mut out = io::stdout();
    execute!(out, EnterAlternateScreen, cursor::Hide)?;
    let mut current: Option<Step> = None;
    let mut scroll = 0usize;
    let mut done = false;
    loop {
        let (width, height) = terminal::size()?;
        let content_rows = height.saturating_sub(7) as usize;
        let content = current.as_ref().map(|s| render::step(s, sim.trace(), sim.config()))
            .unwrap_or_else(|| "Ready. Press n to execute the first action.\n\nLane addresses -> storage sectors/cache lines or shared-memory banks.\nModel: serial workgroups, contiguous subgroups, fully associative LRU cache.\nMetrics describe this virtual GPU; they do not predict device timing.".into());
        let lines: Vec<_> = content.lines().collect();
        scroll = scroll.min(lines.len().saturating_sub(content_rows.max(1)));
        queue!(out, cursor::MoveTo(0, 0), Clear(ClearType::All))?;
        let title = format!(
            "FUSOR / {}   SG {}   WG {}   step {}{}",
            render::clean(&sim.trace().name),
            sim.config().subgroup_size,
            sim.trace().workgroup_size,
            sim.cursor(),
            if done { " [complete]" } else { "" }
        );
        draw(&mut out, 0, width, &title)?;
        draw(
            &mut out,
            1,
            width,
            "n next | p previous | r reset | e end | j/k scroll | PgUp/PgDn | q quit",
        )?;
        for (i, line) in lines.iter().skip(scroll).take(content_rows).enumerate() {
            draw(&mut out, i as u16 + 3, width, line)?;
        }
        for (i, line) in render::summary(sim.totals()).lines().enumerate() {
            if height >= 3 {
                draw(&mut out, height - 3 + i as u16, width, line)?;
            }
        }
        out.flush()?;
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => break,
                KeyCode::Char('n' | ' ') | KeyCode::Right => {
                    if let Some(step) = sim.step() {
                        current = Some(step);
                        scroll = 0;
                    } else {
                        done = true;
                    }
                }
                KeyCode::Char('p') | KeyCode::Left => {
                    current = sim.seek(sim.cursor().saturating_sub(1));
                    scroll = 0;
                    done = false;
                }
                KeyCode::Char('r') | KeyCode::Home => {
                    sim.reset();
                    current = None;
                    scroll = 0;
                    done = false;
                }
                KeyCode::Char('e') | KeyCode::End => {
                    while let Some(step) = sim.step() {
                        current = Some(step);
                    }
                    done = true;
                    scroll = 0;
                }
                KeyCode::Char('j') | KeyCode::Down => scroll = scroll.saturating_add(1),
                KeyCode::Char('k') | KeyCode::Up => scroll = scroll.saturating_sub(1),
                KeyCode::PageDown => scroll = scroll.saturating_add(content_rows.max(1)),
                KeyCode::PageUp => scroll = scroll.saturating_sub(content_rows.max(1)),
                _ => {}
            },
            _ => {}
        }
    }
    Ok(())
}

fn draw(out: &mut impl Write, row: u16, width: u16, text: &str) -> io::Result<()> {
    // A single-width ASCII viewport avoids control sequences and wide glyph spill.
    let text: String = text
        .chars()
        .map(|c| {
            if c.is_ascii() && !c.is_control() {
                c
            } else {
                '?'
            }
        })
        .take(width.saturating_sub(1) as usize)
        .collect();
    queue!(out, cursor::MoveTo(0, row), Print(text))
}
