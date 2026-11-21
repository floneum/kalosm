import { memo, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { useTraining, parseCheckpoint, encodeCheckpoint } from './useTraining';
import { DEPTHS, HIDDENS, WIDTHS, modelKey, modelLabel, parameterCount } from './engine/architectures';
import type { ArenaSnapshot } from './engine/protocol';
import { Icon } from './components/Icon';
import { Board, EvaluationBar } from './components/Board';
import { LossChart } from './components/Chart';
import { Play } from './components/Play';

type Tab = 'training' | 'play' | 'about';
const number = (value: number) => Math.round(value).toLocaleString();
const compact = (value: number) =>
  value >= 1e6
    ? `${(value / 1e6).toFixed(2)}m`
    : value >= 1e4
      ? `${(value / 1e3).toFixed(1)}k`
      : number(value);
const duration = (seconds: number) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`;
const scoreLabel = (score: number) => {
  const pawns = Math.atanh(Math.max(-0.99, Math.min(0.99, score))) * 8;
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(1)}`;
};

function Modal({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`modal ${wide ? 'wide-modal' : ''}`}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-label={title}
    >
      <div className="modal-inner">
        <div className="modal-heading">
          <h2>{title}</h2>
          <button className="icon-button" aria-label="Close dialog" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        {children}
      </div>
    </dialog>
  );
}

const ArenaCard = memo(function ArenaCard({
  arena,
  onInspect,
  running,
}: {
  arena: ArenaSnapshot;
  onInspect: (id: number) => void;
  running: boolean;
}) {
  return (
    <button
      className="arena-card"
      onClick={() => onInspect(arena.id)}
      aria-label={`Inspect self-play board ${arena.id + 1}, game ${arena.episode}, move ${Math.ceil(arena.ply / 2)}`}
    >
      <div className="arena-top">
        <span>
          <span className={`arena-dot ${running ? '' : 'paused'}`} />
          BOARD {String(arena.id + 1).padStart(2, '0')}
        </span>
        <span className="arena-ply">
          {arena.ply ? `Move ${Math.ceil(arena.ply / 2)}` : 'Ready'}{' '}
          <Icon name="expand" size={12} />
        </span>
      </div>
      <Board fen={arena.fen} lastMove={arena.lastMove} label={`Self-play board ${arena.id + 1}`} />
      <div className="arena-bottom">
        <span>
          <span
            className={
              arena.fen.split(' ')[1] === 'w' ? 'piece-indicator white' : 'piece-indicator'
            }
          />
          {arena.lastSan}
        </span>
        <span className="arena-score">{scoreLabel(arena.score)}</span>
      </div>
      <EvaluationBar score={arena.score} />
    </button>
  );
});

function App() {
  const training = useTraining();
  const [draftModel, setDraftModel] = useState(training.model);
  useEffect(() => setDraftModel(training.model), [training.model]);
  const PARAM_COUNT = parameterCount(training.model),
    MODEL_BYTES = PARAM_COUNT * 4;
  const draftParameters = parameterCount(draftModel);
  const {
    settings,
    setSettings,
    running,
    setRunning,
    metrics,
    arenas,
    rates,
    generation,
    seconds,
    history,
  } = training;
  const [tab, setTab] = useState<Tab>('training');
  const [inspected, setInspected] = useState<number | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const uploadRef = useRef<HTMLInputElement>(null);
  const currentArena = arenas.find((arena) => arena.id === inspected);
  useEffect(() => {
    if (notice) {
      const timer = setTimeout(() => setNotice(''), 5500);
      return () => clearTimeout(timer);
    }
  }, [notice]);

  async function download() {
    let data;
    try {
      data = await training.checkpoint();
    } catch (error) {
      setNotice(String(error));
      return;
    }
    const blob = new Blob([encodeCheckpoint(data)], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob),
      link = document.createElement('a');
    link.href = url;
    link.download = `rookie-generation-${data.generation}.rookie`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice(`Generation ${data.generation} saved.`);
  }
  async function importFile(file?: File) {
    if (!file) return;
    try {
      if (file.size > 64000000) throw new Error('That file is too large for a Rookie checkpoint.');
      training.load(parseCheckpoint(await file.arrayBuffer()));
      setNotice('Model restored. Let the learning continue.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not read that checkpoint.');
    }
    if (uploadRef.current) uploadRef.current.value = '';
  }

  return (
    <>
      <header className="site-header">
        <div className="header-inner">
          <a
            className="brand"
            href="#"
            aria-label="Rookie home"
            onClick={(e) => {
              e.preventDefault();
              setTab('training');
            }}
          >
            <span className="brand-mark">
              <Icon name="rook" size={24} />
            </span>
            rookie<span className="brand-period">.</span>
            <span className="lab-label">CHESS LAB</span>
          </a>
          <nav aria-label="Main navigation">
            {(
              [
                { key: 'training', label: 'Training floor', icon: 'grid' },
                { key: 'play', label: 'Play Rookie', icon: 'play' },
                { key: 'about', label: 'How it learns', icon: 'tree' },
              ] as const
            ).map((item) => (
              <button
                key={item.key}
                className={tab === item.key ? 'nav-link active' : 'nav-link'}
                aria-current={tab === item.key ? 'page' : undefined}
                onClick={() => setTab(item.key)}
              >
                <Icon name={item.icon} size={15} />
                {item.label}
                {item.key === 'training' && running && <span className="nav-dot" />}
              </button>
            ))}
          </nav>
          <div className="local-status">
            <span className="status-dot" />
            All in your browser <Icon name="cpu" size={16} />
          </div>
        </div>
      </header>

      <main>
        <section className="intro">
          <div>
            <div className="eyebrow">
              <span className="little-star">✳</span> A CHESS ENGINE THAT TRAINS IN YOUR BROWSER
            </div>
            <h1>
              {tab === 'play' ? (
                <>
                  Play
                  <br />
                  <em>Rookie.</em>
                </>
              ) : tab === 'about' ? (
                <>
                  How
                  <br />
                  <em>it works.</em>
                </>
              ) : (
                <>
                  Rookie learns chess
                  <br />
                  <em>by playing itself.</em>
                </>
              )}
            </h1>
          </div>
          <div className="intro-aside">
            <span className="intro-number">
              {tab === 'training'
                ? '01 / OBSERVE'
                : tab === 'play'
                  ? '02 / CHALLENGE'
                  : '03 / UNDERSTAND'}
            </span>
            <p>
              {tab === 'training'
                ? 'Rookie starts knowing only the rules. It plays games against itself, and a small neural network learns to score positions from what the search finds. About a minute of training gets it to club strength.'
                : tab === 'play'
                  ? 'Play the model as it is right now. It keeps training while you are away from this tab, so it gets stronger between games.'
                  : 'Everything runs on this device: self-play and search on the CPU, training on the GPU. Nothing is uploaded.'}
            </p>
            <div className="intro-actions">
              {tab === 'training' ? (
                <>
                  <button className="button primary" onClick={() => setRunning(!running)}>
                    <Icon name={running ? 'pause' : 'play'} size={15} />
                    {running ? 'Pause training' : 'Resume training'}
                  </button>
                  <button className="button secondary" onClick={download}>
                    <Icon name="download" size={15} />
                    Save model
                  </button>
                </>
              ) : (
                <button className="button secondary" onClick={() => setTab('training')}>
                  Back to training <Icon name="arrow" size={16} />
                </button>
              )}
            </div>
          </div>
        </section>

        {training.error && (
          <div className="error-banner" role="alert">
            {training.error}
          </div>
        )}

        <div hidden={tab !== 'training'}>
          <section className="metrics-strip" aria-label="Live training metrics">
            <div className="metric">
              <span className="metric-label">
                Search positions <Icon name="tree" size={14} />
              </span>
              <div>
                {compact(metrics.nodes)}
                <span className="metric-detail">explored</span>
              </div>
              <small>positions searched in self-play</small>
            </div>
            <div className="metric">
              <span className="metric-label">
                Completed games <Icon name="rook" size={14} />
              </span>
              <div>
                {compact(metrics.games)}
                <span className="metric-detail">self-play</span>
              </div>
              <small>{number(metrics.truncated)} extra episodes reached the move limit</small>
            </div>
            <div className="metric">
              <span className="metric-label">
                Learning updates <Icon name="cpu" size={14} />
              </span>
              <div>
                {compact(metrics.batches)}
                <span className="metric-detail">batches</span>
              </div>
              <small>{compact(rates.batches)} updates / sec · {training.games} positions / batch</small>
            </div>
            <div className="metric">
              <span className="metric-label">
                Training speed <Icon name="bolt" size={14} />
              </span>
              <div>
                {compact(rates.batches * training.games)}
                <span className="metric-detail">samples / sec</span>
              </div>
              <small>
                <span className={running ? 'tiny-dot' : 'tiny-dot stopped'} />
                {running
                  ? `${compact(rates.positions)} moves / sec · ${compact(rates.nodes)} leaves / sec`
                  : 'Paused'}
              </small>
            </div>
          </section>

          <section className="training-layout">
            <div className="floor">
              <div className="section-heading">
                <div>
                  <h2>
                    Self-play{' '}
                    <span className={`live-label ${running ? '' : 'is-paused'}`}>
                      <span />
                      {running ? 'LIVE' : 'PAUSED'}
                    </span>
                  </h2>
                  <p>{training.games} worker threads playing games with the current weights.</p>
                </div>
                <div className="floor-meta">
                  <Icon name="grid" size={14} />
                  <span>{Math.min(24, training.games)} live views</span>
                  <span className="meta-divider" />
                  <span>Click to inspect</span>
                </div>
              </div>
              <div className={`arena-grid ${training.games > 12 ? 'dense-grid' : ''}`}>
                {arenas.map((arena) => (
                  <ArenaCard
                    key={arena.id}
                    arena={arena}
                    running={running}
                    onInspect={setInspected}
                  />
                ))}
                {!arenas.length && (
                  <div className="floor-loading">
                    <Icon name="rook" size={30} />
                    Setting out the pieces…
                  </div>
                )}
              </div>
              <div className="floor-caption">
                <span>
                  <span className="tiny-dot" />
                  These are the games being played right now.
                </span>
                <span>Latest finished games from the self-play workers</span>
              </div>
              <div className="challenge-card">
                <div className="challenge-illustration">
                  <Icon name="rook" size={48} />
                  <span>?</span>
                </div>
                <div>
                  <span className="eyebrow">PLAY IT</span>
                  <h3>Play the current model</h3>
                  <p>It uses the weights as they are now.</p>
                </div>
                <button className="button dark-button" onClick={() => setTab('play')}>
                  Play Rookie <Icon name="arrow" size={17} />
                </button>
              </div>
            </div>

            <aside className="training-sidebar">
              <div className="panel engine-panel">
                <div className="panel-heading">
                  <h3>Model</h3>
                  <Icon name="settings" size={17} />
                </div>
                <div className="model-identity">
                  <span className="model-icon">
                    <Icon name="cpu" size={25} />
                  </span>
                  <div>
                    <strong>
                      Rookie <span>Gen. {generation}</span>
                    </strong>
                    <small>Fusor · WebGPU training · CPU search</small>
                  </div>
                  <span className={`status-dot ${running ? '' : 'stopped'}`} />
                </div>
                <p className="small-note" role="status">
                  {training.backend}
                </p>
                <div className="model-facts">
                  <div>
                    <strong>
                      {(MODEL_BYTES / 1024 / 1024).toFixed(2)} <span>MiB</span>
                    </strong>
                    <small>model weights</small>
                  </div>
                  <div>
                    <strong>{number(PARAM_COUNT)}</strong>
                    <small>learned parameters</small>
                  </div>
                </div>
                <div className="model-picker">
                  <div className="model-dimensions">
                    {(
                      [
                        ['width', 'First layer', WIDTHS, 'neurons'],
                        ['depth', 'Depth', DEPTHS, 'layers'],
                        ['hidden', 'Hidden width', HIDDENS, 'neurons'],
                      ] as const
                    ).map(([key, label, options, unit]) => (
                      <div className="field" key={key}>
                        <label htmlFor={`model-${key}`}>{label}</label>
                        <select
                          id={`model-${key}`}
                          value={draftModel[key]}
                          onChange={(e) =>
                            setDraftModel((m) => ({ ...m, [key]: Number(e.target.value) }))
                          }
                        >
                          {options.map((n) => (
                            <option key={n} value={n}>
                              {n} {unit}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                  <div className="model-preview">
                    <strong>{compact(draftParameters)} parameters</strong>
                    <span>{((draftParameters * 4) / 1048576).toFixed(2)} MiB</span>
                  </div>
                  <button
                    className="button secondary full-width"
                    disabled={
                      training.switching ||
                      (!training.ready && !training.error) ||
                      modelKey(draftModel) === modelKey(training.model)
                    }
                    onClick={() => void training.switchModel(draftModel)}
                  >
                    {training.switching
                      ? 'Saving the current model…'
                      : modelKey(draftModel) === modelKey(training.model)
                        ? 'Current model'
                        : 'Switch to this size'}
                  </button>
                  <p className="small-note">
                    Each size keeps its own checkpoint, so you can train several and compare them.
                    A wider first layer costs little in search speed; more depth or hidden width
                    costs more.
                  </p>
                </div>
                <div className="field">
                  <label htmlFor="search-budget">
                    Self-play search <span>alpha-beta</span>
                  </label>
                  <select
                    id="search-budget"
                    value={settings.nodes}
                    onChange={(e) => setSettings((s) => ({ ...s, nodes: Number(e.target.value) }))}
                  >
                    {[100, 200, 400, 800, 1600].map((n) => (
                      <option value={n} key={n}>
                        {n} positions / move
                      </option>
                    ))}
                  </select>
                </div>
                <div className="session-status">
                  <span>
                    <span className={`tiny-dot ${running ? '' : 'stopped'}`} />
                    {!training.ready
                      ? 'Compiling for your GPU…'
                      : running
                        ? 'Training'
                        : 'Training paused'}
                  </span>
                  <code>{duration(seconds)}</code>
                </div>
              </div>

              <div className="panel learning-panel">
                <div className="panel-heading">
                  <h3>Loss</h3>
                  <span className="chart-legend">
                    <span />
                    Value loss
                  </span>
                </div>
                <div className="loss-value">
                  {metrics.batches ? metrics.valueLoss.toFixed(4) : '—'}
                  <span>latest batch</span>
                </div>
                <LossChart history={history} />
                <p className="small-note">
                  How closely the model matches its training targets. Lower loss alone doesn’t
                  establish stronger play.
                </p>
                <div className="outcomes">
                  <div>
                    <span className="outcome-dot white" />
                    White wins <strong>{number(metrics.whiteWins)}</strong>
                  </div>
                  <div>
                    <span className="outcome-dot" />
                    Black wins <strong>{number(metrics.blackWins)}</strong>
                  </div>
                  <div>
                    <span className="outcome-dot draw" />
                    Draws <strong>{number(metrics.draws)}</strong>
                  </div>
                </div>
              </div>
              <div className="local-note">
                <Icon name="sun" size={21} />
                <div>
                  <strong>Stored on this device.</strong>
                  <p>
                    Nothing is uploaded.{' '}
                    {training.saved
                      ? 'Your model is saved in this browser.'
                      : 'Download a checkpoint to keep your progress.'}
                  </p>
                </div>
              </div>
              <div className="model-actions">
                <button onClick={() => uploadRef.current?.click()}>
                  <Icon name="upload" size={13} />
                  Load model
                </button>
                <button onClick={() => setResetOpen(true)}>
                  <Icon name="reset" size={13} />
                  Start fresh
                </button>
              </div>
            </aside>
          </section>
        </div>

        <div hidden={tab !== 'play'}>
          <Play training={training} active={tab === 'play'} />
        </div>

        {tab === 'about' && (
          <section className="about-layout">
            <div className="learning-loop">
              <span className="eyebrow">THE TRAINING LOOP</span>
              <h2>
                Play, search,
                <br />
                <em>learn.</em>
              </h2>
              <p>
                The same idea as self-play engines like AlphaZero, with an alpha-beta search and a
                network small enough to evaluate on the CPU.
              </p>
              <div className="loop-steps">
                {[
                  {
                    icon: 'grid',
                    title: '01 · Play',
                    text: 'Worker threads play games against themselves with the current weights. The first few moves of each game are random so the games differ.',
                  },
                  {
                    icon: 'tree',
                    title: '02 · Search',
                    text: 'Each move comes from an alpha-beta search that scores positions with the network alone. Nothing about chess beyond the rules is written into it.',
                  },
                  {
                    icon: 'cpu',
                    title: '03 · Learn',
                    text: 'Every searched position becomes a training row: its inputs, and a target that is mostly the search score with a little of the game result. The GPU trains on a window of recent rows.',
                  },
                ].map((step) => (
                  <div className="loop-step" key={step.title}>
                    <span>
                      <Icon name={step.icon as 'grid' | 'tree' | 'cpu'} size={25} />
                    </span>
                    <div>
                      <h3>{step.title}</h3>
                      <p>{step.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="about-details">
              <div className="panel model-blueprint">
                <span className="eyebrow">THE MODEL</span>
                <div className="blueprint-number">
                  {number(PARAM_COUNT)}
                  <span>parameters</span>
                </div>
                <div className="architecture-diagram">
                  <div>
                    832 inputs
                    <span>pieces by square, castling, en passant, clocks, piece counts</span>
                  </div>
                  <span className="diagram-stem" />
                  <div>
                    MLP
                    <span>{modelLabel(training.model)}</span>
                  </div>
                  <span className="diagram-stem" />
                  <div className="search-node">
                    <Icon name="tree" />
                    One value, used by alpha-beta search
                  </div>
                </div>
                <p>
                  The network is built and trained with Kalosm’s Fusor. Training runs on your GPU;
                  the search runs the same network on the CPU, where the first layer is updated
                  incrementally as pieces move.
                </p>
              </div>
              <div className="panel honest-notes">
                <h3>What to expect</h3>
                <p>
                  A fresh model plays close to randomly. On a recent laptop, a minute of training
                  reaches roughly 1750 Elo against Stockfish limited to 1800, at 300 ms per move.
                  That figure comes from native runs; a browser is somewhat slower.
                </p>
                <p>
                  Loss measures how well the network fits its own search targets. It does not
                  measure playing strength, which only games can.
                </p>
                <p>
                  The model size counts raw weights. Checkpoints also store the optimizer state, so
                  a saved model resumes training where it stopped.
                </p>
              </div>
            </div>
          </section>
        )}

        <footer>
          <a
            className="footer-brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setTab('training');
            }}
          >
            rookie.
          </a>
          <span>Runs entirely in your browser.</span>
          <div>
            <span className="tiny-dot" />
            Built with Kalosm’s Fusor.
          </div>
        </footer>
      </main>

      <input
        ref={uploadRef}
        className="visually-hidden"
        type="file"
        accept=".rookie,application/octet-stream"
        aria-label="Load Rookie model checkpoint"
        onChange={(e) => void importFile(e.target.files?.[0])}
      />
      {notice && (
        <div className="toast" role="status">
          <Icon name="info" size={17} />
          <span>{notice}</span>
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setNotice('')}
          >
            <Icon name="close" size={15} />
          </button>
        </div>
      )}
      {resetOpen && (
        <Modal title="Start over?" onClose={() => setResetOpen(false)}>
          <p>
            Reset the learned weights, training history, and games to the starting model. Save your
            current model first if you want to return to it.
          </p>
          <div className="dialog-actions">
            <button className="button secondary" onClick={download}>
              Save current model
            </button>
            <button
              className="button primary"
              onClick={() => {
                training.load(null);
                setResetOpen(false);
                setNotice('Training reset.');
              }}
            >
              Reset training
            </button>
          </div>
        </Modal>
      )}
      {currentArena && (
        <Modal
          title={`Board ${String(currentArena.id + 1).padStart(2, '0')} · A closer look`}
          onClose={() => setInspected(null)}
          wide
        >
          <div className="inspect-layout">
            <div>
              <Board
                fen={currentArena.fen}
                lastMove={currentArena.lastMove}
                coordinates
                label="Live enlarged self-play board"
              />
              <EvaluationBar score={currentArena.score} />
            </div>
            <div>
              <span className="eyebrow">LIVE SELF-PLAY</span>
              <h3>Game {currentArena.episode}</h3>
              <p>
                {currentArena.status} · Move {Math.ceil(currentArena.ply / 2)}
              </p>
              <div className="inspect-eval">
                <strong>{scoreLabel(currentArena.score)}</strong>
                <span>
                  search evaluation
                  <br />
                  from White’s perspective
                </span>
              </div>
              <h4>Where it looked</h4>
              <p className="small-note">
                Top candidates from the search before the last move. Bars show each move's
                share of a softmax over its search score, not probabilities of winning.
              </p>
              {currentArena.branches.map((branch) => (
                <div className="candidate" key={branch.uci}>
                  <code>{branch.uci}</code>
                  <div>
                    <span
                      style={{
                        width: `${Math.max(2, (branch.visits / Math.max(1, currentArena.branches[0]?.visits || 1)) * 100)}%`,
                      }}
                    />
                  </div>
                  <span>{branch.visits}</span>
                </div>
              ))}
              <div className="analysis-meta">
                <span>{currentArena.nodes} positions</span>
                <span>{currentArena.depth} ply explored</span>
              </div>
              <button className="button secondary full-width" onClick={() => setRunning(!running)}>
                <Icon name={running ? 'pause' : 'play'} size={14} />
                {running ? 'Pause to explore' : 'Resume self-play'}
              </button>
            </div>
          </div>
        </Modal>
      )}
    </>
  );
}

export default App;
