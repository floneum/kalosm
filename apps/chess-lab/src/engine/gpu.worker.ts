import init, { ChessGpu } from '../wasm/rookie_fusor';
import type { ModelConfig } from './architectures';
import { DEFAULT_SETTINGS, emptyMetrics } from './protocol';
import type { ArenaSnapshot, Metrics, Settings } from './protocol';
import type { SearchResult } from './search';
import { decodeSnapshot, encodeHuman, decodeAnswer } from './gpu-state';
import { BatchBuffers, INPUT } from './features';
import type { TrainingSample } from './features';
import { Position } from './chess';
import type { SelfPlayRequest, SelfPlayResponse } from './selfplay.worker';
export type Request =
  | {
      type: 'init';
      model: ModelConfig;
      settings: Settings;
      running: boolean;
      state?: Float32Array;
      metrics?: Metrics;
    }
  | { type: 'running'; running: boolean }
  | { type: 'settings'; settings: Settings }
  | { type: 'save'; id: number }
  | { type: 'play'; id: number; fen: string; history: string[]; millis: number }
  // Think about the human's position while they decide (fills the persistent
  // transposition table that the reply search then starts from); stopped by
  // the next play request or an explicit stop.
  | { type: 'ponder'; fen: string; history: string[] }
  | { type: 'stop_ponder' }
  // A finished human game: each position and the result for White (1, 0 or -1).
  | { type: 'learn'; fens: string[]; moves: string[]; result: number };
export type Response =
  | { type: 'ready'; backend: string; games: number }
  | { type: 'error'; error: string; id?: number }
  | { type: 'snapshot'; arenas: ArenaSnapshot[]; metrics: Metrics; step: number; fill: number }
  | { type: 'save'; id: number; state: Float32Array }
  | { type: 'play'; id: number; result: SearchResult };
const send = (response: Response, transfer: Transferable[] = []) =>
  postMessage(response, { transfer });
let gpu: ChessGpu | undefined,
  running = false,
  busy = false,
  initialized = false,
  settings = DEFAULT_SETTINGS;
let base = emptyMetrics(),
  metrics = emptyMetrics(),
  arenas: ArenaSnapshot[] = [],
  step = 0,
  lastLoss = 0,
  lastSnapshot = 0;
const jobs: Extract<Request, { type: 'save' | 'play' | 'learn' }>[] = [];
let ponder: { fen: string; history: string[] } | null = null;
// One pondering slice: short, so a play request is answered promptly.
const PONDER_SLICE = 100;
// Training batches spent on each finished human game.
const HUMAN_STEPS = 4;
// Self-play: worker threads play games with the CPU search while this worker
// trains on their positions. Settings measured natively: 5% of the target
// from the game result, 6 random opening plies, 4 batches per 1024 new rows
// from the last 65,536 rows.
const CPU_RESULT = 0.05;
const CPU_LAMBDA = 0;
const CPU_RANDOM = 6;
const CPU_STEPS = 4;
const CPU_WINDOW = 65536;
const ROW = INPUT + 1;
const helpers: Worker[] = [];
const replay = new Float32Array(CPU_WINDOW * ROW);
let replayCount = 0,
  replayNext = 0,
  fresh = 0,
  lastPublish = 0,
  seed = 0x2545f491;
const latest: Uint32Array[] = [];
const random = () => {
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return seed >>> 0;
};
function startHelpers(model: ModelConfig) {
  const count = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 2));
  for (let i = 0; i < count; i++) {
    const worker = new Worker(new URL('./selfplay.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<SelfPlayResponse>) => {
      if (event.data.type === 'game') ingest(event.data.data);
      else send({ type: 'error', error: `CPU self-play: ${event.data.error}` });
    };
    const post = (m: SelfPlayRequest) => worker.postMessage(m);
    post({ type: 'init', width: model.width, depth: model.depth, hidden: model.hidden, seed: 0x9e3779b9 * (i + 1) });
    post(searchSettings());
    post({ type: 'running', running });
    helpers.push(worker);
  }
}
const searchSettings = (): SelfPlayRequest => ({
  type: 'settings',
  nodes: settings.nodes,
  resultWeight: CPU_RESULT,
  lambda: CPU_LAMBDA,
  randomPlies: CPU_RANDOM,
});
function ingest(data: Float32Array) {
  const rows = (data.length - 769) / ROW;
  for (let r = 0; r < rows; r++) {
    replay.set(data.subarray(r * ROW, (r + 1) * ROW), replayNext * ROW);
    replayNext = (replayNext + 1) % CPU_WINDOW;
    replayCount = Math.min(replayCount + 1, CPU_WINDOW);
  }
  fresh += rows;
  latest.push(new Uint32Array(data.buffer, rows * ROW * 4, 768).slice());
  if (latest.length > 24) latest.shift();
  const result = data[data.length - 1];
  metrics.games += 1;
  metrics.positions += rows;
  metrics.nodes += rows * settings.nodes;
  if (result > 0) metrics.whiteWins += 1;
  else if (result < 0) metrics.blackWins += 1;
  else metrics.draws += 1;
}
async function publishWeights(force = false) {
  if (!gpu || (!force && performance.now() - lastPublish < 250)) return;
  lastPublish = performance.now();
  const weights = await gpu.value_weights();
  for (const worker of helpers)
    worker.postMessage({ type: 'weights', weights } satisfies SelfPlayRequest);
}
async function trainOnReplay() {
  if (!gpu) return;
  const batch = gpu.batch(),
    x = new Float32Array(batch * INPUT),
    values = new Float32Array(batch);
  for (let s = 0; s < CPU_STEPS; s++) {
    for (let b = 0; b < batch; b++) {
      const at = (random() % replayCount) * ROW;
      x.set(replay.subarray(at, at + INPUT), b * INPUT);
      values[b] = replay[at + INPUT];
    }
    await gpu.train(x, values, 1);
    step += 1;
  }
}
function snapshot() {
  const states = new Uint32Array(latest.length * 768);
  latest.forEach((s, i) => states.set(s, i * 768));
  arenas = latest.length ? decodeSnapshot(states).arenas : [];
  send({ type: 'snapshot', arenas, metrics: { ...metrics, batches: step }, step, fill: 1 });
  lastSnapshot = performance.now();
}
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
async function refreshLoss() {
  if (!gpu || performance.now() - lastLoss < 750) return;
  const loss = await gpu.losses();
  if (loss.some((v) => !Number.isFinite(v))) throw new Error('GPU training returned a non-finite loss');
  metrics.valueLoss = loss[0];
  lastLoss = performance.now();
}
async function pump() {
  if (busy || !gpu) return;
  busy = true;
  try {
    while (jobs.length || running || ponder) {
      const job = jobs.shift();
      if (job?.type === 'save') {
        const state = await gpu.save();
        send({ type: 'save', id: job.id, state }, [state.buffer]);
      } else if (job?.type === 'learn') {
        // Train on the game just played: each position toward its result.
        const rows: TrainingSample[] = job.fens.map((fen) => {
          const position = new Position(fen);
          return { position, value: job.result * position.turn };
        });
        if (rows.length) {
          const buffers = new BatchBuffers(gpu.batch());
          buffers.training(rows);
          await gpu.train(buffers.x, buffers.values, HUMAN_STEPS);
          step += HUMAN_STEPS;
        }
      } else if (job?.type === 'play') {
        ponder = null;
        const states = await gpu.play(encodeHuman(job.fen, job.history, 1), job.millis);
        send({
          type: 'play',
          id: job.id,
          result: decodeAnswer(states, job.fen.split(' ')[1] === 'w' ? 1 : -1),
        });
      } else if (ponder) {
        // The result is discarded; the table keeps what the search learned.
        await gpu.play(encodeHuman(ponder.fen, ponder.history, 1), PONDER_SLICE);
        if (fresh >= gpu.batch() && replayCount >= gpu.batch()) {
          fresh -= gpu.batch();
          await trainOnReplay();
        }
        await publishWeights();
      } else {
        // Train once enough new positions arrived, keep the workers' weights
        // fresh, and show the latest finished games.
        if (fresh >= gpu.batch() && replayCount >= gpu.batch()) {
          fresh -= gpu.batch();
          await trainOnReplay();
        } else {
          await delay(2);
        }
        await publishWeights();
        if (performance.now() - lastSnapshot > 250) {
          await refreshLoss();
          snapshot();
        }
      }
      // Yield so settings and play requests are handled between calls.
      await delay(0);
    }
  } catch (error) {
    running = false;
    send({ type: 'error', error: String(error) });
  } finally {
    busy = false;
  }
}
self.onmessage = async (event: MessageEvent<Request>) => {
  const m = event.data;
  if (m.type === 'init') {
    if (initialized) return;
    initialized = true;
    settings = m.settings;
    running = m.running;
    base = m.metrics || emptyMetrics();
    metrics = { ...base };
    try {
      await init();
      gpu = await ChessGpu.create(m.model.width, m.model.depth, m.model.hidden);
      if (m.state) {
        gpu.load(m.state);
        step = m.state[0];
      }
      startHelpers(m.model);
      await publishWeights(true);
      send({ type: 'ready', backend: gpu.info(), games: helpers.length });
      void pump();
    } catch (error) {
      running = false;
      send({ type: 'error', error: `Fusor could not start: ${String(error)}` });
    }
  } else if (m.type === 'running') {
    running = m.running;
    for (const worker of helpers)
      worker.postMessage({ type: 'running', running } satisfies SelfPlayRequest);
    void pump();
  } else if (m.type === 'settings') {
    settings = m.settings;
    for (const worker of helpers) worker.postMessage(searchSettings());
  } else if (m.type === 'ponder') {
    ponder = m;
    void pump();
  } else if (m.type === 'stop_ponder') {
    ponder = null;
  } else {
    jobs.push(m);
    void pump();
  }
};
