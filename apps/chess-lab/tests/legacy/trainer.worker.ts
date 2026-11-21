import { Position, notation, result } from '../../src/engine/chess';
import { TinyModel } from './model';
import type { Sample } from './model';
import { search } from './search';
import { DEFAULT_SETTINGS, emptyMetrics } from '../../src/engine/protocol';
import type {
  ArenaSnapshot,
  Settings,
  TrainerRequest,
  TrainerResponse,
} from '../../src/engine/protocol';

interface Arena {
  position: Position;
  keys: number[];
  samples: Sample[];
  view: ArenaSnapshot;
  finishedAt: number;
  before?: Position;
}
let model = new TinyModel();
let settings: Settings = DEFAULT_SETTINGS;
let running = false,
  syncing = false,
  timer: ReturnType<typeof setTimeout> | undefined;
let arenas: Arena[] = [],
  cursor = 0,
  pending: Sample[] = [],
  replay: Sample[] = [],
  replayIndex = 0;
let metrics = emptyMetrics(),
  lastSnapshot = 0;
const send = (message: TrainerResponse) => postMessage(message);

function newArena(id: number, episode = 1): Arena {
  const position = new Position();
  return {
    position,
    keys: [position.key()],
    samples: [],
    finishedAt: 0,
    view: {
      id,
      episode,
      fen: position.fen(),
      ply: 0,
      lastMove: null,
      lastSan: 'Starting position',
      score: 0,
      branches: [],
      depth: 0,
      nodes: 0,
      status: 'Self-play',
    },
  };
}

function remember(sample: Sample) {
  if (replay.length < 4096) replay.push(sample);
  else {
    replay[replayIndex] = sample;
    replayIndex = (replayIndex + 1) % 4096;
  }
  pending.push(sample);
}

function step(arena: Arena) {
  if (arena.finishedAt) {
    // Only the explicitly slow watch mode holds a completed board on screen.
    // Fast modes never wait for presentation: training and display are decoupled.
    const next = newArena(arena.view.id, arena.view.episode + 1);
    Object.assign(arena, next);
  }
  const p = arena.position;
  const thinking = search(p, model, arena.keys, {
    simulations: settings.nodes,
    exploration: true,
  });
  metrics.nodes += thinking.nodes;
  if (!thinking.move) return;
  const sample: Sample = {
    position: p.clone(),
    moves: thinking.moves,
    policy: thinking.policy,
    value: thinking.value * p.turn,
  };
  arena.samples.push(sample);
  remember(sample);
  arena.before = sample.position;
  p.push(thinking.move);
  arena.keys.push(p.key());
  metrics.positions++;
  const outcome = result(p, arena.keys);
  const truncated = !outcome && p.ply >= 200;
  arena.view.ply = p.ply;
  arena.view.lastMove = thinking.move;
  arena.view.branches = thinking.branches;
  arena.view.depth = thinking.depth;
  arena.view.nodes = thinking.nodes;
  arena.view.status = outcome?.reason || (truncated ? 'Move limit · bootstrapped' : 'Self-play');
  if (outcome || truncated) {
    if (outcome) {
      metrics.games++;
      if (outcome.winner === 1) metrics.whiteWins++;
      else if (outcome.winner === -1) metrics.blackWins++;
      else metrics.draws++;
      // Retain search supervision and blend in the actual game outcome.
      // Samples are shared with replay, so terminal results correct earlier targets.
      for (const past of arena.samples) past.value = 0.5 * past.value + 0.5 * outcome.winner;
    } else metrics.truncated++;
    arena.finishedAt = performance.now();
  }
  if (pending.length >= 32) {
    const fresh = pending.splice(0, 32);
    const batch = [
      ...fresh,
      ...Array.from({ length: 32 }, () => replay[Math.floor(Math.random() * replay.length)]),
    ];
    const loss = model.train(batch);
    metrics.valueLoss = metrics.batches
      ? 0.92 * metrics.valueLoss + 0.08 * loss.valueLoss
      : loss.valueLoss;
    metrics.batches++;
  }
}

function snapshot() {
  // SAN, FEN and displayed evaluation are computed at display frequency only.
  send({
    type: 'snapshot',
    arenas: arenas.map((a) => ({
      ...a.view,
      fen: a.position.fen(),
      score: model.value(a.position),
      lastSan:
        a.before && a.view.lastMove ? notation(a.before, a.view.lastMove) : 'Starting position',
    })),
    metrics,
  });
  lastSnapshot = performance.now();
}
function schedule() {
  if (timer !== undefined || !running || syncing) return;
  timer = setTimeout(tick, 0);
}
function tick() {
  timer = undefined;
  if (!running || syncing) return;
  const until = performance.now() + 40;
  do {
    step(arenas[cursor++ % arenas.length]);
  } while (performance.now() < until);
  if (performance.now() - lastSnapshot >= 160) snapshot();
  schedule();
}

self.onmessage = (event: MessageEvent<TrainerRequest>) => {
  const message = event.data;
  if (message.type === 'init') {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    arenas = message.ids.map((id) => newArena(id));
    settings = message.settings;
    model = new TinyModel(message.weights);
    running = message.running;
    syncing = false;
    metrics = emptyMetrics();
    pending = [];
    replay = [];
    replayIndex = 0;
    cursor = 0;
    snapshot();
    schedule();
  } else if (message.type === 'running') {
    running = message.running;
    snapshot();
    schedule();
  } else if (message.type === 'settings') settings = message.settings;
  else if (message.type === 'sync') {
    syncing = true;
    snapshot();
    send({ type: 'sync', round: message.round, weights: Array.from(model.weights) });
  } else if (message.type === 'weights') {
    model = new TinyModel(message.weights);
    syncing = false;
    schedule();
  }
};
