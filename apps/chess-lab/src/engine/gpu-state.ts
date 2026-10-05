import { Position, moveKey } from './chess';
import type { Move } from './chess';
import { emptyMetrics } from './protocol';
import type { ArenaSnapshot } from './protocol';
import type { SearchResult } from './search';
export const GAME_WORDS = 768;
const floating = new DataView(new ArrayBuffer(4));
const f32 = (word: number) => {
  floating.setUint32(0, word, true);
  return floating.getFloat32(0, true);
};
export const decodeMove = (m: number): Move => ({
  from: m & 127,
  to: (m >>> 7) & 127,
  promotion: (m >>> 14) & 7 || undefined,
  flags: (m >>> 17) & 7,
});
function position(state: Uint32Array, offset: number): Position {
  const p = new Position();
  for (let i = 0; i < 128; i++) p.board[i] = state[offset + i] | 0;
  p.turn = (state[offset + 128] | 0) as 1 | -1;
  p.castle = state[offset + 129];
  p.ep = state[offset + 130] | 0;
  p.half = state[offset + 131];
  p.ply = state[offset + 132];
  p.kings = [state[offset + 133], state[offset + 134]];
  return p;
}
export function decodeSnapshot(state: Uint32Array) {
  const metrics = emptyMetrics(),
    arenas: ArenaSnapshot[] = [];
  for (let g = 0; g < state.length / GAME_WORDS; g++) {
    const b = g * GAME_WORDS;
    metrics.nodes += state[b + 138];
    metrics.positions += state[b + 139];
    metrics.games += state[b + 140];
    metrics.whiteWins += state[b + 141];
    metrics.blackWins += state[b + 142];
    metrics.draws += state[b + 143];
    metrics.truncated += state[b + 144];
    if (g < 24) {
      const p = position(state, b),
        last = state[b + 136];
      arenas.push({
        id: g,
        fen: p.fen(),
        ply: p.ply,
        episode: state[b + 135],
        lastMove: last ? decodeMove(last) : null,
        lastSan: last ? moveKey(decodeMove(last)) : 'Starting position',
        score: f32(state[b + 152]),
        depth: state[b + 154],
        nodes: state[b + 155],
        branches: branches(state, b),
        status:
          ['GPU self-play', 'White wins', 'Black wins', 'Draw', 'Move limit · bootstrapped'][
            state[b + 137]
          ] || 'GPU self-play',
      });
    }
  }
  return { metrics, arenas };
}
function branches(state: Uint32Array, offset: number) {
  return Array.from({ length: 5 }, (_, i) => {
    const b = offset + 736 + i * 4,
      m = state[b];
    return {
      move: decodeMove(m),
      uci: m ? moveKey(decodeMove(m)) : '',
      visits: state[b + 1],
      score: f32(state[b + 2]),
      prior: f32(state[b + 3]),
    };
  }).filter((b) => !!b.uci);
}
const hash = (n: number) => {
  n = (n ^ (n >>> 16)) >>> 0;
  n = Math.imul(n, 2146121005);
  n ^= n >>> 15;
  n = Math.imul(n, 2221713035);
  return (n ^ (n >>> 16)) >>> 0;
};
function key(p: Position): [number, number] {
  let lo = hash(p.castle + 32 * (p.turn === 1 ? 0 : 1) + 77),
    hi = hash(lo + 17);
  for (let s = 0; s < 128; s++) {
    if (s & 0x88) {
      s += 7;
      continue;
    }
    if (p.board[s]) {
      const n = (p.board[s] + 6) * 128 + s;
      lo ^= hash(n + 101);
      hi ^= hash(n + 98765);
    }
  }
  if (p.ep >= 0 && p.legalMoves().some((m) => m.flags & 1)) {
    lo ^= hash(p.ep + 99999);
    hi ^= hash(p.ep + 77777);
  }
  return [lo >>> 0, hi >>> 0];
}
// Human input is uploaded once per move, copied to every one of `batch` games.
// Self-play never calls the CPU rules.
export function encodeHuman(fen: string, history: string[], batch: number): Uint32Array {
  const p = new Position(fen),
    row = new Uint32Array(GAME_WORDS),
    states = new Uint32Array(GAME_WORDS * batch);
  for (let i = 0; i < 128; i++) row[i] = p.board[i];
  row[128] = p.turn;
  row[129] = p.castle;
  row[130] = p.ep;
  row[131] = p.half;
  row[132] = p.ply;
  row[133] = p.kings[0];
  row[134] = p.kings[1];
  row[135] = 1;
  for (const fen of history) {
    const old = new Position(fen);
    if (old.ply <= 240) row.set(key(old), 224 + old.ply * 2);
  }
  for (let i = 0; i < batch; i++) states.set(row, i * GAME_WORDS);
  return states;
}
export function decodeAnswer(state: Uint32Array, beforeTurn: number): SearchResult {
  const move = state[136] ? decodeMove(state[136]) : null,
    found = branches(state, 0);
  return {
    move,
    moves: found.map((b) => b.move),
    policy: found.map((b) => b.visits / Math.max(1, state[153])),
    value: f32(state[152]) * beforeTurn,
    nodes: state[155],
    depth: state[154],
    branches: found,
  };
}
