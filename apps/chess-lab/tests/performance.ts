import assert from 'node:assert/strict';
import { Position, result } from '../src/engine/chess';
import { TinyModel } from './legacy/model';
import type { Sample } from './legacy/model';
import { search } from './legacy/search';

// Fixed workload and RNG: optimizations must retain 48 simulations per move.
// Run separately from correctness tests; elapsed-time gates are machine-specific.
let seed = 17;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
  return (seed >>> 0) / 4294967296;
};
const model = new TinyModel();
const arenas = Array.from({ length: 12 }, () => {
  const p = new Position();
  return { p, keys: [p.key()] };
});
let nodes = 0,
  batch: Sample[] = [];
const count = 4000;
const started = performance.now();
for (let i = 0; i < count; i++) {
  const arena = arenas[i % arenas.length];
  const answer = search(arena.p, model, arena.keys, { simulations: 48, exploration: true, random });
  assert.ok(answer.move);
  batch.push({
    position: arena.p.clone(),
    moves: answer.moves,
    policy: answer.policy,
    value: answer.value * arena.p.turn,
  });
  arena.p.make(answer.move!);
  arena.keys.push(arena.p.key());
  nodes += answer.nodes;
  if (result(arena.p, arena.keys) || arena.p.ply >= 200) {
    arena.p = new Position();
    arena.keys = [arena.p.key()];
  }
  if (batch.length === 32) {
    model.train(batch);
    batch = [];
  }
}
const seconds = (performance.now() - started) / 1000,
  rate = count / seconds;
console.log(
  JSON.stringify(
    {
      positions: count,
      simulations: 48,
      seconds: +seconds.toFixed(3),
      movesPerSecond: Math.round(rate),
      searchPositionsPerSecond: Math.round(nodes / seconds),
    },
    null,
    2,
  ),
);
assert.ok(
  rate >= Number(process.env.MIN_MOVES_PER_SECOND || 5000),
  `Performance target missed: ${Math.round(rate)} moves/sec (target ${process.env.MIN_MOVES_PER_SECOND || 5000}).`,
);
