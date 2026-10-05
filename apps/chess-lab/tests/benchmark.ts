import { Position, result } from '../src/engine/chess';
import { TinyModel, MODEL_BYTES } from './legacy/model';
import type { Sample } from './legacy/model';
import { search } from './legacy/search';

const model = new TinyModel();
const arenas = Array.from({ length: 12 }, () => {
  const p = new Position();
  return { p, keys: [p.key()] };
});
const start = performance.now();
let positions = 0,
  nodes = 0,
  batches = 0;
let batch: Sample[] = [];
while (performance.now() - start < 5000) {
  const arena = arenas[positions % arenas.length];
  const answer = search(arena.p, model, arena.keys, { simulations: 48, exploration: true });
  if (answer.move) {
    batch.push({
      position: arena.p.clone(),
      moves: answer.moves,
      policy: answer.policy,
      value: answer.value * arena.p.turn,
    });
    arena.p.make(answer.move);
    arena.keys.push(arena.p.key());
    positions++;
    nodes += answer.nodes;
  }
  if (result(arena.p, arena.keys) || arena.p.ply >= 200) {
    arena.p = new Position();
    arena.keys = [arena.p.key()];
  }
  if (batch.length >= 32) {
    model.train(batch);
    batches++;
    batch = [];
  }
}
const seconds = (performance.now() - start) / 1000;
console.log(
  JSON.stringify(
    {
      seconds: +seconds.toFixed(2),
      positions,
      positionsPerSecond: Math.round(positions / seconds),
      nodesPerSecond: Math.round(nodes / seconds),
      batches,
      modelBytes: MODEL_BYTES,
    },
    null,
    2,
  ),
);
