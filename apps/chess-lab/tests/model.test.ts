import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BatchBuffers, encode, INPUT } from '../src/engine/features';
import { Position } from '../src/engine/chess';
import { parameterCount, validModel } from '../src/engine/architectures';
import { emptyMetrics } from '../src/engine/protocol';
import { encodeCheckpoint, parseCheckpoint } from '../src/useTraining';
import type { Checkpoint } from '../src/useTraining';

test('neural features and rights are side-relative; batches repeat rows', () => {
  const w = new Position('4k3/8/8/8/8/8/P7/4K3 w - - 0 1');
  const b = new Position('4k3/p7/8/8/8/8/8/4K3 b - - 0 1');
  const wa = new Float32Array(INPUT),
    ba = new Float32Array(INPUT);
  encode(w, wa);
  encode(b, ba);
  assert.deepEqual(wa.slice(0, 780), ba.slice(0, 780));
  // Batches repeat the given rows to fill every slot.
  const buffers = new BatchBuffers(128);
  buffers.training([{ position: w, value: 0.25 }]);
  assert.deepEqual(buffers.x.slice(127 * INPUT, 128 * INPUT), wa);
  assert.ok(buffers.values.every((v) => v === 0.25));
});
test('model sizes report real parameter counts and reject invalid sizes', () => {
  assert.equal(parameterCount({ width: 128, depth: 2, hidden: 32 }), 111456);
  assert.equal(parameterCount({ width: 256, depth: 4, hidden: 64 }), 238464);
  assert.equal(validModel({ width: 128, depth: 2, hidden: 33 }), false);
  assert.equal(validModel({ width: 100, depth: 2, hidden: 32 }), false);
  assert.equal(validModel(null), false);
});
test('binary checkpoints round-trip and reject other files', () => {
  const model = { width: 64, depth: 3, hidden: 16 };
  const state = new Float32Array(1 + parameterCount(model) * 3).map((_, i) => (i % 97) / 97);
  state[0] = 12;
  const metrics = emptyMetrics();
  metrics.games = 5;
  const original: Checkpoint = { format: 'rookie-fusor-v5', model, state, metrics, seconds: 3.5, generation: 12 };
  const bytes = encodeCheckpoint(original);
  assert.equal(bytes.byteLength % 4, 0);
  const parsed = parseCheckpoint(bytes);
  assert.deepEqual(parsed.model, model);
  assert.deepEqual(Array.from(parsed.state), Array.from(state));
  assert.equal(parsed.metrics.games, 5);
  assert.equal(parsed.seconds, 3.5);
  // JSON text, a truncated state and a non-finite weight are all refused.
  assert.throws(() => parseCheckpoint(new TextEncoder().encode('{"format":"rookie-fusor-v3"}').buffer));
  assert.throws(() => parseCheckpoint(bytes.slice(0, bytes.byteLength - 4)));
  const corrupt = bytes.slice(0);
  new DataView(corrupt).setFloat32(bytes.byteLength - 4, Number.NaN, true);
  assert.throws(() => parseCheckpoint(corrupt));
});
