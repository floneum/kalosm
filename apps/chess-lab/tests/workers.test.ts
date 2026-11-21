import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Worker } from 'node:worker_threads';
import { Position, moveKey, result } from '../src/engine/chess';
import { TinyModel, PARAM_COUNT } from './legacy/model';
import { DEFAULT_SETTINGS } from '../src/engine/protocol';
import type { TrainerResponse } from '../src/engine/protocol';
import type { SearchResult } from './legacy/search';

function workerHarness(kind: 'trainer' | 'play') {
  const worker = new Worker(new URL('./worker-harness.mjs', import.meta.url), { workerData: kind });
  const messages: any[] = [];
  const waiters = new Set<() => void>();
  worker.on('message', (message) => {
    messages.push(message);
    for (const wake of waiters) wake();
  });
  function next<T = any>(predicate: (message: T) => boolean): Promise<T> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        waiters.delete(check);
        reject(new Error(`Timed out waiting for ${kind} worker`));
      }, 10000);
      const check = () => {
        const index = messages.findIndex(predicate);
        if (index !== -1) {
          clearTimeout(timer);
          waiters.delete(check);
          resolve(messages.splice(index, 1)[0]);
        }
      };
      waiters.add(check);
      check();
    });
  }
  return { worker, next };
}

test('legacy training worker advances arenas, updates weights, pauses and synchronizes', async () => {
  const { worker, next } = workerHarness('trainer');
  try {
    await next((m) => m.type === 'ready');
    const original = Array.from(new TinyModel().weights);
    worker.postMessage({
      type: 'init',
      ids: [0, 4, 8],
      settings: DEFAULT_SETTINGS,
      weights: original,
      running: false,
    });
    const initial = await next<Extract<TrainerResponse, { type: 'snapshot' }>>(
      (m) => m.type === 'snapshot',
    );
    assert.equal(initial.metrics.positions, 0);
    assert.deepEqual(
      initial.arenas.map((a) => a.id),
      [0, 4, 8],
    );
    worker.postMessage({ type: 'running', running: true });
    const trained = await next<Extract<TrainerResponse, { type: 'snapshot' }>>(
      (m) => m.type === 'snapshot' && m.metrics.batches > 4,
    );
    assert.ok(trained.metrics.positions >= 160);
    assert.ok(trained.metrics.nodes > trained.metrics.positions * 20);
    assert.ok(trained.arenas.every((a) => a.ply > 0 || a.episode > 1));
    for (const arena of trained.arenas)
      assert.ok(
        new Position(arena.fen).legalMoves().length ||
          arena.status === 'Checkmate' ||
          arena.status === 'Stalemate',
      );
    worker.postMessage({ type: 'running', running: false });
    worker.postMessage({ type: 'sync', round: 7 });
    const weights = await next<Extract<TrainerResponse, { type: 'sync' }>>(
      (m) => m.type === 'sync' && m.round === 7,
    );
    assert.equal(weights.weights.length, PARAM_COUNT);
    assert.ok(weights.weights.every(Number.isFinite));
    assert.ok(weights.weights.some((w, i) => original[i] !== w));
    worker.postMessage({ type: 'weights', weights: original });
    worker.postMessage({ type: 'sync', round: 8 });
    const restored = await next<Extract<TrainerResponse, { type: 'sync' }>>(
      (m) => m.type === 'sync' && m.round === 8,
    );
    assert.deepEqual(
      restored.weights,
      original,
      'paused worker must not silently resume when weights synchronize',
    );
    worker.postMessage({ type: 'weights', weights: weights.weights });
    worker.postMessage({ type: 'running', running: true });
    const resumed = await next<Extract<TrainerResponse, { type: 'snapshot' }>>(
      (m) => m.type === 'snapshot' && m.metrics.positions > trained.metrics.positions + 1000,
    );
    assert.equal(
      resumed.metrics.games,
      resumed.metrics.whiteWins + resumed.metrics.blackWins + resumed.metrics.draws,
    );
  } finally {
    await worker.terminate();
  }
});

test('legacy play worker finds mate and replies legally to a human opening', async () => {
  const { worker, next } = workerHarness('play');
  try {
    await next((m) => m.type === 'ready');
    const weights = Array.from(new TinyModel().weights);
    const mate = new Position('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
    worker.postMessage({ id: 1, fen: mate.fen(), keys: [mate.key()], weights, simulations: 768 });
    const answer = await next<{ id: number; result: SearchResult }>((m) => m.id === 1);
    assert.ok(answer.result.move);
    mate.push(answer.result.move!);
    assert.equal(result(mate, [])?.reason, 'Checkmate');
    const opening = new Position(),
      keys = [opening.key()];
    opening.push(opening.legalMoves().find((m) => moveKey(m) === 'e2e4')!);
    keys.push(opening.key());
    worker.postMessage({ id: 2, fen: opening.fen(), keys, weights, simulations: 768 });
    const reply = await next<{ id: number; result: SearchResult }>((m) => m.id === 2);
    assert.ok(reply.result.move);
    assert.ok(opening.legalMoves().some((m) => moveKey(m) === moveKey(reply.result.move!)));
    assert.ok(reply.result.branches.length > 1);
  } finally {
    await worker.terminate();
  }
});

