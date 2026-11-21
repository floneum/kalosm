import { Worker } from 'node:worker_threads';
import { DEFAULT_SETTINGS } from '../src/engine/protocol';
import type { Metrics } from '../src/engine/protocol';
import { TinyModel } from './legacy/model';

const count = Number(process.env.WORKERS || 4);
const metrics = new Map<number, Metrics>();
const workers: Worker[] = [];
let started = 0;
await Promise.all(
  Array.from(
    { length: count },
    (_, id) =>
      new Promise<void>((resolve, reject) => {
        const worker = new Worker(new URL('./worker-harness.mjs', import.meta.url), {
          workerData: 'trainer',
        });
        workers.push(worker);
        worker.on('error', reject);
        worker.on('message', (message) => {
          if (message.type === 'ready') resolve();
          if (message.type === 'snapshot') metrics.set(id, message.metrics);
        });
      }),
  ),
);
started = performance.now();
for (let id = 0; id < count; id++)
  workers[id].postMessage({
    type: 'init',
    ids: [id * 3, id * 3 + 1, id * 3 + 2],
    settings: DEFAULT_SETTINGS,
    weights: Array.from(new TinyModel().weights),
    running: true,
  });
await new Promise((resolve) => setTimeout(resolve, 5000));
await Promise.all(
  workers.map(
    (worker) =>
      new Promise<void>((resolve) => {
        worker.on('message', (message) => {
          if (message.type === 'sync') resolve();
        });
        worker.postMessage({ type: 'sync', round: 1 });
      }),
  ),
);
const seconds = (performance.now() - started) / 1000;
const values = [...metrics.values()];
const sum = (key: keyof Metrics) => values.reduce((total, m) => total + m[key], 0);
console.log(
  JSON.stringify(
    {
      workers: count,
      arenas: count * 3,
      simulationsPerMove: 48,
      seconds: +seconds.toFixed(3),
      movesPerSecond: Math.round(sum('positions') / seconds),
      searchPositionsPerSecond: Math.round(sum('nodes') / seconds),
      trainingBatchesPerSecond: Math.round(sum('batches') / seconds),
      trainingSamplesPerSecond: Math.round((sum('batches') * 64) / seconds),
      completedGames: sum('games'),
      truncatedEpisodes: sum('truncated'),
    },
    null,
    2,
  ),
);
await Promise.all(workers.map((w) => w.terminate()));
