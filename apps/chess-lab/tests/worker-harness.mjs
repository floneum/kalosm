// Run the exact browser worker modules under Node's worker_threads for protocol
// integration tests. Only the browser transport is adapted; engine code is shared.
import { parentPort, workerData } from 'node:worker_threads';
import { tsImport } from 'tsx/esm/api';
globalThis.self = globalThis;
globalThis.postMessage = (message) => parentPort.postMessage(message);
await tsImport(`./legacy/${workerData}.worker.ts`, import.meta.url);
parentPort.on('message', (data) => globalThis.onmessage({ data }));
parentPort.postMessage({ type: 'ready' });
