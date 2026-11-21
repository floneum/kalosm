import { useEffect, useRef, useState } from 'react';
import {
  DEFAULT_MODEL,
  modelKey,
  parameterCount,
  validModel,
} from './engine/architectures';
import type { ModelConfig } from './engine/architectures';
import { DEFAULT_SETTINGS, emptyMetrics } from './engine/protocol';
import type { ArenaSnapshot, Metrics, Settings } from './engine/protocol';
import type { Request, Response } from './engine/gpu.worker';
import type { SearchResult } from './engine/search';

export interface HistoryPoint {
  time: number;
  loss: number;
  positions: number;
}
export interface Checkpoint {
  format: 'rookie-fusor-v5';
  model: ModelConfig;
  /** `ChessGpu.save()`: the step, then weights and both AdamW moments. */
  state: Float32Array;
  metrics: Metrics;
  seconds: number;
  generation: number;
}
// A checkpoint file is binary: a fixed little-endian header (magic, version,
// width, depth, hidden width, generation, seconds, statistics), then the
// state as f32.
const MAGIC = 0x35464b52; // "RKF5"
const VERSION = 5;
const METRIC_KEYS = Object.keys(emptyMetrics()) as (keyof Metrics)[];
const HEADER_BYTES = 6 * 4 + 8 + 4 + METRIC_KEYS.length * 8;
function checkState(state: Float32Array, model: ModelConfig) {
  if (state.length !== 1 + parameterCount(model) * 3) throw new Error('Invalid Fusor checkpoint size');
  if (state.some((v) => !Number.isFinite(v)) || state[0] < 0 || !Number.isInteger(state[0]))
    throw new Error('Checkpoint contains invalid weights or optimizer state');
}
export function encodeCheckpoint(data: Checkpoint): ArrayBuffer {
  const buffer = new ArrayBuffer(HEADER_BYTES + data.state.byteLength),
    view = new DataView(buffer);
  const words = [MAGIC, VERSION, data.model.width, data.model.depth, data.model.hidden, data.generation];
  words.forEach((w, i) => view.setUint32(i * 4, w, true));
  view.setFloat64(24, data.seconds, true);
  view.setUint32(32, METRIC_KEYS.length, true);
  METRIC_KEYS.forEach((key, i) => view.setFloat64(36 + i * 8, data.metrics[key], true));
  new Float32Array(buffer, HEADER_BYTES).set(data.state);
  return buffer;
}
export function parseCheckpoint(buffer: ArrayBuffer): Checkpoint {
  if (buffer.byteLength < HEADER_BYTES) throw new Error('That is not a Rookie checkpoint.');
  const view = new DataView(buffer),
    word = (i: number) => view.getUint32(i * 4, true);
  if (word(0) !== MAGIC || word(1) !== VERSION || view.getUint32(32, true) !== METRIC_KEYS.length)
    throw new Error('Choose a Rookie v5 checkpoint. Earlier checkpoints used a different model.');
  const model = { width: word(2), depth: word(3), hidden: word(4) };
  if (!validModel(model)) throw new Error('Invalid checkpoint model size');
  if ((buffer.byteLength - HEADER_BYTES) % 4) throw new Error('Invalid Fusor checkpoint size');
  const state = new Float32Array(buffer.slice(HEADER_BYTES));
  checkState(state, model);
  const metrics = emptyMetrics();
  METRIC_KEYS.forEach((key, i) => (metrics[key] = view.getFloat64(36 + i * 8, true)));
  const seconds = view.getFloat64(24, true),
    generation = word(5);
  if (Object.values(metrics).some((v) => !Number.isFinite(v) || v < 0))
    throw new Error('Invalid training statistics');
  if (!Number.isFinite(seconds) || seconds < 0 || generation !== state[0])
    throw new Error('Invalid checkpoint metadata');
  return { format: 'rookie-fusor-v5', model, state, metrics, seconds, generation };
}
async function storage(write?: Checkpoint, key = 'latest'): Promise<Checkpoint | null> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('rookie-fusor-v5', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('checkpoints');
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result,
        tx = db.transaction('checkpoints', write ? 'readwrite' : 'readonly'),
        store = tx.objectStore('checkpoints');
      if (write) store.put(write, modelKey(write.model));
      const operation = write ? store.put(write, 'latest') : store.get(key);
      let result: Checkpoint | null = null;
      operation.onsuccess = () => {
        result = write || operation.result || null;
      };
      tx.oncomplete = () => {
        db.close();
        resolve(result);
      };
      tx.onerror = () => {
        db.close();
        reject(tx.error);
      };
    };
  });
}
export function useTraining() {
  const [model, setModel] = useState<ModelConfig>(DEFAULT_MODEL),
    [switching, setSwitching] = useState(false);
  const modelRef = useRef(model);
  modelRef.current = model;
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS),
    [running, setRunning] = useState(true);
  const [visible, setVisible] = useState(!document.hidden);
  const [arenas, setArenas] = useState<ArenaSnapshot[]>([]),
    [metrics, setMetrics] = useState(emptyMetrics);
  const [seconds, setSeconds] = useState(0),
    [generation, setGeneration] = useState(0),
    [history, setHistory] = useState<HistoryPoint[]>([]);
  const [rates, setRates] = useState({ positions: 0, nodes: 0, batches: 0 }),
    [error, setError] = useState(''),
    [saved, setSaved] = useState(false);
  const [backend, setBackend] = useState('Compiling Fusor for your GPU…'),
    // Simultaneous games; Fusor sizes the batch to the architecture.
    [games, setGames] = useState(0),
    [ready, setReady] = useState(false),
    [fill, setFill] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const worker = useRef<Worker | null>(null),
    pending = useRef(
      new Map<number, { resolve: (r: Response) => void; reject: (e: Error) => void }>(),
    ),
    id = useRef(0);
  const initial = useRef<Checkpoint | null | undefined>(undefined),
    metricsRef = useRef(metrics),
    secondsRef = useRef(0),
    generationRef = useRef(0);
  // Self-play keeps training while a human game is open.
  const actualRunning = running && visible,
    runningRef = useRef(actualRunning),
    settingsRef = useRef(settings);
  runningRef.current = actualRunning;
  settingsRef.current = settings;
  function request(message: Extract<Request, { type: 'save' | 'play' }>): Promise<Response> {
    return new Promise((resolve, reject) => {
      if (!worker.current) {
        reject(new Error('GPU is starting'));
        return;
      }
      pending.current.set(message.id, { resolve, reject });
      worker.current.postMessage(message);
    });
  }
  // Learn from a finished human game (see the worker's 'learn' request).
  function learn(fens: string[], moves: string[], result: number) {
    worker.current?.postMessage({ type: 'learn', fens, moves, result } satisfies Request);
  }
  async function checkpoint(): Promise<Checkpoint> {
    const response = await request({ type: 'save', id: ++id.current });
    if (response.type !== 'save') throw new Error('Checkpoint failed');
    return {
      format: 'rookie-fusor-v5',
      model: { ...modelRef.current },
      state: response.state,
      metrics: { ...metricsRef.current },
      seconds: secondsRef.current,
      generation: response.state[0],
    };
  }
  const checkpointRef = useRef(checkpoint);
  checkpointRef.current = checkpoint;
  useEffect(() => {
    const listener = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', listener);
    return () => document.removeEventListener('visibilitychange', listener);
  }, []);
  useEffect(() => {
    let live = true,
      autoSaving = false;
    setReady(false);
    setBackend('Compiling Fusor for your GPU…');
    const rejectPending = (message: string) => {
      for (const task of pending.current.values()) task.reject(new Error(message));
      pending.current.clear();
    };
    void (async () => {
      if (initial.current === undefined) {
        try {
          initial.current = await storage();
        } catch {
          initial.current = null;
        }
      }
      if (!live) return;
      const restored = initial.current;
      let state: Float32Array | undefined;
      if (restored) {
        try {
          checkState(restored.state, restored.model);
          state = restored.state;
          setModel(restored.model);
          modelRef.current = restored.model;
        } catch {
          initial.current = null;
        }
      }
      if (state && restored) {
        metricsRef.current = restored.metrics;
        setMetrics(restored.metrics);
        secondsRef.current = restored.seconds;
        setSeconds(restored.seconds);
        setSaved(true);
      }
      const w = new Worker(new URL('./engine/gpu.worker.ts', import.meta.url), { type: 'module' });
      worker.current = w;
      w.onmessage = (event: MessageEvent<Response>) => {
        if (!live) return;
        const data = event.data;
        if (data.type === 'ready') {
          setReady(true);
          setBackend(data.backend);
          setGames(data.games);
        } else if (data.type === 'snapshot') {
          metricsRef.current = data.metrics;
          setMetrics(data.metrics);
          setArenas(data.arenas);
          generationRef.current = data.step;
          setGeneration(data.step);
          setFill(data.fill);
        } else if (data.type === 'error') {
          setError(data.error);
          setRunning(false);
          rejectPending(data.error);
        } else {
          pending.current.get(data.id)?.resolve(data);
          pending.current.delete(data.id);
        }
      };
      w.onerror = (e) => {
        setError(e.message || 'The GPU worker failed');
        setRunning(false);
        rejectPending('The GPU worker failed');
      };
      w.postMessage({
        type: 'init',
        model: modelRef.current,
        settings: settingsRef.current,
        running: runningRef.current,
        state,
        metrics: metricsRef.current,
      } satisfies Request);
    })();
    const timer = setInterval(() => {
      if (!worker.current || autoSaving) return;
      autoSaving = true;
      void checkpointRef
        .current()
        .then(storage)
        .then(() => {
          if (live) setSaved(true);
        })
        .catch(() => {
          if (live) setSaved(false);
        })
        .finally(() => {
          autoSaving = false;
        });
    }, 30000);
    return () => {
      live = false;
      clearInterval(timer);
      worker.current?.terminate();
      worker.current = null;
      rejectPending('Training session changed');
    };
  }, [epoch]);
  useEffect(() => {
    worker.current?.postMessage({ type: 'running', running: actualRunning } satisfies Request);
  }, [actualRunning]);
  useEffect(() => {
    worker.current?.postMessage({ type: 'settings', settings } satisfies Request);
  }, [settings]);
  useEffect(() => {
    let previous = metricsRef.current,
      last = performance.now();
    const timer = setInterval(() => {
      const now = performance.now(),
        dt = (now - last) / 1000,
        current = metricsRef.current;
      last = now;
      setRates({
        positions: Math.max(0, (current.positions - previous.positions) / dt),
        nodes: Math.max(0, (current.nodes - previous.nodes) / dt),
        batches: Math.max(0, (current.batches - previous.batches) / dt),
      });
      previous = current;
      if (runningRef.current) {
        secondsRef.current += dt;
        setSeconds(secondsRef.current);
      }
      if (runningRef.current && current.batches)
        setHistory((h) =>
          [
            ...h,
            { time: secondsRef.current, loss: current.valueLoss, positions: current.positions },
          ].slice(-90),
        );
    }, 1000);
    return () => clearInterval(timer);
  }, []);
  function load(data: Checkpoint | null, config = modelRef.current) {
    setModel(data?.model || config);
    modelRef.current = data?.model || config;
    initial.current = data;
    metricsRef.current = data?.metrics || emptyMetrics();
    setMetrics(metricsRef.current);
    secondsRef.current = data?.seconds || 0;
    setSeconds(secondsRef.current);
    generationRef.current = data?.generation || 0;
    setGeneration(generationRef.current);
    setHistory([]);
    setRates({ positions: 0, nodes: 0, batches: 0 });
    setError('');
    setSaved(false);
    setEpoch((e) => e + 1);
  }
  async function switchModel(next: ModelConfig) {
    if (!validModel(next) || modelKey(next) === modelKey(modelRef.current)) return;
    setSwitching(true);
    try {
      if (ready) await storage(await checkpoint());
      const existing = await storage(undefined, modelKey(next));
      load(existing, next);
    } catch (error) {
      setError(`Could not switch models: ${String(error)}`);
    } finally {
      setSwitching(false);
    }
  }
  // The opponent searches for at most `millis` of wall time.
  // Keep searching the human's position until they move (see the worker).
  function ponder(fen: string, history: string[]) {
    worker.current?.postMessage({ type: 'ponder', fen, history } satisfies Request);
  }
  function stopPonder() {
    worker.current?.postMessage({ type: 'stop_ponder' } satisfies Request);
  }
  async function play(fen: string, history: string[], millis: number): Promise<SearchResult> {
    const response = await request({ type: 'play', id: ++id.current, fen, history, millis });
    if (response.type !== 'play') throw new Error('Search failed');
    return response.result;
  }
  return {
    settings,
    setSettings,
    running,
    setRunning,
    visible,
    arenas,
    metrics,
    rates,
    seconds,
    generation,
    history,
    checkpoint,
    learn,
    load,
    error,
    saved,
    backend,
    games,
    ready,
    fill,
    play,
    ponder,
    stopPonder,
    model,
    switchModel,
    switching,
    workerCount: 1,
  };
}
export type Training = ReturnType<typeof useTraining>;
