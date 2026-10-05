// A CPU self-play worker: its own Wasm instance plays games with the latest
// weights and sends each game's training rows and final position back.
import init, { CpuEngine } from '../wasm/rookie_fusor';
export type SelfPlayRequest =
  | { type: 'init'; width: number; depth: number; hidden: number; seed: number }
  | { type: 'weights'; weights: Float32Array }
  | { type: 'settings'; nodes: number; resultWeight: number; lambda: number; randomPlies: number }
  | { type: 'running'; running: boolean };
export type SelfPlayResponse =
  // 833 numbers per position (features, target), then the final game state
  // (768 words as f32 bits) and the result for White.
  | { type: 'game'; data: Float32Array }
  | { type: 'error'; error: string };
let engine: CpuEngine | undefined,
  running = false,
  busy = false,
  nodes = 400,
  resultWeight = 0.05,
  lambda = 0,
  randomPlies = 6;
const send = (r: SelfPlayResponse, transfer: Transferable[] = []) => postMessage(r, { transfer });
const yieldNow = () => new Promise((resolve) => setTimeout(resolve, 0));
async function loop() {
  if (busy || !engine) return;
  busy = true;
  try {
    while (running) {
      const data = engine.play_game(nodes, resultWeight, lambda, randomPlies);
      if (data.length) send({ type: 'game', data }, [data.buffer]);
      // Let weight updates and settings in between games.
      await yieldNow();
    }
  } catch (error) {
    send({ type: 'error', error: String(error) });
  } finally {
    busy = false;
  }
}
onmessage = async (event: MessageEvent<SelfPlayRequest>) => {
  const m = event.data;
  if (m.type === 'init') {
    await init();
    engine = new CpuEngine(m.width, m.depth, m.hidden, m.seed);
  } else if (m.type === 'weights') {
    engine?.set_weights(m.weights);
    void loop();
  } else if (m.type === 'settings') {
    ({ nodes, resultWeight, lambda, randomPlies } = m);
  } else if (m.type === 'running') {
    running = m.running;
    void loop();
  }
};
