import init, { ChessGpu } from './wasm/rookie_fusor';
import { INPUT } from './engine/features';
import { Position, result } from './engine/chess';
import { encodeHuman, decodeAnswer } from './engine/gpu-state';
const report = document.querySelector('#report')!;
function log(text: string) {
  report.textContent += '\n' + text;
}
function check(ok: boolean, message: string) {
  if (!ok) throw new Error(message);
  log('PASS · ' + message);
}
try {
  await init();
  const gpu = await ChessGpu.create(128, 2, 32),
    BATCH = gpu.batch();
  log(gpu.info());
  const x = new Float32Array(BATCH * INPUT),
    values = new Float32Array(BATCH);
  for (let i = 0; i < BATCH; i++) {
    const c = i % 2;
    x[i * INPUT + c] = 1;
    x[i * INPUT + 831] = 1;
    x[i * INPUT + 16 + (i % 32)] = 0.5;
    values[i] = c === 0 ? 0.35 : -0.35;
  }
  const held = x.slice();
  for (let i = 0; i < BATCH; i++) {
    held[i * INPUT + 16 + (i % 32)] = 0;
    held[i * INPUT + 64 + (i % 32)] = 0.5;
  }
  // Sign agreement with the held-out class, and value error.
  const grade = (p: Float32Array) => {
    let hits = 0,
      mse = 0;
    for (let i = 0; i < BATCH; i++) {
      if (p[i] > 0 === values[i] > 0) hits++;
      mse += (p[i] - values[i]) ** 2;
    }
    return { hits, mse: mse / BATCH };
  };
  const before = grade(await gpu.predict(held));
  for (let i = 0; i < 4; i++) await gpu.train(x, values, 32);
  await gpu.losses();
  const start = performance.now();
  for (let i = 0; i < 8; i++) await gpu.train(x, values, 32);
  const loss = await gpu.losses(),
    seconds = (performance.now() - start) / 1000;
  const afterPrediction = await gpu.predict(held),
    after = grade(afterPrediction);
  log(
    `Training: ${Math.round((256 * BATCH) / seconds).toLocaleString()} positions/sec, ${seconds.toFixed(3)}s for 256 complete AdamW steps (GPU completion included).`,
  );
  log(
    `Held-out synthetic pattern: ${before.hits}/${BATCH} → ${after.hits}/${BATCH}; value MSE ${before.mse.toFixed(6)} → ${after.mse.toFixed(6)}. This tests graph learning, not chess strength.`,
  );
  check(
    after.hits >= BATCH * 0.9 && after.mse < before.mse * 0.2,
    'the network learns unseen pattern variants',
  );
  check(loss.every(Number.isFinite), 'training loss remains finite');
  const state = await gpu.save();
  await gpu.train(x, values, 1);
  gpu.load(state);
  const restored = await gpu.predict(held);
  check(
    restored.every((v, i) => Math.abs(v - afterPrediction[i]) < 1e-5),
    'GPU weight publication and full AdamW checkpoint restore',
  );
  const mate = new Position('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
  const answer = decodeAnswer(
    await gpu.play(encodeHuman(mate.fen(), [mate.fen()], 1), 192),
    mate.turn,
  );
  check(!!answer.move, 'the search produces a legal move');
  mate.push(answer.move!);
  check(result(mate, [])?.reason === 'Checkmate', 'the search finds mate');
  log('ALL CHECKS PASSED');
} catch (error) {
  log('FAILED · ' + String(error));
}
