import { Position } from '../../src/engine/chess';
import { TinyModel } from './model';
import { search } from './search';

self.onmessage = (
  event: MessageEvent<{
    id: number;
    fen: string;
    keys: number[];
    weights: number[];
    simulations: number;
  }>,
) => {
  const { id, fen, keys, weights, simulations } = event.data;
  const started = performance.now();
  const result = search(new Position(fen), new TinyModel(weights), keys, {
    simulations,
    deadline: started + 1400,
  });
  postMessage({ id, result, elapsed: performance.now() - started });
};
