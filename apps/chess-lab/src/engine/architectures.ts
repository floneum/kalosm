// The value network's size: an NNUE-shaped MLP whose first layer of `width`
// neurons is the accumulator the search updates incrementally, with `depth`
// layers in all and `hidden` neurons in every layer after the first.
export interface ModelConfig {
  width: number;
  depth: number;
  hidden: number;
}
export const WIDTHS = [64, 128, 256, 512] as const;
export const DEPTHS = [2, 3, 4] as const;
export const HIDDENS = [16, 32, 64] as const;
export const DEFAULT_MODEL: ModelConfig = { width: 128, depth: 2, hidden: 32 };
export const modelKey = (m: ModelConfig) => `mlp-${m.width}-${m.depth}-${m.hidden}`;
export const modelLabel = (m: ModelConfig) => `${m.width} × ${m.depth}, hidden ${m.hidden}`;
export function validModel(m: unknown): m is ModelConfig {
  if (!m || typeof m !== 'object') return false;
  const c = m as ModelConfig;
  return (
    (WIDTHS as readonly number[]).includes(c.width) &&
    (DEPTHS as readonly number[]).includes(c.depth) &&
    (HIDDENS as readonly number[]).includes(c.hidden)
  );
}
// First layer, hidden layers, value head, and the linear input-to-value path.
export function parameterCount(m: ModelConfig): number {
  return 832 * m.width + m.hidden * m.width + (m.depth - 2) * m.hidden * m.hidden + m.hidden + 832;
}
