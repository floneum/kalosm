# Rookie — a chess engine that trains in your browser

Rookie starts knowing only the rules of chess. It plays games against itself, and a
small value network learns to score positions from what its search finds. Training
and play run entirely on the local machine.

## Run

```sh
cd apps/chess-lab
npm ci
# wasm-bindgen CLI must match Cargo.lock (currently 0.2.128).
cargo install wasm-bindgen-cli --version 0.2.128 --locked
npm run build:wasm
npm run dev -- --port 5173
```

`WASM_BINDGEN=/path/to/wasm-bindgen npm run build:wasm` can select a task-local CLI.
Requires a WebGPU browser. `npm run build` builds the site after the Wasm backend.

## How it works

Every forward and backward pass runs through Fusor.

- **Training** runs on the GPU through a Fusor `Session`: one graph holds the
  network, a bound-aware value loss, its gradients and the AdamW update. A step sets
  the input leaves, resolves the loss and the updated state, and each state leaf
  adopts the buffer its update landed in.
- **Search** runs on the CPU, natively and in WebAssembly. The same network is
  built on Fusor's CPU device as two compiled programs (`fusor::CpuProgram`):
  `update` moves one perspective's first-layer accumulator by a sparse change of the
  inputs, and `head` runs the rest of the network from an accumulator. The search
  keeps accumulators per ply and computes one only when a position below it is
  evaluated.
- **Self-play** runs on worker threads (one WebAssembly instance each, about
  `hardwareConcurrency - 2`), each playing games with the latest weights. The GPU
  worker trains on their positions as they arrive: 4 batches of 1,024 per 1,024 new
  rows, drawn from the last 65,536. A value target is the root search value with 5%
  of the game result blended in; the first six plies of a game are random.

The search is iterative-deepening alpha-beta with a transposition table, killer and
history ordering, principal-variation search, late-move reductions and null-move
pruning. Leaves extend up to three plies along moves the model's own linear values
rate as gains, and the frontier prunes moves the model says cannot reach alpha.
Against a human the engine ponders their position in 100 ms slices into a table kept
for the whole game.

The engine contains no chess knowledge beyond the rules: no piece values, no
positional terms, no capture ordering. Checkmate, stalemate, repetition, the
fifty-move rule and insufficient material are decided by the rules; everything else
is the network, so a fresh model plays close to randomly.

## Model sizes

The network is an NNUE-shaped MLP over 832 side-to-move-relative inputs (pieces by
square, castling rights, en passant file, two clocks, piece counts, a bias), with a
linear path from the inputs straight to the value.

| Setting | Choices | Default |
|---|---|---|
| First layer (the accumulator) | 64 / 128 / 256 / 512 | 128 |
| Depth (layers in all) | 2 / 3 / 4 | 2 |
| Hidden width (layers after the first) | 16 / 32 / 64 | 32 |

Each size keeps its own IndexedDB checkpoint, including optimizer state, so several
can be trained and compared. Downloads are binary `.rookie` files: a fixed header
(size, generation, training statistics) followed by the state as little-endian f32.
AdamW uses automatic warm-up, inverse-square-root decay, weight decay and gradient
clipping.

## Validation and measurement

```sh
npm test
npm run build
# CPU rules against fixtures, the CPU programs against the GPU model, and
# incremental accumulators against from-scratch ones:
node --import tsx tests/gpu-fixtures.ts
cargo run --release --manifest-path fusor-chess/Cargo.toml --bin cpu_check -- /private/tmp/rookie-gpu-fixtures.json checkpoint.bin openings.json
# Train for a fixed time, then measure agreement with a Stockfish-scored
# position set (measurement only; see `evalset`):
cargo run --release --manifest-path fusor-chess/Cargo.toml --bin learn_cpu
# Search speed, engines head to head, and against Stockfish limited by UCI_Elo:
cargo run --release --manifest-path fusor-chess/Cargo.toml --bin cpu_speed -- openings.json a.bin
cargo run --release --manifest-path fusor-chess/Cargo.toml --bin cpu_match -- openings.json 200 a.bin 300ms lmr+null b.bin 300ms lmr+null
ROOKIE_PONDER_MS=3000 cargo run --release --manifest-path fusor-chess/Cargo.toml --bin cpu_anchor -- openings.json a.bin 1800 96 300
```

Measured on an M2 Max with the default size: the search runs about 900k positions/s
natively and about 520k in WebAssembly (under Node). After 60 s of training from
random weights, playing at 300 ms per move, the model scores about 43% against
Stockfish 19 limited to UCI_Elo 1800 (about 1750; 96 games, one run).

Open `/verify.html` on the dev server to run the browser learning, checkpoint and
mate-search checks. `tests/legacy` preserves the original JavaScript learner for
comparison only; the app does not import it.
