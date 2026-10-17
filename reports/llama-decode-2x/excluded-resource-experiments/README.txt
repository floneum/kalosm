Excluded after memory exhaustion, 2026-10-11

The user killed the benchmark processes after the two-resident-model
interleaved A/A experiment exhausted physical memory, compressed roughly
23 GiB, and drove heavy swapping. Free memory fell to about 64 MiB and
disk space to about 117 MiB. No valid timing comparison was produced.

Serializing model loading and GPU execution did not prevent simultaneous
model residency. The old VM check ran only before measured tokens, too late
to protect loading. This design is rejected for the limited-memory machine.

interleaved_wall.py.txt and pair_qmatmul.py.txt preserve the excluded source.
Their original entry points now exit immediately without launching anything.
profile-stepped.bin.disabled is the preserved historical executable, with
execution permission removed. Its full-model stdin control was removed from
active source. Other historical executable copies are not rebuilt this turn.

Full-model wall_ab.py, probe.py and model_ab.py now use the shared sequential
benchmark_guard.py with startup checks, a cross-worktree lock, continuous
memory/disk/swap checks, and cleanup of only its own child process group.
The tests use tiny Python subprocesses and simulated pressure, never a GPU
or a model. They do not constitute another real-model performance run.

No further model benchmark or build was started during this remediation.
Goal is paused. Original 208-token decode target remains below 2x; accepted
long-context and Q4-unpack results are unchanged. Do not cite this failed
experiment as performance evidence.
