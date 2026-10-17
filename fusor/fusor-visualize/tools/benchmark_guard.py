"""Sequential macOS benchmark runner with startup and continuous pressure checks.

This is a conservative stop policy, not an OS-enforced GPU memory quota. The
shared lock covers callers of this runner across worktrees. Direct executable
launches are not covered. Never retry a resource failure automatically.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024**3
LOCK_PATH = Path(f"/tmp/fusor-webgpu-benchmark-{os.getuid()}.lock")


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    available_bytes: int
    compressed_bytes: int
    swapout_bytes: int
    disk_free_bytes: int


def read_snapshot(directory):
    if sys.platform != "darwin":
        raise ResourceLimitError("resource monitor supports macOS only; refusing unmonitored run")
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
    page_match = re.search(r"page size of (\d+) bytes", raw)
    if page_match is None:
        raise ResourceLimitError("cannot parse memory page size")
    page_bytes = int(page_match[1])
    pages = {name.strip().strip('"'): int(value) for name, value in
             re.findall(r"^([^:\n]+):\s+(\d+)\.", raw, re.M)}
    # Inactive/speculative pages are potentially reclaimable, not a promise of
    # available GPU storage. Do not count compressor space as free memory.
    return Snapshot(
        available_bytes=sum(pages[name] for name in
                            ["Pages free", "Pages inactive", "Pages speculative"]) * page_bytes,
        compressed_bytes=pages["Pages occupied by compressor"] * page_bytes,
        swapout_bytes=pages["Swapouts"] * page_bytes,
        disk_free_bytes=shutil.disk_usage(directory).free,
    )


def check_resources(current, *, initial=None, model_bytes=0):
    if model_bytes < 0:
        raise ValueError("model size must be nonnegative")
    # Loading needs room for native weights, GPU weights/staging and the OS.
    # Runtime checks start immediately after spawn, before model initialization.
    required = max(12*GIB, 2*model_bytes + 6*GIB) if initial is None else 6*GIB
    if current.available_bytes < required:
        raise ResourceLimitError(
            f"memory headroom {current.available_bytes/GIB:.2f} GiB < {required/GIB:.2f} GiB")
    disk_floor = 8*GIB if initial is None else 4*GIB
    if current.disk_free_bytes < disk_floor:
        raise ResourceLimitError(
            f"disk headroom {current.disk_free_bytes/GIB:.2f} GiB < {disk_floor/GIB:.0f} GiB")
    if current.compressed_bytes > 2*GIB:
        raise ResourceLimitError("compressor exceeds 2 GiB; leave memory for user applications")
    if initial is not None:
        if current.compressed_bytes - initial.compressed_bytes > GIB//2:
            raise ResourceLimitError("compressor grew by more than 512 MiB during benchmark")
        if current.swapout_bytes - initial.swapout_bytes > 64*1024**2:
            raise ResourceLimitError("swapouts grew by more than 64 MiB during benchmark")


@contextmanager
def single_benchmark(path=LOCK_PATH):
    # Never unlink this file: removing a locked inode lets another process enter.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceLimitError("a guarded benchmark is already running; refusing overlap") from exc
        yield fd
    finally:
        # The child inherits the descriptor, retaining the lock if its parent
        # is killed abruptly. Normal cleanup reaps the child before closing it.
        os.close(fd)


@contextmanager
def cleanup_on_signals():
    def interrupted(signum, _frame):
        raise InterruptedError(f"benchmark interrupted by signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGTERM, signal.SIGHUP]}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_child(child):
    # start_new_session gives this command its own group; never signal any
    # process discovered by name, or any unrelated application.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=2)


def run_guarded(command, *, cwd, env=None, stdout=None, stderr=None, timeout=600,
                model_bytes=0, report_path=None, lock_path=LOCK_PATH, poll_seconds=.5,
                record_child_rss=False):
    """Run one process, abort/reap on pressure, error, timeout or interruption."""
    if timeout <= 0 or not 0 < poll_seconds <= 1:
        raise ValueError("positive timeout and polling interval <= 1 second required")
    if stdout == subprocess.PIPE or stderr == subprocess.PIPE:
        raise ValueError("stream output to files; pipes can block a monitored child")
    child = None
    report = None
    started = time.monotonic()

    def record(event, **values):
        if report:
            report.write(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started,
                                         **values)) + "\n")
            report.flush()

    with single_benchmark(lock_path) as lock_fd, cleanup_on_signals():
        try:
            if report_path is not None:
                report = Path(report_path).open("a")
            initial = read_snapshot(cwd)
            record("preflight", snapshot=asdict(initial), command=list(map(str, command)),
                   model_bytes=model_bytes)
            check_resources(initial, model_bytes=model_bytes)
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True, pass_fds=(lock_fd,))
            record("started", pid=child.pid)
            while True:
                current = read_snapshot(cwd)
                child_rss = None
                if record_child_rss and child.poll() is None:
                    sample = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            timeout=2)
                    if sample.returncode == 0 and sample.stdout.strip().isdigit():
                        child_rss = int(sample.stdout.strip()) * 1024
                    elif child.poll() is None:
                        raise ResourceLimitError("cannot sample owned benchmark process RSS")
                record("sample", snapshot=asdict(current), child_rss_bytes=child_rss)
                check_resources(current, initial=initial)
                remaining = timeout - (time.monotonic()-started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = child.wait(timeout=min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            # Catch pressure that arose during the last poll interval as well.
            check_resources(read_snapshot(cwd), initial=initial)
            if code:
                raise subprocess.CalledProcessError(code, command)
            record("complete", returncode=code)
            return subprocess.CompletedProcess(command, code)
        except BaseException as exc:
            record("aborted", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            # Cleanup precedes releasing the cross-worktree lock on every exit.
            if child is not None:
                stop_child(child)
            if report:
                report.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only benchmark resource preflight; starts no workload.")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    snapshot = read_snapshot(Path.cwd())
    print(json.dumps(asdict(snapshot), indent=2))
    try:
        check_resources(snapshot, model_bytes=args.model.stat().st_size if args.model else 0)
    except ResourceLimitError as error:
        parser.exit(1, f"REFUSED: {error}\n")
