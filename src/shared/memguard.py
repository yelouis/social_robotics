from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import httpx

from config import DATA_ROOT
from models_config import get_model

logger = logging.getLogger("social_robotics.memguard")

FLOOR: int = 8 * (1024**3)  # 8 GB floor per 03_eval_harness.md §12

HEAVY_STEPS: Dict[str, int] = {
    # Declared peaks = measured × 1.15 rounded up to GB per 03_eval_harness.md §12
    # Measured designer October 10, 2026, M4 Max (1.56 GB * 1.15 rounded up)
    "sr_siglip": 2 * (1024**3),
    # Measured designer October 10, 2026, M4 Max (4.76 GB * 1.15 rounded up)
    "sr_e2v": 6 * (1024**3),
    # Measured designer October 10, 2026, M4 Max (7.72 GB * 1.15 rounded up)
    "sr_judge_load": 9 * (1024**3),
    # Measured in A7c step 6: 28.10 GB RSS on 50 items (drop 24.96 GB; 28.10 × 1.15 = 32.32 GB)
    "sr_judge_large_load": 33 * (1024**3),
}

DEFAULT_MEM_WAIT_S: float = 1800.0

_HELD_DEPTH: int = 0


class MemoryDeferred(Exception):
    """Raised when memory guard cannot admit a heavy step within the time limit or stops it."""

    pass


def read_memory() -> Tuple[int, int]:
    """Read available memory in bytes and pressure level from macOS kernel.

    Formula per 03_eval_harness.md §12:
    available = hw.memsize * kern.memorystatus_level / 100
    pressure = kern.memorystatus_vm_pressure_level (1 normal, 2 warning, 4 critical)
    """
    fake_override = os.environ.get("SR_MEMGUARD_FAKE_AVAILABLE_GB")
    if fake_override is not None:
        try:
            fake_gb = float(fake_override)
            sys.stderr.write(f"memory guard: FAKE MEMORY READING ({fake_gb:.1f} GB)\n")
            sys.stderr.flush()
            return int(fake_gb * (1024**3)), 1
        except ValueError:
            pass

    try:
        out = subprocess.check_output(
            [
                "sysctl",
                "-n",
                "hw.memsize",
                "kern.memorystatus_level",
                "kern.memorystatus_vm_pressure_level",
            ],
            text=True,
        ).split()
        if len(out) < 3:
            raise ValueError(f"expected >= 3 fields from sysctl, got {len(out)}: {out}")
        memsize = int(out[0])
        level = int(out[1])
        pressure = int(out[2])
        available_bytes = int(memsize * level / 100)
        return available_bytes, pressure
    except Exception as e:
        sys.stderr.write(f"memory guard: READ FAILED ({e})\n")
        sys.stderr.flush()
        return 0, 4



def lock_dir() -> Path:
    """Return directory for machine-wide heavy lock shared with animated_infographics."""
    env_dir = os.environ.get("INFOGRAPHICS_LOCK_DIR")
    if env_dir:
        p = Path(env_dir)
    else:
        p = Path.home() / ".cache" / "animated_infographics" / "locks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_heavy_lock_holder() -> Tuple[Optional[int], Optional[str]]:
    """Read holder PID and step of heavy.lock if currently held, else (None, None)."""
    lock_file = lock_dir() / "heavy.lock"
    if not lock_file.exists():
        return None, None
    try:
        with open(lock_file, "r+", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return None, None
            except (BlockingIOError, OSError):
                content = f.read().strip()
                parts = content.split()
                step = parts[0] if parts else None
                pid = None
                if "pid" in parts:
                    idx = parts.index("pid")
                    if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        pid = int(parts[idx + 1])
                return pid, step
    except Exception:
        return None, None


def log_event(
    step: str,
    action: str,
    waited_ms: int,
    available_gb: float,
    pressure: int,
    data_root: Optional[Union[str, Path]] = None,
) -> None:
    """Appends one event line to DATA_ROOT/runs/memguard.log per §12."""
    root = Path(data_root or DATA_ROOT)
    log_path = root / "runs" / "memguard.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"ts={ts} pid={os.getpid()} step={step} action={action} "
        f"waited_ms={waited_ms} available_gb={available_gb:.1f} pressure={pressure}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def is_model_loaded(model: str, endpoint: Optional[str] = None) -> bool:
    """Check if a model is loaded in Ollama via GET /api/ps."""
    base = endpoint or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        resp = httpx.get(f"{base}/api/ps", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for m in models:
                name = m.get("name") or m.get("model") or ""
                if name == model or name.startswith(f"{model}:") or model.startswith(f"{name}:"):
                    return True
    except Exception:
        pass
    return False


def is_our_judge_loaded(endpoint: Optional[str] = None) -> bool:
    """Check if our judge model is loaded in Ollama via GET /api/ps."""
    return is_model_loaded(get_model("vlm_judge"), endpoint=endpoint)


def unload_own_judge(endpoint: Optional[str] = None, timeout_s: float = 30.0) -> bool:
    """Unloads ONLY get_model('vlm_judge') from Ollama per §12. Never touches other models."""
    our_model = get_model("vlm_judge")
    base = endpoint or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        resp = httpx.post(
            f"{base}/api/generate",
            json={"model": our_model, "keep_alive": 0},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return False
    except Exception:
        return False

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not is_our_judge_loaded(endpoint=base):
            waited_ms = int((time.monotonic() - t0) * 1000)
            avail, press = read_memory()
            log_event("sr_judge_load", "unload", waited_ms, avail / (1024**3), press)
            return True
        time.sleep(0.5)
    return False


def _log_waiting_line(
    step: str,
    peak_gb: int,
    available: int,
    pressure: int,
    holder_pid: Optional[int],
    last_log_time: float,
) -> float:
    now = time.monotonic()
    if last_log_time == 0.0 or now - last_log_time >= 30.0:
        avail_gb = available / (1024**3)
        holder_str = f"held by pid {holder_pid}" if holder_pid is not None else "free"
        msg = (
            f"memory guard: waiting for {step}: need {peak_gb} GB + floor 8 GB, "
            f"available {avail_gb:.1f} GB, pressure {pressure}, heavy lock {holder_str}"
        )
        print(msg, flush=True)
        logger.info(msg)
        return now
    return last_log_time


def _handle_timeout(step: str, peak_gb: int, timeout_s: float, t_start: float) -> None:
    available, pressure = read_memory()
    avail_gb = available / (1024**3)
    waited_ms = int((time.monotonic() - t_start) * 1000)
    log_event(step, "deferred", waited_ms, avail_gb, pressure)
    raise MemoryDeferred(
        f"memory guard: timed out waiting {timeout_s:.0f}s for {step}: "
        f"need {peak_gb} GB + floor 8 GB, available {avail_gb:.1f} GB"
    )


@contextmanager
def guard(step: str, timeout_s: Optional[float] = None) -> Iterator[None]:
    """Admission context manager implementing 03_eval_harness.md §12 steps 1-6."""
    global _HELD_DEPTH
    if timeout_s is None:
        if "SR_MEM_WAIT_S" in os.environ:
            timeout_s = float(os.environ["SR_MEM_WAIT_S"])
        elif os.environ.get("SR_MEMGUARD_FAKE_AVAILABLE_GB") is not None:
            timeout_s = 3.0
        else:
            timeout_s = float(DEFAULT_MEM_WAIT_S)
    peak = HEAVY_STEPS.get(step, 0)
    peak_gb = int(round(peak / (1024**3)))
    t_start = time.monotonic()

    # Nested path: already hold the heavy lock in this process
    if _HELD_DEPTH > 0:
        last_log_time = 0.0
        waited = False
        while time.monotonic() - t_start < timeout_s:
            available, pressure = read_memory()
            if available - peak >= FLOOR:
                if not waited:
                    waited_ms = 0
                else:
                    waited_ms = int((time.monotonic() - t_start) * 1000)
                avail_gb = available / (1024**3)
                log_event(step, "admit", waited_ms, avail_gb, pressure)
                _HELD_DEPTH += 1
                try:
                    yield
                    return
                finally:
                    _HELD_DEPTH -= 1

            waited = True
            if step != "sr_judge_load" and is_our_judge_loaded():
                unload_own_judge()
                available, pressure = read_memory()
                if available - peak >= FLOOR:
                    continue

            last_log_time = _log_waiting_line(
                step=step,
                peak_gb=peak_gb,
                available=available,
                pressure=pressure,
                holder_pid=os.getpid(),
                last_log_time=last_log_time,
            )
            time.sleep(min(5.0, max(0.1, timeout_s - (time.monotonic() - t_start))))

        _handle_timeout(step, peak_gb, timeout_s, t_start)
        return

    # Outer path: acquire the heavy lock
    lock_file = lock_dir() / "heavy.lock"
    lock_fd = open(lock_file, "a+", encoding="utf-8")
    acquired_lock = False
    last_log_time = 0.0
    waited = False

    try:
        while time.monotonic() - t_start < timeout_s:
            holder_pid, _ = get_heavy_lock_holder()
            available, pressure = read_memory()

            if holder_pid is None and (available - peak >= FLOOR):
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired_lock = True
                    lock_fd.seek(0)
                    lock_fd.truncate()
                    lock_fd.write(f"{step} pid {os.getpid()}\n")
                    lock_fd.flush()

                    available, pressure = read_memory()
                    if available - peak >= FLOOR:
                        if not waited:
                            waited_ms = 0
                        else:
                            waited_ms = int((time.monotonic() - t_start) * 1000)
                        avail_gb = available / (1024**3)
                        log_event(step, "admit", waited_ms, avail_gb, pressure)
                        _HELD_DEPTH += 1
                        try:
                            yield
                            return
                        finally:
                            _HELD_DEPTH -= 1
                    else:
                        lock_fd.seek(0)
                        lock_fd.truncate()
                        lock_fd.flush()
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                        acquired_lock = False
                except (BlockingIOError, OSError):
                    acquired_lock = False

            waited = True

            if step != "sr_judge_load" and is_our_judge_loaded():
                unload_own_judge()
                available, pressure = read_memory()
                if available - peak >= FLOOR:
                    continue

            last_log_time = _log_waiting_line(
                step=step,
                peak_gb=peak_gb,
                available=available,
                pressure=pressure,
                holder_pid=holder_pid,
                last_log_time=last_log_time,
            )

            if holder_pid is not None:
                time.sleep(min(0.5, max(0.05, timeout_s - (time.monotonic() - t_start))))
            else:
                time.sleep(min(5.0, max(0.1, timeout_s - (time.monotonic() - t_start))))

        _handle_timeout(step, peak_gb, timeout_s, t_start)
    finally:
        if acquired_lock:
            try:
                lock_fd.seek(0)
                lock_fd.truncate()
                lock_fd.flush()
            except Exception:
                pass
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            lock_fd.close()
        except Exception:
            pass


def check(
    step: str,
    release: Callable[[], None],
    reload: Callable[[], None],
) -> None:
    """Between-items check per 03_eval_harness.md §12."""
    available, pressure = read_memory()
    if pressure == 4 or available < (FLOOR // 2):
        release()
        log_event(step, "stop", 0, available / (1024**3), pressure)
        raise MemoryDeferred(
            f"memory guard: critical pressure level ({pressure}) or available memory "
            f"({available / (1024**3):.1f} GB < {FLOOR / (1024**3) / 2:.1f} GB)"
        )
    elif pressure == 2 or available < FLOOR:
        log_event(step, "pause", 0, available / (1024**3), pressure)
        release()
        with guard(step):
            reload()
        avail_after, press_after = read_memory()
        log_event(step, "resume", 0, avail_after / (1024**3), press_after)


def check_shared(
    step: str,
    data_root: Optional[Union[str, Path]] = None,
) -> None:
    """Between-items check for shared models per 03_eval_harness.md §8 and §12.

    On warning or critical pressure/memory, logs action=stop and raises MemoryDeferred.
    Never releases or unloads anything.
    """
    available, pressure = read_memory()
    if pressure >= 2 or available < FLOOR:
        log_event(step, "stop", 0, available / (1024**3), pressure, data_root=data_root)
        raise MemoryDeferred(
            f"memory guard: shared step '{step}' stopped due to memory pressure ({pressure}) "
            f"or available memory ({available / (1024**3):.1f} GB < {FLOOR / (1024**3):.1f} GB)"
        )


def status() -> None:
    """Prints memory guard status per §12."""
    available, pressure = read_memory()
    avail_gb = available / (1024**3)
    holder_pid, holder_step = get_heavy_lock_holder()
    if holder_pid is not None:
        lock_status = f"held by pid {holder_pid} (step: {holder_step or 'unknown'})"
    else:
        lock_status = "free"

    our_model = get_model("vlm_judge")
    print(f"Available memory: {avail_gb:.1f} GB (pressure level: {pressure})")
    print(f"Heavy lock: {lock_status}")
    print("Ollama models:")

    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        resp = httpx.get(f"{base}/api/ps", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if not models:
                print("  (none loaded)")
            for m in models:
                name = m.get("name") or m.get("model") or "unknown"
                size = m.get("size")
                size_str = f"{size / (1024**3):.1f} GB" if isinstance(size, (int, float)) else "unknown size"
                marker = (
                    " * (ours)"
                    if (name == our_model or name.startswith(f"{our_model}:") or our_model.startswith(f"{name}:"))
                    else ""
                )
                print(f"  - {name} ({size_str}){marker}")
        else:
            print(f"  (error querying /api/ps: status {resp.status_code})")
    except Exception as e:
        print(f"  (could not connect to Ollama: {e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory guard status and management")
    parser.add_argument("--status", action="store_true", help="Print current memory guard status")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
