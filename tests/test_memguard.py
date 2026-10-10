import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List

import pytest

from shared.memguard import (
    MemoryDeferred,
    check,
    get_heavy_lock_holder,
    guard,
    lock_dir,
    read_memory,
    unload_own_judge,
)


# ---------------------------------------------------------------------------
# Admission & Lock tests
# ---------------------------------------------------------------------------

def test_memguard_h_default_lock_path(monkeypatch):
    """(h) Assert the lock-path string with INFOGRAPHICS_LOCK_DIR unset equals

    os.path.expanduser('~/.cache/animated_infographics/locks/heavy.lock').
    """
    monkeypatch.delenv("INFOGRAPHICS_LOCK_DIR", raising=False)
    expected = Path(os.path.expanduser("~/.cache/animated_infographics/locks/heavy.lock"))
    assert lock_dir() / "heavy.lock" == expected


def test_memguard_a_admits_at_once(tmp_path: Path, monkeypatch):
    """(a) Admits at once when available - peak >= 8 GB."""
    lock_d = tmp_path / "locks"
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)

    # 32 GB available, pressure 1
    monkeypatch.setattr("shared.memguard.read_memory", lambda: (32 * 1024**3, 1))

    admitted = False
    with guard("sr_siglip"):
        admitted = True

    assert admitted

    # Check memguard.log
    log_file = tmp_path / "runs" / "memguard.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "step=sr_siglip" in content
    assert "action=admit" in content
    assert "waited_ms=0" in content


def test_memguard_b_waits_and_admits(tmp_path: Path, monkeypatch, capsys):
    """(b) With a fake reader that rises from 10 -> 20 GB, waits and then admits sr_e2v.

    The waiting log line matches §12 verbatim.
    """
    lock_d = tmp_path / "locks"
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)

    # Reader returns 10 GB on first call, then 20 GB
    call_count = [0]

    def fake_read():
        call_count[0] += 1
        if call_count[0] == 1:
            return (10 * 1024**3, 1)
        return (20 * 1024**3, 1)

    monkeypatch.setattr("shared.memguard.read_memory", fake_read)
    # Fast sleep
    monkeypatch.setattr("time.sleep", lambda s: None)

    admitted = False
    with guard("sr_e2v", timeout_s=5.0):
        admitted = True

    assert admitted
    captured = capsys.readouterr()
    expected_line = (
        "memory guard: waiting for sr_e2v: need 6 GB + floor 8 GB, available 10.0 GB, "
        "pressure 1, heavy lock free"
    )
    assert expected_line in captured.out


def test_memguard_c_timeout_raises_memory_deferred(tmp_path: Path, monkeypatch):
    """(c) With SR_MEM_WAIT_S=1 and memory never sufficient, raises MemoryDeferred within about 1-2 s."""
    lock_d = tmp_path / "locks"
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)
    monkeypatch.setenv("SR_MEM_WAIT_S", "1")

    # Only 5 GB available: need 6 + 8 = 14 GB for sr_e2v
    monkeypatch.setattr("shared.memguard.read_memory", lambda: (5 * 1024**3, 1))

    t0 = time.time()
    with pytest.raises(MemoryDeferred) as exc_info:
        with guard("sr_e2v"):
            pass
    elapsed = time.time() - t0

    assert 0.9 <= elapsed <= 3.0
    assert "timed out waiting" in str(exc_info.value)

    # Log should contain action=deferred
    log_file = tmp_path / "runs" / "memguard.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "step=sr_e2v" in content
    assert "action=deferred" in content


# ---------------------------------------------------------------------------
# Unload own model tests
# ---------------------------------------------------------------------------

def test_memguard_d_unloads_only_our_model(tmp_path: Path, monkeypatch):
    """(d) If /api/ps lists both qwen2.5vl:7b and gemma4:26b, admission for sr_e2v unloads ONLY qwen2.5vl:7b."""
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)

    posted_models: List[str] = []

    class MockResponse:
        def __init__(self, status_code: int, json_data: dict):
            self.status_code = status_code
            self._json = json_data

        def json(self):
            return self._json

    ps_calls = [0]

    def mock_get(url, timeout=5.0):
        ps_calls[0] += 1
        # First call: both models loaded. Subsequent calls: only gemma4 loaded
        if ps_calls[0] == 1:
            models = [{"name": "qwen2.5vl:7b"}, {"name": "gemma4:26b"}]
        else:
            models = [{"name": "gemma4:26b"}]
        return MockResponse(200, {"models": models})

    def mock_post(url, json=None, timeout=10.0, **kwargs):
        payload = json or kwargs.get("json")
        if payload and "model" in payload:
            posted_models.append(payload["model"])
        return MockResponse(200, {})

    monkeypatch.setattr("httpx.get", mock_get)
    monkeypatch.setattr("httpx.post", mock_post)

    success = unload_own_judge(endpoint="http://mock-ollama:11434")
    assert success
    assert "qwen2.5vl:7b" in posted_models
    assert "gemma4:26b" not in posted_models
    assert len(posted_models) == 1


# ---------------------------------------------------------------------------
# Between-items checks
# ---------------------------------------------------------------------------

def test_memguard_e_between_items_warning(tmp_path: Path, monkeypatch):
    """(e) A warning reading calls release(), re-admits, then calls reload()."""
    lock_d = tmp_path / "locks"
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)

    calls = {"release": 0, "reload": 0}

    def mock_release():
        calls["release"] += 1

    def mock_reload():
        calls["reload"] += 1

    # Warning: pressure == 2; during re-admission inside guard(), provide ample memory
    read_count = [0]

    def mock_read():
        read_count[0] += 1
        if read_count[0] == 1:
            return (6 * 1024**3, 2)  # Warning pressure
        return (32 * 1024**3, 1)  # Plentiful memory on re-admission

    monkeypatch.setattr("shared.memguard.read_memory", mock_read)

    check("sr_siglip", release=mock_release, reload=mock_reload)

    assert calls["release"] == 1
    assert calls["reload"] == 1

    log_file = tmp_path / "runs" / "memguard.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "action=pause" in content
    assert "action=resume" in content


def test_memguard_f_between_items_critical(tmp_path: Path, monkeypatch):
    """(f) A critical reading calls release() and raises MemoryDeferred."""
    lock_d = tmp_path / "locks"
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)

    calls = {"release": 0, "reload": 0}

    def mock_release():
        calls["release"] += 1

    def mock_reload():
        calls["reload"] += 1

    # Critical: pressure == 4
    monkeypatch.setattr("shared.memguard.read_memory", lambda: (2 * 1024**3, 4))

    with pytest.raises(MemoryDeferred):
        check("sr_siglip", release=mock_release, reload=mock_reload)

    assert calls["release"] == 1
    assert calls["reload"] == 0

    log_file = tmp_path / "runs" / "memguard.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "action=stop" in content


# ---------------------------------------------------------------------------
# Lock sharing with child process
# ---------------------------------------------------------------------------

def test_memguard_g_lock_shared_with_child_process(tmp_path: Path, monkeypatch):
    """(g) A child process holds <lock dir>/heavy.lock through plain fcntl.flock.

    guard() blocks until the child exits.
    """
    lock_d = tmp_path / "locks"
    lock_d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INFOGRAPHICS_LOCK_DIR", str(lock_d))
    monkeypatch.setattr("shared.memguard.DATA_ROOT", tmp_path)
    monkeypatch.setattr("shared.memguard.read_memory", lambda: (32 * 1024**3, 1))

    lock_file = lock_d / "heavy.lock"

    # Spawn child process that holds the lock for 1 second
    child_code = f"""
import fcntl, time, os
fd = open({repr(str(lock_file))}, "a+")
fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
fd.seek(0)
fd.truncate()
fd.write(f"test_child pid {{os.getpid()}}\\n")
fd.flush()
time.sleep(1.0)
fd.close()
"""
    p = subprocess.Popen([sys.executable, "-c", child_code])
    # Give child a moment to acquire lock
    time.sleep(0.2)

    # Holder should be visible
    holder_pid, holder_step = get_heavy_lock_holder()
    assert holder_pid == p.pid
    assert holder_step == "test_child"

    # Parent enters guard: must block until child exits
    t0 = time.time()
    with guard("sr_siglip", timeout_s=5.0):
        t_admitted = time.time()

    p.wait()
    assert t_admitted - t0 >= 0.7  # waited for child to release


# ---------------------------------------------------------------------------
# Fake memory override & CLI drill
# ---------------------------------------------------------------------------

def test_fake_memory_reading_override(monkeypatch, capsys):
    monkeypatch.setenv("SR_MEMGUARD_FAKE_AVAILABLE_GB", "2.5")
    avail, press = read_memory()
    assert avail == int(2.5 * 1024**3)
    assert press == 1
    captured = capsys.readouterr()
    assert "FAKE MEMORY READING (2.5 GB)" in captured.err


def test_supervisor_handling(tmp_path: Path):
    """Supervisor drill:

    - exits 75 three times and then 0 finishes DONE with SR_MEMWAIT_SLEEP_S=1
    - exits 1 twice with no progress aborts.
    """
    sup_script = Path("tools/run_supervised.sh").resolve()
    progress_file = tmp_path / "progress.json"
    progress_file.write_text("[]", encoding="utf-8")

    # Script that exits 75 three times, then writes 1 item and exits 0
    counter_file = tmp_path / "counter.txt"
    runner_code = f"""
import json, sys
counter_path = {repr(str(counter_file))}
p_path = {repr(str(progress_file))}
c = 0
try:
    c = int(open(counter_path).read().strip())
except Exception:
    pass
c += 1
open(counter_path, "w").write(str(c))
if c <= 3:
    sys.exit(75)
else:
    json.dump(["item1"], open(p_path, "w"))
    sys.exit(0)
"""
    log_file = tmp_path / "test_sup.log"
    env = dict(os.environ)
    env["SR_MEMWAIT_SLEEP_S"] = "1"
    env["SR_SUPERVISE_LOG"] = str(log_file)

    res = subprocess.run(
        [str(sup_script), str(progress_file), sys.executable, "-c", runner_code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "runner exited 0 (clean)" in res.stderr or "DONE" in res.stderr
    assert "runner deferred by memory guard (exit 75, deferral 3/" in res.stderr

    # Script that exits 1 twice with no progress -> aborts with exit 1
    fail_runner = "import sys; sys.exit(1)"
    res_fail = subprocess.run(
        [str(sup_script), str(progress_file), sys.executable, "-c", fail_runner],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res_fail.returncode == 1
    assert "ABORT: 2 consecutive relaunches with zero new records" in res_fail.stderr


# ---------------------------------------------------------------------------
# Slow footprint test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_memguard_release_footprint():
    """Release is real (slow): after FrameEncoder + NonverbalAudioEncoder load, encode

    and release(), the process footprint is within 1.5 GB of its pre-load value.
    """
    import gc
    import psutil
    from features.audio import NonverbalAudioEncoder
    from features.visual import FrameEncoder
    from PIL import Image

    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    # Load and encode with FrameEncoder
    v_encoder = FrameEncoder()
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    _ = v_encoder.encode_frames([dummy_img, dummy_img])
    v_encoder.release()
    del v_encoder

    # Load and encode with NonverbalAudioEncoder
    import tempfile
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)

    try:
        a_encoder = NonverbalAudioEncoder()
        _ = a_encoder.encode_wav(wav_path)
        a_encoder.release()
        del a_encoder
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    gc.collect()
    rss_after = proc.memory_info().rss
    diff_gb = (rss_after - rss_before) / (1024**3)
    assert diff_gb < 1.5, f"Footprint grew by {diff_gb:.2f} GB after release"
