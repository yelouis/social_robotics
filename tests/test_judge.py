from __future__ import annotations

from pathlib import Path
import subprocess
import pytest

from harness.items import Item, write_items
from harness.splits import make_group_split
from judge.vlm_judge import (
    DATASET_QUESTIONS,
    OllamaJudge,
    build_prompt,
    parse_prob,
    prompt_hash,
    run_judge,
)


EXPECTED_OOPS_PROMPT = (
    "You are shown 4 frames, in time order, from a video of a person doing something.\n"
    "Context: A short clip from a home video.\n"
    "Question: Is everything still going as the person intended at this point, with no accident or failure yet?\n"
    "Answer with a single integer from 0 to 100: your probability (in percent) that the answer is YES. Output only the integer."
)


# ---------------------------------------------------------------------------
# Fast tests
# ---------------------------------------------------------------------------

def test_parse_prob():
    assert parse_prob("73") == 0.73
    assert parse_prob("I think 85.") == 0.85
    assert parse_prob("100") == 1.0
    assert parse_prob("0") == 0.0
    assert parse_prob("probability: 7%") == 0.07
    assert parse_prob("none") is None
    assert parse_prob("250") is None


def test_rendered_prompt_matches_literal_expected():
    context_text = "A short clip from a home video."
    question = DATASET_QUESTIONS["oops"]
    rendered = build_prompt(context_text=context_text, question=question, n=4)
    assert rendered == EXPECTED_OOPS_PROMPT


def test_prompt_hash_changes_with_question():
    h_oops = prompt_hash(DATASET_QUESTIONS["oops"])
    h_holo = prompt_hash(DATASET_QUESTIONS["holoassist"])
    h_custom = prompt_hash("Is this person cooking?")
    assert len(h_oops) == 12
    assert len(h_holo) == 12
    assert h_oops != h_holo
    assert h_oops != h_custom


def test_payload_contains_only_prompt_and_four_images(tmp_path: Path, monkeypatch):
    # Create a small dummy video
    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    recorded_calls = []

    def mock_ollama_chat(model, prompt, image_paths=None, options=None, timeout=None, host=None, fmt=None):
        recorded_calls.append({
            "model": model,
            "prompt": prompt,
            "image_paths": image_paths,
            "options": options,
        })
        return "85"

    monkeypatch.setattr("judge.vlm_judge.ollama_chat", mock_ollama_chat)

    judge = OllamaJudge(model="test_model")
    item = Item("test:1:0", "oops", "g1", str(vid_path), [0.0, 1.0], [1.0, 2.0], 1, "A short clip.", "c")

    prob, raw, attempts, elapsed_ms = judge.judge_item(
        video_path=item.video_path,
        window=item.action_window_sec,
        context_text=item.context_text,
        question=DATASET_QUESTIONS["oops"],
    )

    assert prob == 0.85
    assert len(recorded_calls) == 1
    call = recorded_calls[0]

    # Payload checks: prompt string, 4 JPEG images, no audio, no label
    assert isinstance(call["prompt"], str)
    assert f"label={item.label}" not in call["prompt"]
    assert "label" not in call["prompt"].lower()

    assert call["image_paths"] is not None
    assert len(call["image_paths"]) == 4
    for p in call["image_paths"]:
        assert p.endswith((".jpg", ".jpeg"))
        assert not p.endswith((".wav", ".mp3", ".m4a", ".aac"))


def test_second_call_on_cached_item_makes_zero_requests(tmp_path: Path, monkeypatch):
    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    items_dir = tmp_path / "items" / "oops"
    items_dir.mkdir(parents=True, exist_ok=True)
    item = Item("oops:vid1:0", "oops", "g1", str(vid_path), [0.0, 1.0], [1.0, 2.0], 1, "clip", "c")
    write_items([item], items_dir / "items.jsonl")

    splits_dir = tmp_path / "splits"
    make_group_split([item], "oops", official={"oops:vid1:0": "test"}, path_dir=splits_dir)

    call_count = [0]

    def mock_ollama_chat(model, prompt, image_paths=None, options=None, timeout=None, host=None, fmt=None):
        call_count[0] += 1
        return "75"

    monkeypatch.setattr("judge.vlm_judge.ollama_chat", mock_ollama_chat)

    # First run: queries model once
    run_judge(
        dataset="oops",
        split="test",
        backend="ollama",
        data_root=tmp_path,
        splits_dir=splits_dir,
    )
    assert call_count[0] == 1

    # Second run: cached, makes 0 requests
    run_judge(
        dataset="oops",
        split="test",
        backend="ollama",
        data_root=tmp_path,
        splits_dir=splits_dir,
    )
    assert call_count[0] == 1


def test_three_unparseable_replies_returns_null_and_three_attempts(tmp_path: Path, monkeypatch):
    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    call_count = [0]

    def mock_ollama_chat(model, prompt, image_paths=None, options=None, timeout=None, host=None, fmt=None):
        call_count[0] += 1
        return "I am unable to answer this question."

    monkeypatch.setattr("judge.vlm_judge.ollama_chat", mock_ollama_chat)

    judge = OllamaJudge(model="test_model")
    prob, raw, attempts, elapsed_ms = judge.judge_item(
        video_path=str(vid_path),
        window=[0.0, 1.0],
        context_text="A home video.",
        question=DATASET_QUESTIONS["oops"],
    )

    assert prob is None
    assert attempts == 3
    assert call_count[0] == 3
    assert "unable to answer" in raw


def test_gemini_judge_retry_parse_success(tmp_path: Path, monkeypatch):
    """Replies 'none', 'none', '42' -> 0.42, attempts 3, and 2nd and 3rd calls had temperature=0.3."""
    from judge.vlm_judge import GeminiJudge

    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    recorded_configs = []

    class MockResp:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    responses = ["none", "none", "42"]

    class MockModels:
        def generate_content(self, model, contents, config=None):
            recorded_configs.append(config)
            return MockResp(responses.pop(0))

    class MockClient:
        def __init__(self):
            self.models = MockModels()

    judge = GeminiJudge(api_key="fake")
    judge.client = MockClient()

    prob, raw, attempts, elapsed_ms = judge.judge_item(
        video_path=str(vid_path),
        window=[0.0, 1.0],
        context_text="A home video.",
        question=DATASET_QUESTIONS["oops"],
    )

    assert prob == 0.42
    assert attempts == 3
    assert len(recorded_configs) == 3
    assert recorded_configs[0].temperature == 0
    assert recorded_configs[1].temperature == 0.3
    assert recorded_configs[2].temperature == 0.3


def test_gemini_judge_retry_parse_failure(tmp_path: Path, monkeypatch):
    """Replies 'none' x 3 -> None, attempts 3."""
    from judge.vlm_judge import GeminiJudge

    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    class MockResp:
        def __init__(self, text):
            self.text = text
            self.candidates = []

    responses = ["none", "none", "none"]

    class MockModels:
        def generate_content(self, model, contents, config=None):
            return MockResp(responses.pop(0))

    class MockClient:
        def __init__(self):
            self.models = MockModels()

    judge = GeminiJudge(api_key="fake")
    judge.client = MockClient()

    prob, raw, attempts, elapsed_ms = judge.judge_item(
        video_path=str(vid_path),
        window=[0.0, 1.0],
        context_text="A home video.",
        question=DATASET_QUESTIONS["oops"],
    )

    assert prob is None
    assert attempts == 3


def test_gemini_judge_429_exhaustion_raises(tmp_path: Path, monkeypatch):
    """Five raised 429 errors -> judge_item raises, and run_judge writes an errors.jsonl line and no cache line."""
    from judge.vlm_judge import GeminiJudge

    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    class RateLimitErr(Exception):
        pass

    class MockModels:
        def generate_content(self, model, contents, config=None):
            raise RateLimitErr("Resource has been exhausted (e.g. check quota) 429")

    class MockClient:
        def __init__(self):
            self.models = MockModels()

    monkeypatch.setattr("time.sleep", lambda s: None)

    judge = GeminiJudge(api_key="fake")
    judge.client = MockClient()

    with pytest.raises(Exception) as exc_info:
        judge.judge_item(
            video_path=str(vid_path),
            window=[0.0, 1.0],
            context_text="A home video.",
            question=DATASET_QUESTIONS["oops"],
        )
    assert "429" in str(exc_info.value)

    # Now verify run_judge writes to errors.jsonl and not to cache
    item = Item("test:1:0", "oops_test", "g1", str(vid_path), [0.0, 1.0], [1.0, 2.0], 1, "Context.", "c")
    items_dir = tmp_path / "items" / "oops_test"
    items_dir.mkdir(parents=True)
    write_items([item], items_dir / "items.jsonl")

    splits_dir = tmp_path / "splits"
    from harness.splits import make_group_split
    make_group_split([item], "oops_test", official={item.item_id: "test"}, path_dir=splits_dir)

    monkeypatch.setattr("judge.vlm_judge.GeminiJudge", lambda *a, **kw: judge)

    run_judge(
        dataset="oops_test",
        split="test",
        backend="gemini",
        limit=1,
        data_root=tmp_path,
        splits_dir=splits_dir,
        question_override=DATASET_QUESTIONS["oops"],
    )

    judge_dir = tmp_path / "judge" / "oops_test" / "gemini-3.6-flash"
    assert judge_dir.exists()
    err_file = judge_dir / "errors.jsonl"
    assert err_file.exists()
    err_content = err_file.read_text(encoding="utf-8")
    assert "429" in err_content
    assert item.item_id in err_content

    # No successful answer cached
    cache_files = [f for f in judge_dir.glob("*.jsonl") if f.name not in ("errors.jsonl", "progress.json")]
    for cf in cache_files:
        assert cf.stat().st_size == 0


def test_payload_large_judge_and_qwen_judge_unchanged(tmp_path: Path, monkeypatch):
    from judge.vlm_judge import make_large_judge

    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    recorded_calls = []

    def mock_ollama_chat(model, prompt, image_paths=None, options=None, timeout=None, host=None, fmt=None, think=None, keep_alive=None):
        recorded_calls.append({
            "model": model,
            "prompt": prompt,
            "image_paths": image_paths,
            "options": options,
            "think": think,
            "keep_alive": keep_alive,
        })
        return "85"

    monkeypatch.setattr("judge.vlm_judge.ollama_chat", mock_ollama_chat)
    monkeypatch.setattr("shared.memguard.is_model_loaded", lambda m, endpoint=None: True)

    # 1. Test large judge payload
    large_judge = make_large_judge()
    assert large_judge.model == "gemma4:26b"
    assert large_judge.options == {"temperature": 0, "num_ctx": 16384, "num_predict": 16}
    assert large_judge.think is False
    assert large_judge.keep_alive == "15m"
    assert large_judge.load_step == "sr_judge_large_load"

    prob, raw, attempts, elapsed_ms = large_judge.judge_item(
        video_path=str(vid_path),
        window=[0.0, 1.0],
        context_text="A short clip.",
        question=DATASET_QUESTIONS["oops"],
    )

    assert prob == 0.85
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["model"] == "gemma4:26b"
    assert call["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 16}
    assert call["think"] is False
    assert call["keep_alive"] == "15m"
    assert len(call["image_paths"]) == 4

    # 2. Test qwen judge payload (neither think nor keep_alive)
    recorded_calls.clear()
    qwen_judge = OllamaJudge()
    prob_q, raw_q, attempts_q, elapsed_q = qwen_judge.judge_item(
        video_path=str(vid_path),
        window=[0.0, 1.0],
        context_text="A short clip.",
        question=DATASET_QUESTIONS["oops"],
    )
    assert prob_q == 0.85
    assert len(recorded_calls) == 1
    q_call = recorded_calls[0]
    assert q_call["think"] is None
    assert q_call["keep_alive"] is None
    assert q_call["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 16}


def test_run_judge_ollama_large_split_train_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="judge-large is test split only"):
        run_judge(
            dataset="oops_test",
            split="train",
            backend="ollama-large",
            data_root=tmp_path,
        )


def test_run_judge_ollama_large_never_unloads_shared_model(tmp_path: Path, monkeypatch):
    from shared.memguard import MemoryDeferred

    vid_path = tmp_path / "dummy.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    items = [
        Item("test:1:0", "oops_test", "g1", str(vid_path), [0.0, 1.0], [1.0, 2.0], 1, "Context 1.", "c"),
        Item("test:2:0", "oops_test", "g2", str(vid_path), [0.0, 1.0], [1.0, 2.0], 0, "Context 2.", "c"),
    ]
    items_dir = tmp_path / "items" / "oops_test"
    items_dir.mkdir(parents=True)
    write_items(items, items_dir / "items.jsonl")

    splits_dir = tmp_path / "splits"
    make_group_split(items, "oops_test", official={it.item_id: "test" for it in items}, path_dir=splits_dir)

    http_calls = []

    def fake_post(url, *args, **kwargs):
        http_calls.append({"method": "POST", "url": url, "json": kwargs.get("json")})
        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "50"}}
        return FakeResp()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("shared.memguard.is_model_loaded", lambda m, endpoint=None: True)

    read_count = [0]
    def fake_read_memory():
        read_count[0] += 1
        if read_count[0] <= 1:
            return 32 * (1024**3), 1  # normal
        return 32 * (1024**3), 2  # warning before second item

    monkeypatch.setattr("shared.memguard.read_memory", fake_read_memory)

    with pytest.raises(MemoryDeferred):
        run_judge(
            dataset="oops_test",
            split="test",
            backend="ollama-large",
            data_root=tmp_path,
            splits_dir=splits_dir,
            question_override=DATASET_QUESTIONS["oops"],
        )

    # Recorded HTTP calls must contain NO /api/generate and NO keep_alive: 0
    for call in http_calls:
        assert "/api/generate" not in call["url"], f"Unexpected call to /api/generate: {call}"
        payload = call.get("json") or {}
        assert payload.get("keep_alive") != 0, f"Unexpected keep_alive: 0: {call}"


# ---------------------------------------------------------------------------
# Slow tests (live local model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_live_local_judge_synthetic_items(tmp_path_factory):
    fn_dir = tmp_path_factory.mktemp("judge_live")
    vid_path = fn_dir / "test_live.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    judge = OllamaJudge()
    # 4 synthetic windows
    windows = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    for idx, win in enumerate(windows):
        prob, raw, attempts, elapsed_ms = judge.judge_item(
            video_path=str(vid_path),
            window=win,
            context_text="A synthetic test pattern video.",
            question=DATASET_QUESTIONS["oops"],
        )
        assert prob is not None, f"Item {idx} returned None probability. Raw: {raw}"
        assert 0.0 <= prob <= 1.0
        assert attempts >= 1


@pytest.mark.slow
def test_live_large_judge_synthetic_items(tmp_path_factory):
    import httpx
    from judge.vlm_judge import make_large_judge

    fn_dir = tmp_path_factory.mktemp("judge_large_live")
    vid_path = fn_dir / "test_live_large.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(vid_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    judge = make_large_judge()
    windows = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    for idx, win in enumerate(windows):
        prob, raw, attempts, elapsed_ms = judge.judge_item(
            video_path=str(vid_path),
            window=win,
            context_text="A synthetic test pattern video.",
            question=DATASET_QUESTIONS["oops"],
        )
        assert prob is not None, f"Item {idx} returned None probability. Raw: {raw}"
        assert 0.0 <= prob <= 1.0
        assert attempts >= 1

    # Check /api/ps lists gemma4:26b with context_length 16384
    resp = httpx.get("http://127.0.0.1:11434/api/ps", timeout=10.0)
    assert resp.status_code == 200
    data = resp.json()
    loaded_models = data.get("models", [])
    gemma_entry = next((m for m in loaded_models if "gemma4:26b" in m.get("name", "") or "gemma4:26b" in m.get("model", "")), None)
    assert gemma_entry is not None, f"gemma4:26b not found in /api/ps: {loaded_models}"
    assert gemma_entry.get("context_length") == 16384, f"Unexpected context_length: {gemma_entry}"
