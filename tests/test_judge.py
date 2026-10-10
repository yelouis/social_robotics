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
