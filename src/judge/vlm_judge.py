from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from config import DATA_ROOT
from harness.items import Item, read_items
from harness.splits import load_split
from models_config import get_model
from shared import memguard
from shared.memguard import MemoryDeferred
from shared.vlm_client import ollama_chat

PROMPT_TEMPLATE = (
    "You are shown {n} frames, in time order, from a video of a person doing something.\n"
    "Context: {context_text}\n"
    "Question: {question}\n"
    "Answer with a single integer from 0 to 100: your probability (in percent) that the answer is YES. Output only the integer."
)

DATASET_QUESTIONS: Dict[str, str] = {
    "holoassist": "Did the person perform this step correctly, without a mistake?",
    "oops": "Is everything still going as the person intended at this point, with no accident or failure yet?",
    "verdict": "Will the person rate this item highly (8 or more out of 10)?",
}

PARSE_RE = re.compile(r"\b(100|[1-9]?\d)\b")


def build_prompt(context_text: str, question: str, n: int = 4) -> str:
    """Renders the evaluation prompt template verbatim."""
    return PROMPT_TEMPLATE.format(n=n, context_text=context_text, question=question)


def prompt_hash(question: str) -> str:
    """Computes prompt hash per 03_eval_harness.md §8: first 12 hex digits of SHA-256."""
    content = PROMPT_TEMPLATE + question + "frames=4;side=768;q=90"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def parse_prob(text: Optional[str]) -> Optional[float]:
    r"""Parses probability from model response using the first match of \b(100|[1-9]?\d)\b."""
    if not text:
        return None
    match = PARSE_RE.search(text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val / 100.0
    return None


def sample_frames(
    video_path: Union[str, Path],
    window: List[float],
    tmp_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Samples 4 frames at t_k = s + (k + 0.5)*(e - s)/4, resized so longer side is 768px,

    JPEG quality 90 under TMPDIR.
    """
    from features.visual import extract_frame

    if tmp_dir is None:
        base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
        target_dir = Path(tempfile.mkdtemp(prefix="judge_frames_", dir=base_tmp))
    else:
        target_dir = Path(tmp_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    s = float(window[0])
    e = float(window[1])
    frame_paths: List[Path] = []

    for k in range(4):
        t_k = s + (k + 0.5) * (e - s) / 4.0
        img = extract_frame(video_path, t_k)
        w, h = img.size
        if w >= h:
            new_w = 768
            new_h = max(1, int(round(h * 768.0 / w)))
        else:
            new_h = 768
            new_w = max(1, int(round(w * 768.0 / h)))

        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        out_path = target_dir / f"frame_{k}.jpg"
        img.save(out_path, format="JPEG", quality=90)
        frame_paths.append(out_path)

    return frame_paths


class OllamaJudge:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self.model = model or get_model("vlm_judge")
        self.host = host

    def judge_item(
        self,
        video_path: Union[str, Path],
        window: List[float],
        context_text: str,
        question: str,
    ) -> Tuple[Optional[float], str, int, float]:
        """Judges an item using local Ollama model.

        Returns (judge_prob, raw_text, attempts, elapsed_ms).
        """
        t0 = time.time()
        base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
        item_tmp_dir = Path(tempfile.mkdtemp(prefix="ollama_judge_", dir=base_tmp))

        try:
            frame_paths = sample_frames(video_path, window, tmp_dir=item_tmp_dir)
            prompt = build_prompt(context_text=context_text, question=question, n=4)
            str_frame_paths = [str(p) for p in frame_paths]

            # Attempt 1: temperature 0 (guarded if model not yet loaded in Ollama)
            is_loaded = memguard.is_our_judge_loaded(endpoint=self.host)
            if not is_loaded:
                with memguard.guard("sr_judge_load"):
                    raw = ollama_chat(
                        model=self.model,
                        prompt=prompt,
                        image_paths=str_frame_paths,
                        options={"temperature": 0, "num_ctx": 8192, "num_predict": 16},
                        timeout=180.0,
                        host=self.host,
                    )
            else:
                raw = ollama_chat(
                    model=self.model,
                    prompt=prompt,
                    image_paths=str_frame_paths,
                    options={"temperature": 0, "num_ctx": 8192, "num_predict": 16},
                    timeout=180.0,
                    host=self.host,
                )
            prob = parse_prob(raw)
            if prob is not None:
                elapsed_ms = (time.time() - t0) * 1000.0
                return prob, raw, 1, elapsed_ms

            # Retries 2 and 3: temperature 0.3
            for attempt in (2, 3):
                raw = ollama_chat(
                    model=self.model,
                    prompt=prompt,
                    image_paths=str_frame_paths,
                    options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 16},
                    timeout=180.0,
                    host=self.host,
                )
                prob = parse_prob(raw)
                if prob is not None:
                    elapsed_ms = (time.time() - t0) * 1000.0
                    return prob, raw, attempt, elapsed_ms

            elapsed_ms = (time.time() - t0) * 1000.0
            return None, raw, 3, elapsed_ms

        finally:
            # Clean up temporary frame files
            for p in item_tmp_dir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                item_tmp_dir.rmdir()
            except Exception:
                pass


class GeminiJudge:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("GOOGLE_API_KEY")
        from google import genai
        self.client = genai.Client(api_key=key) if key else None
        self.model = "gemini-3.6-flash"

    def judge_item(
        self,
        video_path: Union[str, Path],
        window: List[float],
        context_text: str,
        question: str,
    ) -> Tuple[Optional[float], str, int, float]:
        """Judges an item using Gemini frontier API.

        Returns (judge_prob, raw_text, attempts, elapsed_ms).
        """
        if self.client is None:
            raise RuntimeError("GOOGLE_API_KEY is not configured for GeminiJudge")

        t0 = time.time()
        base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
        item_tmp_dir = Path(tempfile.mkdtemp(prefix="gemini_judge_", dir=base_tmp))

        try:
            frame_paths = sample_frames(video_path, window, tmp_dir=item_tmp_dir)
            prompt = build_prompt(context_text=context_text, question=question, n=4)

            from google.genai import types

            image_parts = []
            for fp in frame_paths:
                with open(fp, "rb") as f:
                    image_parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

            contents = image_parts + [prompt]
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16,
            )

            # Retry loop on HTTP 429 or 5xx: up to 5 tries with 30s sleep
            last_err = ""
            for attempt in range(1, 6):
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    raw = resp.text or ""
                    prob = parse_prob(raw)
                    elapsed_ms = (time.time() - t0) * 1000.0
                    return prob, raw, attempt, elapsed_ms
                except Exception as exc:
                    err_str = str(exc)
                    last_err = err_str
                    # Check for rate limit or server error
                    if "429" in err_str or "500" in err_str or "503" in err_str or "504" in err_str:
                        if attempt < 5:
                            time.sleep(30.0)
                            continue
                    raise

            elapsed_ms = (time.time() - t0) * 1000.0
            return None, last_err, 5, elapsed_ms

        finally:
            for p in item_tmp_dir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                item_tmp_dir.rmdir()
            except Exception:
                pass


def run_judge(
    dataset: str,
    split: str,
    backend: str = "ollama",
    limit: Optional[int] = None,
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    question_override: Optional[str] = None,
    judge_instance: Optional[Any] = None,
) -> None:
    start_time = time.time()
    root = Path(data_root or DATA_ROOT)

    question = question_override or DATASET_QUESTIONS.get(dataset)
    if not question:
        raise ValueError(f"No default question for dataset '{dataset}'. Provide question_override.")

    p_hash = prompt_hash(question)

    if judge_instance is not None:
        judge = judge_instance
        if backend == "ollama":
            model_tag = getattr(judge, "model", "ollama").replace(":", "_")
        else:
            model_tag = getattr(judge, "model", "gemini-3.6-flash")
    elif backend == "ollama":
        judge = OllamaJudge()
        model_tag = judge.model.replace(":", "_")
    elif backend == "gemini":
        if split != "test":
            raise ValueError("GeminiJudge is test split only")
        judge = GeminiJudge()
        model_tag = "gemini-3.6-flash"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Load items
    items_path = root / "items" / dataset / "items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    all_items = read_items(items_path)

    # Filter items by split
    split_data = load_split(dataset, path_dir=splits_dir or "splits")
    allowed_ids = set(split_data[split])
    items: List[Item] = [it for it in all_items if it.item_id in allowed_ids]

    # For Gemini on test split, cap at 2,000 items with seeded default_rng(0)
    if backend == "gemini" and len(items) > 2000:
        rng = np.random.default_rng(0)
        sampled_indices = rng.choice(len(items), size=2000, replace=False)
        items = [items[i] for i in sorted(sampled_indices)]

    if limit is not None:
        items = items[:limit]

    cache_dir = root / "judge" / dataset / model_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{p_hash}.jsonl"
    progress_file = cache_dir / "progress.json"
    errors_path = cache_dir / "errors.jsonl"

    cached_entries: Dict[str, Dict[str, Any]] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    cached_entries[entry["item_id"]] = entry
                except Exception:
                    pass

    finished_ids: List[str] = list(cached_entries.keys())
    finished_set = set(finished_ids)

    items_in = len(items)
    items_out = 0
    excluded = 0
    parse_failures = 0
    api_errors = 0

    try:
        for item in items:
            # Check cache
            if item.item_id in cached_entries:
                entry = cached_entries[item.item_id]
                if entry.get("judge_prob") is not None:
                    items_out += 1
                else:
                    excluded += 1
                    parse_failures += 1
                continue

            # Between-items watchdog for local Ollama judge
            if backend == "ollama":
                memguard.check("sr_judge_load", memguard.unload_own_judge, lambda: None)

            try:
                # Action window only
                prob, raw, attempts, elapsed_ms = judge.judge_item(
                    video_path=item.video_path,
                    window=item.action_window_sec,
                    context_text=item.context_text,
                    question=question,
                )

                record = {
                    "item_id": item.item_id,
                    "judge_prob": prob,
                    "raw": raw,
                    "attempts": attempts,
                    "elapsed_ms": elapsed_ms,
                }

                with open(cache_file, "a", encoding="utf-8") as cf:
                    cf.write(json.dumps(record) + "\n")
                    cf.flush()
                    os.fsync(cf.fileno())

                cached_entries[item.item_id] = record
                if item.item_id not in finished_set:
                    finished_ids.append(item.item_id)
                    finished_set.add(item.item_id)

                if prob is not None:
                    items_out += 1
                else:
                    excluded += 1
                    parse_failures += 1

                # Update progress.json
                with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False, encoding="utf-8") as tf:
                    temp_progress = tf.name
                    json.dump(finished_ids, tf)
                    tf.flush()
                    os.fsync(tf.fileno())
                os.replace(temp_progress, progress_file)

            except MemoryDeferred:
                raise
            except Exception as exc:
                api_errors += 1
                excluded += 1
                tb_str = traceback.format_exc()
                err_record = {
                    "item_id": item.item_id,
                    "error": str(exc),
                    "traceback": tb_str,
                    "ts": time.time(),
                }
                with open(errors_path, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps(err_record) + "\n")
                    ef.flush()
                    os.fsync(ef.fileno())

    except MemoryDeferred:
        elapsed_s = time.time() - start_time
        print(
            f"items_in={items_in} items_out={items_out} excluded={excluded} "
            f"elapsed_s={elapsed_s:.2f} parse_failures={parse_failures} api_errors={api_errors} "
            f"deferred_by_memory_guard=1"
        )
        raise
    finally:
        if backend == "ollama":
            memguard.unload_own_judge()

    elapsed_s = time.time() - start_time
    print(
        f"items_in={items_in} items_out={items_out} excluded={excluded} "
        f"elapsed_s={elapsed_s:.2f} parse_failures={parse_failures} api_errors={api_errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Judge (local + frontier)")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", required=True, choices=["train", "test"], help="Dataset split")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "gemini"], help="Judge backend")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = parser.parse_args()

    try:
        run_judge(
            dataset=args.dataset,
            split=args.split,
            backend=args.backend,
            limit=args.limit,
        )
    except MemoryDeferred:
        sys.exit(75)


if __name__ == "__main__":
    main()
