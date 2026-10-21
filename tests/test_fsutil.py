from pathlib import Path
import json

from shared.fsutil import data_glob, is_appledouble


def test_data_glob_excludes_appledouble(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (tmp_path / "errors.jsonl").write_text('{"err": 1}\n', encoding="utf-8")
    (tmp_path / "._a.jsonl").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)

    matched = data_glob(tmp_path, "*.jsonl")
    assert matched == [tmp_path / "a.jsonl", tmp_path / "errors.jsonl"]
    assert is_appledouble(tmp_path / "._a.jsonl")
    assert not is_appledouble(tmp_path / "a.jsonl")


def test_oops_report_judge_diagnostics_unchanged_by_planted_appledouble(tmp_path: Path):
    from tools.report_oops import compute_oops_wall_clocks

    judge_dir = tmp_path / "judge" / "oops" / "qwen2.5vl_7b"
    judge_dir.mkdir(parents=True)

    valid_jsonl = judge_dir / "2409b2876016.jsonl"
    valid_jsonl.write_text(
        json.dumps({"item_id": "oops:clip1:pre", "judge_prob": 1.0, "elapsed_ms": 1500.0}) + "\n",
        encoding="utf-8",
    )
    # Plant AppleDouble twin file with binary content that would fail utf-8 decoding or corrupt counts
    planted_ad = judge_dir / "._2409b2876016.jsonl"
    planted_ad.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)

    timings = compute_oops_wall_clocks(tmp_path)
    # 1500 ms = 1.5 s
    assert timings.get("Judge (`qwen2.5vl_7b`)") == "1.5 s (0.0 min)"

    # Also verify judge answer files (tools/report_oops:196, :285) ignore the planted file
    ans_files = [f for f in data_glob(judge_dir, "*.jsonl") if f.name != "errors.jsonl"]
    assert ans_files == [valid_jsonl]
