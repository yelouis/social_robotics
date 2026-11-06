from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools.build_paper_facts import load_latest_scorecard_rows, validate_recorded_facts
from tools.check_paper_numbers import check_draft_numbers


def test_checker_rejects_invented_number(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {"a": 0.7352, "b": 1208},
        "recorded": {"c": {"value": 0.954, "source": "s", "quote": "q"}},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")

    draft_text = (
        "On November 1, 2026, window length scored 0.735 on 1,208 items (95.4%); "
        "see §11.10 and L18, as in 2020; the face scored 0.736.\n"
    )
    draft_path.write_text(draft_text, encoding="utf-8")

    code, out_text, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert unmatched == ["0.736"]
    assert "unmatched=1" in out_text
    assert "0.736" in out_text


def test_checker_table_drift(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_path.write_text(json.dumps({"computed": {}, "recorded": {}}), encoding="utf-8")
    t1_text = "| col1 | col2 |\n|---|---|\n| a | 0.500 |\n"
    (tables_dir / "T1.md").write_text(t1_text, encoding="utf-8")

    # Change one character inside the table block (0.500 -> 0.501)
    drifted = "| col1 | col2 |\n|---|---|\n| a | 0.501 |\n"
    draft_path.write_text(
        f"Intro paragraph.\n\n<!-- table:T1 -->\n{drifted}<!-- /table:T1 -->\n",
        encoding="utf-8",
    )

    code, out_text, _ = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert "T1" in out_text


def test_recorded_quote_must_exist(tmp_path: Path) -> None:
    src_file = tmp_path / "doc.md"
    src_file.write_text("Some unrelated text without the quote.\n", encoding="utf-8")

    bad_recorded = {
        "test/key": {
            "value": 0.123,
            "source": "doc.md §1",
            "quote": "This exact quote is missing.",
        }
    }
    with pytest.raises(ValueError, match="quote"):
        validate_recorded_facts(bad_recorded, repo_root=tmp_path)


def test_latest_row_wins(tmp_path: Path) -> None:
    sc_path = tmp_path / "scorecard.jsonl"
    rows = [
        {
            "ts": "2026-10-10T10:00:00Z",
            "hypothesis": "H1",
            "dataset": "oops",
            "split": "test",
            "condition": "judge",
            "metric": "auroc",
            "value": 0.400,
            "ci_low": 0.350,
            "ci_high": 0.450,
            "n_items": 100,
            "n_groups": 10,
        },
        {
            "ts": "2026-10-21T12:00:00Z",
            "hypothesis": "H1",
            "dataset": "oops",
            "split": "test",
            "condition": "judge",
            "metric": "auroc",
            "value": 0.472,
            "ci_low": 0.440,
            "ci_high": 0.498,
            "n_items": 1072,
            "n_groups": 90,
        },
    ]
    sc_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    latest = load_latest_scorecard_rows(sc_path)
    assert latest["oops"]["judge"]["value"] == pytest.approx(0.472)
    assert latest["oops"]["judge"]["n_items"] == 1072
