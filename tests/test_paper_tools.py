from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools.build_paper_facts import (
    RECORDED_FACTS,
    load_latest_scorecard_rows,
    t3_cells,
    validate_recorded_facts,
)
from tools.check_paper_numbers import check_draft_numbers


def test_checker_rejects_invented_number(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {"a": {"b": {"x": 0.7352, "n": 1208}}},
        "recorded": {"c": {"value": 0.954, "source": "s", "quotes": ["q"]}},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")

    draft_text = (
        "On November 1, 2026, window length scored 0.735 on 1,208 items (95.4%); "
        "see §11.10 and L18, as in 2020; the face scored 0.736.\n"
        "<!-- facts: computed/a/b; recorded/c/value -->\n"
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
            "quotes": ["This exact quote is missing (0.123)."],
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


def test_checker_scopes_numbers_to_paragraph_tags(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {
            "scorecard": {
                "oops": {"fusion_minus_action_fusion": {"value": 0.0149}},
                "holoassist_audible": {"fusion_minus_action_fusion": {"value": 0.0206}},
            },
            "constants": {"chance_auroc": 0.5},
        },
        "recorded": {},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")

    draft_text = (
        "Oops gained 0.015, not 0.021, over chance 0.5.\n"
        "<!-- facts: computed/scorecard/oops; computed/constants/chance_auroc -->\n"
    )
    draft_path.write_text(draft_text, encoding="utf-8")

    code, out_text, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert unmatched == ["0.021"]
    assert "0.021" in out_text


def test_checker_constants_need_a_tag(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {
            "constants": {"wave_d_draws_per_budget": 20},
            "scorecard": {"reflex": {"judge": {"value": 0.72}}},
        },
        "recorded": {},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")

    draft_text = (
        "We evaluate up to 20 conditions; the judge scored 0.720.\n"
        "<!-- facts: computed/scorecard/reflex/judge -->\n"
    )
    draft_path.write_text(draft_text, encoding="utf-8")

    code, _, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert unmatched == ["20"]

    draft_tagged = (
        "We evaluate up to 20 conditions; the judge scored 0.720.\n"
        "<!-- facts: computed/scorecard/reflex/judge; computed/constants/wave_d_draws_per_budget -->\n"
    )
    draft_path.write_text(draft_tagged, encoding="utf-8")

    code2, _, unmatched2 = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code2 == 0
    assert unmatched2 == []


def test_checker_requires_a_tag(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {"scorecard": {"reflex": {"react-face": {"value": 0.7352}}}},
        "recorded": {},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")
    draft_path.write_text("The face scored 0.735.\n", encoding="utf-8")

    code, out_text, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert "0.735" in unmatched
    assert "no facts tag" in out_text


def test_checker_rejects_short_prefix(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {"scorecard": {"oops": {"judge": {"value": 0.472}}}},
        "recorded": {},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")
    draft_path.write_text(
        "<!-- facts: computed/scorecard -->\nOn Oops!, the judge scored 0.472.\n",
        encoding="utf-8",
    )

    code, out_text, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 1
    assert "0.472" in unmatched
    assert "prefix too short" in out_text


def test_checker_reads_code_spans_and_captions(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_data = {
        "computed": {
            "scorecard": {"oops": {"judge": {"value": 0.472}}},
            "constants": {"chance_auroc": 0.5},
        },
        "recorded": {},
    }
    facts_path.write_text(json.dumps(facts_data), encoding="utf-8")

    # Whitespace-free identifier in backticks passes with no tag
    draft_ok = "Judged by `qwen2.5vl:7b`.\n"
    draft_path.write_text(draft_ok, encoding="utf-8")
    code_ok, _, unmatched_ok = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code_ok == 0
    assert unmatched_ok == []

    # Code span with whitespace: 2.0 unmatched
    draft_code = "The window is `[a_end - 0.5, a_end + 2.0]`.\n<!-- facts: computed/scorecard/oops -->\n"
    draft_path.write_text(draft_code, encoding="utf-8")
    code_c, _, unmatched_c = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code_c == 1
    assert "2.0" in unmatched_c or "+2.0" in unmatched_c

    # Caption: 999 unmatched
    draft_cap = "![Figure 1: n_test = 999](figures/f.png)\n<!-- facts: computed/scorecard/oops -->\n"
    draft_path.write_text(draft_cap, encoding="utf-8")
    code_cap, _, unmatched_cap = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code_cap == 1
    assert "999" in unmatched_cap


def test_checker_exempts_reference_numbers(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.json"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    draft_path = tmp_path / "draft.md"

    facts_path.write_text(json.dumps({"computed": {}, "recorded": {}}), encoding="utf-8")
    draft_text = (
        "Decisions 2, 12, and 13; Issue 12 = A; Figure 2; doi:10.5281/zenodo.14590226; "
        "ERR@HRI 3.0; CC BY 4.0; CC BY-NC-SA 4.0; `label = 1`; [CITATION NEEDED: Gemma 4 reference].\n"
        "\n## 4 Data\n\n1. First item.\n"
    )
    draft_path.write_text(draft_text, encoding="utf-8")

    code, out_text, unmatched = check_draft_numbers(draft_path, facts_path, tables_dir)
    assert code == 0
    assert unmatched == []
    assert "checked=0" in out_text


def test_repo_recorded_facts_are_quoted() -> None:
    assert "holoassist/pre_amendment_reference" in RECORDED_FACTS
    res = validate_recorded_facts(RECORDED_FACTS)
    assert res is RECORDED_FACTS


def test_recorded_values_must_be_quoted(tmp_path: Path) -> None:
    src_file = tmp_path / "doc.md"
    src_file.write_text("The norms are 0.99 and 26.9.\n", encoding="utf-8")

    bad_recorded = {
        "test/unquoted": {
            "value": {"a": 0.99, "b": 0.33},
            "source": "doc.md §1",
            "quotes": ["The norms are 0.99 and 26.9."],
        }
    }
    with pytest.raises(ValueError, match="0.33"):
        validate_recorded_facts(bad_recorded, repo_root=tmp_path)


def test_t3_cells_drop_paths() -> None:
    line1 = (
        '- **Clause 1, "x":** '
        "react-nonverbal not run: 0 of 3000 items cached in /Volumes/X; rerun: y vs step-prior 0.751 "
        "→ not tested: not run: 0 of 3000 items cached in /Volumes/X"
    )
    assert t3_cells(line1) == ('Clause 1, "x"', "react-nonverbal not run", "not tested")

    line2 = '- **Clause 2, "Δ ≈ 0":** ablation Δ 0.015 [-0.001, 0.029] → held'
    assert t3_cells(line2) == ('Clause 2, "Δ ≈ 0"', "ablation Δ 0.015 [-0.001, 0.029]", "held")
