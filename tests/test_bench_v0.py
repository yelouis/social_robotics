"""SRB v0 toolchain tests — the pure-logic pieces (validation + skip logic,
single-rater finalize rules, item templating/eligibility, F3 pairing rules)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from adjudicate import validate, accepted  # noqa: E402
from build_items import build, f1_gold_class, gerundize  # noqa: E402
import pair_items  # noqa: E402


def _row(mid="m1", **kw):
    base = {"moment_id": mid, "A1": "yes", "A2": "yes", "A3": "",
            "B1": "hands over a card", "B2": "yes", "B3": "yes", "B4": "wearer",
            "B5": "approving", "B6": "face; head_gesture", "B7": "continues",
            "B8": "confident", "B9": "", "C1": "", "seconds_spent": "70"}
    base.update(kw)
    return base


def test_validate_accepts_clean_row():
    clean, errors = validate([_row()])
    assert not errors and len(clean) == 1
    assert clean[0]["B5"] == "approving"


def test_validate_skip_logic():
    # B3=no must leave B4/B5 blank
    _, errors = validate([_row(B3="no")])
    assert errors and any("skip logic" in e for _, _, errs in errors for e in errs)
    clean, errors = validate([_row(B3="no", B4="", B5="", B6="", B7="")])
    assert not errors and clean[0]["B3"] == "no"
    # B4=something_else must leave B5 blank
    _, errors = validate([_row(B4="something_else")])
    assert errors
    clean, errors = validate([_row(B4="something_else", B5="")])
    assert not errors


def test_validate_enum_and_duplicate():
    _, errors = validate([_row(B5="happy")])
    assert errors
    _, errors = validate([_row(), _row()])
    assert any("duplicate" in e for _, _, errs in errors for e in errs)


def test_triage_and_finalize_rules():
    rows, _ = validate([
        _row("keep"),
        _row("guess", B8="guessing"),
        _row("flagged", A3="minor"),
        _row("dark", A2="no", B1="", B2="", B3="", B4="", B5="", B6="", B7="", B8=""),
    ])
    acc = accepted(rows)
    ids = {r["moment_id"] for r in acc}
    assert ids == {"keep", "guess"}          # flagged + dark excluded at triage


def test_f1_gold_class_routing():
    g = dict(reaction="yes", audience="yes", directedness="wearer", valence="approving")
    assert f1_gold_class(g) == "approving"
    g = dict(reaction="no", audience="yes", directedness=None, valence=None)
    assert f1_gold_class(g) == "no_reaction"                  # control class
    g = dict(reaction="yes", audience="yes", directedness="something_else", valence=None)
    assert f1_gold_class(g) == "no_reaction"                  # not directed at wearer
    g = dict(reaction="yes", audience="yes", directedness="wearer", valence="mixed")
    assert f1_gold_class(g) is None                           # excluded from F1


def test_gerundize():
    assert gerundize("points at the signboard") == "is pointing at the signboard"
    assert gerundize("hands over a card") == "is handing over a card"
    assert gerundize("listening") == "is listening"


def _golden(mid, action, valence, reaction="yes", directed="wearer", audience="yes"):
    return {"moment_id": mid, "wearer_action": action, "audience": audience,
            "reaction": reaction, "directedness": directed, "valence": valence,
            "channels": ["face"], "next_action": "continues",
            "confidence": "confident", "notes": None,
            "rater": "maintainer", "regime": "v0_single_rater"}


def _moment(mid, clip, pid=1):
    return {"moment_id": mid, "clip_id": clip, "corpus": "ego4d",
            "t_climax_sec": 10.0, "window_sec": [9.0, 13.0], "source": "engine",
            "is_control": False, "control_type": None,
            "engine_prefill": {"per_person": [{"person_id": pid}]},
            "task_label_hint": "x"}


def test_build_items_unclear_excluded_and_dedup():
    golden = [
        _golden("m1", "hands over a card", "approving"),
        _golden("m2", "unclear", "approving"),                       # excluded
        _golden("m3", "waves at neighbor", None, reaction="no", directed=None),  # no_reaction
    ]
    moments = [_moment("m1", "c1"), _moment("m2", "c2"), _moment("m3", "c3")]
    i1, g1, i2, g2, report = build(golden, moments, seed=1)
    assert {g["moment_id"] for g in g1} == {"m1", "m3"}
    # same (clip, bystander) twice within a family -> second deduped
    golden.append(_golden("m4", "shuffles the deck", "disapproving"))
    moments.append(_moment("m4", "c1", pid=1))                        # same clip+person as m1
    i1b, g1b, *_ = build(golden, moments, seed=1)
    assert not any(g["moment_id"] == "m4" for g in g1b)


def test_pair_rules(tmp_path, monkeypatch):
    golden = [
        _golden("p1", "hands over a card", "approving"),
        _golden("p2", "hands over a mug", None, reaction="no", directed=None),   # no_reaction
        _golden("p3", "hands over a plate", "disapproving"),
        _golden("p4", "conversation", "approving"),                   # sentinel: excluded
    ]
    moments = [_moment(f"p{i}", f"c{i}") for i in range(1, 5)]
    from srb_common import write_jsonl
    import srb_common
    monkeypatch.setattr(pair_items, "BENCH_DATA", tmp_path)
    write_jsonl(tmp_path / "golden_labels.jsonl", golden)
    write_jsonl(tmp_path / "candidate_moments.jsonl", moments)
    monkeypatch.setattr(sys, "argv", ["pair_items.py", "--seed", "1"])
    pair_items.main()
    import json
    items = [json.loads(l) for l in open(tmp_path / "items" / "items_f3_track_a.jsonl")]
    gold = [json.loads(l) for l in open(tmp_path / "items" / "gold_f3.jsonl")]
    # approving(0) vs no_reaction(2) = non-adjacent OK; approving vs disapproving OK;
    # no_reaction(2) vs disapproving(3) adjacent -> excluded; sentinel excluded.
    assert len(items) == 2
    for it, g in zip(items, gold):
        assert set(it["moment_ids"]) <= {"p1", "p2", "p3"}
        better = it["moment_ids"][g["gold_index"]]
        assert better in ("p1",) or (better == "p2" and "p3" in it["moment_ids"])
