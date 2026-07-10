"""SRB v0 — blind-baseline gate (docs/07 §5.4.1).

A TEXT-ONLY local LLM answers each item from {question + options} alone (no
clip, no observation block), 5 samples. Items it gets right too often are
answerable from text priors and are dropped:
  >=60% (4-way F1) / >=70% (3-way F2) / >=75% (pairwise F3).
The kill rate is reported per family (a wave statistic).

Applies the drop to BOTH tracks of an item (same underlying moment).

Usage: python bench/blind_gate.py [--model MODEL] [--samples 5]
Writes items back FILTERED (originals kept as *.pregate.jsonl) + blind_gate_report.json
"""
import argparse
import json
import re
import sys

from srb_common import BENCH_DATA, REPO_ROOT, read_jsonl, write_jsonl

sys.path.insert(0, str(REPO_ROOT / "src"))
from shared.vlm_client import ollama_chat  # noqa: E402  (shared/, sanctioned)

THRESH = {"f1": 0.60, "f2": 0.70, "f3": 0.75}
LETTERS = "abcdefgh"


def ask(model, question, options, samples):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    prompt = (f"{question}\n{opts}\n"
              "You cannot see the video. Guess the single most likely answer "
              "from the text alone. Reply with ONLY the option letter.")
    picks = []
    for k in range(samples):
        out = ollama_chat(model, prompt,
                          options={"temperature": 1.0 if samples > 1 else 0.0,
                                   "seed": k, "num_predict": 4},
                          timeout=60)
        m = re.search(rf"[{LETTERS[:len(options)]}]", out.lower())
        picks.append(LETTERS.index(m.group(0)) if m else -1)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:latest")
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args()

    out_dir = BENCH_DATA / "items"
    report = {}
    killed_moments = set()
    for fam in ("f1", "f2", "f3"):
        src = out_dir / f"items_{fam}_track_a.jsonl"
        if not src.exists():
            continue
        items = read_jsonl(src)
        gold = {g["item_id"]: g["gold_index"] for g in read_jsonl(out_dir / f"gold_{fam}.jsonl")}
        kills = []
        for it in items:
            picks = ask(a.model, it["question"], it["options"], a.samples)
            acc = sum(1 for p in picks if p == gold[it["item_id"]]) / len(picks)
            if acc >= THRESH[fam]:
                kills.append(it["item_id"])
                for m in (it.get("moment_ids") or [it["moment_id"]]):
                    killed_moments.add((fam, m))
        report[fam] = {"n_items": len(items), "n_killed": len(kills),
                       "kill_rate": round(len(kills) / max(1, len(items)), 3),
                       "killed": kills}
        # filter both tracks
        for track in ("a", "b"):
            p = out_dir / f"items_{fam}_track_{track}.jsonl"
            gp = out_dir / (f"gold_{fam}.jsonl" if track == "a" else f"gold_{fam}b.jsonl")
            if not p.exists():
                continue
            all_items = read_jsonl(p)
            keep_ids = {it["item_id"] for it in all_items
                        if it["item_id"].replace(f"-{fam}b-", f"-{fam}-") not in kills}
            p.rename(p.with_suffix(".pregate.jsonl"))
            write_jsonl(p, [it for it in all_items if it["item_id"] in keep_ids])
            if gp.exists():
                gp.rename(gp.with_suffix(".pregate.jsonl"))
                write_jsonl(gp, [g for g in read_jsonl(gp.with_suffix(".pregate.jsonl"))
                                 if g["item_id"] in keep_ids])
    survival = 1 - (sum(r["n_killed"] for r in report.values())
                    / max(1, sum(r["n_items"] for r in report.values())))
    report["overall_survival_rate"] = round(survival, 3)
    report["v0_gate_survival>=0.50"] = survival >= 0.50
    (BENCH_DATA / "blind_gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (v if k.startswith("overall") or k.startswith("v0") else
                          {kk: vv for kk, vv in v.items() if kk != "killed"})
                      for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
