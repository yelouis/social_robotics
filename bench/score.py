"""SRB v0 — deterministic scorer (docs/07 §6). No LLM judge.

Predictions format (one file per model+track):
    {"item_id": "...", "answer": "a"}        # option letter, case-insensitive
Usage:
    python bench/score.py predictions.jsonl [--label qwen2.5vl-7b]
Reports per-family accuracy (micro + macro over gold classes) and the
Social Hallucination Rate (SHR): on gold no-reaction/not-directed items,
the fraction of answers asserting evaluative feedback (F1 a/b, F2 a).
"""
import argparse
import json
from collections import Counter, defaultdict

from srb_common import BENCH_DATA, read_jsonl

LETTERS = "abcdefgh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()
    preds = {p["item_id"]: LETTERS.index(p["answer"].strip().lower()[0])
             for p in read_jsonl(a.predictions)}

    out_dir = BENCH_DATA / "items"
    result = {"label": a.label}
    for fam in ("f1", "f2", "f3"):
        for track, gfile in (("a", f"gold_{fam}.jsonl"), ("b", f"gold_{fam}b.jsonl")):
            gp = out_dir / gfile
            if not gp.exists():
                continue
            gold = read_jsonl(gp)
            answered = [g for g in gold if g["item_id"] in preds]
            if not answered:
                continue
            per_class = defaultdict(lambda: [0, 0])
            shr_num = shr_den = 0
            correct = 0
            for g in answered:
                pick = preds[g["item_id"]]
                cls = g.get("gold_class", str(g["gold_index"]))
                per_class[cls][1] += 1
                if pick == g["gold_index"]:
                    correct += 1
                    per_class[cls][0] += 1
                if fam == "f1" and cls == "no_reaction":
                    shr_den += 1
                    shr_num += pick in (0, 1)          # asserted approving/disapproving
                if fam == "f2" and cls in ("something_else", "cant_tell"):
                    shr_den += 1
                    shr_num += pick == 0               # asserted "responds to wearer"
            micro = correct / len(answered)
            macro = sum(c / t for c, t in per_class.values()) / len(per_class)
            key = f"{fam.upper()}_track{track.upper()}"
            result[key] = {
                "n": len(answered), "micro_acc": round(micro, 3),
                "macro_acc": round(macro, 3),
                "per_class": {k: f"{c}/{t}" for k, (c, t) in sorted(per_class.items())},
            }
            if shr_den:
                result[key]["SHR"] = round(shr_num / shr_den, 3)
    # A/B delta per family (perception vs reasoning deficit, §1.3)
    for fam in ("F1", "F2", "F3"):
        ta, tb = result.get(f"{fam}_trackA"), result.get(f"{fam}_trackB")
        if ta and tb:
            result[f"{fam}_AB_delta"] = round(ta["micro_acc"] - tb["micro_acc"], 3)
    print(json.dumps(result, indent=2))
    out = BENCH_DATA / f"scores_{a.label}.json"
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
