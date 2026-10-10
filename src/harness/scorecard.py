from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from harness.metrics import (
    _item_auroc_ci,
    auroc_ci,
    delta_auroc_ci,
)


@dataclass
class ScorecardRow:
    ts: str
    git_sha: str
    dirty: bool
    hypothesis: str
    dataset: str
    split: str
    condition: str
    metric: str
    value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_items: int
    n_groups: int
    n_excluded: int
    config_hash: str
    notes: str


def _get_git_info() -> Tuple[str, bool]:
    git_bin = "/usr/bin/git" if os.path.exists("/usr/bin/git") else "git"
    try:
        sha_proc = subprocess.run(
            [git_bin, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = sha_proc.stdout.strip()
    except Exception:
        git_sha = "unknown"

    try:
        dirty_proc = subprocess.run(
            [git_bin, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(dirty_proc.stdout.strip())
    except Exception:
        dirty = False

    return git_sha, dirty


def make_row(
    hypothesis: str,
    dataset: str,
    split: str,
    condition: str,
    metric: str,
    value: Optional[float],
    ci_low: Optional[float],
    ci_high: Optional[float],
    n_items: int,
    n_groups: int,
    n_excluded: int,
    config_hash: str,
    notes: str = "",
) -> ScorecardRow:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_sha, dirty = _get_git_info()
    return ScorecardRow(
        ts=ts,
        git_sha=git_sha,
        dirty=dirty,
        hypothesis=hypothesis,
        dataset=dataset,
        split=split,
        condition=condition,
        metric=metric,
        value=round(value, 4) if value is not None else None,
        ci_low=round(ci_low, 4) if ci_low is not None else None,
        ci_high=round(ci_high, 4) if ci_high is not None else None,
        n_items=n_items,
        n_groups=n_groups,
        n_excluded=n_excluded,
        config_hash=config_hash,
        notes=notes,
    )


def append_rows(
    rows: List[ScorecardRow],
    path: Union[str, Path] = "results/scorecard.jsonl",
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(
    path: Union[str, Path] = "results/scorecard.jsonl",
) -> List[ScorecardRow]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(ScorecardRow(**data))
    return rows


def config_hash(config: Dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def run_selftest() -> int:
    rng = np.random.default_rng(0)
    all_pass = True

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tf:
        temp_path = tf.name

    try:
        # (a) Planted signal: 60 groups × 20 items, labels Bernoulli(0.5), score = label + 0.8*N(0,1)
        groups_a = np.repeat(np.arange(60), 20)
        y_a = rng.binomial(1, 0.5, size=1200)
        score_a = y_a + 0.8 * rng.standard_normal(size=1200)
        res_a = auroc_ci(y_a, score_a, groups_a, n_boot=1000, seed=0)
        pass_a = bool(res_a.point >= 0.75 and res_a.ci_low > 0.5)
        print(f"check (a) planted signal: {'PASS' if pass_a else 'FAIL'} (auroc={res_a.point:.3f}, ci=[{res_a.ci_low:.3f}, {res_a.ci_high:.3f}])")
        if not pass_a:
            all_pass = False

        # (b) Null: same groups, score ~ N(0,1) independent of label
        score_null = rng.standard_normal(size=1200)
        res_b = auroc_ci(y_a, score_null, groups_a, n_boot=1000, seed=0)
        pass_b = bool(res_b.ci_low <= 0.5 <= res_b.ci_high)
        print(f"check (b) null: {'PASS' if pass_b else 'FAIL'} (ci=[{res_b.ci_low:.3f}, {res_b.ci_high:.3f}])")
        if not pass_b:
            all_pass = False

        # (c) Grouping is real: 40 groups × 25 items; label constant within group; score = group_offset + 0.1*N(0,1)
        groups_c = np.repeat(np.arange(40), 25)
        group_labels = rng.binomial(1, 0.5, size=40)
        group_offsets = group_labels + rng.standard_normal(size=40)
        y_c = np.repeat(group_labels, 25)
        group_offset_items = np.repeat(group_offsets, 25)
        score_c = group_offset_items + 0.1 * rng.standard_normal(size=1000)

        res_c_grouped = auroc_ci(y_c, score_c, groups_c, n_boot=1000, seed=0)
        res_c_item = _item_auroc_ci(y_c, score_c, n_boot=1000, seed=0)
        width_grouped = res_c_grouped.ci_high - res_c_grouped.ci_low
        width_item = res_c_item[2] - res_c_item[1]
        pass_c = bool(width_grouped >= 1.5 * width_item)
        print(f"check (c) grouping is real: {'PASS' if pass_c else 'FAIL'} (grouped_width={width_grouped:.3f}, item_width={width_item:.3f}, ratio={width_grouped/max(width_item, 1e-9):.2f})")
        if not pass_c:
            all_pass = False

        # (d) Paired Δ: score_b = score_a + 1.5*label -> delta ci_low > 0; score_b = score_a -> delta == 0 exactly, CI=(0, 0)
        score_d_b = score_a + 1.5 * y_a
        res_d_diff = delta_auroc_ci(y_a, score_a, score_d_b, groups_a, n_boot=1000, seed=0)
        res_d_same = delta_auroc_ci(y_a, score_a, score_a, groups_a, n_boot=1000, seed=0)
        pass_d = bool(
            res_d_diff.ci_low > 0
            and res_d_same.point == 0.0
            and res_d_same.ci_low == 0.0
            and res_d_same.ci_high == 0.0
        )
        print(f"check (d) paired delta: {'PASS' if pass_d else 'FAIL'} (diff ci_low={res_d_diff.ci_low:.3f}, same delta={res_d_same.point}, ci=[{res_d_same.ci_low}, {res_d_same.ci_high}])")
        if not pass_d:
            all_pass = False

        # Write test rows only to temp_path
        test_rows = [
            make_row("H1", "synthetic", "test", "planted", "auroc", res_a.point, res_a.ci_low, res_a.ci_high, 1200, 60, res_a.n_excluded, "000000000000", "selftest"),
            make_row("H1", "synthetic", "test", "null", "auroc", res_b.point, res_b.ci_low, res_b.ci_high, 1200, 60, res_b.n_excluded, "000000000000", "selftest"),
        ]
        append_rows(test_rows, path=temp_path)
        read_back = read_rows(temp_path)
        if len(read_back) != 2:
            all_pass = False

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return 0 if all_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation Scorecard CLI")
    parser.add_argument("--selftest", action="store_true", help="Run harness metric self-tests")
    parser.add_argument("--history", type=str, default=None, help="Print all rows for a dataset in time order")
    parser.add_argument("--path", type=str, default="results/scorecard.jsonl", help="Scorecard file path")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    rows = read_rows(args.path)
    if not rows:
        print(f"No rows found in {args.path}")
        return

    if args.history:
        dataset_rows = [r for r in rows if r.dataset == args.history]
        if not dataset_rows:
            print(f"No rows found for dataset: {args.history}")
            return
        dataset_rows.sort(key=lambda r: r.ts)
        _print_table(dataset_rows)
        return

    # No arguments -> latest row per (hypothesis, dataset, split, condition, metric), newest first
    latest_map: Dict[Tuple[str, str, str, str, str], ScorecardRow] = {}
    for r in rows:
        key = (r.hypothesis, r.dataset, r.split, r.condition, r.metric)
        if key not in latest_map or r.ts > latest_map[key].ts:
            latest_map[key] = r

    latest_rows = list(latest_map.values())
    latest_rows.sort(key=lambda r: r.ts, reverse=True)
    _print_table(latest_rows)


def _print_table(rows: List[ScorecardRow]) -> None:
    headers = ["ts", "hyp", "dataset", "split", "condition", "metric", "value", "ci_95", "n_items", "n_grp", "n_excl", "cfg_hash", "notes"]
    table_data = []
    for r in rows:
        val_str = f"{r.value:.4f}" if r.value is not None else "null"
        ci_str = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]" if r.ci_low is not None and r.ci_high is not None else "null"
        table_data.append([
            r.ts,
            r.hypothesis,
            r.dataset,
            r.split,
            r.condition,
            r.metric,
            val_str,
            ci_str,
            str(r.n_items),
            str(r.n_groups),
            str(r.n_excluded),
            r.config_hash,
            r.notes,
        ])

    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in table_data:
        print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))


if __name__ == "__main__":
    main()
