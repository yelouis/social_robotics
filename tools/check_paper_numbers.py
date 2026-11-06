#!/usr/bin/env python3
"""tools/check_paper_numbers.py — Verify every prose number and embedded table in a paper draft.

Implements the Item E1 number-checker contract (06_writeup.md §4–§5, agent_execution_guide.md §3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, List, Optional, Set, Tuple


TABLE_BLOCK_RE = re.compile(r"<!-- table:(T\d+) -->\n?(.*?)<!-- /table:\1 -->", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
LINK_TARGET_RE = re.compile(r"\]\([^)\n]*\)")
IMAGE_LINE_RE = re.compile(r"^\s*!\[.*$", re.MULTILINE)
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z?)?\b")
WRITTEN_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
)
NUM_TOKEN_RE = re.compile(r"(?<![\w.§])[+−-]?\d[\d,]*(?:\.\d+)?%?(?![\w])")


def _collect_numbers(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        out.add(abs(val))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_numbers(v, out)


def _token_matches_facts(token: str, fact_numbers: Set[float]) -> bool:
    is_pct = token.endswith("%")
    raw = token[:-1] if is_pct else token
    raw = raw.replace(",", "").replace("−", "-")
    if raw.startswith(("+", "-")):
        unsigned = raw[1:]
    else:
        unsigned = raw

    if "." in unsigned:
        decimals = len(unsigned.split(".", 1)[1])
    else:
        decimals = 0

    tok_abs = abs(float(unsigned))
    target_direct = round(tok_abs, decimals)
    for f in fact_numbers:
        if abs(round(f, decimals) - target_direct) < 1e-12:
            return True

    if is_pct:
        decimals_div = decimals + 2
        target_div = round(tok_abs / 100.0, decimals_div)
        for f in fact_numbers:
            if abs(round(f, decimals_div) - target_div) < 1e-12:
                return True

    return False


def _is_exempt_year(token: str) -> bool:
    if re.fullmatch(r"\d{4}", token):
        yr = int(token)
        return 1990 <= yr <= 2030
    return False


def check_draft_numbers(
    draft_path: Path,
    facts_path: Path = Path("docs/paper/facts.json"),
    tables_dir: Path = Path("docs/paper/tables"),
) -> Tuple[int, str, List[str]]:
    draft_text = draft_path.read_text(encoding="utf-8")
    orig_lines = draft_text.splitlines()

    # (a) Tables first: verify byte-for-byte equality with docs/paper/tables/Tn.md, then drop
    table_errors: List[str] = []

    def _replace_table(m: re.Match) -> str:
        tn = m.group(1)
        body = m.group(2)
        expected_path = tables_dir / f"{tn}.md"
        if not expected_path.exists():
            table_errors.append(f"table {tn}: missing table file {expected_path}")
        else:
            expected_text = expected_path.read_text(encoding="utf-8")
            if body != expected_text:
                table_errors.append(f"table {tn}: content inside markers does not match {expected_path} byte for byte")
        return "\n" * m.group(0).count("\n")

    cleaned = TABLE_BLOCK_RE.sub(_replace_table, draft_text)

    # (b) Drop fenced code blocks, inline code spans, link targets, image lines, arXiv IDs, ISO & written dates
    cleaned = FENCED_CODE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), cleaned)
    cleaned = IMAGE_LINE_RE.sub("", cleaned)
    cleaned = INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = LINK_TARGET_RE.sub("]", cleaned)
    cleaned = ARXIV_RE.sub(" ", cleaned)
    cleaned = ISO_DATE_RE.sub(" ", cleaned)
    cleaned = WRITTEN_DATE_RE.sub(" ", cleaned)

    facts_obj = json.loads(facts_path.read_text(encoding="utf-8"))
    fact_numbers: Set[float] = set()
    _collect_numbers(facts_obj, fact_numbers)

    checked = 0
    matched = 0
    unmatched_tokens: List[str] = []
    unmatched_details: List[str] = []

    cleaned_lines = cleaned.splitlines()
    for line_idx, line in enumerate(cleaned_lines):
        for m in NUM_TOKEN_RE.finditer(line):
            tok = m.group(0)
            if _is_exempt_year(tok):
                continue
            checked += 1
            if _token_matches_facts(tok, fact_numbers):
                matched += 1
            else:
                unmatched_tokens.append(tok)
                orig_line = orig_lines[line_idx] if line_idx < len(orig_lines) else line
                unmatched_details.append(f"line {line_idx + 1}: unmatched token {tok!r} in: {orig_line}")

    unmatched_count = len(unmatched_tokens)
    out_lines = [f"checked={checked} matched={matched} unmatched={unmatched_count}"]
    for err in table_errors:
        out_lines.append(err)
    for det in unmatched_details:
        out_lines.append(det)

    exit_code = 1 if (table_errors or unmatched_count > 0) else 0
    return exit_code, "\n".join(out_lines), unmatched_tokens


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check paper numbers and tables against facts.json.")
    parser.add_argument("draft", type=Path, help="Path to draft markdown file")
    parser.add_argument(
        "--facts",
        type=Path,
        default=Path("docs/paper/facts.json"),
        help="Path to facts.json",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("docs/paper/tables"),
        help="Directory containing T1.md..T6.md",
    )
    args = parser.parse_args(argv)

    code, out_text, _ = check_draft_numbers(args.draft, args.facts, args.tables_dir)
    print(out_text)
    return code


if __name__ == "__main__":
    sys.exit(main())
