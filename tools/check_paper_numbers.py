#!/usr/bin/env python3
"""tools/check_paper_numbers.py — Verify every prose number and embedded table in a paper draft.

Implements the Item E1 & E1b number-checker contract (06_writeup.md §4–§5, agent_execution_guide.md §3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


TABLE_BLOCK_RE = re.compile(r"<!-- table:((?:T|B)\d+) -->\n?(.*?)<!-- /table:\1 -->", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
FACTS_TAG_RE = re.compile(r"<!--\s*facts:\s*(.*?)\s*-->", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
LINK_TARGET_RE = re.compile(r"\]\([^)\n]*\)")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
CITATION_NEEDED_RE = re.compile(r"\[CITATION NEEDED:[^\]]*\]")
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z?)?\b")
WRITTEN_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
)
DOC_REF_RE = re.compile(
    r"\b(?:Issues?|Decisions?|[Ee]ntr(?:y|ies))\s+\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*[–-]\s*)\d+)*"
)
FIG_TABLE_RE = re.compile(r"\b(?:Figure|Table)\s+\d+")
DOI_RE = re.compile(r"\bdoi:\S+")
ERR_HRI_RE = re.compile(r"ERR@HRI\s+\d+(?:\.\d+)?")
CC_BY_RE = re.compile(r"CC\s+BY(?:-NC)?\s+\d+(?:\.\d+)?")
LABEL_EQ_RE = re.compile(r"`?\blabel\s*=\s*[01]\b`?")
HEADING_NUM_RE = re.compile(r"^(\s*#{1,6}\s+)\d+(?:\.\d+)*\b")
LIST_NUM_RE = re.compile(r"^(\s*)\d+\.\s+")
NUM_TOKEN_RE = re.compile(r"(?<![\w.§])[+−-]?\d[\d,]*(?:\.\d+)?%?(?![\w])")


def flatten_facts(obj: Any) -> Dict[str, float]:
    """Recursively flatten facts.json into a map of slash-separated paths to abs(float(v)).

    Ignores strings, None, and booleans; never parses numbers out of strings.
    """
    out: Dict[str, float] = {}

    def _walk(prefix: str, cur: Any) -> None:
        if cur is None or isinstance(cur, (bool, str)):
            return
        if isinstance(cur, (int, float)):
            out[prefix] = abs(float(cur))
            return
        if isinstance(cur, dict):
            for k, v in cur.items():
                next_p = f"{prefix}/{k}" if prefix else str(k)
                _walk(next_p, v)
        elif isinstance(cur, (list, tuple)):
            for i, x in enumerate(cur):
                next_p = f"{prefix}/{i}" if prefix else str(i)
                _walk(next_p, x)

    _walk("", obj)
    return out


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


def _replace_inline_code(m: re.Match) -> str:
    inner = m.group(1)
    if re.fullmatch(r"\S+", inner):
        return " "
    return f" {inner} "


def _clean_line_for_tokens(line: str) -> str:
    # (e) Delete link targets (...) after ] (also removes image paths), arXiv ids, ISO dates, written dates
    s = LINK_TARGET_RE.sub("]", line)
    s = ARXIV_RE.sub(" ", s)
    s = ISO_DATE_RE.sub(" ", s)
    s = WRITTEN_DATE_RE.sub(" ", s)
    # Delete exemption patterns
    s = CITATION_NEEDED_RE.sub(" ", s)
    s = DOC_REF_RE.sub(" ", s)
    s = FIG_TABLE_RE.sub(" ", s)
    s = DOI_RE.sub(" ", s)
    s = ERR_HRI_RE.sub(" ", s)
    s = CC_BY_RE.sub(" ", s)
    s = LABEL_EQ_RE.sub(" ", s)
    # Inline code spans: no whitespace -> deleted; with whitespace -> loses only backticks
    s = INLINE_CODE_RE.sub(_replace_inline_code, s)
    s = LABEL_EQ_RE.sub(" ", s)
    # Strip markdown heading numbers and numbered list prefixes
    s = HEADING_NUM_RE.sub(r"\1 ", s)
    s = LIST_NUM_RE.sub(r"\1 ", s)
    return s


def check_draft_numbers(
    draft_path: Path,
    facts_path: Path = Path("docs/paper/facts.json"),
    tables_dir: Path = Path("docs/paper/tables"),
) -> Tuple[int, str, List[str]]:
    draft_text = draft_path.read_text(encoding="utf-8")
    orig_lines = draft_text.splitlines()

    # (a) Tables first: verify byte-for-byte equality with docs/paper/tables/Xn.md, then blank preserving line count
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
    # (b) Blank fenced code blocks
    cleaned = FENCED_CODE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), cleaned)

    facts_obj = json.loads(facts_path.read_text(encoding="utf-8"))
    flat_facts = flatten_facts(facts_obj)
    const_numbers: Set[float] = {
        v for k, v in flat_facts.items() if k.startswith("computed/constants/")
    }

    # (c) Split the rest into blocks: maximal runs of non-blank lines
    cleaned_lines = cleaned.splitlines()
    blocks: List[List[Tuple[int, str]]] = []
    cur_block: List[Tuple[int, str]] = []
    for idx, line in enumerate(cleaned_lines):
        if line.strip() == "":
            if cur_block:
                blocks.append(cur_block)
                cur_block = []
        else:
            cur_block.append((idx + 1, line))
    if cur_block:
        blocks.append(cur_block)

    checked = 0
    matched = 0
    unmatched_tokens: List[str] = []
    error_lines: List[str] = []
    b_num = 0
    b_tagged = 0

    for block in blocks:
        block_start = block[0][0]
        block_raw = "\n".join(line for _, line in block)
        # (d) Read each block's tags
        tag_matches = FACTS_TAG_RE.findall(block_raw)
        prefixes: List[str] = []
        for tm in tag_matches:
            for p in tm.split(";"):
                ps = p.strip().rstrip("/")
                if ps:
                    prefixes.append(ps)

        # (e) Remove all HTML comments preserving line count within the block
        block_no_comments = HTML_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), block_raw)
        block_lines_no_comments = block_no_comments.split("\n")

        # (f) Extract tokens
        block_tokens: List[Tuple[int, str, str]] = []
        for (line_no, _), line_nc in zip(block, block_lines_no_comments):
            line_clean = _clean_line_for_tokens(line_nc)
            for m in NUM_TOKEN_RE.finditer(line_clean):
                tok = m.group(0)
                if _is_exempt_year(tok):
                    continue
                orig_line = orig_lines[line_no - 1] if (line_no - 1) < len(orig_lines) else line_nc
                block_tokens.append((line_no, tok, orig_line))

        # Check prefix validity if any prefixes are given
        prefix_has_error = False
        for p in prefixes:
            if len(p.split("/")) < 3:
                error_lines.append(f"line {block_start}: prefix too short: {p}")
                prefix_has_error = True
            elif not any(k == p or k.startswith(p + "/") for k in flat_facts):
                error_lines.append(f"line {block_start}: prefix matches nothing: {p}")
                prefix_has_error = True

        if not block_tokens:
            continue

        b_num += 1
        checked += len(block_tokens)

        if not prefixes:
            error_lines.append(f"line {block_start}: no facts tag")
            for line_no, tok, orig_line in block_tokens:
                unmatched_tokens.append(tok)
                error_lines.append(f"line {line_no}: unmatched token {tok!r} in: {orig_line}")
            continue

        if not prefix_has_error:
            b_tagged += 1

        allowed_numbers: Set[float] = set(const_numbers)
        for k, v in flat_facts.items():
            if any(k == p or k.startswith(p + "/") for p in prefixes):
                allowed_numbers.add(v)

        for line_no, tok, orig_line in block_tokens:
            if not prefix_has_error and _token_matches_facts(tok, allowed_numbers):
                matched += 1
            else:
                unmatched_tokens.append(tok)
                error_lines.append(f"line {line_no}: unmatched token {tok!r} in: {orig_line}")

    unmatched_count = len(unmatched_tokens)
    out_lines = [
        f"checked={checked} matched={matched} unmatched={unmatched_count} blocks={b_tagged}/{b_num}"
    ]
    for err in table_errors:
        out_lines.append(err)
    for err in error_lines:
        out_lines.append(err)

    exit_code = 0 if (unmatched_count == 0 and b_tagged == b_num and not table_errors and not error_lines) else 1
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
        help="Directory containing T1.md..T6.md and B1.md..B7.md",
    )
    args = parser.parse_args(argv)

    code, out_text, _ = check_draft_numbers(args.draft, args.facts, args.tables_dir)
    print(out_text)
    return code


if __name__ == "__main__":
    sys.exit(main())
