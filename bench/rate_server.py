"""SRB v0 — local rating UI server (docs/07 ⚠️ Issue 3, Option A.2).

An ergonomic front-end over the SAME git-native CSV workflow: it reads and
writes `ratings_maintainer.csv` (or the retest CSV) with the exact
`RATING_COLUMNS` schema, so `bench/adjudicate.py --validate/--make-retest/
--finalize` consume its output unchanged. Nothing downstream knows the
ratings came from a UI rather than a spreadsheet.

Why a server and not a bare `file://` page: browsers cannot write to local
files, and Safari refuses to play video without HTTP range support. This
serves the kits with proper 206 responses and autosaves every answer.

Stdlib only. Binds 127.0.0.1 (never exposed) — kits are Ego4D pixels and
stay internal-only (docs/07 §4.2).

Usage:
    python bench/rate_server.py                 # main rating round
    python bench/rate_server.py --retest        # blind washout re-rate
    python bench/rate_server.py --port 8080
"""
import argparse
import csv
import json
import os
import re
import socketserver
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from srb_common import BENCH_DATA, RATING_COLUMNS

HERE = Path(__file__).resolve().parent
MULTI_FIELDS = ("A3", "B6")


def blank_row(moment_id):
    row = {c: "" for c in RATING_COLUMNS}
    row["moment_id"] = moment_id
    return row


def apply_skip_logic(answer: dict) -> dict:
    """Return a CSV row with docs/07 §4.4 skip logic enforced: fields the form
    skips are written BLANK, never stale. Mirrors adjudicate.validate()'s
    expectations exactly, so a UI-produced CSV always validates."""
    row = blank_row(answer.get("moment_id", ""))
    for c in RATING_COLUMNS:
        if c in answer and answer[c] is not None:
            v = answer[c]
            row[c] = ";".join(v) if isinstance(v, list) else str(v).strip()

    triaged_out = row["A1"] != "yes" or row["A2"] != "yes"
    if triaged_out:
        for f in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
            row[f] = ""
        return row
    if row["B3"] != "yes":
        for f in ("B4", "B5", "B6", "B7"):
            row[f] = ""
        return row
    if row["B4"] != "wearer":
        row["B5"] = ""
    return row


class Store:
    """CSV-backed rating store: load on start, atomic rewrite on every save."""

    def __init__(self, csv_path: Path, moment_ids):
        self.path = Path(csv_path)
        self.order = list(moment_ids)
        self.rows = {m: blank_row(m) for m in self.order}
        self.reload()

    def reload(self):
        """Re-read the CSV from disk. Called before every write so edits made
        in a spreadsheet (or by another tool) while the server runs are merged
        rather than clobbered — the README promises the two are mixable."""
        if not self.path.exists():
            return
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                mid = (r.get("moment_id") or "").strip()
                if not mid:
                    continue
                self.rows[mid] = {c: (r.get(c) or "") for c in RATING_COLUMNS}
                self.rows[mid]["moment_id"] = mid
                if mid not in self.order:
                    self.order.append(mid)

    def save(self, answer: dict) -> dict:
        self.reload()
        mid = answer.get("moment_id")
        if mid not in self.rows:
            raise KeyError(mid)
        row = apply_skip_logic(answer)
        # Preserve the FIRST completion time — it is the docs/07 Issue-3
        # measurement; re-edits must not inflate it.
        prev = self.rows[mid].get("seconds_spent", "")
        if prev and not answer.get("_override_time"):
            row["seconds_spent"] = prev
        self.rows[mid] = row
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
            w.writeheader()
            for m in self.order:
                w.writerow(self.rows[m])
        os.replace(tmp, self.path)
        return row


def make_handler(store: Store, moments: list, kits_dir: Path, mode: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                     # quiet

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _serve_media(self, path: Path):
            """Range-aware file serving — required for <video> seeking."""
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            size = path.stat().st_size
            ctype = "video/mp4" if path.suffix == ".mp4" else "image/jpeg"
            rng = self.headers.get("Range")
            if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                length = max(0, end - start + 1)
                with open(path, "rb") as f:
                    f.seek(start)
                    chunk = f.read(length)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self._send(200, path.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._send(200, (HERE / "rater.html").read_bytes(), "text/html")
            if p == "/api/state":
                payload = {"mode": mode, "csv": str(store.path),
                           "moments": moments, "ratings": store.rows}
                return self._send(200, json.dumps(payload).encode())
            if p.startswith("/kits/"):
                rel = p[len("/kits/"):]
                if ".." in rel:
                    return self._send(403, b"nope", "text/plain")
                return self._serve_media(kits_dir / rel)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/rating":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            answer = json.loads(self.rfile.read(n) or b"{}")
            try:
                row = store.save(answer)
            except KeyError:
                return self._send(400, json.dumps({"error": "unknown moment"}).encode())
            return self._send(200, json.dumps({"ok": True, "row": row}).encode())

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retest", action="store_true",
                    help="Blind washout re-rate: serves kits_retest/ and writes "
                         "ratings_maintainer_retest.csv (round-one answers are "
                         "never loaded or shown).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    kits_dir = BENCH_DATA / ("kits_retest" if a.retest else "kits")
    csv_path = BENCH_DATA / ("ratings_maintainer_retest.csv" if a.retest
                             else "ratings_maintainer.csv")
    if not kits_dir.exists():
        raise SystemExit(f"[rate] no kits at {kits_dir} — run make_rating_kit.py first")

    with open(kits_dir / "kit_manifest.csv", newline="") as f:
        moments = [{"moment_id": r["moment_id"], "clip_id": r["clip_id"],
                    "t_climax_sec": float(r["t_climax_sec"]),
                    "has_splus": str(r.get("has_splus", "")).lower() == "true",
                    "task_label_hint": r.get("task_label_hint", "")}
                   for r in csv.DictReader(f)]

    store = Store(csv_path, [m["moment_id"] for m in moments])
    handler = make_handler(store, moments, kits_dir, "retest" if a.retest else "main")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), handler) as httpd:
        url = f"http://127.0.0.1:{a.port}/"
        done = sum(1 for r in store.rows.values() if r.get("A1"))
        print(f"[rate] {len(moments)} moments, {done} already rated ({'RETEST' if a.retest else 'main'} round)")
        print(f"[rate] writing -> {csv_path}")
        print(f"[rate] open {url}   (Ctrl-C to stop; every answer is saved immediately)")
        if not a.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[rate] stopped — CSV is up to date.")


if __name__ == "__main__":
    main()
