"""Top-K-by-score retention reservoir for Node 01 acquisition.

The full Ego4D corpus (and its social-passing subset) far exceeds the 2 TB
working SSD, so acquisition keeps only a bounded, highest-value set: at most
`cap` social-passing videos — those with the highest `social_presence_score`
(the summed bystander detection confidence, docs/02) — subject to a hard
`disk_budget_bytes` ceiling on the kept video files. When a new passer arrives
and a limit is exceeded, the lowest-scored entry is evicted and its file
deleted. Score is total social presence, so a longer clip is retained over a
shorter one only when it actually carries more bystander-detection mass.

State persists to JSON so acquisition is resumable (paired with the
orchestrator's `processed_uids` save point); the kept set is exported as
`filtered_manifest.json` for Layer 03. See docs/01_dataset_acquisition.md.
"""
import json
import os
from pathlib import Path

DEFAULT_CAP = 1000
DEFAULT_DISK_BUDGET_BYTES = 950 * 2**30  # ~950 GiB; headroom under the 1.1 TiB free


class ScoreReservoir:
    def __init__(self, cap=DEFAULT_CAP, disk_budget_bytes=DEFAULT_DISK_BUDGET_BYTES,
                 state_path=None, dry_run=False):
        self.cap = int(cap)
        self.disk_budget_bytes = int(disk_budget_bytes)
        self.state_path = Path(state_path) if state_path else None
        self.dry_run = dry_run
        self.entries = {}      # uid -> {uid, score, file_path, file_size, manifest}
        self.evicted_log = []  # [(uid, score), ...] evicted/rejected this session (diagnostics)

    # --- persistence -------------------------------------------------------
    def load(self):
        if self.state_path and self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
            self.entries = {e["uid"]: e for e in data.get("entries", [])}
        return self

    def save(self):
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({
                "cap": self.cap,
                "disk_budget_bytes": self.disk_budget_bytes,
                "kept": len(self.entries),
                "total_bytes": self.total_bytes(),
                "entries": list(self.entries.values()),
            }, f, indent=2)
        os.replace(tmp, self.state_path)  # atomic swap

    # --- queries -----------------------------------------------------------
    def total_bytes(self):
        return sum(int(e.get("file_size", 0)) for e in self.entries.values())

    def _lowest_uid(self):
        if not self.entries:
            return None
        return min(self.entries.values(), key=lambda e: e["score"])["uid"]

    def min_score(self):
        return min((e["score"] for e in self.entries.values()), default=None)

    # --- core --------------------------------------------------------------
    def _delete_file(self, path):
        if self.dry_run or not path:
            return
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def _evict_lowest(self):
        uid = self._lowest_uid()
        if uid is None:
            return None
        entry = self.entries.pop(uid)
        self._delete_file(entry.get("file_path"))
        self.evicted_log.append((uid, entry["score"]))
        return uid

    def consider(self, uid, score, file_path, file_size, manifest_entry):
        """Offer a *passing* video (score > 0) to the reservoir.

        Returns (status, displaced_uids):
          'kept'     -> the candidate is retained; displaced_uids lists any
                        previously-kept entries evicted to make room (files deleted).
          'rejected' -> the candidate did not make the cut; its file is deleted
                        here, so the caller need not purge it again.
        Idempotent on uid (re-considering updates the entry in place).
        """
        self.entries[uid] = {
            "uid": uid,
            "score": float(score),
            "file_path": str(file_path),
            "file_size": int(file_size),
            "manifest": manifest_entry,
        }
        displaced = []
        while len(self.entries) > self.cap or self.total_bytes() > self.disk_budget_bytes:
            ev = self._evict_lowest()
            if ev is None:
                break
            if ev == uid:
                return ("rejected", [])
            displaced.append(ev)
        if uid not in self.entries:
            return ("rejected", [])
        return ("kept", displaced)

    def export_manifest(self, manifest_path):
        """Write filtered_manifest.json = the kept set's manifest records,
        highest score first. This is the Layer 03 save-state."""
        manifest = [e["manifest"] for e in
                    sorted(self.entries.values(), key=lambda x: -x["score"])]
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, manifest_path)
        return len(manifest)
