"""SRB v0 — Claude pre-seeding support (docs/07 §4.6 deviation, user-directed).

The maintainer asked Claude to pre-rate every moment with a written rationale
so the rating round becomes review-and-correct rather than blank-form entry.
This is an ACKNOWLEDGED deviation from the anti-anchoring policy; the
mitigations live in the data model:

  * Seeds are stored in `claude_seeds.jsonl` — a SEPARATE file, never inside
    ratings_maintainer.csv. The human's answers remain the only thing
    adjudicate.py reads.
  * The UI records, per field, whether the human CHANGED the seed or accepted
    it unchanged (`accepted_fields` / `changed_fields` in the seed-audit CSV),
    so post-hoc anchoring analysis and honest disclosure are possible.
  * Claude cannot hear audio (frames only), so voice-channel evidence is a
    known systematic blind spot in the seeds — recorded in every seed as
    `blind:["voice"]` rather than silently absent.

CLI (used by Claude across checkpointed sessions — see PRESEED_INSTRUCTIONS.md):
    python bench/preseed.py --status
    python bench/preseed.py --next 8            # work-order for the next batch
    python bench/preseed.py --frames <moment_id> [--after]
    python bench/preseed.py --record seeds.json # append validated seeds
"""
import argparse
import json
import subprocess
from pathlib import Path

from srb_common import BENCH_DATA, ENUMS, read_jsonl, write_jsonl

SEED_DIR = BENCH_DATA / "seeds"          # one JSONL per seeding model
FRAME_DIR = BENCH_DATA / "_preseed_frames"
SEED_FIELDS = ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
FRAMES_PER_CLIP = 6


def seed_path(model: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model)
    return SEED_DIR / f"{safe}.jsonl"


def load_seeds(model: str = None):
    """Seeds for one model, or {moment_id: {model: seed}} across all models."""
    if model:
        p = seed_path(model)
        return {s["moment_id"]: s for s in (read_jsonl(p) if p.exists() else [])}
    out = {}
    for p in sorted(SEED_DIR.glob("*.jsonl")) if SEED_DIR.exists() else []:
        if p.name.startswith("."):
            continue
        for s in read_jsonl(p):
            out.setdefault(s["moment_id"], {})[s.get("model", p.stem)] = s
    return out


def kit_moments():
    kits = BENCH_DATA / "kits"
    import csv
    with open(kits / "kit_manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def extract_frames(moment_id, after=False):
    """Even frames across the clip, written as JPEGs for Claude to view."""
    src = BENCH_DATA / "kits" / moment_id / ("clip_splus.mp4" if after else "clip_s.mp4")
    if not src.exists():
        return []
    out = FRAME_DIR / moment_id / ("after" if after else "moment")
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    if len(existing) >= FRAMES_PER_CLIP:
        return [str(p) for p in existing]
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src),
                    "-vf", f"fps={FRAMES_PER_CLIP}/7,scale=-2:360",
                    "-frames:v", str(FRAMES_PER_CLIP),
                    str(out / "f%02d.jpg"), "-y"], capture_output=True)
    return [str(p) for p in sorted(out.glob("*.jpg"))]


def validate_seed(s, modality="frames"):
    """Same enum + skip-logic contract the human form uses, so a seed can never
    pre-fill an invalid combination. `modality` additionally gates the voice
    channel: a frame-based seeder cannot have heard anything."""
    errs = []
    mid = s.get("moment_id")
    if not mid:
        return ["missing moment_id"]
    if not s.get("rationale"):
        errs.append("missing rationale (every seeded field needs a stated reason)")
    if modality == "frames":
        b6 = s.get("B6") or ""
        b6 = ";".join(b6) if isinstance(b6, list) else str(b6)
        if "voice" in b6.lower():
            errs.append("B6 lists 'voice' but this seed is frame-based (no audio was "
                        "available) — use --modality video only if the model actually "
                        "received the clip's audio track")
    for f in ("A1", "A2"):
        if s.get(f) not in ENUMS[f]:
            errs.append(f"{f}={s.get(f)!r} invalid")
    triaged_out = s.get("A1") != "yes" or s.get("A2") != "yes"
    if not triaged_out:
        if not s.get("B1"):
            errs.append("B1 required")
        for f in ("B2", "B3", "B8"):
            if s.get(f) not in ENUMS[f]:
                errs.append(f"{f}={s.get(f)!r} invalid")
        if s.get("B3") == "yes":
            if s.get("B4") not in ENUMS["B4"]:
                errs.append("B4 required when B3=yes")
            if s.get("B4") == "wearer" and s.get("B5") not in ENUMS["B5"]:
                errs.append("B5 required when B4=wearer")
            if s.get("B4") != "wearer" and s.get("B5"):
                errs.append("B5 must be empty when B4!=wearer")
        else:
            for f in ("B4", "B5", "B7"):
                if s.get(f):
                    errs.append(f"{f} must be empty when B3={s.get('B3')}")
    return errs


def agreement_report():
    """Where do the seeding models disagree? Those moments are the ones worth
    the human's attention first — and they are deliberately NOT pre-filled."""
    all_seeds = load_seeds()
    fields = ("A1", "A2", "B2", "B3", "B4", "B5", "B7")
    per_field = {f: {"agree": 0, "conflict": 0} for f in fields}
    conflicts = []
    for mid, by_model in all_seeds.items():
        if len(by_model) < 2:
            continue
        bad = []
        for f in fields:
            vals = {str(s.get(f) or "").lower() for s in by_model.values()}
            vals.discard("")
            if len(vals) > 1:
                per_field[f]["conflict"] += 1
                bad.append(f)
            elif vals:
                per_field[f]["agree"] += 1
        if bad:
            conflicts.append({"moment_id": mid, "fields": bad,
                              "models": sorted(by_model)})
    return {"n_moments_multi_seeded": sum(1 for v in all_seeds.values() if len(v) > 1),
            "per_field": per_field, "n_conflicting_moments": len(conflicts),
            "conflicts": conflicts[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Seeding model id, e.g. claude-opus-5 / gpt-5 / gemini-3-pro. "
                         "Required for --next and --record; each model writes its own file.")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--next", type=int, metavar="N")
    ap.add_argument("--frames", metavar="MOMENT_ID")
    ap.add_argument("--after", action="store_true")
    ap.add_argument("--record", metavar="SEEDS_JSON")
    ap.add_argument("--agreement", action="store_true",
                    help="Cross-model agreement + the list of conflicting moments.")
    ap.add_argument("--modality", choices=("video", "frames"), default="video",
                    help="Seeding is VIDEO-ONLY by policy (PRESEED_PROMPT.md): the model "
                         "must watch the mp4 with its audio, because tone routinely decides "
                         "B3/B5/B6. 'frames' is a deprecated escape hatch for a seeder that "
                         "genuinely cannot receive video; it forbids voice evidence and "
                         "marks the seeds blind.")
    a = ap.parse_args()

    moments = kit_moments()

    if a.agreement:
        print(json.dumps(agreement_report(), indent=2))
        return

    if a.status:
        all_seeds = load_seeds()
        by_model = {}
        for by in all_seeds.values():
            for m in by:
                by_model[m] = by_model.get(m, 0) + 1
        mine = len(load_seeds(a.model)) if a.model else None
        print(json.dumps({"kits": len(moments), "seeded_by_model": by_model,
                          "any_seed": len(all_seeds),
                          "this_model": {"model": a.model, "seeded": mine,
                                         "remaining": (len(moments) - mine) if mine is not None else None},
                          "seed_dir": str(SEED_DIR)}, indent=2))
        return

    if a.next:
        if not a.model:
            raise SystemExit("--next requires --model (seeds are per-model)")
        done = load_seeds(a.model)
        todo = [m for m in moments if m["moment_id"] not in done][:a.next]
        for m in todo:
            kit = BENCH_DATA / "kits" / m["moment_id"]
            # Video-capable models take these directly (audio included); frame-only
            # models use the extracted stills. Both are always provided.
            m["clip_moment"] = str(kit / "clip_s.mp4")
            m["clip_after"] = str(kit / "clip_splus.mp4") if (kit / "clip_splus.mp4").exists() else None
            if a.modality == "frames":
                m["frames_moment"] = extract_frames(m["moment_id"])
                m["frames_after"] = extract_frames(m["moment_id"], after=True)
        print(json.dumps(todo, indent=1))
        return

    if a.frames:
        print(json.dumps({"moment": extract_frames(a.frames),
                          "after": extract_frames(a.frames, after=True)}, indent=1))
        return

    if a.record:
        if not a.model:
            raise SystemExit("--record requires --model (seeds are per-model)")
        incoming = json.loads(Path(a.record).read_text())
        incoming = incoming if isinstance(incoming, list) else [incoming]
        bad = [{"moment_id": s.get("moment_id"), "errors": e}
               for s in incoming if (e := validate_seed(s, a.modality))]
        if bad:
            print(json.dumps({"rejected": bad}, indent=2))
            raise SystemExit(1)
        if a.modality == "frames":
            print("[preseed] WARNING: recording FRAME-BASED seeds. Policy is video-only "
                  "(PRESEED_PROMPT.md) — these seeds cannot carry voice evidence and are "
                  "stamped blind:['voice']. Use only if the seeder truly cannot take video.")
        known = {m["moment_id"] for m in moments}
        seeds = load_seeds(a.model)
        for s in incoming:
            if s["moment_id"] not in known:
                raise SystemExit(f"unknown moment_id {s['moment_id']}")
            s["modality"] = a.modality
            # Frame-based seeding is deaf; native-video seeding is not.
            s["blind"] = [] if a.modality == "video" else ["voice"]
            s["model"] = a.model
            seeds[s["moment_id"]] = s
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        ordered = [seeds[m["moment_id"]] for m in moments if m["moment_id"] in seeds]
        write_jsonl(seed_path(a.model), ordered)
        remaining = len(moments) - len(ordered)
        # Durable checkpoint trail: a resuming agent (or a human) can read the
        # whole history without reconstructing it from the seed file.
        import time
        with open(SEED_DIR / f"{a.model}.progress.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  +{len(incoming):3d}  "
                    f"total={len(ordered):4d}  remaining={remaining:4d}  "
                    f"modality={a.modality}\n")
        print(json.dumps({"model": a.model, "accepted": len(incoming),
                          "total_seeded": len(ordered),
                          "remaining": remaining,
                          "checkpoint_log": str(SEED_DIR / f"{a.model}.progress.log")}, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
