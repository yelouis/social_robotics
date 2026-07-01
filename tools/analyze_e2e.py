"""Analyze a v4 E2E filter run: pass rates, gate attribution, best/worst.

Usage: python analyze_e2e.py --results RESULTS.json --log DETECTOR.log
Gate attribution is recovered from the detector's stdout rejection lines:
  "[SocialPresenceDetector] Stereo gate rejected {uid}.mp4: ..."
  "[SocialPresenceDetector] VLM gate rejected {uid}.mp4: X/Y confirmed across N attempts"
A failed video with neither line had no YOLO-pose head+shoulder positive (or all
detections were chin/limb-filtered before the VLM gate).
"""
import argparse
import json
import re
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    recs = json.load(open(args.results))
    log = open(args.log, errors="ignore").read()

    stereo_uids = set(re.findall(r"Stereo gate rejected ([0-9a-zA-Z_\-]+)\.mp4", log))
    vlm_rej = dict(re.findall(r"VLM gate rejected ([0-9a-zA-Z_\-]+)\.mp4: (\d+/\d+ confirmed across \d+ attempts)", log))

    total = len(recs)
    skipped = [r for r in recs if r.get("skipped")]
    errors = [r for r in recs if r.get("error")]
    processed = [r for r in recs if "passed_filter" in r]
    passed = [r for r in processed if r["passed_filter"]]
    failed = [r for r in processed if not r["passed_filter"]]
    def uid8(r):
        return (r["video_id"] or "")[:8]

    # gate attribution for failed videos
    stereo_fail = [r for r in failed if r["video_id"] in stereo_uids]
    vlm_fail = [r for r in failed if r["video_id"] in vlm_rej and r["video_id"] not in stereo_uids]
    nopose_fail = [r for r in failed if r["video_id"] not in stereo_uids and r["video_id"] not in vlm_rej]

    print(f"=== E2E ANALYSIS: {args.results} ===")
    print(f"total entries:        {total}")
    print(f"  skipped:            {len(skipped)} ({sum(1 for s in skipped if s['skipped']=='metadata_solo')} solo, "
          f"{sum(1 for s in skipped if s['skipped']=='unsupported_dataset')} unsupported)")
    print(f"  errors:             {len(errors)}")
    print(f"  processed (YOLO+VLM): {len(processed)}")
    print(f"    PASSED:           {len(passed)}")
    print(f"    failed:           {len(failed)}")
    print(f"      stereo-gate:    {len(stereo_fail)}")
    print(f"      multi-person VLM gate: {len(vlm_fail)}")
    print(f"      no YOLO-pose positive / all-filtered: {len(nopose_fail)}")

    ego_proc = processed
    ego_pass = [r for r in ego_proc if r["passed_filter"]]
    print(f"\nEgo4D pass rate: {len(ego_pass)}/{len(ego_proc)} "
          f"({100*len(ego_pass)/max(1,len(ego_proc)):.1f}%)")

    if processed:
        times = [r["elapsed_sec"] for r in processed if "elapsed_sec" in r]
        if times:
            print(f"timing (processed): mean {st.mean(times):.1f}s median {st.median(times):.1f}s "
                  f"max {max(times):.1f}s total {sum(times)/60:.1f}min")

    # best pass / worst (most damaging) cases among Ego4D
    if ego_pass:
        best = max(ego_pass, key=lambda r: r["social_presence_score"])
        print(f"\nBEST PASS (Ego4D): {best['video_id']} score={best['social_presence_score']} "
              f"bystanders={best['bystander_count']} frames={best['frame_detections_total']} t={best['elapsed_sec']}s")
        print(f"  path: {best['file_path']}")
    # lowest-score pass (borderline) and the VLM-rejected-but-closest (1/2) cases
    low_pass = sorted(ego_pass, key=lambda r: r["social_presence_score"])[:5]
    print("\nlowest-score PASSES (possible false positives to spot-check):")
    for r in low_pass:
        print(f"  {r['video_id']} score={r['social_presence_score']} bys={r['bystander_count']} t={r['elapsed_sec']}s")
    near_miss = [r for r in vlm_fail if vlm_rej.get(r["video_id"], "").startswith("1/")]
    print("\nclosest VLM-gate FAILS (1/2 confirmed — possible false negatives to spot-check):")
    for r in near_miss[:8]:
        print(f"  {r['video_id']} [{vlm_rej[r['video_id']]}] t={r['elapsed_sec']}s")


if __name__ == "__main__":
    main()
