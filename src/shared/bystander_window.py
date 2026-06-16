"""Shared bystander measurement-window anchoring (cross-layer).

THE RECURRING PROBLEM
---------------------
The reaction-bearing 03* layers want to measure a bystander's response around a
task event. The task's `task_reaction_window_sec` is anchored to the **wearer's**
optical-flow climax (±~1 s), but Node-02's bystander detections arrive at a
**median ~6 s cadence** and sit a **median ~7–10 s from that climax**. So the
strict reaction window typically holds **0–1 bystander detections**, starving any
metric that needs the bystander to be present/sampled there. This same
wearer-vs-bystander mismatch was hit and fixed independently in 03b (Resolved
#2), 03d (Resolved #1/#3), and 03e (Resolved #1) — this module is the single
canonical implementation those layers (and 03f) share.

THE ANCHORING RULE
------------------
1. If the strict reaction window already holds `>= min_in_reaction_window` of the
   relevant samples, keep it unchanged (`"reaction_window"`).
2. Otherwise re-anchor to the bystander DETECTION nearest the climax, widened by
   `± anchor_span_detections` detection indices and padded by `± pad_sec`
   seconds (`"bystander_anchored"`) — where the detections (and thus the
   bystander's resolvable signal) actually are.
3. Guard rails: a track too sparse to anchor returns a skip reason
   (`no_detection_reason`); an anchored span wider than `max_anchor_span_sec`
   returns `"span_capped"` (a delta/measurement over minutes is locomotion/drift,
   not a task reaction — 03d Resolved #3).

WHY IT IS PARAMETERIZED (the layers genuinely differ)
-----------------------------------------------------
- **03d** (proxemic delta) counts DETECTIONS for the dense-window test
  (`min_in_reaction_window=2` — a delta needs 2 endpoints), no padding, and
  treats <2 detections as unmeasurable (`allow_single_detection=False`,
  `no_detection_reason="single_detection"`).
- **03e** (gesture signal) counts the upstream 03a TRACE samples for the
  dense-window test (`keep_timestamps=trace`, `min_in_reaction_window=5`), pads by
  03a's ±2 s sampling halo, and a single detection's padded span is enough
  (`allow_single_detection=True`, `no_detection_reason="insufficient_trace"`).
- **03f** (motor resonance) is the next intended consumer (docs/03f Issue 1).

Returns `(start, end, source)` on success — `source` ∈ {`"reaction_window"`,
`"bystander_anchored"`} — or `(None, None, reason)` on skip, where `reason` ∈
{`no_detection_reason`, `"single_detection"`, `"span_capped"`}.
"""

__all__ = ["bystander_measurement_window"]


def bystander_measurement_window(
    detection_timestamps,
    climax_sec,
    reaction_start_sec,
    reaction_end_sec,
    *,
    keep_timestamps=None,
    min_in_reaction_window=2,
    anchor_span_detections=1,
    pad_sec=0.0,
    max_anchor_span_sec=30.0,
    allow_single_detection=False,
    no_detection_reason="single_detection",
):
    """Anchor a measurement window to the bystander near the climax.

    Args:
        detection_timestamps: the bystander's Node-02 detection times — what the
            anchored window is built from.
        climax_sec: the task's climax (wearer optical-flow peak).
        reaction_start_sec, reaction_end_sec: the strict reaction window.
        keep_timestamps: times counted for the "reaction window already dense"
            test. Defaults to `detection_timestamps` (03d); 03e passes the 03a
            trace timestamps instead.
        min_in_reaction_window: keep the strict window when it holds at least this
            many `keep_timestamps`.
        anchor_span_detections: widen ± this many detection indices around the
            climax-nearest detection.
        pad_sec: pad the anchored span by ± this many seconds (03e's 03a-sampling
            halo; 0 for 03d's index-exact span).
        max_anchor_span_sec: anchored spans wider than this are skipped
            (`"span_capped"`).
        allow_single_detection: if False, a track with <2 detections is
            unmeasurable (`no_detection_reason`); if True, a single detection's
            padded span is acceptable.
        no_detection_reason: skip reason when there is no usable detection to
            anchor on (03d: "single_detection"; 03e: "insufficient_trace").

    Returns:
        (start, end, source) or (None, None, reason). See module docstring.
    """
    keep = keep_timestamps if keep_timestamps is not None else detection_timestamps
    in_window = sum(1 for t in keep if reaction_start_sec <= t <= reaction_end_sec)
    if in_window >= min_in_reaction_window:
        return reaction_start_sec, reaction_end_sec, "reaction_window"

    ts = sorted(detection_timestamps)
    if not ts:
        return None, None, no_detection_reason
    if len(ts) < 2 and not allow_single_detection:
        return None, None, "single_detection"

    i = min(range(len(ts)), key=lambda k: abs(ts[k] - climax_sec))
    lo = max(0, i - anchor_span_detections)
    hi = min(len(ts) - 1, i + anchor_span_detections)
    ws = ts[lo] - pad_sec
    we = ts[hi] + pad_sec

    if we <= ws:
        # Degenerate span (e.g. a lone or duplicated detection with no padding):
        # there is no interval to measure over.
        return None, None, "single_detection"
    if (we - ws) > max_anchor_span_sec:
        return None, None, "span_capped"
    return ws, we, "bystander_anchored"
