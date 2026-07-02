"""Cross-layer reaction-window consumption helpers + back-compat wrappers.

The climax DETECTOR now lives in **Layer 02b** (`layer_02b_task_climax.pipeline`)
— run `python -m layer_02b_task_climax.pipeline <manifest>` after Node 02 to
annotate the manifest explicitly. History of the move:

- Per docs/02 Resolved Issue #8 (May 17), Node 02 emits `identified_tasks`
  with an empty `task_temporal_metadata = {}` and the first Layer 03 pipeline
  to run filled it lazily via `populate_climax_for_manifest()`.
- The June 23 multi-window rework (docs/02 Resolved #22) made the detector
  bystander-aware: one reaction *segment* per bystander-detection cluster,
  consumed via `iter_reaction_windows()` / `expand_task_segments()` below.
- The June 30 A/B eval replaced the per-cluster optical-flow peak (proven to
  track ego-motion, not social moments) with the bbox-kernel detector and
  promoted the whole stage to Layer 02b (docs/02b_task_climax_layer.md).

This module keeps (a) the consumption helpers every Layer 03 imports, and
(b) thin wrappers so existing call sites — 03b's lazy populate, tools/ — keep
working against the Layer 02b implementation unchanged.
"""

from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple

try:
    from layer_02b_task_climax.pipeline import (  # noqa: F401  (re-exported)
        CLIMAX_CLUSTER_GAP_SEC,
        CLIMAX_MAX_CLUSTER_SPAN_SEC,
        CLIMAX_MIN_CLUSTER_DETECTIONS,
        CLIMAX_MAX_SEGMENTS,
        CLIMAX_CHECKPOINT_EVERY,
        TaskClimaxPipeline,
        compute_task_climax_for_video,
        _cluster_timestamps,
        _bystander_timestamps_in,
    )
except ImportError:
    from src.layer_02b_task_climax.pipeline import (  # noqa: F401
        CLIMAX_CLUSTER_GAP_SEC,
        CLIMAX_MAX_CLUSTER_SPAN_SEC,
        CLIMAX_MIN_CLUSTER_DETECTIONS,
        CLIMAX_MAX_SEGMENTS,
        CLIMAX_CHECKPOINT_EVERY,
        TaskClimaxPipeline,
        compute_task_climax_for_video,
        _cluster_timestamps,
        _bystander_timestamps_in,
    )


def iter_reaction_windows(task: dict) -> Iterator[Tuple[float, list]]:
    """Yield `(climax_sec, [start, end])` for each reaction segment of a task.

    This is the canonical way for Layer 03 pipelines to consume reaction windows:
    it transparently handles the multi-segment schema (one window per bystander
    cluster) and the legacy single-window format. A task with no usable window
    yields nothing."""
    meta = task.get('task_temporal_metadata') or {}
    segments = meta.get('reaction_segments')
    if segments:
        for s in segments:
            w = s.get('task_reaction_window_sec')
            if w and len(w) == 2:
                yield s.get('task_climax_sec', (w[0] + w[1]) / 2.0), w
    else:
        w = meta.get('task_reaction_window_sec')
        if w and len(w) == 2:
            yield meta.get('task_climax_sec', (w[0] + w[1]) / 2.0), w


def expand_task_segments(tasks: Iterable[dict]) -> Iterator[dict]:
    """Yield one shallow-copied 'pseudo-task' per reaction segment, each carrying
    a *single-window* `task_temporal_metadata`, so an existing per-task loop
    becomes per-segment with no restructuring — replace `for task in tasks:` with
    `for task in expand_task_segments(tasks):` and the body is unchanged. Each
    yielded task keeps all original fields (`task_id`, `task_start_sec`,
    `task_velocity`, …) and adds `segment_index` / `n_segments`. A task whose
    range produced no usable reaction window yields nothing (correctly skipped)."""
    for task in tasks:
        windows = list(iter_reaction_windows(task))
        n = len(windows)
        for i, (climax_sec, window) in enumerate(windows):
            pseudo = dict(task)
            base_meta = dict(task.get('task_temporal_metadata') or {})
            base_meta.pop('reaction_segments', None)  # collapse to a single-window view
            base_meta['task_reaction_window_sec'] = window
            base_meta['task_climax_sec'] = climax_sec
            pseudo['task_temporal_metadata'] = base_meta
            pseudo['segment_index'] = i
            pseudo['n_segments'] = n
            yield pseudo


def populate_climax_for_manifest(
    manifest_path: Path,
    vlm_model: Optional[str] = None,   # deprecated (flow-era) — accepted, ignored
    skip_vlm: bool = False,            # deprecated (flow-era) — accepted, ignored
    workers: Optional[int] = None,
    entry_filter: Optional[Callable[[dict], bool]] = None,
) -> int:
    """Back-compat lazy-populate wrapper over the Layer 02b pipeline.

    Kept so Layer 03 pipelines (03b) and existing tools keep working on a
    manifest that was not explicitly pre-annotated; the supported production
    path is running Layer 02b first (`python -m layer_02b_task_climax.pipeline`),
    after which this is a cheap no-op. Returns the number of entries updated.
    """
    return TaskClimaxPipeline(
        manifest_path, workers=workers, entry_filter=entry_filter,
    ).run()
