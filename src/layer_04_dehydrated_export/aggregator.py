import json
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Any, Iterable
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer summary registry (Resolved Issue 10)
# ---------------------------------------------------------------------------
# Each entry maps a layer name to a list of (path, aggregation, suffix)
# tuples. The aggregator iterates this registry and emits a Parquet-queryable
# scalar column `<layer_name>_<suffix>` for every entry.
#
# Path syntax:
#   "field"               — task-level scalar inside each tasks_analyzed entry
#   "per_person.field"    — scalar inside the per_person array of each task
#
# Aggregations:
#   "mean"        — arithmetic mean of all collected values (numeric only)
#   "max"         — maximum value
#   "max_abs"     — value with the largest absolute magnitude (signed)
#   "any"         — True if any value is truthy
#   "any_eq:V"    — True if any value equals the literal V
LAYER_SUMMARY_REGISTRY = {
    "03b_reasonable_emotion": [
        ("task_aggregate_score", "mean", "avg_task_score"),
    ],
    "03c_acoustic_prosody": [
        ("prosody_scalar", "mean", "avg_prosody_scalar"),
    ],
    "03d_proxemic_kinematics": [
        ("per_person.proxemic_confidence", "max", "max_proxemic_confidence"),
        ("per_person.proxemic_vector", "max_abs", "max_abs_proxemic_vector"),
        ("per_person.classified_action", "any_eq:Approach", "any_approach_detected"),
        ("per_person.classified_action", "any_eq:Retreat", "any_retreat_detected"),
    ],
    "03e_affirmation_gesture": [
        ("per_person.confidence", "max", "max_gesture_confidence"),
        ("per_person.gesture_detected", "any_eq:affirming_nod", "any_nod_detected"),
        ("per_person.gesture_detected", "any_eq:negating_shake", "any_shake_detected"),
    ],
    "03f_motor_resonance": [
        ("ego_kinetic_chaos_score", "max", "max_ego_chaos_score"),
        ("per_person.empathy_scalar", "max", "max_empathy_scalar"),
        ("per_person.motor_resonance_detected", "any", "any_motor_resonance"),
        ("per_person.mirroring_scalar", "max", "max_mirroring_scalar"),
        ("per_person.mirroring_detected", "any", "any_mirroring"),
    ],
    "03g_shared_reality": [
        ("social_reference_sought", "any", "any_social_reference"),
        ("bystander_centered_in_fov", "any", "any_bystander_centered"),
    ],
}

# Resolved Issue 13: Pandas DataFrame in-memory footprint is typically 5-10x
# the source JSON byte size due to Python object overhead, string interning,
# and column dtype promotion. Using a midpoint-of-range constant prevents the
# memory check from drastically underestimating Pandas state. Adjust if a
# specific dataset's column dtype distribution shifts the empirical multiplier.
PANDAS_INFLATION_FACTOR = 8


def _collect_layer_values(item: dict, path: str) -> Iterable[Any]:
    """Yield the scalar values addressed by ``path`` from a single layer record.

    For a record with shape ``{"tasks_analyzed": [{...}, ...]}``, "field" yields
    the value of ``field`` from each task; "per_person.field" yields the value
    of ``field`` from each per-person entry across all tasks. Missing or
    non-iterable intermediate keys are skipped silently.
    """
    tasks = item.get('tasks_analyzed', [])
    if not isinstance(tasks, list):
        return
    if path.startswith("per_person."):
        sub_key = path[len("per_person."):]
        for task in tasks:
            if not isinstance(task, dict):
                continue
            persons = task.get('per_person', [])
            if not isinstance(persons, list):
                continue
            for p in persons:
                if isinstance(p, dict) and sub_key in p:
                    yield p[sub_key]
    else:
        for task in tasks:
            if isinstance(task, dict) and path in task:
                yield task[path]


def _apply_aggregation(values: list, agg: str) -> Optional[Any]:
    if not values:
        return None
    if agg == "mean":
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not nums:
            return None
        return round(sum(nums) / len(nums), 4)
    if agg == "max":
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not nums:
            return None
        return max(nums)
    if agg == "max_abs":
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not nums:
            return None
        return max(nums, key=lambda x: abs(x))
    if agg == "any":
        return bool(any(values))
    if agg.startswith("any_eq:"):
        target = agg.split(":", 1)[1]
        return bool(any(v == target for v in values))
    raise ValueError(f"Unknown aggregation '{agg}'")


class DataAggregator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "filtered_manifest.json"

    def aggregate(self, output_parquet_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Outer-join the manifest and every ``03*_result.json`` into a single
        wide DataFrame, with per-layer summary scalars surfaced as columns.

        When ``output_parquet_path`` is provided AND the dataset triggers the
        Dask path, the merged result is streamed to a partitioned Parquet
        directory and ``None`` is returned (memory-bounded). Otherwise the
        merged DataFrame is materialized into Pandas and returned.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, 'r') as f:
            manifest_data = json.load(f)

        base_records = []
        for item in manifest_data:
            identified_tasks = item.get('identified_tasks', [])
            task_labels = ", ".join(
                t.get('task_label', 'Unknown') for t in identified_tasks
            ) if identified_tasks else item.get('task_label', 'Unknown')

            base_records.append({
                'video_id': item.get('video_id'),
                'source_dataset': item.get('source_dataset', 'Unknown'),
                'task_labels': task_labels,
                'duration_sec': item.get('duration_sec'),
                'fps': item.get('fps'),
            })

        # Memory check — see PANDAS_INFLATION_FACTOR comment for rationale.
        try:
            import psutil
            mem_available = psutil.virtual_memory().available
        except ImportError:
            mem_available = 8 * 1024 ** 3  # Fallback estimate: 8GB

        layer_files = sorted(self.data_dir.glob("03*_result.json"))
        total_size = self.manifest_path.stat().st_size + sum(f.stat().st_size for f in layer_files)
        effective_size = total_size * PANDAS_INFLATION_FACTOR

        use_dask = effective_size > mem_available * 0.5
        npartitions = 1
        if use_dask:
            # Resolved Issue 12: size each partition at ~10% of available
            # memory so the merge shuffles partition-by-partition rather than
            # materializing the whole dataset at once.
            npartitions = max(2, int(effective_size / max(1, int(mem_available * 0.1))))
            logger.warning(
                "Dataset effective size (%.2f GB) exceeds 50%% of available "
                "memory; falling back to Dask with %d partitions.",
                effective_size / (1024 ** 3), npartitions,
            )
            try:
                import dask.dataframe as dd
            except ImportError:
                logger.warning("Dask is not installed. Proceeding with Pandas; OOM risk.")
                use_dask = False
                npartitions = 1

        df = pd.DataFrame(base_records)
        if use_dask:
            df = dd.from_pandas(df, npartitions=npartitions)

        for layer_file in layer_files:
            layer_name = layer_file.stem.replace("_result", "")
            with open(layer_file, 'r') as f:
                try:
                    layer_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse {layer_file}. Skipping.")
                    continue

            layer_records = []
            for item in layer_data:
                # Resolved Issue 15: a record missing video_id used to raise
                # KeyError and abort the entire aggregation. Skip & log instead.
                vid = item.get('video_id') if isinstance(item, dict) else None
                if not vid:
                    logger.warning(
                        "Skipping record without video_id in %s", layer_file.name,
                    )
                    continue
                record = {'video_id': vid}

                # --- Schema A: layers with `aggregate` / `per_person` ---
                if 'aggregate' in item and isinstance(item['aggregate'], dict):
                    for k, v in item['aggregate'].items():
                        record[f"{layer_name}_{k}"] = v

                if 'per_person' in item:
                    record[f"{layer_name}_per_person_raw"] = json.dumps(item['per_person'])

                # --- Schema B: layers with `tasks_analyzed` ---
                if 'tasks_analyzed' in item:
                    record[f"{layer_name}_tasks_analyzed_raw"] = json.dumps(item['tasks_analyzed'])

                # Per-layer summary scalars (Resolved Issue 10). The registry
                # is the single source of truth for which fields are surfaced
                # as Parquet-queryable columns; downstream researchers can
                # filter and aggregate without parsing the *_raw JSON columns.
                for path, agg, suffix in LAYER_SUMMARY_REGISTRY.get(layer_name, []):
                    values = list(_collect_layer_values(item, path))
                    summary = _apply_aggregation(values, agg)
                    if summary is not None:
                        record[f"{layer_name}_{suffix}"] = summary

                # Top-level provenance dicts (Resolved Issue 11). One level of
                # flattening: <layer>_<dict_key>_<sub_key>. Any deeper nesting
                # is JSON-stringified to keep the Parquet schema bounded.
                special_keys = {
                    'video_id', 'layer', 'aggregate', 'per_person',
                    'per_person_raw', 'tasks_analyzed',
                }
                for k, v in item.items():
                    if k in special_keys:
                        continue
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            if isinstance(sv, (dict, list)):
                                record[f"{layer_name}_{k}_{sk}"] = json.dumps(sv)
                            else:
                                record[f"{layer_name}_{k}_{sk}"] = sv
                    elif isinstance(v, list):
                        # Lists at the top level (other than per_person /
                        # tasks_analyzed) get JSON-stringified to preserve them.
                        record[f"{layer_name}_{k}_raw"] = json.dumps(v)
                    else:
                        record[f"{layer_name}_{k}"] = v

                layer_records.append(record)

            layer_df = pd.DataFrame(layer_records)
            if not layer_df.empty:
                if use_dask:
                    layer_dd = dd.from_pandas(layer_df, npartitions=npartitions)
                    df = dd.merge(df, layer_dd, on='video_id', how='outer')
                else:
                    df = df.merge(layer_df, on='video_id', how='outer')

        if use_dask:
            if output_parquet_path is not None:
                # Resolved Issue 12: stream-write the partitioned merge result
                # without materializing the full DataFrame in memory.
                df.to_parquet(str(output_parquet_path), engine='pyarrow', write_index=False)
                return None
            df = df.compute()

        return df

    def add_export_metadata(self, df: pd.DataFrame, active_layers: List[str], git_sha: str = "unknown") -> pd.DataFrame:
        """Attach export provenance to ``df.attrs``. The export script reads
        these attrs and writes them to ``export_metadata.json``."""
        column_hash = hashlib.sha256(
            ",".join(sorted(df.columns)).encode("utf-8")
        ).hexdigest()[:6]
        # Resolved Issue 14: the SemVer prefix preserves human-readable
        # versioning for documented column add/remove changes; the hash
        # suffix lets downstream consumers detect schema drift mechanically
        # by diffing the columns hash between two exports.
        df.attrs['schema_version'] = f"1.0.0+{column_hash}"
        df.attrs['export_timestamp'] = datetime.now(timezone.utc).isoformat()
        df.attrs['active_layers'] = active_layers
        df.attrs['pipeline_git_sha'] = git_sha
        return df
