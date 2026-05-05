import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class DataAggregator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "filtered_manifest.json"
        
    def aggregate(self) -> pd.DataFrame:
        """
        Scans for filtered_manifest.json and 03*_result.json files,
        merges them into a single pandas DataFrame, and flattens nested structures.

        The real filtered_manifest.json has a deeply nested schema where
        `identified_tasks` is a list of task objects, each containing a
        `task_label`.  The aggregator extracts a comma-separated summary of
        all task labels into a single `task_labels` column, along with the
        standard top-level fields.

        Layer result JSONs vary in schema:
        - 03a, 03d, 03e, 03f, 03g use `aggregate` / `per_person` keys.
        - 03b, 03c use `tasks_analyzed` (a list of per-task results).
        Both shapes are handled: `aggregate` dict is flattened into prefixed
        columns, `per_person` and `tasks_analyzed` arrays are JSON-stringified.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        with open(self.manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        # Extract base dataframe from manifest
        base_records = []
        for item in manifest_data:
            # The real manifest stores task labels inside `identified_tasks`
            # as nested objects.  We flatten them into a comma-separated string.
            identified_tasks = item.get('identified_tasks', [])
            task_labels = ", ".join(
                t.get('task_label', 'Unknown') for t in identified_tasks
            ) if identified_tasks else item.get('task_label', 'Unknown')
            
            base_records.append({
                'video_id': item.get('video_id'),
                'source_dataset': item.get('source_dataset', 'Unknown'),
                'task_labels': task_labels,
                'duration_sec': item.get('duration_sec'),
                'fps': item.get('fps')
            })
        
        df = pd.DataFrame(base_records)
        
        # Scan for layer results
        layer_files = sorted(self.data_dir.glob("03*_result.json"))
        
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
                record = {'video_id': item['video_id']}
                
                # --- Schema A: layers with `aggregate` / `per_person` ---
                if 'aggregate' in item and isinstance(item['aggregate'], dict):
                    for k, v in item['aggregate'].items():
                        record[f"{layer_name}_{k}"] = v
                
                # Stringify per_person arrays
                if 'per_person' in item:
                    record[f"{layer_name}_per_person_raw"] = json.dumps(item['per_person'])
                
                # --- Schema B: layers with `tasks_analyzed` (e.g. 03b, 03c) ---
                if 'tasks_analyzed' in item:
                    record[f"{layer_name}_tasks_analyzed_raw"] = json.dumps(item['tasks_analyzed'])
                    # Also extract a summary scalar if possible
                    tasks = item['tasks_analyzed']
                    if tasks:
                        # For 03b: average task_aggregate_score across tasks
                        scores = [t.get('task_aggregate_score') for t in tasks if t.get('task_aggregate_score') is not None]
                        if scores:
                            record[f"{layer_name}_avg_task_score"] = round(sum(scores) / len(scores), 4)
                        # For 03c: average prosody_scalar across tasks
                        pscalars = [t.get('prosody_scalar') for t in tasks if t.get('prosody_scalar') is not None]
                        if pscalars:
                            record[f"{layer_name}_avg_prosody_scalar"] = round(sum(pscalars) / len(pscalars), 4)
                
                # Handle remaining scalar top-level fields
                for k, v in item.items():
                    if k in ('video_id', 'layer', 'aggregate', 'per_person', 'per_person_raw', 'tasks_analyzed'):
                        continue
                    if not isinstance(v, (dict, list)):
                        record[f"{layer_name}_{k}"] = v
                
                layer_records.append(record)
                
            layer_df = pd.DataFrame(layer_records)
            if not layer_df.empty:
                df = df.merge(layer_df, on='video_id', how='outer')
                
        return df

    def add_export_metadata(self, df: pd.DataFrame, active_layers: List[str], git_sha: str = "unknown") -> pd.DataFrame:
        """
        Attach export provenance to the DataFrame's attrs.
        The export script reads these attrs and writes them to export_metadata.json.
        """
        df.attrs['schema_version'] = "1.0.0"
        df.attrs['export_timestamp'] = datetime.now(timezone.utc).isoformat()
        df.attrs['active_layers'] = active_layers
        df.attrs['pipeline_git_sha'] = git_sha
        return df
