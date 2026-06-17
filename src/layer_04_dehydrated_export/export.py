import re
import json
from pathlib import Path

import pandas as pd

try:
    from .aggregator import schema_column_hash
except ImportError:  # when imported with src/ on the path rather than as a package
    from layer_04_dehydrated_export.aggregator import schema_column_hash

# The internal layer-id token the pipeline uses on every column (03a_, 03f_, ...).
# Stripped at publish time so the released dataset does not carry the repo's
# private layer-numbering convention.
_LAYER_ID_PREFIX = re.compile(r'^03[a-z]_')


def strip_layer_id_prefix(column: str) -> str:
    """Drop the leading ``03x_`` layer-id token from a column header for
    publication, keeping the descriptive layer name and the metric:
    ``03f_motor_resonance_max_ego_chaos_score`` -> ``motor_resonance_max_ego_chaos_score``.
    Only the repo-internal layer NUMBER is removed; manifest columns
    (``video_id``, ``fps``, ...) have no prefix and are returned unchanged."""
    return _LAYER_ID_PREFIX.sub('', column)


class DehydratedExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_parquet(self, df: pd.DataFrame, filename: str = "social_metadata.parquet"):
        """
        Exports the aggregated DataFrame to a Parquet file.
        Also embeds the metadata into a separate JSON since Parquet metadata
        can be tricky to retrieve across different readers.

        Publishing rename: the internal ``03x_`` layer-id token is stripped from
        every column header (``03c_acoustic_prosody_avg_prosody_scalar`` ->
        ``acoustic_prosody_avg_prosody_scalar``) so the released dataset does not
        leak the repo's layer-numbering convention. The descriptive layer name
        and metric are kept; the schema hash is recomputed on the published
        column names so ``export_metadata.json`` matches the Parquet.
        """
        # Capture provenance before the publishing column rename (df.attrs is not
        # guaranteed to survive a rename across pandas versions).
        base_schema = str(df.attrs.get("schema_version", "1.0.0")).split("+", 1)[0]
        export_timestamp = df.attrs.get("export_timestamp", "")
        active_layers = df.attrs.get("active_layers", [])
        pipeline_git_sha = df.attrs.get("pipeline_git_sha", "unknown")

        # Strip the 03x_ layer-id prefix from column headers for publication.
        rename_map = {c: strip_layer_id_prefix(c) for c in df.columns}
        published = list(rename_map.values())
        if len(set(published)) != len(published):
            from collections import Counter
            dupes = sorted(n for n, c in Counter(published).items() if c > 1)
            raise ValueError(
                f"Publishing rename collided after stripping the 03x_ prefix: {dupes}. "
                "Two layers share the same descriptive name + metric; keep the prefix "
                "or rename the offending metric in LAYER_SUMMARY_REGISTRY."
            )
        df = df.rename(columns=rename_map)

        # Validate that no synthetic validation videos are included in the export
        if 'video_id' in df.columns:
            synthetic_ids = df[df['video_id'].astype(str).str.startswith("synthetic_")]['video_id'].tolist()
            if synthetic_ids:
                raise ValueError(f"Dehydration validation failed: DataFrame contains synthetic validation video IDs: {synthetic_ids}!")

        try:
            import numpy as np
        except ImportError:
            np = None

        base64_img_pattern = r'^data:image/[a-z]+;base64,'
        path_pattern = r'^/[Vv]olumes/|^/Users/|^/tmp/'

        # Validate dehydration rule: Ensure no byte arrays, numpy arrays, or raw images
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna()
                if not sample.empty:
                    if any(isinstance(v, bytes) for v in sample):
                        raise ValueError(f"Dehydration validation failed: Column {col} contains raw bytes!")
                    if np is not None and any(isinstance(v, np.ndarray) for v in sample):
                        raise ValueError(f"Dehydration validation failed: Column {col} contains numpy array!")

                    str_sample = sample[sample.apply(lambda x: isinstance(x, str))]
                    if not str_sample.empty:
                        if str_sample.str.contains(base64_img_pattern, regex=True).any():
                            raise ValueError(f"Dehydration validation failed: Column {col} contains base64 image!")
                        if str_sample.str.contains(path_pattern, regex=True).any():
                            raise ValueError(f"Dehydration validation failed: Column {col} contains raw file path leak!")

        output_path = self.output_dir / filename
        df.to_parquet(output_path, engine='pyarrow', index=False)

        # Recompute the schema hash on the PUBLISHED (renamed) column names so the
        # metadata fingerprint matches the Parquet consumers actually receive.
        metadata_path = self.output_dir / "export_metadata.json"
        metadata = {
            "schema_version": f"{base_schema}+{schema_column_hash(df.columns)}",
            "export_timestamp": export_timestamp,
            "active_layers": active_layers,
            "pipeline_git_sha": pipeline_git_sha,
            "total_videos": len(df)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return output_path, metadata_path
