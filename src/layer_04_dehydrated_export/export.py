import pandas as pd
from pathlib import Path
import json

class DehydratedExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_parquet(self, df: pd.DataFrame, filename: str = "social_metadata.parquet"):
        """
        Exports the aggregated DataFrame to a Parquet file.
        Also embeds the metadata into a separate JSON since Parquet metadata
        can be tricky to retrieve across different readers.
        """
        import re
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
        
        # Save metadata separately to ensure it is always accessible
        metadata_path = self.output_dir / "export_metadata.json"
        metadata = {
            "schema_version": df.attrs.get("schema_version", "1.0.0"),
            "export_timestamp": df.attrs.get("export_timestamp", ""),
            "active_layers": df.attrs.get("active_layers", []),
            "pipeline_git_sha": df.attrs.get("pipeline_git_sha", "unknown"),
            "total_videos": len(df)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return output_path, metadata_path
