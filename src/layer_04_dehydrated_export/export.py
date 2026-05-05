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
        # Validate dehydration rule: Ensure no byte arrays or raw images
        for col in df.columns:
            # Check for generic object types that might hide bytes
            if df[col].dtype == object:
                sample = df[col].dropna()
                if not sample.empty and isinstance(sample.iloc[0], bytes):
                    raise ValueError(f"Dehydration validation failed: Column {col} contains raw bytes!")
                    
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
