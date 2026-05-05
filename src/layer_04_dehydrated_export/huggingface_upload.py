import os
import logging
from pathlib import Path
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

def upload_to_huggingface(output_dir: str, repo_id: str, token: str = None):
    """
    Uploads the dehydrated metadata to Hugging Face.
    """
    if not token:
        token = os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "HF_TOKEN environment variable or token parameter is missing. "
                "Skipping Hugging Face upload. To enable upload, set the HF_TOKEN environment variable. "
                "Local files will remain intact."
            )
            return
            
    output_dir = Path(output_dir)
    parquet_path = output_dir / "social_metadata.parquet"
    metadata_path = output_dir / "export_metadata.json"
    rehydrate_path = Path(__file__).parent / "rehydrate_dataset.py"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    api = HfApi()
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.warning(f"Hugging Face authentication failed: {e}. Skipping upload. Local files remain intact.")
            return
        logger.warning(f"Could not create repo (might already exist): {e}")

    # Generate Dataset Card (README.md)
    readme_content = f"""---
license: mit
task_categories:
- video-classification
tags:
- egocentric
- social-interactions
- metadata
---

# Social Robotics Dehydrated Metadata

This dataset contains **dehydrated metadata** representing social features extracted from egocentric videos.
It does **NOT** contain any raw video pixels or audio tracks, strictly adhering to our privacy and copyright protocols.

## How to use

1. Download the `social_metadata.parquet`.
2. Use `rehydrate_dataset.py` along with your legally obtained copies of the source videos (e.g., Ego4D) to map these features back to the raw footage.
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    logger.info(f"Uploading files to {repo_id}...")
    
    files_to_upload = [parquet_path, metadata_path, readme_path, rehydrate_path]
    
    try:
        for file_path in files_to_upload:
            if file_path.exists():
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
                logger.info(f"Uploaded {file_path.name}")
                
        logger.info("Upload complete.")
    except Exception as e:
        logger.warning(f"Hugging Face upload failed: {e}. Local files remain intact at {output_dir}.")
