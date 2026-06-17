import logging
from pathlib import Path

from huggingface_hub import HfApi, get_token

logger = logging.getLogger(__name__)


def upload_to_huggingface(output_dir: str, repo_id: str, token: str = None):
    """Upload the dehydrated dataset bundle to a Hugging Face dataset repo.

    Token resolution order: the explicit ``token`` argument, then
    ``huggingface_hub.get_token()`` — which reads the ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` env vars AND the ``hf auth login`` credential
    store. If no token is found anywhere, publishing is treated as unconfigured
    and skipped (returns ``None``).

    Failures are loud, not swallowed: a token that is present but rejected, an
    unreadable Parquet, or any repo-create / file-upload error propagates, so an
    automated caller cannot mistake a failed publish for a successful one.

    Returns the list of uploaded filenames on success, or ``None`` when skipped.
    """
    if not token:
        # Falls back to the `hf auth login` credential store, not just HF_TOKEN.
        token = get_token()
        if not token:
            logger.warning(
                "No Hugging Face token found (pass token=, set HF_TOKEN, or run "
                "`hf auth login`). Skipping upload; local files remain intact."
            )
            return None

    output_dir = Path(output_dir)
    parquet_path = output_dir / "social_metadata.parquet"
    metadata_path = output_dir / "export_metadata.json"
    rehydrate_path = Path(__file__).parent / "rehydrate_dataset.py"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    # Dehydration gate: never publish synthetic-validation rows. A Parquet we
    # cannot even read is likewise unsafe to publish, so read errors propagate
    # rather than being downgraded to a warning that proceeds with the upload.
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if 'video_id' in df.columns:
        synthetic_ids = df[df['video_id'].astype(str).str.startswith("synthetic_")]['video_id'].tolist()
        if synthetic_ids:
            raise ValueError(
                f"HF upload validation failed: Parquet file contains synthetic validation video IDs: {synthetic_ids}!"
            )

    api = HfApi()
    # exist_ok keeps re-publishing idempotent. A 401/permission error (or any
    # other create_repo failure) is a real, loud failure now — previously it was
    # caught and turned into a silent warning + early return.
    api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)

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
    uploaded = []
    # Upload errors propagate — a swallowed failure looks like success to a
    # caller that only checks for an exception.
    for file_path in files_to_upload:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info(f"Uploaded {file_path.name}")
            uploaded.append(file_path.name)

    logger.info("Upload complete.")
    return uploaded
