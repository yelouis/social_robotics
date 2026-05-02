import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage configuration
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/Volumes/Extreme SSD/social_robotics/raw_videos"))

# Ego4D configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
EGO4D_METADATA_PATH = Path(os.getenv("EGO4D_METADATA_PATH", "/Volumes/Extreme SSD/ego4d_data/v2/annotations/ego4d.json"))

# EPIC-KITCHENS-100 configuration
EPIC_KITCHENS_USER = os.getenv("EPIC_KITCHENS_USER")
EPIC_KITCHENS_PASSWORD = os.getenv("EPIC_KITCHENS_PASSWORD")

# Charades-Ego
CHARADES_EGO_URL = os.getenv("CHARADES_EGO_URL", "https://prior-datasets.s3.us-east-2.amazonaws.com/charades/Charades_Ego_v1.zip")

# EgoProceL
# Default updated to working repository
EGOPROCEL_REPO = os.getenv("EGOPROCEL_REPO", "https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning")

# Known dataset locations on Extreme SSD to avoid redownloading
DATASET_PATHS = {
    "charades_ego": [
        Path("/Volumes/Extreme SSD/charades_ego_data/ego_videos/CharadesEgo_v1_480"),
        OUTPUT_DIR / "charades_ego"
    ],
    "ego4d": [
        Path("/Volumes/Extreme SSD/ego4d_data/videos"),
        OUTPUT_DIR / "ego4d"
    ],
    "epic_kitchens": [
        OUTPUT_DIR / "epic_kitchens"
    ],
    "egoprocel": [
        OUTPUT_DIR / "egoprocel"
    ]
}

def ensure_dirs():
    """Ensure that necessary directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"AWS_ACCESS_KEY_ID: {'SET' if AWS_ACCESS_KEY_ID else 'NOT SET'}")
