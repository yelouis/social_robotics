import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# All raw video, features and results live on the external SSD, never the
# internal disk (docs/LESSONS_v0.md, "Operations").
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/Volumes/Extreme SSD/social_robotics"))


def ensure_dirs():
    """Ensure that necessary directories exist."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_ROOT: {DATA_ROOT}")
