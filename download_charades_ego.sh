#!/bin/bash

# Add Homebrew to PATH for M-series Macs
export PATH="/opt/homebrew/bin:$PATH"

# Configuration: Target the external SSD to prevent wear on the internal drive.
OUTPUT_DIR="/Volumes/Extreme SSD/charades_ego_data"

# --- Guard: External drive check (MUST run before mkdir) ---
if [[ "$OUTPUT_DIR" != /Volumes/* ]]; then
  echo "ERROR: OUTPUT_DIR must be on an external drive (/Volumes/...)." && exit 1
fi

mkdir -p "$OUTPUT_DIR"/{annotations,ego_videos,tp_videos}

echo "Starting download to $OUTPUT_DIR..."

# 1. Download & extract annotations (CSV + class mapping, ~5MB)
echo "Downloading annotations..."
wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo.zip
unzip -o "$OUTPUT_DIR/CharadesEgo.zip" -d "$OUTPUT_DIR/annotations"

# 2. Download egocentric videos at 480p (~22GB)
echo "Downloading egocentric videos (~22GB)..."
wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar
echo "Extracting egocentric videos..."
tar -xf "$OUTPUT_DIR/CharadesEgo_v1_480.tar" -C "$OUTPUT_DIR/ego_videos"

# 3. Download original Charades third-person videos at 480p (~15GB)
echo "Downloading third-person videos (~15GB)..."
wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
echo "Extracting third-person videos..."
unzip -o "$OUTPUT_DIR/Charades_v1_480.zip" -d "$OUTPUT_DIR/tp_videos"

echo "Download and extraction complete. Total expected: ~44GB of video + annotations."
