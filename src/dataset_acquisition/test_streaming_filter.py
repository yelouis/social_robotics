import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from dataset_acquisition.downloader import CharadesEgoDownloader

def test_filter_integration():
    print("--- Testing Streaming Filter Integration ---")
    
    # Use a dummy downloader to access filter_and_purge
    dl = CharadesEgoDownloader()
    
    # 1. Find a known video from the SSD
    sample_video = None
    for p in config.DATASET_PATHS.get("charades_ego", []):
        if p.exists():
            videos = list(p.rglob("*.mp4"))
            if videos:
                sample_video = videos[0]
                break
    
    if not sample_video:
        print("No sample video found to test with.")
        return

    print(f"Using sample video: {sample_video}")
    
    # 2. Create a temporary test file
    test_dir = config.BASE_DIR / "test_filter_run"
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_social.mp4"
    shutil.copy2(sample_video, test_file)
    
    print(f"Created test file: {test_file}")
    
    # 3. Run filter_and_purge
    # This should LOAD THE MODEL and evaluate the video
    result = dl.filter_and_purge(test_file)
    
    print(f"Filter result: {'KEEP' if result else 'PURGE'}")
    
    if result:
        if test_file.exists():
            print("SUCCESS: Video kept and file exists.")
        else:
            print("FAILURE: Video kept but file was deleted.")
    else:
        if not test_file.exists():
            print("SUCCESS: Video purged and file deleted.")
        else:
            print("FAILURE: Video purged but file still exists.")

    # 4. Test with a "empty" or invalid video (simulated)
    # We'll just use a tiny text file renamed to mp4
    fake_video = test_dir / "test_empty.mp4"
    with open(fake_video, "w") as f:
        f.write("not a video")
    
    print(f"\nTesting with invalid video: {fake_video}")
    result_empty = dl.filter_and_purge(fake_video)
    print(f"Filter result (invalid): {'KEEP' if result_empty else 'PURGE'}")
    
    if not result_empty and not fake_video.exists():
        print("SUCCESS: Invalid video purged correctly.")
    else:
        print("FAILURE: Invalid video handling failed.")

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_filter_integration()
