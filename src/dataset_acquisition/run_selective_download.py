import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset_acquisition.downloader import Ego4DDownloader, EpicKitchensDownloader, EgoProceLDownloader

from tqdm import tqdm
import time

def run_selective():
    # Increase open file limit to avoid OSError: [Errno 24]
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to set to hard limit, or at least a much higher value
        target_limit = min(hard, 65535) # 65k is usually plenty
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
        print(f"[System] Increased open file limit: {soft} -> {target_limit}")
    except Exception as e:
        print(f"[Warning] Could not increase open file limit: {e}")

    tasks = [
        ("Ego4D", True), # (Name, Force)
        # ("EPIC-KITCHENS-100", False),
        # ("EgoProceL", False)
    ]
    
    with tqdm(total=len(tasks), desc="Overall Acquisition Progress") as pbar:
        # 1. Ego4D
        pbar.set_postfix(dataset="Ego4D")
        ego4d = Ego4DDownloader()
        # ego4d.filter_on_the_fly = False # Disable for debug if needed
        
        all_uids = ego4d.get_all_uids()
        processed_uids = ego4d.get_processed_uids()
        
        to_process = [uid for uid in all_uids if uid not in processed_uids]
        print(f"\n[Ego4D Status] Total: {len(all_uids)}, Processed: {len(processed_uids)}, Remaining: {len(to_process)}")
        
        if to_process:
            batch_size = 50
            num_batches = (len(to_process) + batch_size - 1) // batch_size
            
            batches_run = 0
            for i in range(0, len(to_process), batch_size):
                    
                batch = to_process[i:i + batch_size]
                current_batch_num = i // batch_size + 1
                print(f"\n>>> Ego4D Batch {current_batch_num}/{num_batches} ({len(batch)} UIDs)")
                
                # Execute batch download and immediate filter
                success = ego4d.run(force=True, video_uids=batch)
                if not success:
                    print("\n[STOP] Ego4D acquisition halted (Check disk space or requirements).")
                    return

                # Optional: Brief sleep to allow system to cool down or user to interrupt
                time.sleep(1)
                batches_run += 1
        else:
            print("Ego4D: All videos already processed.")
            
        pbar.update(1)

        # 2. EPIC-KITCHENS-100 (Disabled for now)
        """
        pbar.set_postfix(dataset="EPIC")
        epic = EpicKitchensDownloader()
        
        all_epic_uids = epic.get_all_uids()
        
        # If repo not cloned, run once to initialize
        if not all_epic_uids:
            print("\nInitializing EPIC-KITCHENS-100 (Cloning scripts)...")
            epic.run()
            all_epic_uids = epic.get_all_uids()
            
        processed_epic_uids = epic.get_processed_uids()
        to_process_epic = [uid for uid in all_epic_uids if uid not in processed_epic_uids]
        
        print(f"\n[EPIC Status] Total: {len(all_epic_uids)}, Processed: {len(processed_epic_uids)}, Remaining: {len(to_process_epic)}")
        
        if to_process_epic:
            batch_size = 50
            num_batches = (len(to_process_epic) + batch_size - 1) // batch_size
            
            for i in range(0, len(to_process_epic), batch_size):
                batch = to_process_epic[i:i + batch_size]
                current_batch_num = i // batch_size + 1
                print(f"\n>>> EPIC Batch {current_batch_num}/{num_batches} ({len(batch)} UIDs)")
                
                # Execute batch download and immediate filter
                success = epic.run(force=True, specific_videos=batch)
                if not success:
                    print("\n[STOP] EPIC acquisition halted (Check disk space or requirements).")
                    return
                time.sleep(1)
        else:
            print("EPIC: All videos already processed or metadata missing.")
        """
        pbar.update(1)

        # 3. EgoProceL (Disabled for now)
        """
        pbar.set_postfix(dataset="EgoProceL")
        egoprocel = EgoProceLDownloader()
        egoprocel.run()
        """
        pbar.update(1)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SOCIAL ROBOTICS: DATASET ACQUISITION PIPELINE")
    print("="*60)
    run_selective()
    print("="*60)
    print("Acquisition Complete.")
    print("="*60 + "\n")
