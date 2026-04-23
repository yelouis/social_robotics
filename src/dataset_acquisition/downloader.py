import os
import sys
import subprocess
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import requests
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from .filterer import StreamingFilter

class DatasetDownloader(ABC):
    def __init__(self, name: str, filter_on_the_fly: bool = True):
        self.name = name
        self.output_path = config.OUTPUT_DIR / name
        self.possible_paths = config.DATASET_PATHS.get(name, [self.output_path])
        self.filter_on_the_fly = filter_on_the_fly
        self.filterer = StreamingFilter() if filter_on_the_fly else None
        self.processed_uids_path = self.output_path / "processed_uids.json"

    def get_processed_uids(self) -> set:
        if not self.processed_uids_path.exists():
            return set()
        try:
            with open(self.processed_uids_path, "r") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Error loading processed UIDs for {self.name}: {e}")
            return set()

    def mark_as_processed(self, uids: list):
        current = self.get_processed_uids()
        current.update(uids)
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            with open(self.processed_uids_path, "w") as f:
                json.dump(list(current), f)
        except Exception as e:
            print(f"Error saving processed UIDs for {self.name}: {e}")

    def get_all_uids(self) -> list:
        """Abstract method to retrieve all UIDs for this dataset."""
        return []

    def filter_and_purge(self, video_path: Path):
        """
        Run the social presence filter on a video. 
        If it fails, delete the video.
        """
        if not self.filter_on_the_fly:
            return True

        print(f"Streaming Filter: Evaluating {video_path.name}...")
        has_presence = self.filterer.check_social_presence(video_path)
        
        if not has_presence:
            print(f"Streaming Filter: No social presence detected. PURGING {video_path.name}")
            try:
                video_path.unlink()
            except Exception as e:
                print(f"Error deleting file: {e}")
            return False
        
        print(f"Streaming Filter: Social presence confirmed. KEEPING {video_path.name}")
        return True

    def check_disk_space(self, min_gb: float = 10.0) -> bool:
        """Check if there is enough space left on the output disk."""
        try:
            usage = shutil.disk_usage(self.output_path.anchor if self.output_path.exists() else self.output_path.parent.anchor)
            free_gb = usage.free / (1024**3)
            if free_gb < min_gb:
                print(f"\n[CRITICAL] Low Disk Space: {free_gb:.2f} GB remaining on {self.output_path.anchor}")
                print(f"Required: at least {min_gb} GB. Stopping acquisition to prevent corruption.")
                return False
            return True
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True # Proceed with caution if check fails

    def is_already_downloaded(self) -> bool:
        """Check if any of the possible paths contain mp4 files."""
        for path in self.possible_paths:
            if path.exists() and any(path.rglob("*.mp4")):
                print(f"Found existing data for {self.name} in {path}")
                return True
        return False

    @abstractmethod
    def check_requirements(self) -> bool:
        """Check if keys or tools are available."""
        pass

    @abstractmethod
    def download(self):
        """Execute the download."""
        pass

    def run(self, force: bool = False, **kwargs) -> bool:
        print(f"\n--- Processing Dataset: {self.name} ---")
        
        if not force and self.is_already_downloaded():
            print(f"Skipping download for {self.name}: Data already exists on disk.")
            return True

        if not self.check_requirements():
            print(f"Skipping {self.name}: Requirements not met.")
            return False
        
        if not self.check_disk_space():
            return False
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.download(**kwargs)
        return True

class Ego4DDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("ego4d")

    def get_all_uids(self) -> list:
        """Retrieve all UIDs from the Ego4D metadata."""
        metadata_path = self.output_path / "ego4d.json"
        # If not there, check one level up (common in Ego4D CLI)
        if not metadata_path.exists():
            metadata_path = self.output_path.parent / "ego4d.json"
            
        if not metadata_path.exists():
            print(f"Ego4D metadata not found at {metadata_path}. Cannot retrieve all UIDs.")
            return []
            
        print(f"Loading Ego4D metadata from {metadata_path}...")
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
                # Ego4D metadata structure usually has a 'videos' list
                if isinstance(data, dict) and "videos" in data:
                    return [v["video_uid"] for v in data["videos"]]
                elif isinstance(data, list):
                    return [v["video_uid"] for v in data if "video_uid" in v]
        except Exception as e:
            print(f"Error parsing Ego4D metadata: {e}")
        return []

    def check_requirements(self) -> bool:
        if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
            print("Missing AWS credentials for Ego4D. (Skipping download but will still index if data exists)")
            return False
        
        # Check for AWS CLI
        aws_found = False
        try:
            subprocess.run(["aws", "--version"], check=True, capture_output=True)
            aws_found = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        # Check for ego4d python package
        ego4d_found = False
        try:
            import ego4d
            ego4d_found = True
        except ImportError:
            # Check in vlm_env
            vlm_python = config.BASE_DIR / "vlm_env" / "bin" / "python"
            if vlm_python.exists():
                try:
                    subprocess.run([str(vlm_python), "-c", "import ego4d"], check=True, capture_output=True)
                    ego4d_found = True
                except subprocess.CalledProcessError:
                    pass

        if not aws_found and not ego4d_found:
            print("Neither AWS CLI nor 'ego4d' python package found. Please install one of them.")
            return False
            
        return True

    def download(self, video_uids=None, **kwargs):
        print(f"Starting Ego4D download to {self.output_path}...")
        
        # Ensure we have a place for temporary AWS credentials if they aren't in ~/.aws
        aws_dir = Path.home() / ".aws"
        aws_creds = aws_dir / "credentials"
        
        # If no credentials file exists, we'll create a temporary one for this session
        temp_aws_dir = config.BASE_DIR / ".aws_temp"
        temp_aws_creds = temp_aws_dir / "credentials"
        
        env = os.environ.copy()
        
        if not aws_creds.exists():
            print("Creating temporary AWS credentials for Ego4D CLI...")
            temp_aws_dir.mkdir(exist_ok=True)
            with open(temp_aws_creds, "w") as f:
                f.write(f"[default]\n")
                f.write(f"aws_access_key_id = {config.AWS_ACCESS_KEY_ID}\n")
                f.write(f"aws_secret_access_key = {config.AWS_SECRET_ACCESS_KEY}\n")
            env["AWS_SHARED_CREDENTIALS_FILE"] = str(temp_aws_creds)
            if config.AWS_DEFAULT_REGION:
                env["AWS_DEFAULT_REGION"] = config.AWS_DEFAULT_REGION

        # Use the ego4d module via the same python executable if possible
        # or fall back to 'ego4d' command if available in path
        python_exe = sys.executable
        # If we're in a venv, sys.executable should be correct. 
        # But for this specific task, we know vlm_env has it.
        vlm_python = config.BASE_DIR / "vlm_env" / "bin" / "python"
        if vlm_python.exists():
            python_exe = str(vlm_python)

        cmd = [
            python_exe, "-m", "ego4d.cli.cli",
            "-o", str(self.output_path),
            "--datasets", "full_scale",
            "-y"
        ]
        
        if video_uids:
            cmd.extend(["--video_uids"] + video_uids)
        
        print(f"Executing Ego4D CLI: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, env=env, check=True)
            print("Ego4D download batch completed.")
            
            if self.filter_on_the_fly:
                print("Running immediate filtering for this batch...")
                # We only want to filter the ones we just tried to download
                # But since the CLI might have skipped some, we check what's actually there
                processed = self.get_processed_uids()
                videos = []
                for v in self.output_path.rglob("*.mp4"):
                    # Check if filename (without extension) is in processed list
                    # or if it's an Apple double file
                    if v.name.startswith("._"):
                        continue
                    if v.stem in processed:
                        continue
                    videos.append(v)

                for video in tqdm(videos, desc="Filtering Batch"):
                    self.filter_and_purge(video)
                
                # Mark as processed regardless of whether they were kept or purged
                if video_uids:
                    self.mark_as_processed(video_uids)
                    
        except subprocess.CalledProcessError as e:
            print(f"Error during Ego4D download: {e}")
        finally:
            # Clean up temp credentials if created
            if temp_aws_creds.exists():
                import shutil
                shutil.rmtree(temp_aws_dir)

class CharadesEgoDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("charades_ego")

    def get_all_uids(self) -> list:
        """For Charades, the UIDs are the video filenames inside the ZIP."""
        # This is a bit chicken-and-egg since we need the ZIP to see UIDs.
        # However, we can often get them from a manifest if we had one.
        # For now, we'll return an empty list until the ZIP is downloaded.
        zip_path = self.output_path / "Charades_Ego_v1.zip"
        if not zip_path.exists():
            return []
        
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                return [Path(m).stem for m in zip_ref.namelist() if m.endswith(".mp4")]
        except Exception:
            return []

    def check_requirements(self) -> bool:
        return True

    def download(self, **kwargs):
        url = config.CHARADES_EGO_URL
        zip_path = self.output_path / "Charades_Ego_v1.zip"
        
        if zip_path.exists():
            print(f"{zip_path} already exists. Skipping download.")
            return

        print(f"Downloading Charades-Ego from {url}...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, "wb") as f, tqdm(
                desc=zip_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            print("Extracting Charades-Ego...")
            import zipfile
            processed = self.get_processed_uids()
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc="Extracting & Filtering"):
                    if member.endswith(".mp4"):
                        uid = Path(member).stem
                        if uid in processed:
                            continue

                        # Extract individual file
                        zip_ref.extract(member, self.output_path)
                        extracted_path = self.output_path / member
                        # Filter immediately
                        self.filter_and_purge(extracted_path)
                        # Mark as processed
                        self.mark_as_processed([uid])
                    else:
                        zip_ref.extract(member, self.output_path)
            
            # Delete the large zip after extraction
            print(f"Cleaning up {zip_path.name}...")
            zip_path.unlink()
            print("Extraction and filtering complete.")
            
        except Exception as e:
            print(f"Error downloading or extracting Charades-Ego: {e}")

class EpicKitchensDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("epic_kitchens")
        self.script_repo = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"

    def get_all_uids(self) -> list:
        repo_dir = self.output_path.parent / ".epic-scripts-repo"
        csv_path = repo_dir / "data" / "epic_100_splits.csv"
        
        if not csv_path.exists():
            return []
            
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # EPIC-KITCHENS video_id is unique
            if "video_id" in df.columns:
                return df["video_id"].unique().tolist()
        except Exception:
            # Fallback to manual parsing if pandas is missing
            try:
                uids = set()
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                    header = lines[0].strip().split(",")
                    if "video_id" in header:
                        idx = header.index("video_id")
                        for line in lines[1:]:
                            uids.add(line.strip().split(",")[idx])
                return list(uids)
            except Exception:
                pass
        return []

    def check_requirements(self) -> bool:
        return True

    def download(self, specific_videos=None, **kwargs):
        repo_dir = self.output_path.parent / ".epic-scripts-repo"
        if not repo_dir.exists():
            print("Cloning epic-kitchens-download-scripts...")
            subprocess.run(["git", "clone", self.script_repo, str(repo_dir)], check=True)
            
        print("Running epic_downloader.py...")
        # The script downloads to <output_path>/EPIC-KITCHENS.
        target_dir = self.output_path.parent
        cmd = [
            sys.executable, str(repo_dir / "epic_downloader.py"),
            "--output-path", str(target_dir),
            "--videos"
        ]
        
        if specific_videos:
            cmd.extend(["--specific-videos", ",".join(specific_videos)])
            
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=str(repo_dir))
        except subprocess.CalledProcessError as e:
            print(f"Error running epic_downloader.py: {e}")
            return
            
        # Move files from EPIC-KITCHENS to epic_kitchens
        downloaded_epic = target_dir / "EPIC-KITCHENS"
        if downloaded_epic.exists():
            import shutil
            print("Running post-download filtering for EPIC-KITCHENS...")
            videos = list(downloaded_epic.rglob("*.mp4"))
            for video in tqdm(videos, desc="Filtering EPIC"):
                uid = video.stem # EPIC filenames are video_ids
                if self.filter_and_purge(video):
                    # Move only if kept
                    rel_path = video.relative_to(downloaded_epic)
                    dest_path = self.output_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(video), str(dest_path))
                
                # Mark as processed regardless
                self.mark_as_processed([uid])
            
            shutil.rmtree(downloaded_epic)
            print(f"Filtering and move complete. Remaining videos in {self.output_path}")

class EgoProceLDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("egoprocel")

    def check_requirements(self) -> bool:
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Git not found. Please install git to clone EgoProceL.")
            return False
        return True

    def download(self, **kwargs):
        print(f"Cloning EgoProceL repository from {config.EGOPROCEL_REPO}...")
        repo_path = self.output_path / "EgoProceL-Repo"
        
        if repo_path.exists():
            print("EgoProceL repository already cloned.")
        else:
            try:
                subprocess.run(["git", "clone", config.EGOPROCEL_REPO, str(repo_path)], check=True)
                print("EgoProceL cloned successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning EgoProceL: {e}")
                return
                
        print("\n" + "="*60)
        print("EgoProceL is a meta-dataset containing links and annotations.")
        print("To download the actual video files (CMU-MMAC, EGTEA, etc.),")
        print(f"please navigate to {repo_path} and follow the instructions in its README.md.")
        print("="*60 + "\n")

def download_all(force: bool = False):
    config.ensure_dirs()
    # Priority Order: Ego4D > Charades-Ego > EPIC-KITCHENS-100 > EgoProceL
    downloaders = [
        Ego4DDownloader(),
        # CharadesEgoDownloader(),
        # EpicKitchensDownloader(),
        # EgoProceLDownloader(),
    ]
    
    for downloader in downloaders:
        downloader.run(force=force)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download datasets for social robotics.")
    parser.add_argument("--force", action="store_true", help="Force download even if data already exists.")
    args = parser.parse_args()
    
    download_all(force=args.force)
