import os
import sys
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
import requests
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

class DatasetDownloader(ABC):
    def __init__(self, name: str):
        self.name = name
        self.output_path = config.OUTPUT_DIR / name
        self.possible_paths = config.DATASET_PATHS.get(name, [self.output_path])

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

    def run(self, **kwargs):
        print(f"\n--- Processing Dataset: {self.name} ---")
        
        if self.is_already_downloaded():
            print(f"Skipping download for {self.name}: Data already exists on disk.")
            return

        if not self.check_requirements():
            print(f"Skipping {self.name}: Requirements not met.")
            return
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.download(**kwargs)

class Ego4DDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("ego4d")

    def check_requirements(self) -> bool:
        if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
            print("Missing AWS credentials for Ego4D. (Skipping download but will still index if data exists)")
            return False
        
        try:
            subprocess.run(["aws", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("AWS CLI not found. Please install it.")
            return False
            
        return True

    def download(self, **kwargs):
        print("Ego4D download logic would go here.")
        # Implementation details...
        pass

class CharadesEgoDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("charades_ego")

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
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_path)
            print("Extraction complete.")
            
        except Exception as e:
            print(f"Error downloading or extracting Charades-Ego: {e}")

class EpicKitchensDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("epic_kitchens")
        self.script_repo = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"

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
            shutil.copytree(downloaded_epic, self.output_path, dirs_exist_ok=True)
            shutil.rmtree(downloaded_epic)
            print(f"Moved videos to {self.output_path}")

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

def download_all():
    config.ensure_dirs()
    downloaders = [
        Ego4DDownloader(),
        CharadesEgoDownloader(),
        EpicKitchensDownloader(),
        EgoProceLDownloader(),
    ]
    
    for downloader in downloaders:
        downloader.run()

if __name__ == "__main__":
    download_all()
