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

    def run(self):
        print(f"\n--- Processing Dataset: {self.name} ---")
        
        if self.is_already_downloaded():
            print(f"Skipping download for {self.name}: Data already exists on disk.")
            return

        if not self.check_requirements():
            print(f"Skipping {self.name}: Requirements not met.")
            return
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.download()

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

    def download(self):
        print("Ego4D download logic would go here.")
        # Implementation details...
        pass

class CharadesEgoDownloader(DatasetDownloader):
    def __init__(self):
        super().__init__("charades_ego")

    def check_requirements(self) -> bool:
        return True

    def download(self):
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

def download_all():
    config.ensure_dirs()
    downloaders = [
        Ego4DDownloader(),
        CharadesEgoDownloader(),
    ]
    
    for downloader in downloaders:
        downloader.run()

if __name__ == "__main__":
    download_all()
