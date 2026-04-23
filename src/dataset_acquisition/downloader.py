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
            print("Ego4D download completed successfully.")
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
