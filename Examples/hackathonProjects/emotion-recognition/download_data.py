"""
Dataset Download Helper for RAVDESS

This script helps download and organize the RAVDESS dataset.
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm


RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DATA_DIR = "./data/ravdess"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_ravdess(output_dir=DATA_DIR):
    """
    Download and extract the RAVDESS dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "ravdess.zip")
    
    # Check if already downloaded
    if os.path.exists(os.path.join(output_dir, "Actor_01")):
        print("RAVDESS dataset already exists. Skipping download.")
        return
    
    print(f"Downloading RAVDESS dataset to {output_dir}...")
    print("This may take a few minutes (approximately 2.5GB)...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(RAVDESS_URL, zip_path, reporthook=t.update_to)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        print(f"Dataset downloaded and extracted to {output_dir}")
        print("\nDataset structure:")
        for actor_dir in sorted(os.listdir(output_dir))[:5]:
            actor_path = os.path.join(output_dir, actor_dir)
            if os.path.isdir(actor_path):
                num_files = len([f for f in os.listdir(actor_path) if f.endswith('.wav')])
                print(f"  {actor_dir}: {num_files} audio files")
        print("  ...")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print(RAVDESS_URL)
        print(f"\nExtract to: {output_dir}")


if __name__ == "__main__":
    download_ravdess()
