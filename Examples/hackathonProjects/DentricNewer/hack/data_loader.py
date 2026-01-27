import os
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import DatasetDict

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

class MedicalImageDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        label = example['label']

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_medical_imaging_dataset(dataset_name="wubingheng/vit-medical-image-classification", max_retries=3):
    """
    Loads the wubingheng/vit-medical-image-classification dataset.
    This is a Chinese medicine herb classification dataset with 24 classes.
    Downloads arrow files directly to bypass broken HuggingFace metadata.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow as pa
    from datasets import Dataset

    print(f"Loading dataset: {dataset_name}...")

    for attempt in range(max_retries):
        try:
            # Download the arrow files directly (correct paths from HF repo)
            print(f"Downloading train data (attempt {attempt + 1})...")
            train_file = hf_hub_download(
                repo_id=dataset_name,
                filename="data/train/data-00000-of-00001.arrow",
                repo_type="dataset"
            )
            print(f"Downloading test data...")
            test_file = hf_hub_download(
                repo_id=dataset_name,
                filename="data/test/data-00000-of-00001.arrow",
                repo_type="dataset"
            )

            # Load arrow files using streaming reader (for Arrow IPC stream format)
            print("Loading arrow files...")
            with pa.memory_map(train_file, 'r') as source:
                train_table = pa.ipc.open_stream(source).read_all()
            with pa.memory_map(test_file, 'r') as source:
                test_table = pa.ipc.open_stream(source).read_all()

            train_dataset = Dataset(train_table)
            test_dataset = Dataset(test_table)

            # Create validation split from test
            test_val = test_dataset.train_test_split(test_size=0.5, seed=42)

            print(f"Dataset loaded: {len(train_dataset)} train, {len(test_val['train'])} val, {len(test_val['test'])} test")

            return DatasetDict({
                'train': train_dataset,
                'validation': test_val['train'],
                'test': test_val['test']
            })
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e

def create_dataloaders(dataset, processor, batch_size=32):
    print("Creating dataloaders...")
    train_dataset = MedicalImageDataset(dataset['train'], processor)

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    }

    for split in ['validation', 'test']:
        if split in dataset:
            loaders[split] = DataLoader(
                MedicalImageDataset(dataset[split], processor),
                batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
            )

    return loaders