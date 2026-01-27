import os
import torch
import time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from torchvision import transforms
from PIL import Image

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

class MedicalImageDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']
        if image.mode != 'RGB': image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        label = example['label']
        return {'pixel_values': inputs['pixel_values'].squeeze(0), 'labels': torch.tensor(label, dtype=torch.long)}

def load_medical_imaging_dataset(max_retries=3):
    for attempt in range(max_retries):
        try:
            raw_dataset = load_dataset("marmal88/skin_cancer", split="train")
            dataset = raw_dataset.train_test_split(test_size=0.3, seed=42)
            val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
            return DatasetDict({
                'train': dataset['train'],
                'validation': val_test['train'],
                'test': val_test['test']
            })
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e

def create_dataloaders(dataset, processor, batch_size=32):
    train_dataset = MedicalImageDataset(dataset['train'], processor)
    loaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)}
    for split in ['validation', 'test']:
        if split in dataset:
            loaders[split] = DataLoader(
                MedicalImageDataset(dataset[split], processor),
                batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
    return loaders
