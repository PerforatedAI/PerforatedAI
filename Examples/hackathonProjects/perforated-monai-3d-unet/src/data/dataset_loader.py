from monai.apps import DecathlonDataset
from monai.data import DataLoader, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ConvertToMultiChannelBasedOnBratsClassesd,
)
import os


def get_dataloaders(
    data_dir,
    batch_size=1,
    num_workers=2,
):
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5)),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        CropForegroundd(keys=["image", "label"], source_key="image"),

        # 🔴 THIS IS THE FIX
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
    ])


    train_files = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="training",
        download=False,
        transform=None,
    ).data

    val_files = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="validation",
        download=False,
        transform=None,
    ).data

    train_ds = PersistentDataset(
        data=train_files,
        transform=transforms,
        cache_dir=cache_dir,
    )

    val_ds = PersistentDataset(
        data=val_files,
        transform=transforms,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader
