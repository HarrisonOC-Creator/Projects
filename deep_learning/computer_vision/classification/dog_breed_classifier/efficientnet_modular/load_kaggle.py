"""Download, prepare and optionally subset the Stanford Dogs dataset.
Labels are remapped to contiguous indices when subsetting."""
from pathlib import Path
import shutil
import copy
import torch
import torchvision
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import kagglehub

def load_kaggle(data_root="data/",
                dataset_name="miljan/stanford-dogs-dataset-traintest",
                num_classes=None,
                batch_size=32,
                seed=42,
                val_split=0.1):
    """
    Download, prepare and optionally subset the Stanford Dogs dataset.
    Labels are remapped to contiguous indices when subsetting.

    Args:
        data_root (str): Root folder to store data.
        dataset_name (str): Kaggle dataset identifier.
        num_classes (int): If provided, randomly subset to this many classes.
        batch_size (int): Batch size for dataloaders.
        seed (int): Random seed for reproducibility.
        val_split (float): Portion of training data to be used as a validation set.

    Returns:
        train_dataloader, val_dataloader, test_dataloader, class_names
    """

    data_path = Path(data_root)
    image_path = data_path / "dog_breeds"

    # Download if not already present
    if not image_path.is_dir():
        print(f"{image_path} directory doesn't exist ... downloading dataset")
        image_path.mkdir(parents=True, exist_ok=True)
        original_path = kagglehub.dataset_download(dataset_name)
        shutil.copytree(original_path, image_path, dirs_exist_ok=True)
    else:
        print(f"{image_path} directory exists already ... skipping download")

    train_dir = image_path / "cropped/cropped/train"
    test_dir = image_path / "cropped/cropped/test"

    # Transforms
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),  
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomErasing(p=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((224,224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


    full_train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    if num_classes is not None:
        random.seed(seed)
        selected_classes = random.sample(full_train_data.classes, num_classes)
        class_to_idx = full_train_data.class_to_idx
        selected_indices = [class_to_idx[c] for c in selected_classes]

        train_subset_indices = [i for i, (_, label) in enumerate(full_train_data.samples) if label in selected_indices]
        test_subset_indices = [i for i, (_, label) in enumerate(test_data.samples) if label in selected_indices]

        full_train_data = RemapSubset(Subset(full_train_data, train_subset_indices), selected_indices)
        test_data = RemapSubset(Subset(test_data, test_subset_indices), selected_indices)

        class_names = [c.split("-", 1)[1] for c in selected_classes]
        if len(class_names) <= 5:
            print(f"Selected Classes: {class_names}")
        else:
            pass
    else:
        class_names = [c.split("-", 1)[1] for c in full_train_data.classes]

    # Split train into train + validation
    val_size = int(len(full_train_data) * val_split)
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(seed))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, class_names
