import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ImageFolderNoLabel(Dataset):
    def __init__(self, root, transform=None):
        self.ds = datasets.ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x, torch.tensor(0, dtype=torch.long)


class ImageDirDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.files = [p for p in self.root.rglob('*') if p.suffix.lower() in exts]
        if not self.files:
            raise FileNotFoundError(f'No images found under: {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(0, dtype=torch.long)


def _default_transforms(image_size: int, randflip: bool):
    ops = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if randflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(ops)


def get_dataset(name: str, data_dir: str, image_size: int, randflip: bool) -> Tuple[Dataset, int]:
    name = name.lower()
    transform = _default_transforms(image_size, randflip)

    if name == 'cifar10':
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        return ds, 10

    if name == 'cifar10_test':
        ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return ds, 10

    if name.startswith('lsun_'):
        lsun_class = name.split('_', 1)[1]
        if lsun_class == 'church':
            classes = ['church_outdoor_train']
        elif lsun_class == 'bedroom':
            classes = ['bedroom_train']
        elif lsun_class == 'cat':
            classes = ['cat_train']
        else:
            raise ValueError(f'Unsupported LSUN class: {lsun_class}')
        ds = datasets.LSUN(root=data_dir, classes=classes, transform=transform)
        return ds, 1

    if name in ['celebahq', 'celebahq256', 'ffhq', 'imagefolder']:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'Data directory not found: {data_dir}')
        has_subdirs = any(p.is_dir() for p in Path(data_dir).iterdir())
        if has_subdirs:
            ds = ImageFolderNoLabel(root=data_dir, transform=transform)
        else:
            ds = ImageDirDataset(root=data_dir, transform=transform)
        return ds, 1

    raise ValueError(f'Unsupported dataset: {name}')
