import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List, Callable, Optional

def mnist_collate(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)

    # Stack images into a 4‑D tensor (N, C, H, W)
    x = torch.stack(imgs, dim=0).float()

    # Standardization of the batch 
    mean = x.mean()
    std = x.std().clamp_min(1e-8)
    x = (x - mean) / std

    # Tensor conversion 
    y = torch.tensor(labels, dtype=torch.long)

    # One‑hot encoding
    y_onehot = F.one_hot(y, num_classes=10).float()

    return x, y, y_onehot

def get_mnist_loaders(
    batch_size: int,
    collate_fn: Callable = mnist_collate,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    
    # Tensor conversion
    tfm = transforms.ToTensor()

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    torch.manual_seed(0)

    train_loader, test_loader = get_mnist_loaders(32)
    x, y, y_onehot = next(iter(train_loader))

    print(x[0,:5,:5]) 
    print(y[0])
    print(y_onehot[0])
