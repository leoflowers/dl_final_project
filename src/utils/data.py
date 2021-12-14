import torch

from torchvision import datasets, models, transforms
from pathlib import Path

BATCH_SIZE = 128

def check_valid(path):
    path = Path(path)
    return not path.stem.startswith('._')


def data_transforms():
    t = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]
            )
    return t


def create_datasets(transformations, check_valid):
    dsets = {x: datasets.ImageFolder(f"../data/CINIC10/{x}",
                                     transformations,
                                     is_valid_file=check_valid)
             for x in ['train', 'test', 'valid']}
    return dsets


def create_dataloaders(dsets):
    dl = {x: torch.utils.data.DataLoader(dsets[x],
                                         batch_size=BATCH_SIZE,
                                         shuffle=(x == 'train'))
          for x in ['train', 'test', 'valid']}
    return dl
