import os
from pathlib import Path
from typing import Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PatchingDataset(Dataset):
    """
    Creates a dataset class to embed patches in batches.

    Parameters
    ----------
    id_dir: str
        The directory containing the patches (this would be the patient ID).

    Returns
    -------
    img: torch.Tensor
        The patch as a Tensor.

    coords:
        The coordinates of the patch containing its coordinates.
    """

    def __init__(self, id_dir: str) -> None:
        self.id_dir = id_dir
        self.patches = os.listdir(id_dir)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[str]]:
        img_name = self.patches[idx]
        img_path = os.path.join(self.id_dir, img_name)

        img = Image.open(img_path)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

        img = preprocess(img)
        coords = Path(img_name).stem

        return img, coords


