from glob import glob
from typing import Union, List

import cv2
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,
                 img_dirs: Union[str, List[str]],
                 classification: bool = False,
                 resize: bool = False):
        super().__init__()
        self.img_paths = None
        self.img_files = None
        self.classification = classification
        self.resize = resize

    def __getitem__(self,
                    item: int):
        img = cv2.imread(self.img_files[item])  # BRG, (H, W, C)
        assert img is not None, f"{self.img_files[item]} doesn't exists."

        if self.resize:
            pass

        if self.classification:
            pass

        return img
