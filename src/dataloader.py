from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import fnmatch

plt.ion()   # interactive mode

class BoomerDataset(Dataset):

    def __init__(self, root_dir, meta_frame, transform):
        self.root_dir = root_dir
        self.transform = transform

        # dataframe of metadata
        self.meta_frame = meta_frame

        self.len = sum(len(files) for _, _, files in os.walk(root_dir))

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = os.path.join(self.root_dir,
                             self.meta_frame.iloc[idx, 0])

        image = io.imread(img_name)
        image = self.transform(image)
        
        return image


class BoomerDatasetContainer(BoomerDataset):

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, 256)
        ])

        root_dir = "../data/boomer_humour/"

        super().__init__(self, root_dir, transform)