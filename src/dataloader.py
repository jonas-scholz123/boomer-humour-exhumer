from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from skimage.color import rgba2rgb, gray2rgb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import fnmatch

from dataOrganiser import MetaData
from utils import imshow

class BoomerDataset(Dataset):

    def __init__(self, meta_frame, transform, is_training):
        self.transform = transform

        # dataframe of metadata
        if is_training is None:
            self.meta_frame = meta_frame
        else:
            training_entries = meta_frame["is_training"] == is_training
            self.meta_frame = meta_frame[training_entries].reset_index()

        self.len = self.meta_frame.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fpath = self.meta_frame.loc[idx, "fpath"]
        label = self.meta_frame.loc[idx, "is_boomer"].astype("int")
        image = io.imread(fpath)

        # turn RGBA to RGB
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        # turn greyscale into RGB
        elif len(image.shape) == 2:
            image = gray2rgb(image)
        
        image = self.transform(image)
        
        return image, label


class BoomerDatasetContainer(BoomerDataset):

    def __init__(self, is_training=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        meta_frame = MetaData("../data/metadata.pkl").df

        super().__init__(meta_frame, transform, is_training)

if __name__ == "__main__":
    boomer_data = BoomerDatasetContainer(is_training=True)

    for i in range(len(boomer_data)):
        boomer_data[i]