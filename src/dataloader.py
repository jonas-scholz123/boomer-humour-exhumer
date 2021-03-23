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
import config

class BoomerDataset(Dataset):
    '''
    Custom pyTorch dataset, loads the metadata file and transforms
    images into the right format
    '''

    def __init__(self, meta_frame, transform, is_training):
        '''
        PARAMS:
            pd.DataFrame meta_frame: metadata dataframe
            torchvision.transform: transforms images into desired shape
            bool is_training: determines whether it loads training/testing data
        '''
        self.transform = transform

        # dataframe of metadata
        if is_training is None:
            self.meta_frame = meta_frame
        else:
            training_entries = meta_frame["is_training"] == is_training
            self.meta_frame = meta_frame[training_entries].reset_index()

        self.len = self.meta_frame.shape[0]
    
    def process_image(self, image):
        # turn RGBA to RGB
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        # turn greyscale into RGB
        elif len(image.shape) == 2:
            image = gray2rgb(image)
        
        return self.transform(image).float()

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        '''
        Retrieves fpath from metadata frame, loads and transforms image
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fpath = self.meta_frame.loc[idx, "fpath"]
        label = self.meta_frame.loc[idx, "is_boomer"].astype("int")
        image = io.imread(fpath)
        image = self.process_image(image)
        
        return image, label


class BoomerDatasetContainer(BoomerDataset):
    '''
    Wrapper that specifies all the details for 
    boomer memes in particular
    '''
    def __init__(self, is_training=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        meta_frame = pd.read_pickle(config.paths["metadata"])

        super().__init__(meta_frame, transform, is_training)

if __name__ == "__main__":
    boomer_data = BoomerDatasetContainer(is_training=True)

    for i in range(len(boomer_data)):
        if i % 50 == 0:
            print(i)
        boomer_data[i]