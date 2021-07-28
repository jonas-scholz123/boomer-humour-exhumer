from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
from skimage.color import rgba2rgb, gray2rgb
from nltk import word_tokenize
import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision import transforms, utils

from dataOrganiser import MetaData
from utils import imshow
import config

def collate_fn(batch):
    '''
    Overrides original collate fn to include padding process
    Padds batch of variable length

    '''
    x_image, x_text, labels = zip(*batch)
    x_image = torch.stack(x_image)
    labels = torch.tensor(labels)
    lengths = torch.tensor([seq.shape[0] for seq in x_text])
    # pad sequences
    x_text = pad_sequence(x_text, batch_first=True)
    # return lengths for later packing of padded sequences
    return x_image, x_text, lengths, labels

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
        self.word2id_path = config.paths["word2id"]
        with open(self.word2id_path, "rb") as f:
            self.word2id = pickle.load(f)
    
    
    def process_image(self, image):
        '''
        Apply self.transform to image to normalise, scale etc
        '''
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
        text = self.meta_frame.loc[idx, "text"]

        if text:
            text = text.lower()
            text_ids = [self.word2id[tok] for tok in word_tokenize(text)]
            # make padding of 0s
        else:
            # indicates no text
            text_ids = [0]
        # else:
            # print("WARNING: None found in df, should have NO_TEXT_DETECTED!")

        # padding = [0 for _ in range(config.params["sentence_length"] - len(text_ids))]
        # text_ids = torch.IntTensor(text_ids + padding)
        text_ids = torch.IntTensor(text_ids)

        image = io.imread(fpath)
        image = self.process_image(image)

        return image, text_ids, label


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