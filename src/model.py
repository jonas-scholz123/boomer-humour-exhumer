import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from utils import ConceptNetDict


class CNN(nn.Module):
    def __init__(self):
        # TODO: pass hyperparameters

        #cnn params
        kernel_size = 5
        pool_size = 2

        #cnn channels
        in_channels = 3
        out_channels_1 = 6
        out_channels_2 = 16

        #linear params
        in_lin1 = 21904 # = 148 ** 2 taken from model.summary()
        out_lin1 = 120

        super().__init__()

        # convolutional layers encode pixels into image features 
        # 3 in channels, 6 out channels, kernel_size=5
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size)
        # 6 in channels, 16 out channels, kernel_size=5
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size)

        # pool
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        # flatten
        self.flat = nn.Flatten()

        # fully connected layers make the actual classification
        # 148 taken from summary, as there exists no utility
        # function for calculating this magic number
        self.fc1 = nn.Linear(in_lin1, out_lin1)
    
    def forward(self, x):
        '''
        INPUTS:
            tensor x: (nr_samples, nr_channels (3), nr_x_pixels (32), nr_y_pixels (32))
        RETURNS:
            boomer probability
        '''
        # Max pooling over 2x2 window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flat(x)
        return F.relu(self.fc1(x))

class RNN(nn.Module):

    def __init__(self):
        self.embedding_matrix_path = config.paths["embedding_matrix"]
        self.embed_dict = ConceptNetDict()
        self.embed_dim = 300
        self.word2id = self._load_word2id()
        embedding_matrix = self._load_embedding_matrix()

        super().__init__()

        self.embed_layer = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)

        # input_size = config.params["sentence_length"]
        # self.gru = nn.GRU(self.embed_dim, 256, batch_first=True)
        self.gru = nn.GRU(self.embed_dim, 256, batch_first=True)
    
    def _load_word2id(self):
        path = config.paths["word2id"]
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def _load_embedding_matrix(self):
        if os.path.exists(self.embedding_matrix_path):
            with open(self.embedding_matrix_path, "rb") as f:
                return pickle.load(f)
        else:
            return self._make_embedding_matrix()
    
    def _make_embedding_matrix(self):
        embedding_matrix = np.zeros((len(self.word2id), self.embed_dim))

        for word, idx in self.word2id.items():
            if word in self.embed_dict:
                embedding_matrix[idx] = self.embed_dict[word]

        embedding_matrix = torch.FloatTensor(embedding_matrix)
        with open(self.embedding_matrix_path, "wb") as f:
            pickle.dump(embedding_matrix, f)

        return embedding_matrix
    
    def forward(self, text_data, text_lengths):

        x = self.embed_layer(text_data)
        x = pack_padded_sequence(x, text_lengths,
                                 batch_first=True, enforce_sorted=False)

        #NOTE: last output not 100% sure is really the last output
        sequence_outputs, last_output = self.gru(x)
        # x, output_lengths = pad_packed_sequence(x, batch_first=True)
        return last_output.view(last_output.shape[1:]) # flatten first dim


class Model(nn.Module):

    def __init__(self):
        # TODO: pass hyperparameters
        # TODO: pass embeddings

        #cnn params
        kernel_size = 5
        pool_size = 2

        #cnn channels
        in_channels = 3
        out_channels_1 = 6
        out_channels_2 = 16

        #linear params
        in_lin1 = 376 #  taken from model.summary()
        out_lin1 = 120
        out_lin2 = 84
        final_out = 1 # boomer probability 

        super(Model, self).__init__()
        self.rnn = RNN()
        self.cnn = CNN()

        # convolutional layers encode pixels into image features 
        # 3 in channels, 6 out channels, kernel_size=5
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size)
        # 6 in channels, 16 out channels, kernel_size=5
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size)

        # pool
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        # flatten
        self.flat = nn.Flatten()

        # fully connected layers make the actual classification
        # 148 taken from summary, as there exists no utility
        # function for calculating this magic number
        self.fc1 = nn.Linear(in_lin1, out_lin1)
        self.fc2 = nn.Linear(out_lin1, out_lin2)

        # output: boomer or not boomer
        self.fc3 = nn.Linear(out_lin2, final_out)
    
    def forward(self, x_image, x_text, text_lengths):
        '''
        INPUTS:
            tensor x: (nr_samples, nr_channels (3), nr_x_pixels (32), nr_y_pixels (32))
        RETURNS:
            boomer probability
        '''
        h_image = self.cnn(x_image)
        h_text = self.rnn(x_text, text_lengths)

        h = torch.cat((h_image, h_text), dim=1)

        #TODO: figure out correct dimensionality of fc1
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h.view(-1)

if __name__ == "__main__":
    model = Model()
    print(model)

    from dataloader import BoomerDatasetContainer
    dataset = BoomerDatasetContainer(is_training=True)

    trainloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=4,
                                                shuffle=True,
                                                num_workers=2)

    iterator = iter(trainloader)
    output = model(iterator.next()[0])
    # test_images = torch.rand(25, 3, 32, 32) # 25 images with 3 channels of 32*32 pixels
    # output = model(test_images)

