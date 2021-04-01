import torch
from model import Model
from torch import optim
from tqdm import tqdm
from torch import nn
import torchvision
from torchvision import transforms

import config
from dataloader import BoomerDatasetContainer

class TrainingSuite:
    '''
    A wrapper around the training process
    '''

    def __init__(self, model, optimiser, loss_fn, trainloader):
        '''
        PARAMS:
            pyTorch model model: the model to be trained. Needs to return a tensor of 
                    shape (batch_size)
            pyTorch optimiser optimiser
            pyTorch.nn.loss loss_fn: a binary loss function (with logits for binary classification)
            pyTorch trainloader trainloader: the custom trainloader object that
                                             takes a dataset and batches it
        '''
        self.model = model
        self.trainloader = trainloader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
    
    def train(self, n_epochs):
        '''
        training loop, gets predictions, computes loss and backpropagates

        PARAMS:
            int n_epochs: number of training epochs
        '''
        for epoch in range(n_epochs):

            running_loss = 0.0

            for i, data in tqdm(enumerate(self.trainloader)):
                image, text, y = data

                self.optimiser.zero_grad()

                y_hat = self.model(image.float()) # either 0 or 1
                predicted = torch.round(y_hat)
                loss = self.loss_fn(y_hat, y.float())
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

                if i % 100 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
    
    def load_model(self, path):
        '''
        Loads model

        PARAMS:
            path: filepath ending in .pth where model is saved
        '''
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        '''
        Saves model

        PARAMS:
            path: filepath ending in .pth to save model to
        '''
        torch.save(self.model.state_dict(), path)
        print("Finished Training")


class TrainingSuiteBoomer(TrainingSuite):
    '''
    Wrapper that configures training suite for Boomer data
    '''

    def __init__(self):
        transform = None
        trainset = BoomerDatasetContainer(is_training=True)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=10,
                                                  shuffle=True,
                                                  num_workers=6)

        model = Model().float()

        optimiser = optim.Adam(model.parameters(), lr=0.001)

        # better than calling sigmoid in model.forward()
        loss_fn = nn.BCEWithLogitsLoss()

        super().__init__(model, optimiser, loss_fn, trainloader)
        

if __name__ == "__main__":
    MODEL_PATH = config.paths["saved_model"] 
    training_suite = TrainingSuiteBoomer()
    training_suite.load_model(MODEL_PATH)
    training_suite.train(n_epochs=1)
    training_suite.save_model(MODEL_PATH)