import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        # TODO: pass hyperparameters

        super(Model, self).__init__()

        # convolutional layers encode pixels into image features 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # pool
        self.pool = nn.MaxPool2d(2, 2)

        # flatten
        self.flat = nn.Flatten()

        # fully connected layers make the actual classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #
        self.fc2 = nn.Linear(120, 84)

        # single output: boomer percentage 
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        '''
        INPUTS:
            tensor x: (nr_samples, nr_channels (3), nr_x_pixels (32), nr_y_pixels (32))
        '''
        # Max pooling over 2x2 window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flat(x)
        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = Model()
    test_images = torch.rand(25, 3, 32, 32) # 25 images with 3 channels of 32*32 pixels
    output = model(test_images)

    print(output)

