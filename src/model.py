import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

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
        in_lin1 = 21904 # = 148 ** 2 taken from summary
        out_lin1 = 120
        out_lin2 = 84
        final_out = 1 # boomer probability 

        super(Model, self).__init__()

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
        # self.fc3 = nn.Linear(out_lin2, final_out)
        self.fc3 = nn.Linear(out_lin2, 2)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        '''
        INPUTS:
            tensor x: (nr_samples, nr_channels (3), nr_x_pixels (32), nr_y_pixels (32))
        '''
        # Max pooling over 2x2 window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return self.sigmoid(x)

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

