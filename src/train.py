import torch
from model import Model
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms

class TrainingSuite:

    def __init__(self, model, optimiser, loss_fn, trainloader):
        self.model = model 
        self.trainloader = trainloader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
    
    def train(self, n_epochs):
        for epoch in range(n_epochs):

            running_loss = 0.0

            for i, data in enumerate(self.trainloader):
                x, y = data

                self.optimiser.zero_grad()

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

                if i % 2000 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Finished Training")

class TrainingSuiteCifar(TrainingSuite):

    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)
        model = Model()

        optimiser = optim.Adam(model.parameters(), lr=0.001)
        # optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        loss_fn = nn.CrossEntropyLoss()

        super().__init__(model, optimiser, loss_fn, trainloader)


        

if __name__ == "__main__":
    MODEL_PATH = '../saved_models/cifar.pth'
    training_suite = TrainingSuiteCifar()
    training_suite.train(n_epochs = 3)
    training_suite.save_model(MODEL_PATH)