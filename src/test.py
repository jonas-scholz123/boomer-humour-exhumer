import torch
import torchvision
from torchvision import transforms

from model import Model
from utils import imshow

class TestSuite:

    def __init__(self, model, testloader, classes):
        self.model = model
        self.testloader = testloader
        self.batch_size = testloader.batch_size
        self.classes = classes
    
    def preview_batch(self):
        dataiter = iter(self.testloader)
        x, y = dataiter.next()

        #print images
        imshow(torchvision.utils.make_grid(x))
        print('GroundTruth: ', ' '.join('%5s' % self.classes[y[j]] for j in range(self.batch_size)))

        y_hat = self.model(x)

        _, predicted = torch.max(y_hat, 1)
        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
                                for j in range(self.batch_size)))
    
    def calculate_acccuracy(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                x, y = data
                y_hat = self.model(x)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

class TestSuiteCifar(TestSuite):

    def __init__(self):

        batch_size = 4
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        model = Model()
        model.load_state_dict(torch.load("../saved_models/cifar.pth"))

        super().__init__(model, testloader, classes)


if __name__ == "__main__":
    test_suite = TestSuiteCifar()
    # test_suite.calculate_acccuracy()
    test_suite.preview_batch()



