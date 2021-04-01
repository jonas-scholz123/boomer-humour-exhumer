import torch
from torch import nn
import torchvision
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from dataloader import BoomerDatasetContainer
from model import Model
from utils import imshow
import config


class TestSuite:
    '''
    Class for evaluating unseen test data and computing performance
    metrics.
    '''

    def __init__(self, model, testloader, metrics):
        '''
        PARAMS:
            pyTorch model model: the model to be evaluated
            pyTorch dataloader testloader: loads test data in batches
            list[function]: list of functions that take (ys, y_hats) as
                arguments and compute a metric.
        '''
        self.model = model
        self.model.eval() # eval mode
        self.testloader = testloader
        self.batch_size = testloader.batch_size
        self.sigmoid = nn.Sigmoid()
        self.metrics = metrics
    
    def preview_batch(self):
        '''
        Plots a batch of images and prints ground truth/prediction
        '''
        dataiter = iter(self.testloader)
        x, y = dataiter.next()

        #print images
        imshow(torchvision.utils.make_grid(x))
        print('GroundTruth: ', ' '.join('%5s' % y[j] for j in range(self.batch_size)))

        y_hat = self.model(x)

        # _, predicted = torch.max(y_hat, 1)
        predicted = torch.round(y_hat)
        print('Predicted: ', ' '.join('%5s' % predicted[j]
                                for j in range(self.batch_size)))
    
    def _get_pairs(self):
        '''
        Generic evaluation that simply returns
        all ys, y_hats for then passing into a metric
        '''
        ys = []
        y_hats = []
        with torch.no_grad():
            for x, y in tqdm(self.testloader):
                y_hat = self.sigmoid(self.model(x))
                ys.append(y)
                y_hats.append(y_hat)
        return torch.cat(ys), torch.cat(y_hats).detach()
    
    def evaluate(self):
        '''
        Evaluates all metrics

        RETURNS:
            list[tuple("name", metric)] results: all evaluated metrics
        '''
        if not self.metrics:
            return

        ys, y_hats = self._get_pairs()

        results = [metric(ys, y_hats) for metric in self.metrics]
        return results


class TestSuiteBoomer(TestSuite):
    '''
    Wrapper that specifies the parameters
    of TestSuite
    '''
    def __init__(self):

        transform = None
        batch_size = 10

        testset = BoomerDatasetContainer(is_training=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=4)

        metrics = [Metrics.accuracy, Metrics.confusion_matrix]

        model = Model()
        #TODO: change old model
        model.load_state_dict(torch.load(config.paths["saved_model"]))

        super().__init__(model, testloader, metrics)

class Metrics:
    '''
    All functions must take only ys, y_hats as args
    '''
    def accuracy(ys, y_hats):
        return "accuracy", float(1 - (ys - y_hats).round().abs().sum()/ys.shape[0])
    
    def confusion_matrix(ys, y_hats):
        return "confusion", confusion_matrix(ys, y_hats.round(), normalize="true")


if __name__ == "__main__":
    test_suite = TestSuiteBoomer()
    results = test_suite.evaluate()
    breakpoint()
    test_suite.preview_batch()



