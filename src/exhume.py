import sys
import os
from skimage import io
import torch

from dataloader import BoomerDatasetContainer
from model import Model
import config

class Exhumer:

    def __init__(self, model, image_processor):
        self.model = model
        self.image_processor = image_processor

        # for mapping nr to probability
        self.sigmoid = torch.nn.Sigmoid()
    
    def exhume(self, im_path):
        if not os.path.exists(im_path):
            print("Invalid Path")
            return

        image = io.imread(im_path)
        image = self.image_processor.process_image(image)
        image = image.unsqueeze(0)
        y_hat = float(self.sigmoid(self.model(image)).flatten()[0])
        print(f"The image is { round(100*y_hat, 2) }% boomerish")


class ExhumerContainer(Exhumer):

    def __init__(self):
        model = Model()
        model.load_state_dict(torch.load(config.paths["saved_model"]))
        model.eval()

        image_processor = BoomerDatasetContainer()

        super().__init__(model, image_processor)

if __name__ == "__main__":

    im_path = sys.argv[1]
    exhumer = ExhumerContainer()
    exhumer.exhume(im_path)


    

        