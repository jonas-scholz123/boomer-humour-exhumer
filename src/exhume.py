import sys
import os
from skimage import io
import torch

from dataloader import BoomerDatasetContainer
from model import Model
import config

if __name__ == "__main__":

    im_path = sys.argv[1]
    if not os.path.exists(im_path):
        print("Invalid Path")
        exit(0)

    model = Model()
    model.load_state_dict(torch.load(config.paths["saved_model"]))
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    image_processor = BoomerDatasetContainer()

    image = io.imread(im_path)
    image = image_processor.process_image(image)
    image = image.unsqueeze(0)
    y_hat = sigmoid(model(image)).flatten()[0]
    print(f"The image is {100*y_hat}% Boomerish")
    

        