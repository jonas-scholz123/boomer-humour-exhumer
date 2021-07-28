import sys
import os
from skimage import io
import torch
import pickle
from nltk import word_tokenize

from dataloader import BoomerDatasetContainer
from ocr import GCloudOCR, TesseractOCR
from model import Model
import config

from testing import TestSuiteBoomer

class Exhumer:

    def __init__(self, model, image_processor, ocr_engines):
        self.model = model
        self.image_processor = image_processor

        # for mapping nr to probability
        self.sigmoid = torch.nn.Sigmoid()

        self.ocr_engines = ocr_engines

        with open(config.paths["word2id"], "rb") as f:
            self.word2id = pickle.load(f)
    
    def exhume(self, im_path):
        '''
            return_data: bool: whether or not to also return text, engine title etc
        '''
        prob, _, _ = self.exhume_with_meta(im_path)
        return prob
    
    def exhume_with_meta(self, im_path):
        if not os.path.exists(im_path):
            print("Invalid Path")
            return
        image = io.imread(im_path)
        image = self.image_processor.process_image(image)
        
        for engine in self.ocr_engines:
            text = engine.extract_text(im_path).lower()
            if engine.is_valid_annotation(text):
                break

        used_engine = engine
        text_ids = [self.word2id[tok] for tok in word_tokenize(text) if tok in self.word2id]

        x_image = image.unsqueeze(0)
        x_text = torch.IntTensor(text_ids).unsqueeze(0)
        length = torch.tensor(len(text_ids)).unsqueeze(0)
        prob = float(self.sigmoid(self.model(x_image, x_text, length)).flatten()[0])
        
        return prob, used_engine, text

class ExhumerContainer(Exhumer):

    def __init__(self):
        model = Model()
        model.load_state_dict(torch.load(config.paths["saved_model"]))
        model.eval()

        image_processor = BoomerDatasetContainer()

        # sorted from first used to last used
        ocr_engines = [TesseractOCR(), GCloudOCR()]
        ocr_engines = [engine for engine in ocr_engines if engine.valid]
        if len(ocr_engines) == 0:
            raise Exception('No ocr engines are valid')

        super().__init__(model, image_processor, ocr_engines)

if __name__ == "__main__":

    im_path = sys.argv[1]
    exhumer = ExhumerContainer()
    prob = exhumer.exhume(im_path)
    print(f"The image is { round(100*prob, 2) }% boomerish")


    

        