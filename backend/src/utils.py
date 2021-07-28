import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class ConceptNetDict:
    def __init__(self):
        path = config.paths["embeddings"] + "en_mini_conceptnet.h5"
        self.df = pd.read_hdf(path, "data")

    def __getitem__(self, idx):
        return self.df.loc[idx].values

    def __contains__(self, idx):
        return self.get(idx) is not None

    def get(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            return