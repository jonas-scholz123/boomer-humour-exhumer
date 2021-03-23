'''
Metadata maker
'''
import pandas as pd
import os
from skimage import io
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

import config

class MetaData:
    '''
    Keeps track of testing/training data filepaths, whether or not they
    are boomer memes and checks that they are all valid in MetaData.df

    Columns: index, fpath, is_boomer, is_training
    '''

    def __init__(self, metadata_path, boomer_roots=[],
                 non_boomer_roots=[], training_frac=0.9):
        '''
        PARAMS:
            str metadata_path: pkl filepath where the metadata is saved
            list[str] boomer_roots: root directories of boomer memes 
            list[str] non_boomer_roots: root directories of non boomer memes 
            float training_frac: what fraction of data is used for training
        '''

        self.fpath = metadata_path
        self.boomer_roots = boomer_roots
        self.non_boomer_roots = non_boomer_roots
        self.training_frac = training_frac

        self.df = self._build_df()
    
    def save_df(self):
        '''
        Saves dataframe 
        '''
        self.df.to_pickle(self.fpath)
    
    def image_is_valid(self, full_path):
        '''
        Use PIL to verify the integrity of images

        PARAMS:
            str full_path: image path
        RETURNS:
            bool validity of image 
        '''
        try:
            im = Image.open(full_path)
            im.verify()
            return True
        except UnidentifiedImageError:
            return False

    
    def _make_df(self, full_paths, is_boomers):
        '''
        Uses parsed data to make pd.DataFrame 

        PARAMS:
            list full_paths: list of full image paths
            list is_boomers: list of bools where is_boomers[i]
                                indicates whether or not the image
                                at full_path[i] is a boomer meme
        RETURNS:
            pd.DataFrame: the complete metadata frame
        '''
        df = pd.DataFrame({
            "fpath": full_paths,
            "is_boomer": is_boomers
        })
        # df.set_index("fpath", inplace=True)
        df["is_training"] = True
        to_change = df.sample(frac= 1 - self.training_frac).index
        df.loc[to_change, "is_training"] = False
        return df
    
    def _build_df(self):
        '''
        Parses data from root directories to make metaDataFrame

        RETURNS:
            pd.DataFrame: the complete metadata frame
        '''
        if not self.boomer_roots + self.non_boomer_roots:
            print("No data roots supplied, can't build dataframe")
            return
        
        full_paths = []
        is_boomers = []

        all_roots = self.boomer_roots + self.non_boomer_roots

        for i, root in enumerate(all_roots, 1):
            print("root: ", root, i, "/", len(all_roots))
            for image_path in tqdm(os.listdir(root)):
                full_path = os.path.join(root, image_path)

                if not os.path.isfile(full_path):
                    # TODO: deal with subfolders
                    continue

                if (full_path.split(".")[-1] not in {"png", "jpg"}
                        or not self.image_is_valid(full_path)):
                    # delete all videos & corrupt images
                    print("deleting: ", full_path)
                    os.remove(full_path)
                
                is_boomers.append(root in self.boomer_roots)
                full_paths.append(full_path)

        return self._make_df(full_paths, is_boomers)
        


if __name__ == "__main__":

    metadata_path = config.paths["metadata"]

    boomer_roots = config.paths["boomer_roots"]

    non_boomer_roots = config.paths["non_boomer_roots"]

    meta = MetaData(
        metadata_path,
        boomer_roots=boomer_roots,
        non_boomer_roots=non_boomer_roots
    )
    
    meta.save_df()

    print("number normal memes: ", (meta.df["is_boomer"] == False).sum())
    print("number boomer memes: ", (meta.df["is_boomer"] == True).sum())