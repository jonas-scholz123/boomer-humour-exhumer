'''
Metadata maker
'''
import pandas as pd
import os
from skimage import io
from tqdm import tqdm
from PIL import Image

class MetaData:
    '''
    Keeps track of testing/training data

    Columns: index, fpath, is_boomer, is_training
    '''

    def __init__(self, metadata_path, boomer_roots=[],
                 non_boomer_roots=[], training_frac=0.9, rebuild=False):
        self.columns = ["fpath, is_boomer, is_training"]
        self.fpath = metadata_path
        self.boomer_roots = boomer_roots
        self.non_boomer_roots = non_boomer_roots
        self.training_frac = training_frac

        if not rebuild:
            self.df = self.load_df()
        else:
            self.df = self.build_df()
    
    def load_df(self):
        if os.path.isfile(self.fpath):
            return pd.read_pickle(self.fpath)
        else:
            return self.build_df()

    def save_df(self):
        self.df.to_pickle(self.fpath)
    
    def image_is_valid(self, full_path):
        im = Image.open(full_path)
        try:
            im.verify()
            return True
        except UnidentifiedImageError:
            return False

    
    def make_df(self, full_paths, is_boomers):
            df = pd.DataFrame({
                "fpath": full_paths,
                "is_boomer": is_boomers
            })
            # df.set_index("fpath", inplace=True)
            df["is_training"] = True
            to_change = df.sample(frac= 1 - self.training_frac).index
            df.loc[to_change, "is_training"] = False
            return df
    
    def build_df(self):
        if not self.boomer_roots + self.non_boomer_roots:
            print("No data roots supplied, can't build dataframe")
            return
        
        full_paths = []
        is_boomers = []

        all_roots = self.boomer_roots + self.non_boomer_roots

        for root in tqdm(all_roots):
            for image_path in tqdm(os.listdir(root)):
                full_path = os.path.join(root, image_path)

                if not os.path.isfile(full_path):
                    # TODO: deal with subfolders
                    continue

                if (full_path.split(".")[-1] not in {"png", "jpg"}
                        or not self.image_is_valid(full_path)):
                    # delete all non videos
                    print("deleting: ", full_path)
                    os.remove(full_path)
                
                is_boomers.append(root in self.boomer_roots)
                full_paths.append(full_path)

        return self.make_df(full_paths, is_boomers)
        


if __name__ == "__main__":
    meta = MetaData(
        "../data/metadata.pkl",
        ["../data/Boomerhumour",
         "../data/boomershumor"],
        ["../data/me_irl",
         "../data/meirl",
         "../data/BlackPeopleTwitter",
         "../data/196",
         "../data/WhitePeopleTwitter"],
        rebuild=True
        )
    
    meta.save_df()

    print((meta.df["is_boomer"] == False).sum())
    print((meta.df["is_boomer"] == True).sum())