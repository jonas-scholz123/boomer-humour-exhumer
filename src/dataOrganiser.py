'''
Metadata maker
'''
import pandas as pd
import os

class MetaData:
    '''
    Keeps track of testing/training data

    Columns: index, fpath, is_boomer, is_training
    '''

    def __init__(self, metadata_path, boomer_roots=[], non_boomer_roots=[], training_frac=0.9):
        self.columns = ["fpath, is_boomer, is_training"]
        self.fpath = metadata_path
        self.boomer_roots = boomer_roots
        self.non_boomer_roots = non_boomer_roots
        self.training_frac = training_frac

        self.df = self.load_df()
    
    def load_df(self):
        if os.path.isfile(self.fpath):
            return pd.read_pickle(self.fpath)
        else:
            return self.build_df()

    def save_df(self):
        self.df.to_pickle(self.fpath)
    
    def build_df(self):
        if not self.boomer_roots + self.non_boomer_roots:
            print("No data roots supplied, can't build dataframe")
            return
        
        full_paths = []
        is_boomers = []

        for root in self.boomer_roots + self.non_boomer_roots:
            for image_path in os.listdir(root):
                # print(image_path)
                full_path = os.path.join(root, image_path) 
                if not os.path.isfile(full_path):
                    # TODO: deal with subfolders
                    continue

                if image_path.split(".")[-1] not in {"png", "jpg"}:
                    # delete all non videos
                    os.remove(full_path)
                    print(image_path.split(".")[-1])

                is_boomers.append(root in self.boomer_roots)
                full_paths.append(full_path)
        
        df = pd.DataFrame({
            "fpath": full_paths,
            "is_boomer": is_boomers
        })
        # df.set_index("fpath", inplace=True)
        df["is_training"] = True
        to_change = df.sample(frac= 1 - self.training_frac).index
        df.loc[to_change, "is_training"] = False
        return df

                



if __name__ == "__main__":
    meta = MetaData(
        "../data/metadata.pkl", 
        ["../data/Boomerhumour", "../data/boomershumor"]
        )
    
    meta.save_df()

    print(meta.df)