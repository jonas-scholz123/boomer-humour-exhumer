# %%
'''
Metadata maker
'''
import pandas as pd
import pickle
import os
from nltk import word_tokenize
from skimage import io
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

import config
from ocr import GCloudOCR, TesseractOCR
from utils import ConceptNetDict
from matplotlib.pyplot import imshow, show


class MetaData:
    '''
    Keeps track of testing/training data filepaths, whether or not they
    are boomer memes and checks that they are all valid in MetaData.df

    Columns: index, fpath, is_boomer, is_training
    '''

    def __init__(self, metadata_path, ocr, boomer_roots=[],
                 non_boomer_roots=[], training_frac=0.9, correct_word_threshold=0.7):
        '''
        PARAMS:
            str metadata_path: pkl filepath where the metadata is saved
            list[str] boomer_roots: root directories of boomer memes 
            list[str] non_boomer_roots: root directories of non boomer memes 
            float training_frac: what fraction of data is used for training
        '''

        self.fpath = metadata_path
        self.ocr = ocr
        self.boomer_roots = boomer_roots
        self.non_boomer_roots = non_boomer_roots
        self.word2id_path = config.paths["word2id"]


        self.training_frac = training_frac
        self.embeds = ConceptNetDict()
        self.correct_word_threshold = correct_word_threshold

        self.df = self._load_df()
        self.word2id = self._load_word2id()
        # in case word2id doesn't exist, we populate from df
        if not self.word2id:
            self._populate_word2id()
    
    def _load_df(self):
        '''
        If exists, doesn't rebuild
        '''
        if os.path.exists(self.fpath):
            print("loaded existing dataframe")
            return pd.read_pickle(self.fpath)
        else:
            print("Building dataframe from scratch")
            self._build_df()
    
    def _save_df(self):
        '''
        Saves dataframe 
        '''
        self.df.to_pickle(self.fpath)

    def _populate_word2id(self):
        "populating word2id!"
        not_nan = self.df["text"].notnull()

        for text in self.df.loc[not_nan, "text"]:
            self.add_to_vocab(text)

    def _load_word2id(self):
        if os.path.exists(self.word2id_path):
            with open(self.word2id_path, "rb") as f:
                return pickle.load(f)
        else:
            return {}
    
    def _save_word2id(self):
        with open(self.word2id_path, "wb") as f:
            pickle.dump(self.word2id, f)
    
    def save(self):
        self._save_df()
        self._save_word2id()
    
    def add_to_vocab(self, sentence):
        tokens = word_tokenize(sentence)
        for token in tokens:
            token = token.lower()
            if token not in self.word2id:
                self.word2id[token] = len(self.word2id)
    
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
    
    def is_valid_annotation(self, annotation):

        valids = 0
        tokens = word_tokenize(annotation)

        if len(tokens) == 0:
            return False

        for t in tokens:
            if t.lower() in self.embeds:
                valids += 1
        
        return valids/len(tokens) > self.correct_word_threshold
    
    def annotate_texts(self, max_extractions, batch=True):
        '''
        Uses self.ocr to extract text from images
        and saves this text in self.df

        PARAMS:
            int max_extractions: won't run more extractions than max_extractions,
                                 to save costs.
            bool batch: whether or not the batched version of the OCR
                        engine should be used.
        RETURNS:
            int invalids: number of invalid (failed) extractions
        '''
        if not self.ocr:
            print("Error: no OCR engine provided.")
        
        # initial setup
        if "text" not in self.df.columns:
            self.df.loc[:, "text"] = None
        if "ocr_engine" not in self.df.columns:
            self.df.loc[:, "ocr_engine"] = None

        nr_extractions = 0

        mask = self.df["text"].isnull()
        # get indices of unannotated entries
        indices = self.df.loc[mask].index[:max_extractions]

        if batch:
            #NOTE: batched doesn't support validation checks (yet)
            valid_count = max_extractions
            fpaths = self.df.loc[indices, "fpath"]
            annotations = self.ocr.batch_extract(fpaths)
            self.df.loc[indices, "text"] = annotations

        else:
            valid_count = 0
            for idx in tqdm(indices):

                annotation = self.ocr.extract_text(self.df.loc[idx, "fpath"])
                self.add_to_vocab(annotation)
                # print("INDEX: ", idx)
                # imshow(io.imread(self.df.loc[idx, "fpath"]))
                # show()
                # print(annotation)
                nr_extractions += 1

                if self.is_valid_annotation(annotation):
                    valid_count += 1
                    self.df.loc[idx, "text"] = annotation

            print(f"{100 * valid_count/max_extractions}% valid.")
        
        self.df.loc[indices, "ocr_engine"] = self.ocr.__class__.__name__
        return max_extractions - valid_count
    
    def print_stats(self):
        print("number normal memes: ", (meta.df["is_boomer"] == False).sum())
        print("number boomer memes: ", (meta.df["is_boomer"] == True).sum())

        if "text" in self.df.columns:
            total_annotated = self.df["text"].notnull().sum()
            print(f"{total_annotated} out of {self.df.shape[0]} entries annotated.")
    
    def sample_annotation(self, idx):
        fpath = self.df.loc[idx, "fpath"]
        text = self.df.loc[idx, "text"]
        engine = self.df.loc[idx, "ocr_engine"]
        image = io.imread(fpath)
        imshow(image)
        print(f"Transcribed by {engine}")
        print(text)


if __name__ == "__main__":

    metadata_path = config.paths["metadata"]
    boomer_roots = config.paths["boomer_roots"]
    non_boomer_roots = config.paths["non_boomer_roots"]
    batch_size = 100
    meta = MetaData(
        metadata_path,
        ocr=None,
        boomer_roots=boomer_roots,
        non_boomer_roots=non_boomer_roots,
        correct_word_threshold=None
    )
    # %%

    while meta.df["text"].isnull().sum() > 0:

        # annotate with tesseract ocr first
        meta.ocr = TesseractOCR()
        meta.correct_word_threshold=0.7

        nr_invalid = meta.annotate_texts(batch_size, batch=False)
        meta.print_stats()
        meta.save()

        meta.ocr = GCloudOCR()
        meta.correct_word_threshold=0.0

        # then annotate failed ones with Google
        meta.annotate_texts(nr_invalid, batch=True)
        meta.print_stats()
        meta.save()