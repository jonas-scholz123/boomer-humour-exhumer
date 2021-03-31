import pytesseract
import cv2
import numpy as np
from google.cloud import vision_v1 as vision
from google.cloud import storage
import io
import os
import json
from tqdm import tqdm

import config


class TesseractOCR:
    '''
    Uses cv2 to preprocess images and tesseract OCR to extract text.
    Is free but not as good as Google cloud.
    '''

    def process_image(self, image):
        '''
        Processes image so that tesseract has a better chance of working

        PARAMS:
            image: an image instance from cv2.imread()
        RETURNS:
            processed image
        '''
        # resize because tesseract can't deal below 300dpi
        # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)


        # image = cv2.blur(image,(5,5))
        # blur for smoothing edges
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.threshold(cv2.medianBlur(image, 3), 240, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


        #make black and white image
        # cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        return image
    
    def batch_extract(self, fpaths):
        '''
        Simple batched version for consistent API
        '''
        return [self.extract_text(fpath) for fpath in fpaths]

    
    def extract_text(self, fpath):
        '''
        Opens image at fpath, processes it and extracts text

        PARAMS:
            str fpath: filepath of image
        '''

        image = cv2.imread(fpath)
        image = self.process_image(image)
        text = pytesseract.image_to_string(image, lang="eng", config="--psm 1")

        return text

class GCloudOCR:
    '''
    Google Cloud OCR interface. Costs money and needs authenticated account etc.
    '''

    def __init__(self):
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.paths["gcloud_credentials"]
        #for annotations
        self.client = vision.ImageAnnotatorClient()

        #for accessing bucket
        self.storage_client = storage.Client() 
        self.bucket = self.storage_client.bucket("meme_bucket")

        self.bucket_root = "gs://meme_bucket/"
        self.image_dir = "to_annotate/"
        self.annotations_dir = "annotations/"
        self.bucket_annotations_path = "gs://meme_bucket/annotations/"

    def extract_text(self, path):
        '''
        Single image text extraction
        PARAMS:
            str path: filepath of image
        '''

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = self.client.text_detection(image=image)

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return response.full_text_annotation.text
    
    def fpath_to_blob(self, fpath):
        '''
        Turns local fpath into gCloud bucket blob
        '''
        fname = fpath.split("/")[-1]
        return self.bucket.blob(self.image_dir + fname)
    
    def upload_to_bucket(self, fpath):
        '''
        Uploads image to appropriate directory in bucket
        '''
        blob = self.fpath_to_blob(fpath)
        blob.upload_from_filename(fpath)
    
    def delete_from_bucket(self, fpath):
        '''
        Deletes blob corresponding to fpath from bucket
        '''
        blob = self.fpath_to_blob(fpath)
        blob.delete()

    def batch_extract(self, fpaths):
        '''
        Uses gClouds batch processing capability and leaves no new files in bucket
        (minimise storage cost)
        '''

        batch_size = len(fpaths)

        print("uploading images to bucket")
        for fpath in tqdm(fpaths):
            self.upload_to_bucket(fpath)
        
        print("Annotating images")
        self.sample_async_batch_annotate_images(fpaths, batch_size)

        json_path = self.annotations_dir + f"output-{1}-to-{batch_size}.json"
        annotations = self.read_bucket_annotations(json_path)

        print("deleting images from bucket")
        for fpath in tqdm(fpaths):
            self.delete_from_bucket(fpath)
        return annotations
    
    def read_bucket_annotations(self, json_path):
        json_blob = self.bucket.blob(json_path)
        data = json.loads(json_blob.download_as_string(client=None))
        # annotations = [entry.get("fullTextAnnotation")["text"] for entry in data["responses"]]

        annotations = []
        for entry in data["responses"]:
            if "fullTextAnnotation" in entry.keys():
                ft_annot = entry["fullTextAnnotation"]
                if "text" in ft_annot.keys():
                    annotations.append(ft_annot["text"])
                    continue
            annotations.append("NO_TEXT_DETECTED")

        #storage costs money -> delete
        json_blob.delete()
        return annotations
    
    def sample_async_batch_annotate_images(self, fpaths, batch_size):
        """Perform async batch image annotation."""

        output_uri = self.bucket_annotations_path

        image_uris = []
        for blob in self.bucket.list_blobs(prefix=self.image_dir):
            if blob.name.endswith("/"):
                # don't want directory
                continue

        blobs = [self.fpath_to_blob(fpath) for fpath in fpaths]
        
        image_uris = [self.bucket_root + blob.name for blob in blobs]

        client = vision.ImageAnnotatorClient()

        features = [
            {"type_": vision.Feature.Type.TEXT_DETECTION},
        ]

        requests = []
        for input_image_uri in image_uris: 
            source = {"image_uri": input_image_uri}
            image = {"source": source}
            requests.append({"image": image, "features": features})
        
        gcs_destination = {"uri": output_uri}

        # The max number of responses to output in each JSON file
        output_config = {"gcs_destination": gcs_destination,
                        "batch_size": batch_size}

        operation = client.async_batch_annotate_images(requests=requests, output_config=output_config)

        print("Waiting for operation to complete...")
        response = operation.result()

        # The output is written to GCS with the provided output_uri as prefix
        gcs_output_uri = response.output_config.gcs_destination.uri
        print("Output written to GCS with prefix: {}".format(gcs_output_uri))




if __name__ == "__main__":
    FPATH = "../images/non_boomer.jpg"
    ROOT = "../data/boomerhumour"


    gocr = GCloudOCR()
    gocr.sample_async_batch_annotate_images(5)
    # fnames, annots = gocr.read_bucket_annotations("annotations/output-1-to-95.json")

    # gocr.sample_async_batch_annotate_images()
    # ocr = TesseractOCR()

    #for fname in os.listdir(ROOT)[105:107]:
        #fpath = os.path.join(ROOT, fname)
        #text = ocr.extract_text(fpath)
        # gtext = gocr.extract_text(fpath)
        #print("TESSERACT:")
        #print(text.replace("\n", " "))
        #print("GOOGLE:")
        #print(gtext.replace("\n", " "))
    
    # fnames = os.listdir(ROOT)[0:3]
    # fpaths = [os.path.join(ROOT, fname) for fname in fnames]

    # annotations = gocr.batch_extract(fpaths)

    # responses = gocr.batch_extract(fpaths)
    # breakpoint()