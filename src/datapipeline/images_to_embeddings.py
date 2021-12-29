import os
import json
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from typing import Tuple
from src.modelling.vggface_model import VGGFace_Model
from tqdm import tqdm


class Create_Embeddings:

    def __init__(self):
        self.cleaned_image_dict = defaultdict(list)
        self.augment = keras.Sequential([
            keras.layers.RandomZoom((-0.2, 0.2), fill_mode='reflect'),
            keras.layers.RandomTranslation(0.2, 0.2, fill_mode='reflect'),
            keras.layers.RandomRotation(0.2, fill_mode='reflect'),
            keras.layers.RandomContrast(0.2)
        ])

    def load_images_paths(self, img_path: str):
        image_formats = {'.jpg', '.png', '.jpeg'}
        for root, dirs, files in os.walk(img_path):
            for f in files:
                filename, file_extension = os.path.splitext(f)
                if file_extension in image_formats:
                    try:
                        sample_img_path = os.path.join(root ,f)
                        img = Image.open(sample_img_path)
                        img.verify()    # Check if image is corrupted
                        self.cleaned_image_dict[os.path.basename(root)] += [sample_img_path]
                    except (IOError, SyntaxError) as e:
                        print('Bad File:', f)

    def get_embeddings(self, image: np.ndarray, model, BGR=True, is_train=False):
        if BGR:
            image = image[:,:,::-1] # Convert from BGR to RGB
        if is_train:
            image = self.augment(image)
        # prepare the face for the model, e.g. center pixels
        image = np.asarray([image], 'float32')
        samples = preprocess_input(image, version=2)
        # perform prediction
        yhat = model.predict(samples)
        return yhat
    
    def build_embedding_db(self, model, save_path: str):
        rows = []
        total = 0
        for k, image_list in self.cleaned_image_dict.items():
            total += len(image_list)
        with tqdm(total=total, desc="embeddding progress") as pbar:
            for k, image_list in self.cleaned_image_dict.items():
                for img_path in image_list:
                    img = np.array(Image.open(img_path))
                    embedding = self.get_embeddings(img, model, BGR=False, is_train=False)
                    rows.append([embedding, k])
                    pbar.update(1)
        df = pd.DataFrame(rows, columns=['embedding', 'class'])
        df.to_csv(save_path, index=False)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    VGG_M = VGGFace_Model()
    model_folder_path = './models'
    model = VGG_M.build_model(model_folder_path)

    CE = Create_Embeddings()
    CE.load_images_paths('./data/raw/output/train')
    # image_path = CE.cleaned_image_dict['eric_kwok'][15]
    # img = np.array(Image.open(image_path))
    # print(CE.get_embeddings(img, model, BGR=False, is_train=True))
    csv_path = './data/embedding.csv'
    CE.build_embedding_db(model, csv_path)

