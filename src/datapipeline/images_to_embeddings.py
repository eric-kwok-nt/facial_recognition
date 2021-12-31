import os
import pickle
from PIL import Image
import numpy as np
from collections import defaultdict
from tensorflow import keras
from keras_vggface.utils import preprocess_input
from src.modelling.vggface_model import VGGFace_Model
from tqdm import tqdm
from typing import Dict


class Create_Embeddings:

    def __init__(self):
        """This class loads images and creates embeddings for images
        """
        self.cleaned_image_dict = defaultdict(list)
        self.augment = keras.Sequential([
            keras.layers.RandomRotation(0.2, fill_mode='reflect'),
            keras.layers.RandomContrast(0.2)
        ])

    def load_images_paths(self, img_path: str):
        """Walk through the files and folders in img_path and save the image paths as values with subfolders as keys (class of image)
        It also checks whether the image is corrupted.

        Args:
            img_path (str): Path of the image
        """
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

    def get_embeddings(self, image: np.ndarray, model: keras.Model, BGR=True, augment=False) -> np.ndarray:
        """Given an image and a backbone model, it outputs the embedding

        Args:
            image (np.ndarray): Image to be converted
            model (keras.Model): Model to use for the conversion
            BGR (bool, optional): Whether the image is in BGR or RGB. Defaults to True.
            augment (bool, optional): Whether to augment the image. Defaults to False.

        Returns:
            np.ndarray: Image Embedding [num_sample, num_features]
        """
        if BGR:
            image = image[:,:,::-1] # Convert from BGR to RGB
        if augment:
            image = self.augment(image)
        # prepare the face for the model, e.g. center pixels
        image = np.asarray([image], 'float32')
        samples = preprocess_input(image, version=2)
        # perform prediction
        yhat = model.predict(samples)
        return yhat
    
    def build_embedding_db(self, model: keras.Model, save_path=None, augment=0) -> Dict[str, list]:
        """Builds an embedding database

        Args:
            model (keras.Model): Model used for obtaining embeddings
            save_path (str, optional): Path to save the embeddings to in the form of pickle format. If None, it will not save into pickle file. Defaults to None.
            augment (int, optional): Number of augmented copy of images per image. Defaults to 0.

        Returns:
            Dict[str, list]: Keys are 'embedding' and 'class'. Each element in the embedding list correspond to the element in the class list
        """
        assert isinstance(augment, int) and (augment >= 0), "Please enter an augment integer >= 0"
        embedding_dict = defaultdict(list)
        total = 0
        for k, image_list in self.cleaned_image_dict.items():
            total += len(image_list) * (augment + 1)
        with tqdm(total=total, desc="embeddding progress") as pbar:
            for k, image_list in self.cleaned_image_dict.items():
                for img_path in image_list:                    
                    img = np.array(Image.open(img_path))
                    aug = False
                    for _ in range(augment+1):
                        embedding = self.get_embeddings(img, model, BGR=False, augment=aug).tolist()
                        embedding_dict['embedding'] += embedding
                        embedding_dict['class'] += [k]
                        aug = True
                        pbar.update(1)
        if save_path is not None:
            folder, _ = os.path.split(save_path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(save_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
        return embedding_dict


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model_path = 'models/base_model.pickle'
    VGG_M = VGGFace_Model(model_path=model_path)
    model = VGG_M.build_model()

    CE = Create_Embeddings()
    CE.load_images_paths('./data/raw/output/train')
    # image_path = CE.cleaned_image_dict['eric_kwok'][15]
    # img = np.array(Image.open(image_path))
    # print(CE.get_embeddings(img, model, BGR=False, is_train=True))
    embedding_path = './data/embedding.pickle'
    CE.build_embedding_db(model, embedding_path, augment=4)

