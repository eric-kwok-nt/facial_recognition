import os
import logging
import pickle
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import numpy as np
from src.utils.download_file import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGGFace_Model:

    def __init__(
        self, 
        url="https://drive.google.com/u/0/uc?id=19_ESwSZPCJ7KzW72PAwc1z0R6ZD6wRPi&export=download",
        model_path='models/base_model.pickle'
        ):
        self.url = url
        self.model_path = model_path

    def download_model(self, target_path: str):
        base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        if not os.path.exists(target_path):
            logger.info(f"{target_path} not found. Creating...")
            os.makedirs(target_path)
        file_name = 'base_model.pickle'
        self.model_path = os.path.join(target_path, file_name)
        with open(self.model_path, 'wb') as f:
            pickle.dump(base_model, f)
    
    def build_model(self, target_path=None):
        if not os.path.exists(self.model_path):
            assert target_path is not None, "Please input target path for base model to be downloaded"
            download(self.url, target_path, 'base_model.pickle')

        with open(self.model_path, 'rb') as f:
            base_model = pickle.load(f)
        model = Model(base_model=base_model)
        return model


class Model(keras.Model):

    def __init__(self, base_model):
        super().__init__()
        self.resize = keras.layers.Resizing(224,224)
        self.base_model = base_model
    
    def call(self, inputs):
        x = self.resize(inputs)
        x = self.base_model(x)
        return x

    
if __name__ == '__main__':
    VGG_M = VGGFace_Model()
    model_folder_path = './models'
    # VGG_M.download_model(target_path=model_folder_path)
    model = VGG_M.build_model(model_folder_path)
    
    def get_embeddings(files, model):
        faces = [f for f in files]
        samples = np.asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = model.predict(samples)
        return yhat
    
    image_path = './data/raw/images/clarence/300_clarence.jpg'
    img = Image.open(image_path)    
    img = np.array(img)
    print(get_embeddings([img], model))