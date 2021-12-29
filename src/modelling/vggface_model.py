import requests
import os
import logging
import pickle
from tqdm import tqdm
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGGFace_Model:

    def __init__(
        self, 
        url="https://drive.google.com/u/0/uc?id=19_ESwSZPCJ7KzW72PAwc1z0R6ZD6wRPi&export=download"
        ):
        self.url = url
        self.model_path = 'models/base_model.pickle'

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
            if not os.path.exists(target_path):
                logger.info(f"{target_path} not found. Creating...")
                os.makedirs(target_path)
            file_name = 'base_model.pickle'
            self.model_path = os.path.join(target_path, file_name)
            r = requests.get(self.url, stream=True)
            total = int(r.headers.get('content-length', 0))
            with open(self.model_path, 'wb') as f, tqdm(
                    desc=f"Download {file_name} progress",
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    ) as bar:
                    for data in r.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)

        with open(self.model_path, 'rb') as f:
            base_model = pickle.load(f)
        model = Model(base_model=base_model)
        return model


class Model(keras.Model):

    def __init__(self, base_model):
        super().__init__()
        self.resize = keras.layers.Resizing(224,224)
        self.augment = keras.Sequential([
            keras.layers.RandomZoom((-0.2, 0.2), fill_mode='reflect'),
            keras.layers.RandomTranslation(0.2, 0.2, fill_mode='reflect'),
            keras.layers.RandomRotation(0.2, fill_mode='reflect'),
            keras.layers.RandomContrast(0.2)
        ])
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        x = self.resize(inputs)
        if training:
            x = self.augment(x)
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