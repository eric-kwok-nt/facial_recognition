import logging
from tensorflow import keras
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGGFace_Model:

    def __init__(self):
        """Builds and save the VGGFace2 model
        """
        self.base_model = None

    def download_model(self):
        """Downloads the VGGFace model from the original source.
        """
        logger.info("Proceed to download model...")
        self.base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    def build_model(self) -> keras.Model:
        """Builds the model using the base model. If base_mode.pickle not found in 

        Returns:
            keras.Model: Model with resize layer
        """
        logger.info("Building model...")
        model = Model(base_model=self.base_model)
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
    model_path = './models/base_model.pickle'
    VGG_M = VGGFace_Model(model_path=model_path)
    VGG_M.download_model()
    model = VGG_M.build_model()
    
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