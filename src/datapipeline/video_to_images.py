import cv2
import logging
import os
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import pdb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Video2Image:

    def __init__(self):
        self.detector = MTCNN()

    def convert(self, vid_path: str, name: str, save_path: str):
        """Convert video to images

        Args:
            vid_path (str): Path to video
            name (str): Name of person
            save_path (str): Saving path
            min_row (int, optional): Cropping height start point. Defaults to 0.
            max_row ([type], optional): Cropping height end point. Defaults to None.
            min_col (int, optional): Cropping width start point. Defaults to 0.
            max_col ([type], optional): Cropping width end point. Defaults to None.
        """
        assert os.path.exists(vid_path), "Video does not exist!"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger.info("Start conversion...")
        vidcap = cv2.VideoCapture(vid_path)
        success, image = vidcap.read()
        count = 0
        while success:
            image = self._image_rotate(image, name)
            face = self.extract_face(image, (224, 224))
            if face is not False:
                image_path = os.path.join(save_path, f"{count}_{name}.jpg")
                cv2.imwrite(image_path, face)
                success, image = vidcap.read()
                count += 1
        
        logger.info(f"Converted {count} images")

    def _image_rotate(self, image, name):
        # Private method to rotate image
        if name == 'eric_kwok':
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
        elif name == 'eric_lee':
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        else:
            return image

    def extract_face(self, image: np.ndarray, required_size=(224, 224)):
        image_rgb = image[:,:,::-1]
        results = self.detector.detect_faces(image_rgb)
        if len(results) == 0:
            return False
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = image[y1:y2, x1:x2]
        # resize pixels to the model size
        face = Image.fromarray(face)
        face = face.resize(required_size)
        face_array = np.asarray(face)
        return face_array


if __name__ == '__main__':
    # vid_paths = ['data/raw/videos/eric_lee.mp4', 'data/raw/videos/clarence.mov', 'data/raw/videos/eric_kwok.mp4']
    # names = ['eric_lee', 'clarence', 'eric_kwok']
    # vid_paths = ['data/raw/videos/clarence.mov', 'data/raw/videos/eric_kwok.mp4']
    # names = ['clarence', 'eric_kwok']
    vid_paths = ['data/raw/videos/eric_kwok.mp4']
    names = ['eric_kwok']
    VI = Video2Image()
    for vid_path, name in zip(vid_paths, names):
        save_path = os.path.join('data/raw/images', name)
        VI.convert(vid_path, name, save_path)
