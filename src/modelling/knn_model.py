import os
import numpy as np
import pickle
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.download_file import download_gdrive
from typing import Union, Any, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNN_Classify:

    def __init__(self, embedding_path='./data/embedding.pickle', knn_path='./models/knn.pickle'):
        """This class fits the KNN model and perform prediction given embeddings

        Args:
            embedding_path (str, optional): Path to the saved embedding pickle file. Defaults to './data/embedding.pickle'.
            knn_path (str, optional): Path to save or loads the knn pickle file. Defaults to './models/knn.pickle'.
        """
        self.embedding_path = embedding_path
        self.knn_path = knn_path
        self.knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
        self.embedding_dict = None
        self.embedding_url = 'https://drive.google.com/u/0/uc?export=download&confirm=eFgn&id=12dbMuLnbK2nzYQO7Hg_Gx-pueGB9HrGo'
        self.knn_url = "https://drive.google.com/u/0/uc?export=download&confirm=9aTU&id=1nzkRQUv2kUiY1r9u_Notep3EAvh8SMHp"

    def download_(self, url: str, file_path: str) -> Any:
        """Downloads model if not found, else load from directory

        Args:
            url (str): URL of the file
            file_path (str): Path of the file

        Returns:
            Any: An object from pickle file
        """
        logger.info(f"Getting {file_path}")
        if not os.path.exists(file_path):
            folder, filename = os.path.split(file_path)
            download_gdrive(url, folder, filename)
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"{file_path} successfully loaded")
        return obj

    def build_model(self, fit_knn=False) -> None:
        """Builds the KNN model

        Args:
            fit_knn (bool, optional): Whether to fit the knn and save to a pickle file. Defaults to False.
        """
        # Load embeddings and knn
        self.embedding_dict = self.download_(self.embedding_url, self.embedding_path)
        self.embedding_dict['embedding'] = np.array(self.embedding_dict['embedding'])
        self.embedding_dict['class'] = np.array(self.embedding_dict['class'])
        if not fit_knn:
            self.knn = self.download_(self.knn_url, self.knn_path)
        else:
            logger.info("Fitting KNN...")
            self.knn.fit(self.embedding_dict['embedding'], self.embedding_dict['class'])
            with open(self.knn_path, 'wb') as f:
                pickle.dump(self.knn, f)

    def predict(self, embedding: Union[np.ndarray, list], threshold=0.36331658291457286) -> Tuple[List[str], List[float]]:
        """Performs prediction given an embedding and a threshold

        Args:
            embedding (Union[np.ndarray, list]): Embedding of the image. [num_sample, num_features]
            threshold (float, optional): Threshold below which to classify the embedding as 'others' class. Defaults to 0.36331658291457286.

        Returns:
            Tuple[List[str], List[float]]: Returns a list of predicted class and a list of the corresponding probabilities
        """
        _, neigh_ind = self.knn.kneighbors(embedding, n_neighbors=1)
        neigh_ind = neigh_ind.ravel()
        neigh_embedding = self.embedding_dict['embedding'][neigh_ind]
        prob = cosine_similarity(embedding, neigh_embedding)
        prob = prob[:,0]
        y_pred = self.embedding_dict['class'][neigh_ind]
        y_pred[prob < threshold] = 'others'
        return y_pred, prob


if __name__ == '__main__':
    KC = KNN_Classify()
    KC.download_()


