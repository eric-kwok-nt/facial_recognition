import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.download_file import download


class KNN_Classify:

    def __init__(self, embedding_path='./data/embedding.pickle', knn_path='./models/knn.pickle'):
        self.embedding_path = embedding_path
        self.knn_path = knn_path
        self.knn = KNeighborsClassifier(1, metric='euclidean', n_jobs=-1)
        self.embedding_dict = None
        self.embedding_url = 'https://drive.google.com/u/0/uc?export=download&confirm=663Q&id=12dbMuLnbK2nzYQO7Hg_Gx-pueGB9HrGo'
        self.knn_url = "https://drive.google.com/u/0/uc?export=download&confirm=OyUD&id=1nzkRQUv2kUiY1r9u_Notep3EAvh8SMHp"

    def download_(self, url, file_path):
        if not os.path.exists(file_path):
            folder, filename = os.path.split(file_path)
            download(url, folder, filename)
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def build_model(self, fit_knn=False):
        # Load embeddings and knn
        self.embedding_dict = self.download_(self.embedding_url, self.embedding_path)
        self.embedding_dict['embedding'] = np.array(self.embedding_dict['embedding'])
        self.embedding_dict['class'] = np.array(self.embedding_dict['class'])
        if not fit_knn:
            self.knn = self.download_(self.knn_url, self.knn_path)
        else:
            self.knn.fit(self.embedding_dict['embedding'], self.embedding_dict['class'])
            with open(self.knn_path, 'wb') as f:
                pickle.dump(self.knn, f)

    def predict(self, embedding: np.ndarray, threshold=0.36331658291457286):
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


