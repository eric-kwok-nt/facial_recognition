import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.download_file import download


class KNN_Classify:

    def __init__(self, csv_path='./data/embedding.csv'):
        self.csv_path = csv_path

    def download_embeddings(self):
        if not os.path.exists(self.csv_path):
            csv_url = 'https://drive.google.com/u/0/uc?id=1WvywpI2jezcoPqFqBOH5hsVCH2Em-4ax&export=download'
            folder, filename = os.path.split(self.csv_path)
            download(csv_url, folder, filename)

if __name__ == '__main__':
    KC = KNN_Classify()
    KC.download_embeddings()


