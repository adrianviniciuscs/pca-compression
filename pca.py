from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit_transform(self, X):

    def inverse_transform(self, X):


def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")
