from models_util import get_cnn_embedding_model
import torch


class Embedding(object):
    def __init__(self):
        self.cnn_model = get_cnn_embedding_model()

    def __call__(self, sample):
        embedding = self.cnn_model.predict(sample.numpy().reshape(-1, 28, 28, 1))
        embedding = embedding.reshape((128))
        embedding = torch.from_numpy(embedding)
        return embedding
