from conv_net import ConvNet
import torch


class CNNFeatureExtractor:
    """
    CNN feature extractor
    """
    def __init__(self):
        self.model = self.get_cnn_embedding_model()

    def forward(self, x):
        return self.model.embed(x)

    @staticmethod
    def get_cnn_embedding_model():
        cnn_model = ConvNet()
        cnn_model.load_state_dict(torch.load("models/cnn_mnist.ckpt", map_location='cpu'))
        cnn_model.eval()
        return cnn_model

