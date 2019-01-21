from cnn import get_cnn_embedding_model


class CNNFeatureExtractor:
    """
    CNN feature extractor
    """
    def __init__(self):
        self.model = get_cnn_embedding_model()

    def forward(self, x):
        return self.model.predict(x)

