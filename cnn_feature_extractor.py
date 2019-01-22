from keras.models import load_model


class CNNFeatureExtractor:
    """
    CNN feature extractor
    """
    def __init__(self):
        self.model = self.get_cnn_embedding_model()

    def forward(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_cnn_embedding_model():
        cnn_model = load_model('models/CNN_mnist_model.h5')
        cnn_model.pop()
        cnn_model.pop()

        return cnn_model

