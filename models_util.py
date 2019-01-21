from keras.models import load_model


def get_cnn_embedding_model():
    cnn_model = load_model('models/CNN_mnist_model.h5')
    cnn_model.pop()
    cnn_model.pop()

    return cnn_model
