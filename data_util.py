import torchvision.transforms as transforms
from modified_mnist_dataset import ModifiedMnistDataset
import torch
import scipy.io
from keras import backend as K


def get_data_loader(image_file_path, batch_size=256, shuffle=False):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    common_trans = [transforms.ToTensor(), normalize]
    test_compose = transforms.Compose(common_trans)

    test_set = ModifiedMnistDataset(image_file_path, transform=test_compose)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return test_loader


def read_data(mat_file):
    mat = scipy.io.loadmat(mat_file)

    x_train = mat['train_x']
    y_train = mat['train_y']

    x_test = mat['test_x']
    y_test = mat['test_y']

    # input image dimensions
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_test, y_test
