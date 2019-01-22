from torch.utils.data import Dataset
import scipy.io
from PIL import Image


class ModifiedMnistDataset(Dataset):
    def __init__(self, file_path, transform=None):
        mat = scipy.io.loadmat(file_path)

        self.x_test = mat['test_x']
        self.y_test = mat['test_y']
        self.transform = transform

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        sample = Image.fromarray(self.x_test[idx].reshape((28, 28)))

        if self.transform:
            sample = self.transform(sample)

        return sample, self.y_test[idx]
