import torchvision.transforms as transforms
from modified_mnist_dataset import ModifiedMnistDataset
import torch


def get_data_loader(image_file_path, batch_size=256, shuffle=False, num_samples=10000):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    common_trans = [transforms.ToTensor(), normalize]
    test_compose = transforms.Compose(common_trans)

    test_set = ModifiedMnistDataset(image_file_path, transform=test_compose)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
                                              sampler=torch.utils.data.SubsetRandomSampler(list(
                                              range(num_samples))))

    return test_loader
