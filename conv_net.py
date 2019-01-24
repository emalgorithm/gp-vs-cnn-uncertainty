import torch.nn as nn


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25))
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 16 * 64, 128)
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=0))

    def forward(self, x):
        embedding = self.embed(x)
        out = self.fc(embedding)
        return out

    def embed(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        embedding = self.layer3(out)
        return embedding

    # def __init__(self, num_classes=10):
    #     super(ConvNet, self).__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.layer3 = nn.Linear(7 * 7 * 32, 128)
    #     self.fc = nn.Linear(128, num_classes)
    #
    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.reshape(out.size(0), -1)
    #     embedding = self.layer3(out)
    #     out = self.fc(embedding)
    #     return out
    #
    # def embed(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.reshape(out.size(0), -1)
    #     embedding = self.layer3(out)
    #     return embedding
