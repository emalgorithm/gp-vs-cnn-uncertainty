from cnn_feature_extractor import CNNFeatureExtractor
from combined_model import CombinedModel
import gpytorch
import torchvision.datasets as dset
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from models_util import get_cnn_embedding_model


cnn_model = get_cnn_embedding_model()

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
common_trans = [transforms.ToTensor(), normalize]
train_compose = transforms.Compose(common_trans)
test_compose = transforms.Compose(common_trans)

d_func = dset.MNIST
train_set = dset.MNIST('data', train=True, transform=train_compose, download=True)
test_set = dset.MNIST('data', train=False, transform=test_compose)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

feature_extractor = CNNFeatureExtractor()
num_features = 128
num_classes = 10

model = CombinedModel(feature_extractor, num_dim=num_features)#.cuda()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim,
                                                    n_classes=num_classes)#.cuda()

n_epochs = 300
lr = 0.1
optimizer = SGD([
    # {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)


def train(epoch):
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer,
                                        num_data=len(train_loader.dataset))

    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (
            epoch, batch_idx + 1, len(train_loader), loss.item()))


def test():
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in test_loader:
        # data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))


for epoch in range(1, n_epochs + 1):
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
        test()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict},
               'models/gp_mnist.dat')
