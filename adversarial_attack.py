from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from conv_net import ConvNet
from models_util import load_combined_model, compute_variance

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

epsilons = [2]#[0, 0.5, 1, 1.5, 2, 2.5, 3] #[0, .05, .1, .15, .2, .25, .3]

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
common_trans = [transforms.ToTensor(), normalize]
test_compose = transforms.Compose(common_trans)

test_set = datasets.MNIST('data', train=False, transform=test_compose)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(list(
                                              range(1000))))

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             ])),
#         batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(list(
#                                               range(100))))

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = "models/cnn_mnist.ckpt"
model = ConvNet().to(device)
# pretrained_model = "models/lenet_mnist_model.pth"
# model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

gp_model, likelihood = load_combined_model('models/gp_mnist.dat')
gp_model.eval()
likelihood.eval()


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def adv_test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    gp_correct = 0
    adv_examples = []
    cnn_correct_vars = []
    cnn_wrong_vars = []
    gp_correct_vars = []
    gp_wrong_vars = []
    perturbed_points = []
    y = []
    i = 0
    cnn_pred = []
    cnn_vars = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            cnn_pred.append(0)
            cnn_vars.append(1 - output.max().item())
            perturbed_points.append(data)
            y.append(target)
            continue

        i += 1

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        perturbed_points.append(perturbed_data)
        y.append(target)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            cnn_pred.append(1)
            cnn_vars.append(1 - output.max().item())
            cnn_correct_vars.append(1 - output.max().item())
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            cnn_wrong_vars.append(1 - output.max().item())
            cnn_pred.append(0)
            cnn_vars.append(1 - output.max().item())
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    X = torch.cat(perturbed_points, dim=0)
    y = torch.cat(y, dim=0)
    gp_output = likelihood(gp_model(X))
    pred = gp_output.probs.argmax(1)
    gp_correct += pred.eq(y).cpu().sum()
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    gp_final_acc = gp_correct/float(len(test_loader))

    gp_correct_vars, gp_wrong_vars, batch_vars = compute_variance(gp_model(X),
                                                                        likelihood.mixing_weights,
                                                                        pred.numpy(),
                                                                                y.numpy())
    gp_correct_vars = [x.item() for x in gp_correct_vars]
    gp_wrong_vars = [x.item() for x in gp_wrong_vars]
    gp_vars = [x.item() for x in batch_vars]
    gp_pred = pred.eq(y).numpy()



    print("Epsilon: {}\tCNN Test Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader),
                                                              final_acc))
    print("Epsilon: {}\tGP Test Accuracy = {} / {} = {}".format(epsilon, gp_correct,
                                                                len(test_loader), gp_final_acc))
    print("CNN Average correct var: {}".format(np.array(cnn_correct_vars).mean()))
    print("CNN Average wrong var:   {}".format(np.array(cnn_wrong_vars).mean()))
    print("GP Average correct var:  {}".format((np.array(gp_correct_vars) ** 2).mean()))
    print("GP Average wrong var:    {}".format((np.array(gp_wrong_vars) ** 2).mean()))

    print()
    print()

    # Return the accuracy and an adversarial example
    return gp_pred, np.array(gp_vars) ** 2, np.array(cnn_pred), np.array(cnn_vars)

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    gp_pred, gp_vars, cnn_pred, cnn_vars = adv_test(model, device, test_loader, eps)

    for penalty in [0, 2, 4, 6, 8, 10]:
        gp_score = 0
        cnn_score = 0

        for i in range(len(gp_pred)):
            if gp_vars[i] >= 0.063:
                gp_score += 1 if gp_pred[i] == 1 else -penalty
            if cnn_vars[i] >= 0.12:
                cnn_score += 1 if cnn_pred[i] == 1 else -penalty

        print("Penalty: {}".format(penalty))
        print("CNN score: {}".format(cnn_score))
        print("GP score: {}".format(gp_score))

    print()
    print()
    print()
