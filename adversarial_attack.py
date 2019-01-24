from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from conv_net import ConvNet
from models_util import load_combined_model, compute_variance

epsilons = [.3]
pretrained_model = "models/cnn_mnist.ckpt"

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
common_trans = [transforms.ToTensor(), normalize]
test_compose = transforms.Compose(common_trans)

test_set = datasets.MNIST('data', train=False, transform=test_compose)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet().to(device)
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
    correct_stds = []
    wrong_stds = []
    adv_examples = []

    i = 0
    # Loop over all examples in test set
    for data, target in test_loader:
        if i >= 1000:
            break

        i += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        # output = model(data)
        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        gp_output = likelihood(gp_model(data))
        init_pred = gp_output.probs.argmax(1)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(gp_output.probs, target)

        # Zero all existing gradients
        # model.zero_grad()
        gp_model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        gp_output = likelihood(gp_model(perturbed_data))
        gp_pred = gp_output.probs.argmax(1)

        one_hot_target = torch.zeros(1, 10)
        one_hot_target[0, target] = 1
        correct_batch_stds, wrong_batch_stds, batch_stds = compute_variance(gp_model(data),
                                                                            likelihood.mixing_weights,
                                                                            gp_pred, one_hot_target)
        correct_stds += correct_batch_stds
        wrong_stds += wrong_batch_stds

        if gp_pred.item() == target.item():
            gp_correct += 1

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(i)
    gp_final_acc = gp_correct / float(i)
    print("Epsilon: {}\tCNN Test Accuracy = {} / {} = {}".format(epsilon, correct, i, final_acc))
    print("Epsilon: {}\tGP Test Accuracy = {} / {} = {}".format(epsilon, gp_correct, i, gp_final_acc))
    print("Average correct variance is {}".format((np.array([std.item() ** 2 for std in
                                                             correct_stds]).mean())))
    print("Average wrong variance is {}".format((np.array([std.item() ** 2 for std in
                                                             wrong_stds]).mean())))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = adv_test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)