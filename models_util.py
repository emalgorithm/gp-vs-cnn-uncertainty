from cnn_feature_extractor import CNNFeatureExtractor
import torch
from combined_model import CombinedModel
import gpytorch
import numpy as np
from data_util import get_data_loader


def load_combined_model(file_path):
    feature_extractor = CNNFeatureExtractor()
    num_features = 128
    num_classes = 10

    state_dicts = torch.load(file_path)

    model = CombinedModel(feature_extractor, num_dim=num_features)
    model.load_state_dict(state_dicts['model'])
    model.eval()

    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim,
                                                        n_classes=num_classes)
    likelihood.load_state_dict(state_dicts['likelihood'])
    likelihood.eval()

    return model, likelihood


def test_model(model, likelihood, test_loader):
    correct = 0
    correct_stds = []
    wrong_stds = []
    for data, target in test_loader:
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct_batch_stds, wrong_batch_stds, _ = compute_variance(model(data),
                                                             likelihood.mixing_weights, pred, target)
            correct_stds += correct_batch_stds
            wrong_stds += wrong_batch_stds
            # if target.numel() != 1:
            #     target = target.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('GP test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))
    correct_variance = np.array(correct_stds) ** 2
    wrong_variance = np.array(wrong_stds) ** 2
    return correct_variance, wrong_variance


def test_model_with_rejection(model, likelihood, test_loader, penalty, threshold=0.04):
    correct = 0
    correct_stds = []
    wrong_stds = []
    for data, target in test_loader:
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct_batch_stds, wrong_batch_stds, batch_stds = compute_variance(model(data),
                                                             likelihood.mixing_weights, pred, target)
            correct_stds += correct_batch_stds
            wrong_stds += wrong_batch_stds
            # target = target.argmax(1)
            for i, _ in enumerate(target):
                if batch_stds[i] < threshold:
                    correct += 1 if target[i] == pred[i] else -penalty
            # correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('GP with rejection: score: {}'.format(correct))
    correct_variance = np.array(correct_stds) ** 2
    wrong_variance = np.array(wrong_stds) ** 2
    return correct, correct_variance, wrong_variance


def test_cnn_model(cnn_model, test_loader, penalty=0):
    correct = 0
    correct_stds = []
    wrong_stds = []
    for data, target in test_loader:
        with torch.no_grad():
            output = cnn_model(data)
            for i, _ in enumerate(output):
                y_pred = output[i].argmax()
                if y_pred.item() == target[i].item():
                    correct += 1
                    correct_stds.append(1 - output[i].max().item())
                else:
                    correct -= penalty
                    wrong_stds.append(1 - output[i].max().item())
    print('CNN test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))
    correct_variance = np.array(correct_stds)
    wrong_variance = np.array(wrong_stds)
    return correct_variance, wrong_variance


def test_cnn_model_with_rejection(cnn_model, test_loader, penalty, threshold=0.05):
    correct = 0
    correct_stds = []
    wrong_stds = []
    for data, target in test_loader:
        with torch.no_grad():
            output = cnn_model(data)
            for i, _ in enumerate(output):
                y_pred = output[i].argmax()
                var = 1 - output[i].max()
                if var <= 0.05:
                    if y_pred.item() == target[i].item():
                        correct += 1
                        correct_stds.append(1 - output[i].max().item())
                    else:
                        correct -= penalty
                        wrong_stds.append(1 - output[i].max().item())
    print('CNN with rejection: score: {}'.format(correct))
    correct_variance = np.array(correct_stds)
    wrong_variance = np.array(wrong_stds)
    return correct


def compare_rejection_models(test_loader, cnn_model, combined_model, likelihood, gp_threshold,
                             cnn_threshold):
    cnn_scores = []
    cnn_rej_scores = []
    gp_scores = []

    for penalty in range(0, 11, 2):
        print("Penalty: {}".format(penalty))
        gp_score, _, _ = test_model_with_rejection(combined_model, likelihood, test_loader,
                                                   threshold=gp_threshold, penalty=penalty)
        cnn_score = test_cnn_model(cnn_model, test_loader, penalty=penalty)
        cnn_rej_score = test_cnn_model_with_rejection(cnn_model, test_loader, penalty=penalty,
                                                      threshold=cnn_threshold)
        cnn_scores.append(cnn_score)
        gp_scores.append(gp_score)
        cnn_rej_scores.append(cnn_rej_score)
        print()
        print()
    return cnn_scores, cnn_rej_scores, gp_scores


def compute_variance(latent_func, mixing_weights, pred, target):
    n_classes = 10
    n_samples = 20
    samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
    samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
    num_features, n_data, _ = samples.size()

    mixed_fs = mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
    softmax = torch.nn.functional.softmax(mixed_fs.t(), 1).view(n_data, n_samples, n_classes)

    classes_stds = softmax.std(1)
    correct_max_class_stds = []
    wrong_max_class_stds = []
    max_class_stds = []
    for i, row in enumerate(classes_stds):
        if pred[i] == target[i]:
            correct_max_class_stds.append(row[pred[i]])
        else:
            wrong_max_class_stds.append(row[pred[i]])
        max_class_stds.append(row[pred[i]])

    return correct_max_class_stds, wrong_max_class_stds, max_class_stds
