from cnn_feature_extractor import CNNFeatureExtractor
import torch
from combined_model import CombinedModel
import gpytorch
import numpy as np
from data_util import get_data_loader, read_data


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
            target = target.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('GP test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))
    correct_variance = np.array(correct_stds) ** 2
    wrong_variance = np.array(wrong_stds) ** 2
    return correct_variance, wrong_variance


def test_model_with_rejection(model, likelihood, test_loader, variance_threshold=0.04, penalty=1):
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
            target = target.argmax(1)
            for i, _ in enumerate(target):
                if batch_stds[i] < variance_threshold:
                    correct += 1 if target[i] == pred[i] else -penalty
            # correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('GP test set: Score: {} on {} samples'.format(
        correct, len(test_loader.dataset)
    ))
    correct_variance = np.array(correct_stds) ** 2
    wrong_variance = np.array(wrong_stds) ** 2
    return correct, correct_variance, wrong_variance


def test_cnn_model(cnn_model, file_path):
    x_test, y_test = read_data(file_path)
    score = cnn_model.evaluate(x_test, y_test, verbose=0)
    print('CNN Test accuracy:', score[1])


def test_cnn_model_with_rejection(cnn_model, file_path, penalty=1):
    x_test, y_test = read_data(file_path)
    y_pred = cnn_model.predict(x_test)
    correct = 0
    for i, _ in enumerate(y_test):
        correct += 1 if np.argmax(y_test[i]) == np.argmax(y_pred[i]) else -penalty
    print('CNN Test Score:', correct)
    return correct


def compare_rejection_models(file_path, cnn_model, combined_model, likelihood):
    cnn_scores = []
    gp_scores = []

    for penalty in range(1, 11, 2):
        print("Penalty: {}".format(penalty))
        test_loader = get_data_loader(file_path)
        gp_score, _, _ = test_model_with_rejection(combined_model, likelihood, test_loader,
                                                   variance_threshold=0.06, penalty=penalty)

        cnn_score = test_cnn_model_with_rejection(cnn_model, file_path, penalty=penalty)
        cnn_scores.append(cnn_score)
        gp_scores.append(gp_score)
        print()
        print()
    return cnn_scores, gp_scores


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
        if pred[i] == target[i].argmax():
            correct_max_class_stds.append(row[pred[i]])
        else:
            wrong_max_class_stds.append(row[pred[i]])
        max_class_stds.append(row[pred[i]])

    return correct_max_class_stds, wrong_max_class_stds, max_class_stds
