from cnn_feature_extractor import CNNFeatureExtractor
import torch
from combined_model import CombinedModel
import gpytorch


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
    stds = []
    for data, target in test_loader:
        # data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            batch_stds = compute_variance(model(data), likelihood.mixing_weights, pred)
            stds += batch_stds.tolist()
            target = target.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))
    return stds


def compute_variance(latent_func, mixing_weights, pred):
    n_classes = 10
    n_samples = 20
    samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
    samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
    num_features, n_data, _ = samples.size()

    mixed_fs = mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
    softmax = torch.nn.functional.softmax(mixed_fs.t(), 1).view(n_data, n_samples, n_classes)

    classes_stds = softmax.std(1)
    max_class_stds = torch.Tensor(n_data)
    for i, row in enumerate(classes_stds):
        max_class_stds[i] = row[pred[i]]

    return max_class_stds
