import gpytorch
from gp_layer import GaussianProcessLayer
import torch


class CombinedModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor.forward(x.numpy().reshape(-1, 28, 28, 1))
        features = torch.from_numpy(features)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0],
                                                       self.grid_bounds[1])
        res = self.gp_layer(features)
        return res

