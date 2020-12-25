import torch
import torch.nn as nn
import numpy as np

class NonlinearEncodingModel(nn.Module):
    def __init__(self, encoding_model, feat_dim, n_voxels):
        super(NonlinearEncodingModel, self).__init__()
        self.model = None
        if encoding_model == 'nonlinear_sharedhidden':
            self.model = nn.Sequential(
                nn.Linear(feat_dim, 640),
                nn.ReLU(),
                nn.Linear(640, n_voxels)
            )
        elif encoding_model == 'nonlinear_separatehidden':
            raise Exception('{} encoding model currently not supported'.format(encoding_model))
        else:
            raise Exception('{} encoding model not recognized'.format(encoding_model))

    def forward(self, features):
        outputs = self.model(features)
        return outputs
