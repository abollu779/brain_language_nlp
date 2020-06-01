import torch
import torch.nn as nn
import numpy as np

class MLPEncodingModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, is_mlp_allvoxels = False):
        super(MLPEncodingModel, self).__init__()
        self.model = None
        if len(hidden_sizes) == 2:
            self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), 
                        nn.ReLU(), 
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size))
        else:
            assert len(hidden_sizes) == 1
            self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), 
                        nn.ReLU(), 
                        nn.Linear(hidden_sizes[0], output_size))
            if is_mlp_allvoxels:
                num_voxels = self.model[2].weight.shape[0]
                linear2_weights = self.model[2].weight.detach().numpy() 
                # Need to convert to numpy and adjust that array.
                # Cannot directly adjust the weights tensor as then pytorch converts it 
                # to a leaf tensor that cannot be detected by the optimizer.
                for i in range(num_voxels):
                    scol = 16*i
                    ecol = scol + 16
                    linear2_weights[:i, scol:ecol] = 0
                    linear2_weights[i+1:, scol:ecol] = 0

    def forward(self, features):
        outputs = self.model(features)
        return outputs