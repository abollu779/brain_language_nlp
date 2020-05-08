import torch
import torch.nn as nn

class MLPEncodingModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
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

    def forward(self, features):
        outputs = self.model(features)
        return outputs