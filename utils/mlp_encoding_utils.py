import torch
import torch.nn as nn

class MLPEncodingModel(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLPEncodingModel, self).__init__()
        self.model = None
        if hidden_2_size is not None:
            self.model = nn.Sequential(nn.Linear(input_size, hidden_1_size), 
                        nn.ReLU(), 
                        nn.Linear(hidden_1_size, hidden_2_size),
                        nn.ReLU(),
                        nn.Linear(hidden_2_size, output_size))
        else:
            self.model = nn.Sequential(nn.Linear(input_size, hidden_1_size), 
                        nn.ReLU(), 
                        nn.Linear(hidden_1_size, output_size))

    def forward(self, features):
        outputs = self.model(features)
        return outputs