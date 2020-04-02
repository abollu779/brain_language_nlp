import torch
import torch.nn as nn

class MLPEncodingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPEncodingModel, self).__init__()
        # self.fc1 = nn.Linear(input_size, output_size) 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, features):
        # features: N x 40
        hiddens = self.fc1(features)
        acts = self.act1(hiddens)
        output = self.fc2(acts) # N x 27905
        return output
        # return hiddens