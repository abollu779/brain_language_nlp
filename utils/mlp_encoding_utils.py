import torch
import torch.nn as nn

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
            import pdb
            pdb.set_trace()
            import time
            s_t1 = time.time()
            self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), 
                        nn.ReLU(), 
                        nn.Linear(hidden_sizes[0], output_size))
            e_t1 = time.time()
            print("Model Init Time: {}s".format(e_t1 - s_t1))
            if is_mlp_allvoxels:
                s_t2 = time.time()
                linear2_weights = self.model[2].weight.transpose(1,0)
                for i in range(linear2_weights.shape[1]):
                    srow = 16*i
                    erow = srow + 16
                    linear2_weights[srow:erow, :i] = 0
                    linear2_weights[srow:erow, i+1:] = 0
                e_t2 = time.time()
                print("Model Block Diagonal Weights Time: {}s".format(e_t2 - s_t2))
                import pdb
                pdb.set_trace()

    def forward(self, features):
        outputs = self.model(features)
        return outputs