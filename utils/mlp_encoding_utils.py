import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.optim import Optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def forward(self, features):
        outputs = self.model(features)
        return outputs


########################################
        # CUSTOM OPTIMIZER #
# Copied SGD optimizer definition from
# pytorch docs and adjusted to use
# optimal lambdas and lrs per voxel
########################################
"""
IMPORTANT:
This criterion should only be used with mlp_allvoxels architecture (where output layer is of size num_voxels).
Using it on any other model would probably result in unpredictable results as it was designed with this use in mind.
"""
class SGD_by_voxel(Optimizer):
    def __init__(self, params, lrs=None, momentum=0, dampening=0,
                 weight_decays=None, nesterov=False):
        if lrs is None or type(lrs).__module__  != np.__name__:
            raise ValueError("Need to enter a valid np array of learning rates: {}".format(lrs))
        if (lrs < 0.0).sum() > 0:
            raise ValueError("Invalid learning rate detected (< 0): {}".format(lrs))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decays is not None and type(weight_decays).__module__  != np.__name__:
            raise ValueError("Weight decays must be provided as a np array: {}".format(weight_decays))
        if weight_decays is not None and (weight_decays < 0.0).sum() > 0:
            raise ValueError("Invalid weight_decay value detected: {}".format(weight_decays))

        lr_mode = stats.mode(lrs)[0][0]
        lrs = torch.from_numpy(lrs).to(device)
        weight_decay_mode = None
        if weight_decays is not None:
            weight_decay_mode = stats.mode(weight_decays)[0][0]
            weight_decays = torch.from_numpy(weight_decays).to(device)

        defaults = dict(lrs=lrs, momentum=momentum, dampening=dampening,
                        weight_decays=weight_decays, nesterov=nesterov, 
                        lr_mode=lr_mode, weight_decay_mode=weight_decay_mode)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_by_voxel, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_by_voxel, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lrs = group['lrs']
            weight_decays = group['weight_decays']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decays is not None:
                    if p.shape[0] == 640: # Input -> Hidden Weights
                        d_p = d_p.add(p, alpha=group['weight_decay_mode'])
                    else: # Hidden -> Output Weights
                        d_p = d_p + (torch.mul(p.T, weight_decays)).T
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if p.shape[0] == 640: # Input -> Hidden Weights
                    p = p.add(d_p, alpha=-group['lr_mode'])
                else: # Hidden -> Output Weights
                    p = p + (torch.mul(d_p.T, -lrs)).T

        return loss