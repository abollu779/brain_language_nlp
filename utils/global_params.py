import numpy as np
####################################################################
# This file stores params shared throughout the codebase.
# If we want to change any of these params, it serves as a
# single location where these variables can be modified when needed.
####################################################################

n_folds = 4
encoding_model_options = ['linear', 'linear_sgd', 'mlp_separatehidden', 'mlp_sharedhidden', 'mlp_forloop', 'mlp_smallerhiddensize', 'mlp_largerhiddensize', 'mlp_additionalhiddenlayer']
n_splits = 10 # When training on each fold, train data is further split up into n_splits to compute model costs and pick an optimal lambda during ridge regression
lr_when_no_regularization = 1e-3
model_checkpoint_dir = 'model_checkpoints/'
allvoxels_minibatch_size = 32

###################################################################
# Learning Rates for Different Models (Experimentally Determined) #
###################################################################
lambdas = np.array([10**i for i in range(-6,10)])

# Without Regularization
sgd_noreg_lrs = {'linear_sgd': 1e-3,
                'mlp_sharedhidden': 1e-2}
sgd_noreg_n_epochs = {'linear_sgd': 27,
                    'mlp_sharedhidden': 169}

# With Regularization
sgd_reg_lrs = {'linear_sgd': np.array([1e-3]*16),
                'mlp_sharedhidden': np.array([1e-2]*2 + [1e-3]*14)}
sgd_reg_n_epochs = {'linear_sgd': np.array([27]*16),
                    'mlp_sharedhidden': np.array([40]*16)}
