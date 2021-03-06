####################################################################
# This file stores params shared throughout the codebase.
# If we want to change any of these params, it serves as a
# single location where these variables can be modified when needed.
####################################################################

n_folds = 4
n_epochs = 10
encoding_model_options = ['linear', 'mlp_initial', 'mlp_smallerhiddensize', 'mlp_largerhiddensize', 'mlp_additionalhiddenlayer', 'mlp_allvoxels']
n_splits = 10 # When training on each fold, train data is further split up into n_splits to compute model costs and pick an optimal lambda during ridge regression

# mlp_allvoxels specific params
mlp_allvoxels_minibatch_size = 32