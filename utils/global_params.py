####################################################################
# This file stores params shared throughout the codebase.
# If we want to change any of these params, it serves as a
# single location where these variables can be modified when needed.
####################################################################

n_folds = 4
n_epochs = 27
encoding_model_options = ['linear', 'linear_sgd', 'mlp_separatehidden', 'mlp_sharedhidden', 'mlp_forloop', 'mlp_smallerhiddensize', 'mlp_largerhiddensize', 'mlp_additionalhiddenlayer']
n_splits = 10 # When training on each fold, train data is further split up into n_splits to compute model costs and pick an optimal lambda during ridge regression
lr_when_no_regularization = 1e-3
model_checkpoint_dir = 'model_checkpoints/'
allvoxels_minibatch_size = 32