import numpy as np
####################################################################
# This file stores params shared throughout the codebase.
# If we want to change any of these params, it serves as a
# single location where these variables can be modified when needed.
####################################################################

n_folds = 4
encoding_model_options = ['linear', 'linear_sgd', 'mlp_separatehidden', 'mlp_sharedhidden', 'mlp_forloop', 'mlp_smallerhiddensize', \
                            'mlp_largerhiddensize', 'mlp_additionalhiddenlayer', 'linear_sklearn', 'linear_sgd_sklearn', 'linear_gd', \
                                'mlp_sharedhidden_gd', 'mlp_forloop_gd', 'mlp_separatehidden_gd']
n_splits = 10 # When training on each fold, train data is further split up into n_splits to compute model costs and pick an optimal lambda during ridge regression
lr_when_no_regularization = 1e-3
model_checkpoint_dir = 'model_checkpoints/'
allvoxels_minibatch_size = 128
forloop_minibatch_size = 32
sharedhidden_minibatch_sizes = [32]*5 + [1]*11
new_lr_window = 10
cooldown_period = 15
min_lr = 1e-8
min_sum_grad_norm = 1e-15

###################################################################
# Learning Rates for Different Models (Experimentally Determined) #
###################################################################
lambdas = np.array([10**i for i in range(-6,10)])

# Without Regularization
sgd_noreg_lrs = {'linear_sgd': 1e-3,
                'mlp_sharedhidden': 1e-2,
                'mlp_separatehidden': 1e-3}
sgd_noreg_n_epochs = {'linear_sgd': 27,
                    'mlp_sharedhidden': 169,
                    'mlp_separatehidden': 20}

# With Regularization
sgd_reg_lrs = {'linear_sgd': np.array([1e-3]*16),
                'mlp_sharedhidden': np.array([1e-2]*4+[1e-3]+[1e-4]*11),
                # 'mlp_sharedhidden': np.array([1e-6]*3+[1e-5]+[1e-1]*4+[1e0]*8),
                'mlp_separatehidden': np.array([1e-2]*16),
                'mlp_forloop': np.array([1e-2]*16),
                'linear_gd': np.array([1e-1]*16),
                'mlp_sharedhidden_gd': np.array([1e-1]*16),
                'mlp_forloop_gd': np.array([1e-1]*16),
                'mlp_separatehidden_gd': np.array([1e-1]*16),}
sgd_reg_n_epochs = {'linear_sgd': np.array([38]*16),
                    'mlp_sharedhidden': np.array([60]*4+[70]+[80]+[8]+[5]*2+[6]*4+[7]*3),
                    # 'mlp_sharedhidden': np.array([25]*4+[350]*4+[450]+[400]*7),
                    'mlp_separatehidden': np.array([20]*16),
                    'mlp_forloop': np.array([20]*16),
                    'linear_gd': np.array([450,450,450,450,450,500,550,650,600,750,800,900,850,950,950,1050]),
                    'mlp_sharedhidden_gd': np.array([40]*16),
                    'mlp_forloop_gd': np.array([500]*16),
                    'mlp_separatehidden_gd': np.array([500]*16),}
