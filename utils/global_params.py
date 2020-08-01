import numpy as np
####################################################################
# This file stores params shared throughout the codebase.
# If we want to change any of these params, it serves as a
# single location where these variables can be modified when needed.
####################################################################

n_folds = 4
encoding_model_options = ['linear', 'linear_sgd', 'mlp_separatehidden', 'mlp_sharedhidden', 'mlp_forloop', 'mlp_smallerhiddensize', \
                            'mlp_largerhiddensize', 'mlp_additionalhiddenlayer', 'linear_sklearn', 'linear_sgd_sklearn', 'linear_gd', \
                                'mlp_sharedhidden_gd', 'mlp_forloop_gd', 'mlp_separatehidden_gd', 'mlp_sharedhidden_onepredmodel', \
                                'mlp_sharedhidden_onepredmodel_singlelambda']
n_splits = 10 # When training on each fold, train data is further split up into n_splits to compute model costs and pick an optimal lambda during ridge regression
new_lr_window = 10
cooldown_period = 15
min_lr = 1e-8
min_sum_grad_norm = 1e-15

##############################################################
# Hyperparameters for each model (Experimentally Determined) #
##############################################################

# LRs + EPOCHS
# Without Regularization
sgd_noreg_lrs = {'linear_sgd': 1e-3,
                'mlp_sharedhidden': 1e-2,
                'mlp_separatehidden': 1e-3,
                'mlp_sharedhidden_onepredmodel': 1e-3,
                'mlp_sharedhidden_onepredmodel_singlelambda': 1e-3}
sgd_noreg_n_epochs = {'linear_sgd': 27,
                    'mlp_sharedhidden': 169,
                    'mlp_separatehidden': 20,
                    'mlp_sharedhidden_onepredmodel': 40,
                    'mlp_sharedhidden_onepredmodel_singlelambda': 40}

# With Regularization
sgd_reg_lrs = {'linear_sgd': np.array([1e-3]*16),
                'mlp_sharedhidden': np.array([1e-2]*2+[1e-3]*14),
                'mlp_separatehidden': np.array([1e-2]*16),
                'mlp_forloop': np.array([1e-2]*16),
                'linear_gd': np.array([1e-1]*16),
                'mlp_sharedhidden_gd': np.array([1e-1]*16),
                'mlp_forloop_gd': np.array([1e-1]*16),
                'mlp_separatehidden_gd': np.array([1e-1]*16),
                'mlp_sharedhidden_onepredmodel': np.array([1e-3]*16),
                'mlp_sharedhidden_onepredmodel_singlelambda': np.array([1e-3]*16)}
sgd_reg_n_epochs = {'linear_sgd': np.array([38]*16),
                    'mlp_sharedhidden': np.array([40]*16),
                    'mlp_separatehidden': np.array([20]*16),
                    'mlp_forloop': np.array([20]*16),
                    'linear_gd': np.array([450,450,450,450,450,500,550,650,600,750,800,900,850,950,950,1050]),
                    'mlp_sharedhidden_gd': np.array([40]*16),
                    'mlp_forloop_gd': np.array([500]*16),
                    'mlp_separatehidden_gd': np.array([500]*16),
                    'mlp_sharedhidden_onepredmodel': np.array([40]*16),
                    'mlp_sharedhidden_onepredmodel_singlelambda': np.array([40]*16)}

# MODEL ARCHITECTURE PARAMETERS
hidden_sizes = {'linear_sgd': (False, [], None),
                'mlp_sharedhidden': (False, [640], None),
                'mlp_separatehidden': (True, None, 16),
                'mlp_forloop': (False, [16], None),
                'mlp_smallerhiddensize': (False, [8], None),
                'mlp_largerhiddensize': (False, [24], None),
                'mlp_additionalhiddenlayer': (False, [16,4], None),
                'linear_gd': (False, [], None),
                'mlp_sharedhidden_gd': (False, [640], None),
                'mlp_forloop_gd': (False, [16], None),
                'mlp_separatehidden_gd': (True, None, 16),
                'mlp_sharedhidden_onepredmodel': (False, [640], None),
                'mlp_sharedhidden_onepredmodel_singlelambda': (False, [640], None)}
output_sizes = {'linear_sgd': None,
                'mlp_sharedhidden': None,
                'mlp_separatehidden': None,
                'mlp_forloop': 1,
                'mlp_smallerhiddensize': 1,
                'mlp_largerhiddensize': 1,
                'mlp_additionalhiddenlayer': 1,
                'linear_gd': None,
                'mlp_sharedhidden_gd': None,
                'mlp_forloop_gd': 1,
                'mlp_separatehidden_gd': None,
                'mlp_sharedhidden_onepredmodel': None,
                'mlp_sharedhidden_onepredmodel_singlelambda': None}
minibatch_sizes = {'linear_sgd': 32,
                    'mlp_sharedhidden': 128,
                    'mlp_separatehidden': 32,
                    'mlp_forloop': 32,
                    'mlp_smallerhiddensize': 32,
                    'mlp_largerhiddensize': 32,
                    'mlp_additionalhiddenlayer': 32,
                    'linear_gd': None,
                    'mlp_sharedhidden_gd': None,
                    'mlp_forloop_gd': None,
                    'mlp_separatehidden_gd': None,
                    'mlp_sharedhidden_onepredmodel': 32,
                    'mlp_sharedhidden_onepredmodel_singlelambda': 32}
