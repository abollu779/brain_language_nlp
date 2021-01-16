import numpy as np

roi_keys = ['PostTemp', 'AntTemp', 'AngularG', 'IFG', 'MFG', 'IFGorb', 'pCingulate', 'dmpfc']

# Training Routine
n_folds = 4
n_splits = 10 # nested CV (inner folds to select optimal lambda)
n_epochs = 25
lambdas = np.array([10**i for i in range(-6,6)])
lrs = np.array([1e-4]*11+[1e-5, 1e-6])

# Model Architecture
encoding_model_options = ['linear', 'nonlinear_sharedhidden', 'nonlinear_sharedhidden_roipartition']

# linear: standard linear model
# nonlinear_sharedhidden: one-hidden layer mlp model for all voxels together; 
#                         if use_ridge is set, lambda selected for most voxels is the lambda chosen for predictions
# nonlinear_sharedhidden_roipartition: one-hidden layer mlp models trained on voxels in each roi at a time.
