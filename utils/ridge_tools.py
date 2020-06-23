from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore
import os

from utils.global_params import n_epochs, n_splits, allvoxels_minibatch_size, lr_when_no_regularization, model_checkpoint_dir
from utils.mlp_encoding_utils import MLPEncodingModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

np.random.seed(0)
torch.manual_seed(0)

def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def R2r(Pred,Real):
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ridge_sk(X,Y,lmbda):
    rd = Ridge(alpha = lmbda)
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas)
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def kernel_ridge(X,Y,lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def cross_val_ridge(train_features,train_data,
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False,
                    no_regularization=False):

    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method]

    if no_regularization:
        lmbda = 0
        min_lambdas = np.array([])
        weights = ridge_2(train_features, train_data, lmbda)
    else:
        n_voxels = train_data.shape[1]
        nL = lambdas.shape[0]
        r_cv = np.zeros((nL, train_data.shape[1]))

        kf = KFold(n_splits=n_splits)
        start_t = time.time()
        for icv, (trn, val) in enumerate(kf.split(train_data)):
            #print('ntrain = {}'.format(train_features[trn].shape[0]))
            cost = ridge_1(train_features[trn],train_data[trn],
                                train_features[val],train_data[val],
                                lambdas=lambdas)
            if do_plot:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(cost,aspect = 'auto')
            r_cv += cost
            #if icv%3 ==0:
            #    print(icv)
            #print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
        if do_plot:
            plt.figure()
            plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

        argmin_lambda = np.argmin(r_cv,axis = 0)
        import pdb
        pdb.set_trace()
        weights = np.zeros((train_features.shape[1],train_data.shape[1]))
        for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
            idx_vox = argmin_lambda == idx_lambda
            weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
        if do_plot:
            plt.figure()
            plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);
        min_lambdas = np.array([lambdas[i] for i in argmin_lambda])

    return weights, min_lambdas

def zero_unused_gradients(grad):
    num_voxels = grad.shape[0]
    for i in range(num_voxels):
        scol = 16*i
        ecol = scol + 16
        grad[:i, scol:ecol] = 0
        grad[i+1:, scol:ecol] = 0
    return grad

def pred_ridge_by_lambda_grad_descent(model_dict, X, Y, Xtest, Ytest, opt_lambdas, opt_lrs, is_mlp_separatehidden=False):
    import pdb
    opt_lambdas_lrs = np.array(list(zip(opt_lambdas, opt_lrs)))
    unique_lambdas_lrs = np.unique(opt_lambdas_lrs, axis=0)
    num_lambdas = unique_lambdas_lrs.shape[0]
    num_voxels = 1 if (len(Y.shape) == 1) else Y.shape[1]

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    Xtest, Ytest = torch.from_numpy(Xtest).float(), torch.from_numpy(Ytest).float()

    minibatch_size = model_dict['minibatch_size']
    train_losses = np.zeros((num_lambdas, n_epochs))
    test_losses = np.zeros((n_epochs, num_voxels))
    final_preds = torch.zeros_like(Ytest).to(device)

    # normalize test data
    Xtest = torch.where(torch.isnan(Xtest), torch.zeros_like(Xtest), Xtest).to(device)
    Ytest = torch.where(torch.isnan(Ytest), torch.zeros_like(Ytest), Ytest).to(device)

    for idx, (lmbda, lr) in enumerate(unique_lambdas_lrs):
        model = MLPEncodingModel(model_dict['input_size'], model_dict['hidden_sizes'], model_dict['output_size'], is_mlp_separatehidden)
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean')
        criterion_test = nn.MSELoss(reduction='none') # store test squared errors for every voxel
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2*lmbda)
        if is_mlp_separatehidden:
            # Register backward hook function for second layer's weights tensor
            model.model[2].weight.register_hook(zero_unused_gradients)

        # # Model checkpoint path
        # if not os.path.exists(model_checkpoint_dir):
        #     os.makedirs(model_checkpoint_dir)
        # checkpoint_path = model_checkpoint_dir + 'checkpoint.pt'
    
        current_voxels = opt_lambdas == lmbda
        for epoch in range(n_epochs):
            model.train()
            permutation = torch.randperm(X.shape[0])
            epoch_loss = 0
            for i in range(0, X.shape[0], minibatch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+minibatch_size]
                batch_X, batch_Y = X[indices], Y[indices]

                # normalize batch data
                batch_X = torch.where(torch.isnan(batch_X), torch.zeros_like(batch_X), batch_X).to(device)
                batch_Y = torch.where(torch.isnan(batch_Y), torch.zeros_like(batch_Y), batch_Y).to(device)

                batch_preds = model(batch_X)
                batch_loss = criterion(batch_preds.squeeze(), batch_Y)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach()

                del batch_X
                del batch_Y
                del batch_preds
                del batch_loss
        
            model.eval()
            preds_test = model(Xtest)
            test_loss = criterion_test(preds_test.squeeze(), Ytest).mean(dim=0)
            del preds_test

            # # Overwrite checkpoint
            # torch.save(model.state_dict(), checkpoint_path)
            train_losses[idx, epoch] = epoch_loss
            test_losses[epoch][current_voxels] = test_loss[current_voxels].detach().cpu()

        # Load checkpoint from previous epoch
        # model.load_state_dict(torch.load(checkpoint_path))
    
        # Generate predictions
        model.eval()
        preds_test = model(Xtest)
        final_preds[:, current_voxels] = preds_test[:, current_voxels]

    pdb.set_trace()
    del Xtest
    del Ytest
    return final_preds, train_losses, test_losses

def ridge_by_lambda_grad_descent(model_dict, X, Y, Xval, Yval, lambdas, lrs, split, is_mlp_separatehidden=False):
    num_lambdas = lambdas.shape[0]
    num_voxels = 1 if (len(Y.shape) == 1) else Y.shape[1]

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    Xval, Yval = torch.from_numpy(Xval).float(), torch.from_numpy(Yval).float()
    
    # normalize validation data
    Xval = torch.where(torch.isnan(Xval), torch.zeros_like(Xval), Xval).to(device)
    Yval = torch.where(torch.isnan(Yval), torch.zeros_like(Yval), Yval).to(device)

    epoch_losses, val_losses = np.zeros((num_lambdas, n_epochs, num_voxels)), np.zeros((num_lambdas, n_epochs, num_voxels))
    minibatch_size = model_dict['minibatch_size']
    
    for idx,lmbda in enumerate(lambdas):
        model = MLPEncodingModel(model_dict['input_size'], model_dict['hidden_sizes'], model_dict['output_size'], is_mlp_separatehidden)
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean')
        criterion_val = nn.MSELoss(reduction='none') # store val squared errors for every voxel
        optimizer = optim.Adam(model.parameters(), lr=lrs[idx], weight_decay=2*lmbda) # adds ridge penalty to above SSE criterion
        if is_mlp_separatehidden:
            # Register backward hook function for second layer's weights tensor
            model.model[2].weight.register_hook(zero_unused_gradients)
        
        num_batches = X.shape[0]//minibatch_size

        for epoch in range(n_epochs):
            model.train()
            permutation = torch.randperm(X.shape[0])
            epoch_loss = 0
            for i in range(0, X.shape[0], minibatch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+minibatch_size]
                batch_X, batch_Y = X[indices], Y[indices]

                # normalize batch data
                batch_X = torch.where(torch.isnan(batch_X), torch.zeros_like(batch_X), batch_X).to(device)
                batch_Y = torch.where(torch.isnan(batch_Y), torch.zeros_like(batch_Y), batch_Y).to(device)

                batch_preds = model(batch_X)
                batch_loss = criterion(batch_preds.squeeze(), batch_Y)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach()

                del batch_X
                del batch_Y
                del batch_preds
                del batch_loss
            
            # Validation loss for current epoch
            model.eval()
            preds_val = model(Xval)
            val_loss = criterion_val(preds_val.squeeze(), Yval).mean(dim=0)
            del preds_val

            epoch_loss /= num_batches
            epoch_losses[idx, epoch] = epoch_loss
            val_losses[idx, epoch] = val_loss.detach().cpu()

        del model

    cost = val_losses[:,-1,:] # (num_lambdas, num_voxels)

    import os
    epoch_losses_path, val_losses_path = 'mlp_losses/train_split{}.npy'.format(split), 'mlp_losses/val_split{}.npy'.format(split)
    os.makedirs('mlp_losses/', exist_ok=True)
    np.save(epoch_losses_path, epoch_losses)
    np.save(val_losses_path, val_losses)
    
    del Xval
    del Yval
    return cost

def cross_val_ridge_mlp_train_and_predict(model_dict, train_X, train_Y, test_X, test_Y, lambdas, lrs, is_mlp_separatehidden=False, no_regularization=True):
    import utils.utils as general_utils
    num_lambdas = lambdas.shape[0]
    num_voxels = 1 if (len(train_Y.shape) == 1) else train_Y.shape[1]
    r_cv = np.zeros((num_lambdas, num_voxels))


    if no_regularization:
        preds, train_losses, test_losses = pred_ridge_by_lambda_grad_descent(model_dict, train_X, train_Y, test_X, test_Y, np.array([0.] * num_voxels), np.array([lr_when_no_regularization] * num_voxels), is_mlp_separatehidden=is_mlp_separatehidden)
    else:
        kf = KFold(n_splits=n_splits)
        # Gather recorded costs from training with each lambda
        for ind_num, (trn, val) in enumerate(kf.split(train_Y)):
            if is_mlp_separatehidden:
                start_t = time.time()
                print("======= Split {} =======".format(ind_num))

            cost = ridge_by_lambda_grad_descent(model_dict, train_X[trn], train_Y[trn], train_X[val], train_Y[val], lambdas, lrs, ind_num, is_mlp_separatehidden=is_mlp_separatehidden) # cost: (num_lambdas, )
            r_cv += cost
            if is_mlp_separatehidden:
                end_t = time.time()
                print("Time Elapsed: {}s".format(end_t - start_t))
                print("========================")
        # Identify optimal lambda and use it to generate predictions
        argmin_lambda = np.argmin(r_cv, axis=0)
        opt_lambdas, opt_lrs = lambdas[argmin_lambda], lrs[argmin_lambda] # opt_lambdas, opt_lrs (num_voxels, )
        preds, train_losses, test_losses = pred_ridge_by_lambda_grad_descent(model_dict, train_X, train_Y, test_X, test_Y, opt_lambdas, opt_lrs, is_mlp_separatehidden=is_mlp_separatehidden)
    return preds, train_losses, test_losses

def cross_val_ridge_mlp(encoding_model, train_features, train_data, test_features, test_data,
                        lambdas = np.array([10**i for i in range(-6,10)]), lrs = np.array([1e-4]*9+[1e-3]*8),
                        no_regularization = True):
    num_voxels = train_data.shape[1]
    feat_dim = train_features.shape[1]
    n_train, n_test = train_data.shape[0], test_data.shape[0]

    # Initialize appropriate encoding model
    if encoding_model == 'linear_sgd':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [], num_voxels, allvoxels_minibatch_size
    if encoding_model == 'mlp_forloop':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [16], 1, n_train//n_splits
    elif encoding_model == 'mlp_smallerhiddensize':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [8], 1, n_train//n_splits
    elif encoding_model == 'mlp_largerhiddensize':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [24], 1, n_train//n_splits
    elif encoding_model == 'mlp_additionalhiddenlayer':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [16,4], 1, n_train//n_splits
    elif encoding_model == 'mlp_separatehidden':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [16*num_voxels], num_voxels, allvoxels_minibatch_size
    elif encoding_model == 'mlp_sharedhidden':
        input_size, hidden_sizes, output_size, minibatch_size = feat_dim, [640], num_voxels, allvoxels_minibatch_size
    model_dict = dict(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, minibatch_size=minibatch_size)

    if encoding_model not in ['linear_sgd', 'mlp_separatehidden', 'mlp_sharedhidden']:
        # Train and predict for one voxel at a time
        preds = torch.zeros((num_voxels, n_test))
        train_losses = np.zeros((num_voxels, n_epochs))
        test_losses = np.zeros((num_voxels, n_epochs))

        start_t = time.time()
        for ivox in range(num_voxels):
            vox_preds, vox_train_losses, vox_test_losses = cross_val_ridge_mlp_train_and_predict(model_dict, train_features, train_data[:, ivox],
                                                                                        test_features, test_data[:, ivox], lambdas, lrs, no_regularization=no_regularization)
            
            # Store predictions and model losses
            preds[ivox] = vox_preds.squeeze()
            train_losses[ivox] = vox_train_losses
            test_losses[ivox] = vox_test_losses

            if (ivox % 1000 == 0):
                end_t = time.time()
                print("{} vox: {}s".format(ivox, end_t - start_t))
                start_t = end_t
        preds = preds.T
        # preds: (N_test, num_voxels)
        # train_losses, test_losses: (num_voxels, n_epochs)
    else:
        is_mlp_separatehidden = (encoding_model == 'mlp_separatehidden')
        # Train and predict for all voxels at once
        preds, train_losses, test_losses = cross_val_ridge_mlp_train_and_predict(model_dict, train_features, train_data, test_features,
                                                                                test_data, lambdas, lrs, is_mlp_separatehidden=is_mlp_separatehidden, no_regularization=no_regularization)
        # preds: (N_test, num_voxels)
        # train_losses, test_losses: (n_epochs, )
    return preds, train_losses, test_losses