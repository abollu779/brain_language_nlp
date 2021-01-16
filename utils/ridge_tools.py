from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time
from scipy.stats import zscore
from collections import Counter
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import math

from utils.global_params import lambdas, lrs, n_splits, n_epochs
from utils.nonlinear_utils import NonlinearEncodingModel

torch.manual_seed(0)
writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

################################################
#              Linear Encoding Model           #
################################################
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

def cross_val_ridge_linear(args_dict, train_features, train_data, test_features, test_data, method='plain'):
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

    if args_dict['use_ridge']:
        feat_dim, n_voxels, n_lambdas = train_features.shape[1], train_data.shape[1], lambdas.shape[0]
        costs_across_splits = np.zeros((n_lambdas, n_voxels))
        kf = KFold(n_splits=n_splits)

        for (trn, val) in kf.split(train_data):
            cost = ridge_1(train_features[trn],train_data[trn],
                                train_features[val],train_data[val],
                                lambdas=lambdas)
            costs_across_splits += cost
        
        argmin_lambda = np.argmin(costs_across_splits,axis = 0)
        for idx, lmbda in enumerate(lambdas):
            writer.add_histogram('Lambda={}/Total Val Cost'.format(lmbda), costs_across_splits[idx], 0)
        if args_dict['roi_only']:
            roi_lambdas = Counter(argmin_lambda)
        else:
            rois = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)
            roi_voxels = np.where(rois.item()[args_dict['subject']]['all'] == 1)[0]
            roi_lambdas = argmin_lambda[roi_voxels]
        print("ROI Lambda Counts:\n", roi_lambdas)

        weights = np.zeros((feat_dim, n_voxels))
        errors = np.zeros((train_data.shape[1],))
        for idx_lambda in range(n_lambdas): # this is much faster than iterating over voxels!
            idx_vox = argmin_lambda == idx_lambda
            weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
            errors[idx_vox] = (1 - R2(np.dot(test_features,weights[:,idx_vox]),test_data[:,idx_vox]))
        min_lambdas = np.array([lambdas[i] for i in argmin_lambda])
        writer.add_histogram('Prediction/Test Cost', errors, 0)
    else:
        lmbda, min_lambdas = 0, np.array([])
        weights = ridge_2(train_features, train_data, lmbda)
        
    # End tensorboard logging
    writer.close()
    return weights, min_lambdas

################################################
#            Nonlinear Encoding Model          #
################################################
def R2_torch(Pred,Real):
    SSres = torch.mean((Real-Pred)**2,0)
    SStot = torch.var(Real,0)
    output = 1-SSres/SStot
    output[torch.isnan(output)] = 0. # torch doesn't provide a nan_to_num equivalent
    return output

def normalize_torch_tensor(t):
    t = torch.where(torch.isnan(t), torch.zeros_like(t), t)
    norm_t = (t - t.mean(0))/t.std(0)
    norm_t = torch.where(torch.isnan(norm_t), torch.zeros_like(norm_t), norm_t)
    return norm_t

def predict_ridge_by_lambda_graddescent(args_dict, X, Y, Xtest, Ytest, opt_lambda, opt_lr):
    # TODO: Currently implemented only for nonlinear_sharedhidden; Need to incorporate separatehidden either here or in separate function
    global writer
    train_size, feat_dim, n_voxels = X.shape[0], X.shape[1], Y.shape[1]
    encoding_model, batch_size = args_dict['encoding_model'], args_dict['batch_size']
    if batch_size is None:
            batch_size = train_size
    n_batches = math.ceil(train_size/batch_size)

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    Xtest, Ytest = torch.from_numpy(Xtest).float(), torch.from_numpy(Ytest).float()
    Xtest, Ytest = normalize_torch_tensor(Xtest), normalize_torch_tensor(Ytest) # normalize test data here, and train data per batch inside loop
    Xtest, Ytest = Xtest.to(device), Ytest.to(device)

    model = NonlinearEncodingModel(encoding_model, feat_dim, n_voxels)
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=opt_lr)

    for epoch in range(n_epochs):
        model.train()
        permutation = torch.randperm(train_size)
        train_cost = 0.
        for b in range(0, train_size, batch_size):
            optimizer.zero_grad()

            batch_indices = permutation[b:b+batch_size]
            Xbatch, Ybatch = X[batch_indices], Y[batch_indices]
            Xbatch, Ybatch = normalize_torch_tensor(Xbatch), normalize_torch_tensor(Ybatch) # normalize train data here
            Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)

            batch_preds = model(Xbatch)
            batch_loss = criterion(batch_preds.squeeze(), Ybatch)
            weights_squared_sum = 0.
            for layer in model.model:
                if isinstance(layer, nn.Linear):
                    weights_squared_sum += ((layer.weight)**2).sum()
            batch_loss += (float(batch_size)/train_size) * opt_lambda * weights_squared_sum
            batch_loss.backward()
            optimizer.step()
            train_cost += (1. - R2_torch(batch_preds.squeeze(), Ybatch)).detach().cpu()
            del Xbatch, Ybatch, batch_preds, batch_loss            
        train_cost /= n_batches # train_cost: (n_voxels, )

        # Monitor test cost
        model.eval()
        test_preds = model(Xtest)
        test_cost = (1. - R2_torch(test_preds.squeeze(), Ytest)).detach().cpu().numpy() # test_cost: (n_voxels, )

        # Tensorboard logging
        writer.add_histogram('Prediction/Train Cost', train_cost, epoch)
        writer.add_histogram('Prediction/Test Cost', test_cost, epoch)
        del test_preds
    
    # Generate predictions
    model.eval()
    test_preds = model(Xtest)

    writer.flush()
    del model, Xtest, Ytest
    return test_preds
    
def ridge_by_lambda_graddescent(split_num, args_dict, X, Y, Xval, Yval):
    global writer
    train_size, feat_dim, n_voxels = X.shape[0], X.shape[1], Y.shape[1]
    encoding_model, batch_size = args_dict['encoding_model'], args_dict['batch_size']
    if batch_size is None:
            batch_size = train_size
    n_batches = math.ceil(train_size/batch_size)

    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    Xval, Yval = torch.from_numpy(Xval).float(), torch.from_numpy(Yval).float()
    Xval, Yval = normalize_torch_tensor(Xval), normalize_torch_tensor(Yval) # normalize val data here, and train data per batch inside loop
    Xval, Yval = Xval.to(device), Yval.to(device)

    costs = []
    for idx, lmbda in enumerate(lambdas):
        start = time.time()
        model = NonlinearEncodingModel(encoding_model, feat_dim, n_voxels)
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=lrs[idx])
        # TODO: nonlinear_separatehidden backward hook for zeroing out gradients goes here
        
        for epoch in range(n_epochs):
            model.train()
            permutation = torch.randperm(train_size)
            train_cost = 0.
            for b in range(0, train_size, batch_size):
                optimizer.zero_grad()

                batch_indices = permutation[b:b+batch_size]
                Xbatch, Ybatch = X[batch_indices], Y[batch_indices]
                Xbatch, Ybatch = normalize_torch_tensor(Xbatch), normalize_torch_tensor(Ybatch) # normalize train data here
                Xbatch, Ybatch = Xbatch.to(device), Ybatch.to(device)

                batch_preds = model(Xbatch)
                batch_loss = criterion(batch_preds.squeeze(), Ybatch)
                weights_squared_sum = 0.
                for layer in model.model:
                    if isinstance(layer, nn.Linear):
                        weights_squared_sum += ((layer.weight)**2).sum()
                batch_loss += (float(batch_size)/train_size) * lmbda * weights_squared_sum
                batch_loss.backward()
                optimizer.step()
                train_cost += (1. - R2_torch(batch_preds.squeeze(), Ybatch)).detach().cpu()
                del Xbatch, Ybatch, batch_preds, batch_loss            
            train_cost /= n_batches # train_cost: (n_voxels, )

            # Monitor validation cost
            model.eval()
            val_preds = model(Xval)
            val_cost = (1. - R2_torch(val_preds.squeeze(), Yval)).detach().cpu().numpy() # val_cost: (n_voxels, )

            # Tensorboard logging
            writer.add_histogram('Lambda={}/Train Cost/Split={}'.format(lmbda, split_num), train_cost, epoch)
            writer.add_histogram('Lambda={}/Val Cost/Split={}'.format(lmbda, split_num), val_cost, epoch)
            del val_preds
        del model
        writer.flush()
        costs.append(val_cost)

    costs = np.array(costs) # costs: (n_lambdas, n_voxels)
    del Xval, Yval
    return costs

def cross_val_ridge_nonlinear(args_dict, train_features, train_data, test_features, test_data):
    import utils.utils as general_utils # Import here to avoid circular import error
    global writer
    encoding_model, use_ridge = args_dict['encoding_model'], args_dict['use_ridge']
    n_words, n_voxels, n_lambdas = train_features.shape[0], train_data.shape[1], lambdas.shape[0]
    costs_across_splits = np.zeros((n_lambdas, n_voxels))
    ind = general_utils.CV_ind(n_words, n_folds=n_splits)

    if use_ridge:
        # Gathering costs across splits
        for split_num in range(n_splits):
            start_time = time.time()
            train_ind, val_ind = (ind!=split_num), (ind==split_num)

            costs = ridge_by_lambda_graddescent(split_num, args_dict, train_features[train_ind], train_data[train_ind], train_features[val_ind], train_data[val_ind])
            costs_across_splits += costs
            print("Split: {} | Time: {}s".format(split_num, time.time() - start_time))
        for idx, lmbda in enumerate(lambdas):
            writer.add_histogram('Lambda={}/Total Val Cost'.format(lmbda), costs_across_splits[idx], 0)
        writer.flush()

        # Choosing a lambda based on total val cost data for ROI voxels
        argmin_lambda = np.argmin(costs_across_splits,axis = 0)
        if args_dict['roi_only']:
            roi_lambdas = Counter(argmin_lambda)            
        else:
            rois = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)
            roi_voxels = np.where(rois.item()[args_dict['subject']]['all'] == 1)[0]
            roi_lambdas = argmin_lambda[roi_voxels]
        print("ROI Lambda Counts:\n", roi_lambdas)
        most_common_idx = Counter(roi_lambdas).most_common(1)[0][0]
        if encoding_model=='nonlinear_sharedhidden' or encoding_model=='nonlinear_sharedhidden_roipartition':
            opt_lambda, opt_lr = lambdas[most_common_idx], lrs[most_common_idx]
        else:
            raise Exception('{} encoding model not recognized'.format(encoding_model))
    else:
        opt_lambda, opt_lr = 0., 1e-4

    # Generating predictions with the chosen lambda
    preds = predict_ridge_by_lambda_graddescent(args_dict, train_features, train_data, test_features, test_data, opt_lambda, opt_lr)

    # End tensorboard logging
    writer.close()
    return preds
