import numpy as np
from sklearn.model_selection import KFold
import sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.global_params import n_splits

np.random.seed(0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def sklearn_ridge(X, Y, lmbda):
    model = Ridge(alpha=lmbda, solver='cholesky')
    model.fit(X, Y)
    return model

def sklearn_ridge_train(X, Y, Xval, Yval, lambdas):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        model = sklearn_ridge(X,Y,lmbda)
        predsval = model.predict(Xval)
        error[idx] = 1 - R2(predsval,Yval)
    return error

def sklearn_cross_val_ridge(train_features,train_data,
                    test_features,test_data,
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False,
                    no_regularization=False):
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        cost = sklearn_ridge_train(train_features[trn], train_data[trn],
                                    train_features[val], train_data[val],
                                    lambdas=lambdas)
        r_cv += cost
    
    argmin_lambda = np.argmin(r_cv, axis=0)
    preds = np.zeros((test_data.shape[0], test_data.shape[1]))
    for idx_lambda in np.unique(argmin_lambda):
        idx_vox = argmin_lambda == idx_lambda
        model = sklearn_ridge(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
        curr_preds = model.predict(test_features)
        preds[:, idx_vox] = curr_preds
    return preds

def sklearn_linear_sgd(X, Y, lmbda):
    """
    Default params for SGDRegressor:
    - tol=1e-3: Training stops when (loss > best_loss - tol) for n_iter_no_change epochs
    - n_iter_no_change=5
    - max_iter=1000
    - learning_rate='invscaling'
    - eta0=1e-2: Initial learning rate
    """
    model = make_pipeline(StandardScaler(), 
                        SGDRegressor(loss='squared_loss', penalty='l2', alpha=lmbda))
    model.fit(X, Y)
    return model

def sklearn_linear_sgd_train(X, Y, Xval, Yval, lambdas):
    cost = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        model = sklearn_linear_sgd(X, Y, lmbda)
        predsval = model.predict(Xval)
        cost[idx] = 1 - R2(predsval, Yval)
    return cost

def sklearn_cross_val_ridge_linear_sgd(train_features, train_data, test_features, test_data,
                                        lambdas=np.array([10**i for i in range(-6, 10)]),
                                        no_regularization=True):
    kf = KFold(n_splits=n_splits)
    # Gather recorded costs from training with each lambda
    for ind_num, (trn, val) in enumerate(kf.split(train_data)):
        cost = sklearn_linear_sgd_train(train_features[trn], train_data[trn],
                                        train_features[val], train_data[val],
                                        lambdas=lambdas)
        r_cv += cost

    argmin_lambda = np.argmin(r_cv, axis=0)
    preds = np.zeros((test_data.shape[0], test_data.shape[1]))
    for idx_lambda in np.unique(argmin_lambda):
        idx_vox = argmin_lambda == idx_lambda
        model = sklearn_linear_sgd(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
        curr_preds = model.predict(test_features)
        preds[:, idx_vox] = curr_preds
    return preds
    

