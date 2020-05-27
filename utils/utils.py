import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time
import csv
import os
import os.path
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter

from utils.global_params import n_folds, n_epochs, n_splits
from utils.ridge_tools import cross_val_ridge, corr, cross_val_ridge_mlp
import time as tm

    
def load_transpose_zscore(file): 
    dat = nibabel.load(file).get_data()
    dat = dat.T
    return zscore(dat,axis = 0)

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None,
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

# train/test is the full NLP feature
# train/test_pca is the NLP feature reduced to 10 dimensions via PCA that has been fit on the training data
# feat_dir is the directory where the NLP features are stored
# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def get_nlp_features_fixed_length(layer, seq_len, feat_type, feat_dir, train_indicator, SKIP_WORDS=20, END_WORDS=5176):
    
    loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + '.npy')
    if feat_type == 'elmo':
        train = loaded[SKIP_WORDS:END_WORDS,:][:,:512][train_indicator]   # only forward LSTM
        test = loaded[SKIP_WORDS:END_WORDS,:][:,:512][~train_indicator]   # only forward LSTM
    elif feat_type == 'bert' or feat_type == 'transformer_xl' or feat_type == 'use':
        train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
        test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options elmo, bert, transformer_xl, use'.format(feat_type))
    
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    return train, test, train_pca, test_pca 

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def TR_to_word_CV_ind(TR_train_indicator,SKIP_WORDS=20,END_WORDS=5176):
    time = np.load('./data/fMRI/time_fmri.npy')
    runs = np.load('./data/fMRI/runs_fmri.npy') 
    time_words = np.load('./data/fMRI/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)    
    words_id = np.zeros([len(time_words)],dtype=int)
    # w=find what TR each word belongs to
    for i in range(len(time_words)):                
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True
    return word_train_indicator        


def prepare_fmri_features(train_features, test_features, word_train_indicator, TR_train_indicator, SKIP_WORDS=20, END_WORDS=5176):
        
    time = np.load('./data/fMRI/time_fmri.npy')
    runs = np.load('./data/fMRI/runs_fmri.npy') 
    time_words = np.load('./data/fMRI/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
    all_features = np.zeros([time_words.shape[0], train_features.shape[1]])
    all_features[word_train_indicator] = train_features
    all_features[~word_train_indicator] = test_features
        
    p = all_features.shape[1]
    tmp = np.zeros([time.shape[0], p])
    for i in range(time.shape[0]):
        tmp[i] = np.mean(all_features[(words_id<=i)*(words_id>=i-1)],0)
    tmp = delay_mat(tmp, np.arange(1,5))

    # remove the edges of each run
    tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)])
    tmp = np.nan_to_num(tmp)
        
    return tmp[TR_train_indicator], tmp[~TR_train_indicator]

def single_fold_run_class_time_CV_fmri_crossval_ridge(ind_num, train_ind, test_ind, data, predict_feat_dict,
                                                        regress_feat_names_list = [],
                                                        method = 'kernel_ridge', 
                                                        lambdas = np.array([0.1,1,10,100,1000]),
                                                        detrend = False, skip=5):
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    encoding_model = predict_feat_dict['encoding_model']
    subject = predict_feat_dict['subject']
    use_all_voxels = predict_feat_dict['use_all_voxels']

    word_CV_ind = TR_to_word_CV_ind(train_ind)
    train_losses, test_losses = None, None

    _,_,tmp_train_features,tmp_test_features = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)
    train_features,test_features = prepare_fmri_features(tmp_train_features, tmp_test_features, word_CV_ind, train_ind)

    # split data
    train_data = data[train_ind]
    test_data = data[test_ind]

    # skip TRs between train and test data
    if ind_num == 0: # just remove from front end
        train_data = train_data[skip:,:]
        train_features = train_features[skip:,:]
    elif ind_num == n_folds-1: # just remove from back end
        train_data = train_data[:-skip,:]
        train_features = train_features[:-skip,:]
    else:
        train_data = train_data[skip:-skip,:]
        train_features = train_features[skip:-skip,:]

    start_time = tm.time()
    if encoding_model == 'linear':
        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data))) # (N_train, num_voxels)
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data))) # (N_test, num_voxels)
        
        train_features = np.nan_to_num(zscore(train_features)) # (N_train, feat_dim)
        test_features = np.nan_to_num(zscore(test_features)) # (N_test, feat_dim)
        
        weights, chosen_lambdas = cross_val_ridge(train_features, train_data, method='plain', do_plot=False)
        preds =  np.dot(test_features, weights)
        # weights: (40, 27905)
        del weights
    else:
        vox_subdirname = 'maxvoxels/' if use_all_voxels else 'roivoxels/'
        assert 'mlp' in encoding_model
        preds_dir = '{}/mlp_fold_preds/subject_{}/{}/layer_{}/seqlen_{}/'.format(encoding_model, subject, vox_subdirname, layer, seq_len)
        preds_path = preds_dir + 'fold_{}.npy'.format(ind_num)
        train_losses_dir = '{}/mlp_fold_train_losses/subject_{}/{}/layer_{}/seqlen_{}/'.format(encoding_model, subject, vox_subdirname, layer, seq_len)
        train_losses_path = train_losses_dir + 'fold_{}.npy'.format(ind_num)
        test_losses_dir = '{}/mlp_fold_test_losses/subject_{}/{}/layer_{}/seqlen_{}/'.format(encoding_model, subject, vox_subdirname, layer, seq_len)
        test_losses_path = test_losses_dir + 'fold_{}.npy'.format(ind_num)

        s_t = tm.time()
        if (os.path.exists(preds_path) and os.path.exists(train_losses_path) and os.path.exists(test_losses_path)):
            preds = np.load(preds_path)
            train_losses = np.load(train_losses_path)
            test_losses = np.load(test_losses_path)
        else:
            preds, train_losses, test_losses = cross_val_ridge_mlp(encoding_model, train_features, train_data, test_features, test_data)
            preds = preds.detach().cpu().numpy()

            os.makedirs(preds_dir, exist_ok=True)
            os.makedirs(train_losses_dir, exist_ok=True)
            os.makedirs(test_losses_dir, exist_ok=True)
            
            np.save(preds_path, preds)
            np.save(train_losses_path, train_losses)
            np.save(test_losses_path, test_losses)
        # preds: (N_test, 27905)
        mlp_time = tm.time() - s_t
        print("-> MLP Training Time for fold {}: {}s".format(ind_num, mlp_time))
    corrs = corr(preds, test_data)
    print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
    return corrs, preds, train_losses, test_losses, test_data

def run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict):
    encoding_model = predict_feat_dict['encoding_model']
    single_fold_computation = predict_feat_dict['single_fold_computation']
    fold_num = predict_feat_dict['fold_num']

    n_words = data.shape[0]
    n_voxels = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds)
    
    if single_fold_computation:
        assert 'mlp' in encoding_model
        train_ind = ind!=fold_num
        test_ind = ind==fold_num
        corrs_d, preds_d, train_losses_d, test_losses_d, all_test_data = single_fold_run_class_time_CV_fmri_crossval_ridge(fold_num, train_ind, test_ind, 
                                                                                                        data, predict_feat_dict)
    else: 
        corrs_d = np.zeros((n_folds, n_voxels))
        preds_d = np.zeros((n_words, n_voxels))
        train_losses_d, test_losses_d = None, None
        if 'mlp' in encoding_model:
            train_losses_d = np.zeros((n_folds, n_voxels, n_epochs))
            test_losses_d = np.zeros((n_folds, n_voxels, n_epochs))
        all_test_data = []
        
        # Train across all folds
        for ind_num in range(n_folds):
            train_ind = ind!=ind_num
            test_ind = ind==ind_num
            corrs, preds, train_losses, test_losses, test_data = single_fold_run_class_time_CV_fmri_crossval_ridge(ind_num, train_ind, test_ind, 
                                                                                                        data, predict_feat_dict)
            all_test_data.append(test_data)
            corrs_d[ind_num,:] = corrs
            preds_d[test_ind] = preds

            if 'mlp' in encoding_model:
                train_losses_d[ind_num,:] = train_losses
                test_losses_d[ind_num,:] = test_losses
            
        all_test_data = np.vstack(all_test_data)

    return corrs_d, preds_d, all_test_data, train_losses_d, test_losses_d

def binary_class(Ypred, Y, n_class=20, nSample = 1000):
    np.random.seed(0)

    # does 1000 samples of 1vs2 classification
    n = Y.shape[0]
    acc = np.zeros((Y.shape[1]))
    for iS in range(nSample):
        idx_real = np.random.choice(n, n_class)
        sample_real = Y[idx_real]
        sample_pred_correct = Ypred[idx_real]
        idx_wrong = np.random.choice(n, n_class)
        sample_pred_incorrect = Ypred[idx_wrong]
        dist_correct = np.sum((sample_real - sample_pred_correct)**2,0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect)**2,0)
        acc += (dist_correct < dist_incorrect)*1.0 + (dist_correct == dist_incorrect)*0.5
    acc = acc/nSample
    return acc

def binary_classify_neighborhoods(Ypred, Y, n_class=20, nSample = 1000,pair_samples = [],neighborhoods=[]):
    np.random.seed(0)

    # n_class = how many words to classify at once
    # nSample = how many words to classify

    voxels = Y.shape[-1]
    neighborhoods = np.asarray(neighborhoods, dtype=int)

    import time as tm

    acc = np.full([nSample, Y.shape[-1]], np.nan)
    acc2 = np.full([nSample, Y.shape[-1]], np.nan)
    test_word_inds = []

    if len(pair_samples)>0:
        Ypred2 = Ypred[pair_samples>=0]
        Y2 = Y[pair_samples>=0]
        pair_samples2 = pair_samples[pair_samples>=0]
    else:
        Ypred2 = Ypred
        Y2 = Y
        pair_samples2 = pair_samples
    n = Y2.shape[0]
    start_time = tm.time()
    for idx in range(nSample):
        
        idx_real = np.random.choice(n, n_class)

        sample_real = Y2[idx_real]
        sample_pred_correct = Ypred2[idx_real]

        if len(pair_samples2) == 0:
            idx_wrong = np.random.choice(n, n_class)
        else:
            idx_wrong = sample_same_but_different(idx_real,pair_samples2)
        sample_pred_incorrect = Ypred2[idx_wrong]

        #print(sample_pred_incorrect.shape)

        # compute distances within neighborhood
        dist_correct = np.sum((sample_real - sample_pred_correct)**2,0)
        dist_incorrect = np.sum((sample_real - sample_pred_incorrect)**2,0)

        neighborhood_dist_correct = np.array([np.sum(dist_correct[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])
        neighborhood_dist_incorrect = np.array([np.sum(dist_incorrect[neighborhoods[v,neighborhoods[v,:]>-1]]) for v in range(voxels)])


        acc[idx,:] = (neighborhood_dist_correct < neighborhood_dist_incorrect)*1.0 + (neighborhood_dist_correct == neighborhood_dist_incorrect)*0.5

        test_word_inds.append(idx_real)
    print('Classification for fold done. Took {} seconds'.format(tm.time()-start_time))
    return np.nanmean(acc,0), np.nanstd(acc,0), acc, np.array(test_word_inds)
