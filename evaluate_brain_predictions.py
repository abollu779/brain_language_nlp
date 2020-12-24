import argparse
import numpy as np
import pickle as pk
import time as tm
import os

from utils.utils import CV_ind, binary_classify_neighborhoods, binary_classify_without_neighborhoods


if __name__ == '__main__':
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subject", default='')
    parser.add_argument("--use_neighborhood", action='store_true')
    args = parser.parse_args()
    print(args)

    start_time = tm.time()

    loaded = np.load(args.input_path, allow_pickle=True)
    preds_t_per_feat = loaded.item()['preds_t']
    test_t_per_feat = loaded.item()['test_t']
    print(test_t_per_feat.shape)
    
    n_class = 20   # how many predictions to classify at the same time
    n_folds = 4
    
    if args.use_neighborhood:
        neighborhoods = np.load('./data/voxel_neighborhoods/' + args.subject + '_ars_auto2.npy')
    n_words, n_voxels = test_t_per_feat.shape
    ind = CV_ind(n_words, n_folds=n_folds)

    accs = np.zeros([n_folds,n_voxels])
    acc_std = np.zeros([n_folds,n_voxels])

    for ind_num in range(n_folds):
        test_ind = ind==ind_num
        if args.use_neighborhood:
            accs[ind_num,:],_,_,_ = binary_classify_neighborhoods(preds_t_per_feat[test_ind,:], test_t_per_feat[test_ind,:], n_class=20, nSample = 1000,pair_samples = [],neighborhoods=neighborhoods)
        else:
            accs[ind_num,:] = binary_classify_without_neighborhoods(preds_t_per_feat[test_ind,:], test_t_per_feat[test_ind,:], n_class=20, nSample=1000)

    neighborhood_subdirname = 'with_neighborhood/' if args.use_neighborhood else 'without_neighborhood/'
    output_dirname = args.output_dir + args.input_path.split('encoder_preds/')[1].split('predict_')[0] + neighborhood_subdirname
    os.makedirs(output_dirname, exist_ok=True)
    output_fname = args.input_path.split('predict_')[1].split('.npy')[0]
    output_path = output_dirname + output_fname

    if n_class < 20:
        output_path = output_path + '_{}v{}_'.format(n_class,n_class)

    with open(output_path + '_accs.pkl','wb') as fout:
        pk.dump(accs,fout)

    print('saved: {}'.format(output_path + '_accs.pkl'))


    
