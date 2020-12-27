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
    preds = loaded.item()['preds']
    test_data = loaded.item()['test_data']
    
    n_class = 20   # how many predictions to classify at the same time
    n_folds = 4
    
    if args.use_neighborhood:
        neighborhoods = np.load('./data/voxel_neighborhoods/' + args.subject + '_ars_auto2.npy')
    n_words, n_voxels = test_data.shape
    ind = CV_ind(n_words, n_folds=n_folds)

    accs = np.zeros([n_folds,n_voxels])
    acc_std = np.zeros([n_folds,n_voxels])

    for ind_num in range(n_folds):
        test_ind = ind==ind_num
        if args.use_neighborhood:
            accs[ind_num,:],_,_,_ = binary_classify_neighborhoods(preds[test_ind,:], test_data[test_ind,:], n_class=20, nSample = 1000,pair_samples = [],neighborhoods=neighborhoods)
        else:
            accs[ind_num,:] = binary_classify_without_neighborhoods(preds[test_ind,:], test_data[test_ind,:], n_class=20, nSample=1000)

    neighborhood_subdirname = 'with_neighborhood/' if args.use_neighborhood else 'without_neighborhood/'
    output_dirname = args.output_dir + neighborhood_subdirname
    os.makedirs(output_dirname, exist_ok=True)
    output_fname = args.input_path.split('/')[2] + '_accs.pkl'
    output_path = output_dirname + output_fname

    with open(output_path,'wb') as fout:
        pk.dump(accs,fout)

    print('saved: {}'.format(output_path))


    
