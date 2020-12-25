import argparse
import numpy as np
import os

from utils.utils import run_class_time_CV_fmri_crossval_ridge
from utils.global_params import encoding_model_options, n_folds

if __name__ == '__main__':
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--encoding_model", required=True, choices=encoding_model_options)
    parser.add_argument("--batch_size", type=int, required=False) # only relevant for nonlinear models (trained with GD); if not entered, no batching done during training
    parser.add_argument("--fold_num", type=int, choices=[i for i in range(n_folds)])
    parser.add_argument("--roi_only", action='store_true')
    parser.add_argument("--use_ridge", action='store_true')
    parser.add_argument("--output_dir", required=True)    
    args = parser.parse_args()
    print(args)

    output_dirname = args.output_dir
    output_dirname += 'roi_voxels/' if args.roi_only else 'all_voxels/'
    ridge_suffix = '' if args.use_ridge else 'noridge'
    output_dirname += 'subject{}_{}_layer{}_len{}_encoder{}_{}/'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length, args.encoding_model, ridge_suffix)
    os.makedirs(output_dirname, exist_ok=True)

    args_dict = {'nlp_feat_type':args.nlp_feat_type,
                'nlp_feat_dir':args.nlp_feat_dir,
                'layer':args.layer,
                'seq_len':args.sequence_length,
                'encoding_model':args.encoding_model,
                'batch_size':args.batch_size,
                'fold_num':args.fold_num,
                'roi_only':args.roi_only,
                'use_ridge':args.use_ridge,
                'output_dir':output_dirname}


    # loading fMRI data
    data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))
    if args.roi_only:
        rois = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)
        data = data[:, np.where(rois.item()[args.subject]['all'] == 1)[0]]
    
    corrs, preds, test_data = run_class_time_CV_fmri_crossval_ridge(data, args_dict)

    output_path = output_dirname + 'final_predictions.npy'
    print('saving: ' + output_path)
    np.save(output_path, {'corrs':corrs,'preds':preds,'test_data':test_data})

    
