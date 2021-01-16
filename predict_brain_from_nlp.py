import argparse
import numpy as np
import os

from utils.utils import run_class_time_CV_fmri_crossval_ridge
from utils.global_params import encoding_model_options, n_folds, roi_keys

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
    ridge_str = '' if args.use_ridge else '_noridge'
    batch_size_str = 'nobatch' if (args.batch_size is None) else 'batch{}'.format(args.batch_size)
    output_dirname += 'subject{}_{}_layer{}_len{}_{}{}_{}/'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length, args.encoding_model, ridge_str, batch_size_str)
    os.makedirs(output_dirname, exist_ok=True)

    args_dict = {'nlp_feat_type':args.nlp_feat_type,
                'nlp_feat_dir':args.nlp_feat_dir,
                'subject':args.subject,
                'layer':args.layer,
                'seq_len':args.sequence_length,
                'encoding_model':args.encoding_model,
                'batch_size':args.batch_size,
                'fold_num':args.fold_num,
                'roi_only':args.roi_only,
                'use_ridge':args.use_ridge,
                'output_dir':output_dirname}


    # loading fMRI data
    allvox_data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))
    rois = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)
    if args.encoding_model == 'nonlinear_sharedhidden':
        if args.roi_only:
            # Only use ROI voxels
            data = allvox_data[:, np.where(rois.item()[args.subject]['all'] == 1)[0]]
        else:
            data = allvox_data
        corrs, preds, test_data = run_class_time_CV_fmri_crossval_ridge(data, args_dict)
    elif args.encoding_model == 'nonlinear_sharedhidden_roipartition':
        if not args.roi_only:
            raise Exception('{} encoding model currently can only be used with --roi_only flag'.format(args.encoding_model))
        n_roivoxels = len(np.where(rois.item()[args.subject]['all'] == 1)[0])
        corrs, preds, test_data = np.zeros((n_folds, n_roivoxels)), np.zeros((allvox_data.shape[0], n_roivoxels)), np.zeros((allvox_data.shape[0], n_roivoxels))
        for key in roi_keys:
            print(key)
            args_dict['roi_key'] = key
            data = allvox_data[:, np.where(rois.item()[args.subject][key] == 1)[0]]
            roi_corrs, roi_preds, roi_test_data = run_class_time_CV_fmri_crossval_ridge(data, args_dict)
            # Get current ROI indices after translating them to fit all ROI voxels array
            roi_indices = np.where(rois.item()[args.subject][key][np.where(rois.item()[args.subject]['all'] == 1)[0]] == 1)[0]
            corrs[:,roi_indices], preds[:,roi_indices], test_data[:,roi_indices] = roi_corrs, roi_preds, roi_test_data
    else:
        raise Exception('{} encoding model not recognized'.format(args.encoding_model))

    output_path = output_dirname + 'final_predictions.npy'
    print('saving: ' + output_path)
    np.save(output_path, {'corrs':corrs,'preds':preds,'test_data':test_data})

    
