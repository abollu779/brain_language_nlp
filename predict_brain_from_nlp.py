import argparse
import numpy as np
import os

from utils.global_params import n_folds, encoding_model_options
from utils.utils import run_class_time_CV_fmri_crossval_ridge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp-feat-type", required=True)
    parser.add_argument("--nlp-feat-dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence-length", type=int, required=False)
    parser.add_argument("--encoding-model", default='linear', choices=encoding_model_options)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-all-voxels", action='store_true')
    parser.add_argument("--single-fold-computation", action='store_true')
    parser.add_argument("--fold-num", type=int, choices=[i for i in range(n_folds)])
    
    args = parser.parse_args()

    if args.single_fold_computation and (args.fold_num is None):
        parser.error("--single-fold-computation requires --fold-num to specify which fold to use.")

    print(args)
        
    predict_feat_dict = {'nlp_feat_type':args.nlp_feat_type,
                         'nlp_feat_dir':args.nlp_feat_dir,
                         'layer':args.layer,
                         'seq_len':args.sequence_length,
                         'encoding_model':args.encoding_model,
                         'subject':args.subject,
                         'single_fold_computation': args.single_fold_computation,
                         'fold_num': args.fold_num,
                         'use_all_voxels': args.use_all_voxels}


    # loading fMRI data
    data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))
    # TODO: If a model can be trained for a third of the data, ensure there's a 
    # loop that trains models for the remaining two-thirds later
    num_voxels = data.shape[1]
    data = data[:, :10]

    # limit to ROI data
    if not args.use_all_voxels:
        rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
        data = data[:, np.where(rois.item()[args.subject]['all'] == 1)[0]]

    corrs_t, preds_t, test_t, train_losses_t, test_losses_t = run_class_time_CV_fmri_crossval_ridge(data,
                                                                predict_feat_dict)

    if not args.single_fold_computation:
        dirname = 'maxvoxels/' if args.use_all_voxels else 'roivoxels/'
        fname = 'predict_{}_with_{}_layer_{}_len_{}_encoder_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length, args.encoding_model)
        print('saving: {}'.format(args.output_dir + dirname + fname))

        os.makedirs(args.output_dir + dirname, exist_ok=True)

        np.save(args.output_dir + dirname + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t,'train_losses_t':train_losses_t,'test_losses_t':test_losses_t})

    
