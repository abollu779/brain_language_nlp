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
    parser.add_argument("--no-regularization", action='store_true')
    
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
                         'use_all_voxels': args.use_all_voxels,
                         'no_regularization': args.no_regularization}


    # loading fMRI data
    data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))
    # # limit to ROI data
    #     if not args.use_all_voxels:
    #         rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
    #         data = data[:, np.where(rois.item()[args.subject]['all'] == 1)[0]]
    num_voxels = data.shape[1]
    chunk_size = num_voxels//5

    corrs_t, preds_t, test_t, train_losses_t, test_losses_t = [], [], [], [], []

    for data_start in range(0, num_voxels, chunk_size):
        data_chunk = data[:, data_start:data_start+chunk_size]

        corrs, preds, test, train_losses, test_losses = run_class_time_CV_fmri_crossval_ridge(data_chunk,
                                                                    predict_feat_dict)
        
        corrs_t.append(corrs)
        preds_t.append(preds)
        test_t.append(test)
        train_losses_t.append(train_losses)
        test_losses_t.append(test_losses)
    
    corrs_t = np.concatenate(corrs_t, axis=0)
    preds_t = np.concatenate(preds_t, axis=1)
    test_t = np.concatenate(test_t, axis=1)
    train_losses_t = np.stack(train_losses_t, axis=0)
    test_losses_t = np.stack(test_losses_t, axis=0)


    if not args.single_fold_computation:
        dirname = 'maxvoxels/' if args.use_all_voxels else 'roivoxels/'
        fname = 'predict_{}_with_{}_layer_{}_len_{}_encoder_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length, args.encoding_model)
        print('saving: {}'.format(args.output_dir + dirname + fname))

        os.makedirs(args.output_dir + dirname, exist_ok=True)

        np.save(args.output_dir + dirname + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t,'train_losses_t':train_losses_t,'test_losses_t':test_losses_t})

    
