import argparse
import numpy as np

from utils.utils import run_class_time_CV_fmri_crossval_ridge

encoding_model_options = ['linear', 'mlp_initial', 'mlp_smallerhiddensize', 'mlp_largerhiddensize', 'mlp_additionalhiddenlayer']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--encoding_model", default="linear", choices=encoding_model_options)
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    print(args)
        
    predict_feat_dict = {'nlp_feat_type':args.nlp_feat_type,
                         'nlp_feat_dir':args.nlp_feat_dir,
                         'layer':args.layer,
                         'seq_len':args.sequence_length,
                         'encoding_model':args.encoding_model,
                         'subject':args.subject}


    # loading fMRI data
    data = np.load('./data/fMRI/data_subject_{}.npy'.format(args.subject))

    # limit to ROI data
    rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
    data = data[:, np.where(rois.item()[args.subject]['all'] == 1)[0]]

    ind_num = 0

    corrs_t, _, _, preds_t, test_t, train_losses_t, test_losses_t = run_class_time_CV_fmri_crossval_ridge(data,
                                                                predict_feat_dict)

    fname = 'predict_{}_with_{}_layer_{}_len_{}_encoder_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length, args.encoding_model)
    print('saving: {}'.format(args.output_dir + fname))

    np.save(args.output_dir + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t,'train_losses_t':train_losses_t,'test_losses_t':test_losses_t})

    
