import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pickle as pk

voxels_type_options = ['ALLROI', 'POSTTEMP', 'ANTTEMP', 'ANGULARG', 'IFG', 'MFG', 'IFGORB', 'PCINGULATE', 'DMPFC']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--voxels_type", default='ALL', choices=voxels_type_options)
    args = parser.parse_args()
    print(args)

#################################### CORRELATIONS TO PLOT #################################################
# linear_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_linear.npy'.format(args.subject), allow_pickle=True)
mlp_initial_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_mlp_initial.npy'.format(args.subject), allow_pickle=True)
# mlp_smallerhiddensize_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_mlp_smallerhiddensize.npy'.format(args.subject), allow_pickle=True)
# mlp_largerhiddensize_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_mlp_largerhiddensize.npy'.format(args.subject), allow_pickle=True)
# mlp_additionalhiddenlayer_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_mlp_additionalhiddenlayer.npy'.format(args.subject), allow_pickle=True)
mlp_allvoxels_output = np.load('encoder_preds/predict_{}_with_bert_layer_8_len_10_encoder_mlp_allvoxels.npy'.format(args.subject), allow_pickle=True)

# linear_corrs = linear_output.item()['corrs_t']
mlp_initial_corrs = mlp_initial_output.item()['corrs_t']
# mlp_smallerhiddensize_corrs = mlp_smallerhiddensize_output.item()['corrs_t']
# mlp_largerhiddensize_corrs = mlp_largerhiddensize_output.item()['corrs_t']
# mlp_additionalhiddenlayer_corrs = mlp_additionalhiddenlayer_output.item()['corrs_t']
mlp_allvoxels_corrs = mlp_allvoxels_output.item()['corrs_t']
##############################################################################################################

def get_X_Y(corrs_probe1,  corrs_probe2):
    corrs_avg_1 = np.nanmean(corrs_probe1, axis=0)
    corrs_avg_2 = np.nanmean(corrs_probe2, axis=0)

    X = corrs_avg_1
    Y = corrs_avg_2

    X[np.where(np.isnan(X) == 1)] = 0
    Y[np.where(np.isnan(Y) == 1)] = 0

    rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)

    if args.voxels_type != 'ALLROI':
        if args.voxels_type == 'POSTTEMP':
            mask = rois.item()[args.subject]['PostTemp'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'ANTTEMP':
            mask = rois.item()[args.subject]['AntTemp'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'ANGULARG':
            mask = rois.item()[args.subject]['AngularG'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'IFG':
            mask = rois.item()[args.subject]['IFG'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'MFG':
            mask = rois.item()[args.subject]['MFG'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'IFGORB':
            mask = rois.item()[args.subject]['IFGorb'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'PCINGULATE':
            mask = rois.item()[args.subject]['pCingulate'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        elif args.voxels_type == 'DMPFC':
            mask = rois.item()[args.subject]['dmpfc'][np.where(rois.item()[args.subject]['all'] > 0)[0]]
        X = np.ma.masked_equal(mask * corrs_avg_1, 0).compressed()
        Y = np.ma.masked_equal(mask * corrs_avg_2, 0).compressed()
    return X, Y

def plot_correlations(X, Y, subject, voxels_type, x_label, y_label):
    delete_indices = np.where((np.isnan(X) == 1) | (np.isnan(Y) == 1))
    X = np.delete(X, delete_indices, 0)
    Y = np.delete(Y, delete_indices, 0)

    # Best fit line data
    X_above_zero = X[(X > 0) | (Y > 0)]
    Y_above_zero = Y[(X > 0) | (Y > 0)]
    xmin = X_above_zero.min()
    xmax = X_above_zero.max()
    slope, y_intercept = np.polyfit(X_above_zero, Y_above_zero, deg=1)
    x_0, y_0, x_1, y_1 = xmin, slope*xmin+y_intercept, xmax, slope*xmax+y_intercept

    fig, ax = plt.subplots()
    
    ax.scatter(X, Y)
    bf_line = mlines.Line2D([x_0, x_1], [y_0, y_1], color='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(bf_line)
    ax.add_line(line)

    plt.title('{} vs {} Correlations: Subject {} - {} Voxels'.format(x_label, y_label, subject, voxels_type))
    plt.xlabel('{} corrs'.format(x_label))
    plt.ylabel('{} corrs'.format(y_label))
    plt.xlim(-0.25, 0.5)
    plt.ylim(-0.25, 0.5)
    plt.show()
    # plt.savefig('plots/corrs_{}_subject_{}_{}-{}.png'.format(voxels_type, subject, x_label, y_label))

# X, Y = get_X_Y(linear_corrs, mlp_initial_corrs)
# plot_correlations(X, Y, args.subject, args.voxels_type, 'linear', 'mlp_initial')
# X, Y = get_X_Y(mlp_initial_corrs, mlp_smallerhiddensize_corrs)
# plot_correlations(X, Y, args.subject, args.voxels_type, 'mlp_initial', 'mlp_smallerhiddensize')
# X, Y = get_X_Y(mlp_initial_corrs, mlp_largerhiddensize_corrs)
# plot_correlations(X, Y, args.subject, args.voxels_type, 'mlp_initial', 'mlp_largerhiddensize')
# X, Y = get_X_Y(mlp_initial_corrs, mlp_additionalhiddenlayer_corrs)
# plot_correlations(X, Y, args.subject, args.voxels_type, 'mlp_initial', 'mlp_additionalhiddenlayer')
X, Y = get_X_Y(mlp_initial_corrs, mlp_allvoxels_corrs)
plot_correlations(X, Y, args.subject, args.voxels_type, 'mlp_initial', 'mlp_allvoxels')


#################################### ACCURACIES TO PLOT #################################################
# linear_accs = pk.load(open('final_accs/no_neighborhood_acc_{}_with_bert_layer_8_len_10_encoder_linear_accs_ROI.pkl'.format(args.subject), 'rb'))
# mlp_accs = pk.load(open('final_accs/no_neighborhood_acc_{}_with_bert_layer_8_len_10_encoder_mlp_initial_accs_ROI.pkl'.format(args.subject), 'rb'))
# linear_accs_avg = linear_accs.mean(0)
# mlp_accs_avg = mlp_accs.mean(0)
# rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
# linear_accs_avg = linear_accs_avg[np.where(rois.item()[args.subject]['all'] == 1)[0]]
# mlp_accs_avg = mlp_accs_avg[np.where(rois.item()[args.subject]['all'] == 1)[0]]
##############################################################################################################

def plot_accuracies(X, Y, subject, voxels_type):
    X = np.delete(X, np.where(np.isnan(Y) == 1), 0)
    Y = np.delete(Y, np.where(np.isnan(Y) == 1), 0)

    # Best fit line data
    X_above_half = X[(X > 0.5) | (Y > 0.5)]
    Y_above_half = Y[(X > 0.5) | (Y > 0.5)]
    xmin = X_above_half.min()
    xmax = X_above_half.max()
    slope, y_intercept = np.polyfit(X_above_half, Y_above_half, deg=1)
    x_0, y_0, x_1, y_1 = xmin, slope*xmin+y_intercept, xmax, slope*xmax+y_intercept

    fig, ax = plt.subplots()
    ax.scatter(X, Y)

    bf_line = mlines.Line2D([x_0, x_1], [y_0, y_1], color='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    # bf_line.set_transform(transform)
    line.set_transform(transform)

    ax.add_line(bf_line)
    ax.add_line(line)

    plt.title('Accuracies (No Neighborhood): Subject {} - {} Voxels'.format(subject, voxels_type))
    plt.xlabel('Linear accs')
    plt.ylabel('MLP accs')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    # plt.savefig('plots/accs_{}_subject_{}.png'.format(voxels_type, subject))

# plot_accuracies(linear_accs_avg, mlp_accs_avg, args.subject, args.voxels_type)