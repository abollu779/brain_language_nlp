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

#################################### CORRELATION/ACCURACIES TO PLOT #################################################
# linear_accs = pk.load(open('final_accs/acc_{}_with_bert_layer_1_len_1_encoder_linear_accs_ROI.pkl'.format(args.subject), 'rb'))
# mlp_accs = pk.load(open('final_accs/acc_{}_with_bert_layer_1_len_1_encoder_mlp_accs_ROI.pkl'.format(args.subject), 'rb'))

# linear_accs_avg = linear_accs.mean(0)
# mlp_accs_avg = mlp_accs.mean(0)

# rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
# linear_accs_avg = linear_accs_avg[np.where(rois.item()[args.subject]['all'] == 1)[0]]
# mlp_accs_avg = mlp_accs_avg[np.where(rois.item()[args.subject]['all'] == 1)[0]]

linear_output = np.load('encoder_preds/predict_{}_with_bert_layer_1_len_1_encoder_linear.npy'.format(args.subject), allow_pickle=True)
mlp_output = np.load('encoder_preds/predict_{}_with_bert_layer_1_len_1_encoder_mlp.npy'.format(args.subject), allow_pickle=True)

linear_corrs = linear_output.item()['corrs_t']
mlp_corrs = mlp_output.item()['corrs_t']

linear_corrs_avg = np.nanmean(linear_corrs, axis=0)
mlp_corrs_avg = np.nanmean(mlp_corrs, axis=0)

X = linear_corrs_avg
Y = mlp_corrs_avg

rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)

if args.voxels_type == 'ALLROI':
    X = np.ma.masked_equal(linear_corrs_avg, 0).compressed()
    Y = np.ma.masked_equal(mlp_corrs_avg, 0).compressed()
else:
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
    X = np.ma.masked_equal(mask * linear_corrs_avg, 0).compressed()
    Y = np.ma.masked_equal(mask * mlp_corrs_avg, 0).compressed()
##############################################################################################################

def plot_correlations(X, Y, subject, voxels_type):
    X = np.delete(X, np.where(np.isnan(Y) == 1), 0)
    Y = np.delete(Y, np.where(np.isnan(Y) == 1), 0)

    # Best fit line data
    X_above_zero = X[(X > 0) & (Y > 0)]
    Y_above_zero = Y[(X > 0) & (Y > 0)]
    xmin = X_above_zero.min()
    xmax = X_above_zero.max()
    slope, y_intercept = np.polyfit(X_above_zero, Y_above_zero, deg=1)
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

    plt.title('Correlations: Subject {} - {} Voxels'.format(subject, voxels_type))
    plt.xlabel('Linear corrs')
    plt.ylabel('MLP corrs')
    plt.xlim(-0.25, 0.5)
    plt.ylim(-0.25, 0.5)
    plt.savefig('plots/corrs_{}_subject_{}.png'.format(voxels_type, subject))

def plot_accuracies(X, Y, subject, voxels_type):
    X = np.delete(X, np.where(np.isnan(Y) == 1), 0)
    Y = np.delete(Y, np.where(np.isnan(Y) == 1), 0)

    # Best fit line data
    xmin = X.min()
    xmax = X.max()
    slope, y_intercept = np.polyfit(X, Y, deg=1)
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

    plt.title('Accuracies: Subject {} - {} Voxels'.format(subject, voxels_type))
    plt.xlabel('Linear accs')
    plt.ylabel('MLP accs')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('plots/accs_{}_subject_{}.png'.format(voxels_type, subject))

plot_accuracies(X, Y, args.subject, args.voxels_type)