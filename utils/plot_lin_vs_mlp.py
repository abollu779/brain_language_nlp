import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pickle as pk

voxels_type_options = ['ALL', 'ALL ROI', 'POSTTEMP', 'ANTTEMP', 'ANGULARG', 'IFG', 'MFG', 'IFGORB', 'PCINGULATE', 'DMPFC']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--voxels_type", default='ALL', choices=voxels_type_options)
    args = parser.parse_args()
    print(args)

#################################### ACCURACIES TO PLOT #################################################
linear_accs = pk.load(open('final_accs/acc_F_with_bert_layer_1_len_1_encoder_linear_accs.pkl', 'rb'))
mlp_accs = pk.load(open('final_accs/acc_F_with_bert_layer_1_len_1_encoder_mlp_accs.pkl', 'rb'))

linear_accs_avg = linear_accs.mean(0)
mlp_accs_avg = mlp_accs.mean(0)

if args.voxels_type == 'ALL':
    X = linear_accs_avg
    Y = mlp_accs_avg
else:
    rois = np.load('../HP_subj_roi_inds.npy', allow_pickle=True)
    if args.voxels_type == 'ALL ROI':
        X = np.ma.masked_equal(rois.item()[args.subject]['all'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['all'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'POSTTEMP':
        X = np.ma.masked_equal(rois.item()[args.subject]['PostTemp'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['PostTemp'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'ANTTEMP':
        X = np.ma.masked_equal(rois.item()[args.subject]['AntTemp'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['AntTemp'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'ANGULARG':
        X = np.ma.masked_equal(rois.item()[args.subject]['AngularG'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['AngularG'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'IFG':
        X = np.ma.masked_equal(rois.item()[args.subject]['IFG'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['IFG'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'MFG':
        X = np.ma.masked_equal(rois.item()[args.subject]['MFG'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['MFG'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'IFGORB':
        X = np.ma.masked_equal(rois.item()[args.subject]['IFGorb'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['IFGorb'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'PCINGULATE':
        X = np.ma.masked_equal(rois.item()[args.subject]['pCingulate'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['pCingulate'] * mlp_accs_avg, 0).compressed()
    elif args.voxels_type == 'DMPFC':
        X = np.ma.masked_equal(rois.item()[args.subject]['dmpfc'] * linear_accs_avg, 0).compressed()
        Y = np.ma.masked_equal(rois.item()[args.subject]['dmpfc'] * mlp_accs_avg, 0).compressed()
##############################################################################################################

def plot_accuracies(X, Y, subject, voxels_type):
    slope, y_intercept = np.polyfit(X, Y, deg=1)
    x_0, y_0, x_1, y_1 = 0, y_intercept, 1, slope+y_intercept

    fig, ax = plt.subplots()
    ax.scatter(X, Y)

    bf_line = mlines.Line2D([x_0, x_1], [y_0, y_1], color='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    bf_line.set_transform(transform)

    ax.add_line(line)
    ax.add_line(bf_line)

    plt.title('{} Voxels Scatterplot'.format(voxels_type))
    plt.xlabel('Linear accs')
    plt.ylabel('MLP accs')
    plt.savefig('plots/subject_{}_{}.png'.format(subject, voxels_type))

plot_accuracies(X, Y, args.subject, args.voxels_type)