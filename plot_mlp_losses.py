import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

num_folds = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--voxel_num", required=True, type=int)
    args = parser.parse_args()
    print(args)

def collect_mlp_losses(encoding_model, subject, voxel_num):
    global num_folds
    fold_losses = []
    for fold in range(num_folds):
        curr_fold_losses = np.load("{}/mlp_fold_losses/subject_{}/fold_{}.npy".format(encoding_model, subject, fold))
        curr_fold_losses = curr_fold_losses[voxel_num]
        fold_losses.append(curr_fold_losses)
    fold_losses = np.array(fold_losses)
    return fold_losses

X = np.arange(1,11)
mlp_initial_losses = collect_mlp_losses('mlp_initial', args.subject, args.voxel_num)
mlp_smallerhiddensize_losses = collect_mlp_losses('mlp_smallerhiddensize', args.subject, args.voxel_num)
mlp_largerhiddensize_losses = collect_mlp_losses('mlp_largerhiddensize', args.subject, args.voxel_num)
mlp_additionalhiddenlayer_losses = collect_mlp_losses('mlp_additionalhiddenlayer', args.subject, args.voxel_num)

fig, axs = plt.subplots(2, 2)
for fold in range(num_folds):
    axs_x, axs_y = fold // 2, fold % 2
    axs[axs_x, axs_y].plot(X, mlp_initial_losses[fold], color='green')
    axs[axs_x, axs_y].plot(X, mlp_smallerhiddensize_losses[fold], color='blue')
    axs[axs_x, axs_y].plot(X, mlp_largerhiddensize_losses[fold], color='red')
    axs[axs_x, axs_y].plot(X, mlp_additionalhiddenlayer_losses[fold], color='black')
    axs[axs_x, axs_y].set_title('Losses: Subject {} - Voxel {} - Fold {}'.format(args.subject, args.voxel_num, fold+1))

for i, ax in enumerate(axs.flat):
    if i // 2 == 0:
        ax.set(ylabel='Test Loss')
    else:
        ax.set(xlabel='Epoch', ylabel='Test Loss')

green_patch = mpatches.Patch(color='green', label='mlp_initial')
blue_patch = mpatches.Patch(color='blue', label='mlp_smallerhiddensize')
red_patch = mpatches.Patch(color='red', label='mlp_largerhiddensize')
black_patch = mpatches.Patch(color='black', label='mlp_additionalhiddenlayer')
plt.legend(handles=[green_patch, blue_patch, red_patch, black_patch])

plt.show()



