import argparse
import pickle as pk
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    args = parser.parse_args()
    print(args)

loaded = pk.load(open('{}_accs_ROI.pkl'.format(args.input_path), 'rb'))
mean_subj_acc_across_folds = loaded.mean(0)
print("Mean Accuracy Across Folds:")
for i in range(len(mean_subj_acc_across_folds)):
    print("{}: {}".format(i, mean_subj_acc_across_folds[i]))