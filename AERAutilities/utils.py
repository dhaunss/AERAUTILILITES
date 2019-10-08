import os
import sys
import numpy as np


def convert_npz_to_dirs(npzs):
    # Convert npz files to python directories and skip empty entries

    if not isinstance(npzs, list):
        data_dir = {}
        data_dir.update(npzs)
        return data_dir

    return [{key: x[key] for key in x} for x in npzs]


def create_and_jump_to_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


def format_label(args):
    if args.label is not None and args.label != "":
        args.label = "_" + args.label if args.label[0] != "_" else args.label
    else:
        args.label = ""


def merge_dicts(dicts):
    # for python 3.5 or greater use z = {**x for x in dicts}
    z = dicts[0].copy()
    for d in dicts[1:]:
        z.update(d)

    return z


def get_binned_data(x, y, x_bins):
    if y.ndim == 1:
        # binning[0] -> bin entries, binning[1] -> bin center, binning[2] -> mean value of each bin,  binning[3] -> std value of each bin

        # n[0]: Array of entries in each bin, n[1]: bin edges
        n = np.histogram(x, bins=x_bins)
        # sy: Array of sum(y) for each bin
        sy, _ = np.histogram(x, bins=x_bins, weights=y)
        sy2, _ = np.histogram(x, bins=x_bins, weights=y ** 2)

        # checking for bad bins
        y_mean_binned = np.zeros(sy.shape)
        y_std_binned = np.zeros(sy.shape)
        null_mask = np.all([sy != 0, n[0] != 0], axis=0)

        # Binned mean values of y
        y_mean_binned[null_mask] = sy[null_mask] / n[0][null_mask]
        # Binned std values of y
        y_std_binned[null_mask] = np.sqrt(np.abs(sy2[null_mask] / n[0][null_mask] - np.power(y_mean_binned[null_mask], 2.)))
        # bin value (center)
        bins = n[1][:-1] + (n[1][1] - n[1][0]) / 2.

        return n[0], bins, y_mean_binned, y_std_binned

    elif y.ndim == 2:
        # binning[0] -> bin entries, binning[1] -> bin center, binning[2] -> mean value of each bin,  binning[3] -> std value of each bin
        N, bins, y_mean_binned, y_std_binned = [], [], [], []
        for idx, elem in enumerate(y):
            # n[0]: Array of entries in each bin, n[1]: bin edges
            n = np.histogram(x, bins=x_bins)
            # sy: Array of sum(y) for each bin
            sy, _ = np.histogram(x, bins=x_bins, weights=elem)
            sy2, _ = np.histogram(x, bins=x_bins, weights=elem ** 2)

            null_mask = (sy != 0)  # checking for bad bins
            # Binned mean values of y
            y_mean_binned.append(np.where(null_mask, sy / n[0], 0))
            # Binned std values of y
            y_std_binned.append(np.where(null_mask, np.sqrt(np.abs(sy2 / n[0] - np.power(np.where(null_mask, sy / n[0], 0), 2.))), 0))
            # bin value (center)
            bins.append(n[1][:-1] + (n[1][1] - n[1][0]) / 2.)
            N.append(n[0])

        return np.array(N), np.array(bins)[0], np.array(y_mean_binned), np.array(y_std_binned)

    else:
        print("Wrong dimension of y")


def get_fine_xs(arr, N=1000):
    return np.linspace(np.amin(arr), np.amax(arr), N)


def get_closest_value_idx(array, value, max_diff=None):
    idx = np.abs(array - value).argmin()
    if max_diff is None:
        return idx
    else:
        if np.abs(array[idx] - value) < max_diff:
            return idx
        else:
            return None


def get_lim(arr):
    up = np.amax(arr) * 1.1 if np.amax(arr) > 0 else np.amax(arr) * 0.9
    low = np.amin(arr) * 0.9 if np.amin(arr) > 0 else np.amin(arr) * 1.1
    return (low, up)
