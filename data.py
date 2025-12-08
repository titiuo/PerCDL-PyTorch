import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "./GaitData"
COLUMN_NAMES = ['TOX', 'TAX', 'TAY', 'RAV', 'RAZ', 'RRY', 'LAV', 'LAZ', 'LRY']

def load_metadata(subject, trial):
    code = f"{subject}-{trial}"
    with open(os.path.join(FOLDER, code + ".json")) as f:
        return json.load(f)

def load_XSens(filename):
    signal = pd.read_csv(filename, delimiter="\t", skiprows=1, header=0)
    signal["PacketCounter"] = (signal["PacketCounter"] - signal["PacketCounter"][0]) / 100
    for axis in ["X","Y","Z"]:
        signal[f"FreeAcc_{axis}"] = signal[f"Acc_{axis}"] - np.mean(signal[f"Acc_{axis}"])
    return signal

def load_signal(subject, trial):
    code = f"{subject}-{trial}"
    base = os.path.join(FOLDER, code)
    signal_lb = load_XSens(base + "_lb.txt")
    signal_lf = load_XSens(base + "_lf.txt")
    signal_rf = load_XSens(base + "_rf.txt")
    t_max = min(len(signal_lb), len(signal_lf), len(signal_rf))
    signal_lb, signal_lf, signal_rf = signal_lb[:t_max], signal_lf[:t_max], signal_rf[:t_max]

    gyr_x = signal_lb['Gyr_X']
    angle_x_full = np.cumsum(gyr_x)/100
    a = np.median(angle_x_full[:len(angle_x_full)//2])
    z = np.median(angle_x_full[len(angle_x_full)//2:])
    angle_x_full = np.sign(z)*(angle_x_full - a)*180/abs(z)

    sig = {
        'Time': signal_lb["PacketCounter"],
        'TOX': angle_x_full,
        'TAX': signal_lb["Acc_X"],
        'TAY': signal_lb["Acc_Y"],
        'RAV': np.sqrt(signal_rf["FreeAcc_X"]**2 + signal_rf["FreeAcc_Y"]**2 + signal_rf["FreeAcc_Z"]**2),
        'RAZ': signal_rf["FreeAcc_Z"],
        'RRY': signal_rf["Gyr_Y"],
        'LAV': np.sqrt(signal_lf["FreeAcc_X"]**2 + signal_lf["FreeAcc_Y"]**2 + signal_lf["FreeAcc_Z"]**2),
        'LAZ': signal_lf["FreeAcc_Z"],
        'LRY': signal_lf["Gyr_Y"]
    }
    return pd.DataFrame(sig)

def show_plot_simple(signal, to_plot, sample_rate=100):
    n_samples = len(signal)
    tt = np.arange(n_samples)/sample_rate
    for dim_name in to_plot:
        plt.figure()
        plt.plot(tt, signal[dim_name])
        plt.xlabel("Time (s)")
        plt.ylabel("m/s²" if dim_name[1]=="A" else "deg/s")
        plt.title(dim_name)
        plt.show()


def build_X(subjects, signal_names, start=500, end=1300,trial=1):
    """
    Initialize X for multiple subjects and selected signals.

    Parameters:
    - subjects: list of int, the subject IDs to include
    - signal_names: list of str, the signal names to include (e.g., ['RAV','RAZ'])
    - trial: int, the trial number to use for each subject

    Returns:
    - X: torch.tensor of shape S x N x P
        S = number of subjects,
        N = number of time samples (truncated to the shortest signal),
        P = number of selected signals
    """
    import torch
    all_signals = []
    min_len = None

    for subj in subjects:
        sig_df = load_signal(subj, trial)
        sig = sig_df[signal_names].values  # shape: N_subject x P
        if min_len is None or len(sig) < min_len:
            min_len = len(sig)
        all_signals.append(sig)

    # truncate all signals to the same length
    truncated = [sig[:min_len] for sig in all_signals]

    # stack into a tensor of shape S x N x P
    X = torch.tensor(np.stack(truncated), dtype=torch.float32)

    X = X[:, start:end, :]
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.std(dim=1, keepdim=True) + 1e-8)

    return X



