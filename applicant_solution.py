import json
import os
import gdown
import numpy as np
import sys
from scipy.io import loadmat
from task_and_baseline import baseline, build_task_helpers, make_bandpass, CENTER, BW, MODEL_LAGS, MODEL_SUBSET, shift_signal, shifted_window

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
if not os.path.exists(downloaded_file):
    print("Downloading challenge.mat...")
    gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)
bp = helpers["score_filter"]
score_kernel = make_bandpass(CENTER, BW, Fs)

def deconvolve_wiener(target, kernel, ridge=1e-4):
    n = len(target)
    m = len(kernel)
    n_fft = 1 << (n + m - 2).bit_length()
    kernel_padded = np.pad(kernel, (0, n_fft - m))
    kernel_padded = np.roll(kernel_padded, -(m // 2))
    kernel_fft = np.fft.fft(kernel_padded)
    target_fft = np.fft.fft(target, n_fft)
    reg = ridge * np.max(np.abs(kernel_fft) ** 2)
    estimate_fft = target_fft * np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + reg)
    return np.fft.ifft(estimate_fft)[:n]

def fit_tx_extended(target_signal, terms):
    x_train = np.column_stack([
        shifted_window(term, lag, MODEL_SUBSET.start, MODEL_SUBSET.stop)
        for term in terms
        for lag in MODEL_LAGS
    ])
    gram = x_train.conj().T @ x_train + 1e-6 * np.eye(x_train.shape[1])
    
    pred = np.zeros_like(target_signal)
    for ch in range(4):
        y_block = bp(target_signal[:, ch])[MODEL_SUBSET]
        coef = np.linalg.solve(gram, x_train.conj().T @ y_block)
        coef = coef.reshape(len(terms), len(MODEL_LAGS))
        
        ch_pred = np.zeros(N, dtype=np.complex128)
        for t_idx, term in enumerate(terms):
            for l_idx, lag in enumerate(MODEL_LAGS):
                ch_pred += coef[t_idx, l_idx] * shift_signal(term, lag)
        pred[:, ch] = ch_pred
    return pred

def your_canceller(tx_n, rx):
    extra_pairs = [(2, 5), (5, 2), (4, 1), (1, 4), (4, 3), (3, 4), (4, 5), (5, 4)]
    all_terms = [bp(tx_n[:, 0] ** 2 * tx_n[:, 1].conj()), 
                 bp(tx_n[:, 1] ** 2 * tx_n[:, 0].conj()),
                 bp(tx_n[:, 0] ** 2 * tx_n[:, 3].conj()),
                 bp(tx_n[:, 3] ** 2 * tx_n[:, 0].conj()),
                 bp(tx_n[:, 1] ** 2 * tx_n[:, 2].conj()),
                 bp(tx_n[:, 2] ** 2 * tx_n[:, 1].conj()),
                 bp(tx_n[:, 3] ** 2 * tx_n[:, 2].conj()),
                 bp(tx_n[:, 2] ** 2 * tx_n[:, 3].conj()),
                 bp(tx_n[:, 0] ** 2 * tx_n[:, 5].conj()),
                 bp(tx_n[:, 5] ** 2 * tx_n[:, 0].conj())]
    
    for i, j in extra_pairs:
        all_terms.append(bp(tx_n[:, i] ** 2 * tx_n[:, j].conj()))

    tx_pred = np.zeros_like(rx)
    rank1_pred = np.zeros_like(rx)
    
    for i in range(2):
        tx_pred = fit_tx_extended(rx - rank1_pred, all_terms)
        
        resid = rx - tx_pred
        resid_band = np.column_stack([bp(resid[:, ch]) for ch in range(4)])
        
        cov = resid_band.conj().T @ resid_band / N
        _, vecs = np.linalg.eigh(cov)
        steering = vecs[:, -1]
        
        shared_band = resid_band @ steering
        shared_broad = deconvolve_wiener(shared_band, score_kernel, ridge=1e-4)
        
        denom = np.vdot(shared_band, shared_band) + 1e-30
        for ch in range(4):
            weight = np.vdot(shared_band, resid_band[:, ch]) / denom
            rank1_pred[:, ch] = 0.98 * weight * shared_broad

    return rx - (tx_pred + rank1_pred)

if __name__ == "__main__":
    print("\n=== Baseline ===")
    baseline_reds, baseline_avg = helpers["score"](
        rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
    )

    print("=== Your Solution ===")
    yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

    results = {
        "baseline": {
            "per_channel_db": baseline_reds,
            "average_db": baseline_avg,
        },
        "yours": {
            "per_channel_db": yours_reds,
            "average_db": yours_avg,
        },
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to results.json")
