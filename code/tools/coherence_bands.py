import numpy as np
import itertools
from scipy.signal import coherence

bands = [
    [0.5, 4], # delta
    [4, 8], # theta
    [8, 12], # alpha
    [12, 30], # beta
    [30, 80], # gamma
    [0.5, 80] # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)


def coherence_bands(data, fs):
    n_channels, _ = data.shape
    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))
    n_freq = int(fs) + 1

    cohers = np.zeros((n_freq, n_edges))

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        freq, pair_coher = coherence(
            data[ch1, :],
            data[ch2, :],
            fs=fs,
            window='hamming',
            nperseg=fs * 2,
            noverlap=fs
            )

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    cohers = cohers[filter_idx]

    coher_bands = np.empty((N_BANDS, n_edges))
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)

        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    return coher_bands
