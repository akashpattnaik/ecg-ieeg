import numpy as np

def artifact_removal(data, fs, threshold=20):
    win_size = int(1 * fs)
    ind_overlap = np.reshape(np.arange(np.floor(data.shape[0] / win_size)*win_size, dtype=int), (-1, int(win_size)))

    # mask indices with nan values
    artifacts = np.isnan(data).values

    for win_inds in ind_overlap:
        is_disconnected = np.sum(np.abs(data.iloc[win_inds]), axis=0) < 1/12
        artifacts[win_inds, :] = is_disconnected

        is_noise = np.sqrt(np.sum(np.power(np.diff(data.iloc[win_inds], axis=0), 2), axis=0)) > 15000
        artifacts[win_inds, :] = is_noise

    return artifacts
