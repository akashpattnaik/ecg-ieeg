from scipy.signal import iirnotch, filtfilt
import numpy as np

def notch_filter(data, fs) -> np.array:
    """_summary_

    Args:
        data (_type_): _description_
        fs (_type_): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    b, a = iirnotch(60, 30, fs)
    data_filt = filtfilt(b, a, data, axis=0)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt