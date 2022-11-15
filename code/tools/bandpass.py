from scipy.signal import butter, filtfilt
import numpy as np

def bandpass(data, fs) -> np.array:
    """_summary_

    Args:
        data (_type_): _description_
        fs (_type_): _description_

    Returns:
        np.array: _description_
    """
    # bandpass between 1 and 120Hz
    # TODO: change the arguments so that 1 and 120 are default
    # TODO: add causal function argument
    # TODO: add optional argument for order
    bandpass_b, bandpass_a = butter(3, [1, 120], btype='bandpass', fs=fs)
    data_filt = filtfilt(bandpass_b, bandpass_a, data, axis=0)

    return data_filt