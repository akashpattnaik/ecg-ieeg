from ieeg.auth import Session

def get_iEEG_duration(username, password_bin_file, iEEG_filename):
    '''
    Returns duration in seconds
    '''

    pwd = open(password_bin_file, 'r').read()
    s = Session(username, pwd)
    ds = s.open_dataset(iEEG_filename)
    return ds.get_time_series_details(ds.ch_labels[0]).duration / 1e6


