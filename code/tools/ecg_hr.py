import sys
import os
sys.path.insert(1, os.path.join("./../tools/"))
from helpers import *
#%%



def lineLength(signal):
    return np.sum(abs(np.diff(signal, axis=0 )))

def outliarDetection(signal, threshold=10000):
    ll = lineLength(signal)
    #
    if ll > threshold:
        return 
    else:
        return signal

def remove_std(signal, std_mul=1.5):
    std_rem = signal.copy()
    mean = np.mean(std_rem)
    std = np.std(std_rem)
    std_rem[(std_rem<(mean+std_mul*std)) & (std_rem>(mean-std_mul*std))]=mean
    return std_rem

# %%


def calc_hrv_param(data, win_s, overlap_s):
    df = data.copy()

    start =df.index[0]
    stop = df.index[-1]

    hrv_windows = calc_windows(start, stop, win_s, overlap_s)

    hrv_df = pd.DataFrame()
    for i in tqdm(range(len(hrv_windows))):
        data = df[(df.index >=hrv_windows[i][0]) & (df.index <hrv_windows[i][1])]['heartRate']
        
        if len(data.dropna())>=2:

            nni_ms = 60000/data.values

            #dropping large nni values > 3000 ms
            nni_ms = nni_ms[nni_ms < 3000]



            # print(len(nni_ms))
            # td_results = td.time_domain(nni=nni_ms, rpeaks =rpeaks_ms, sampling_rate=fs,plot=False)
            parameters = ['rmssd', 'hf']

            hrv_params = np.ones([len(data), len(parameters)])*np.nan
            hrv_params[-1,0] =  int(td.rmssd(nni=nni_ms)[0])
            
            # 
            try:
                freq = fd.welch_psd(nni=nni_ms, show=False, mode='dev')[0]['fft_abs']
                hrv_params[-1,1] =  int(freq[2])
            except : 
                hrv_params[-1,1] =  0
            
            hrv_df = hrv_df.append(pd.DataFrame(data={'heartRate' : data.values, 'rmssd' : hrv_params[:,0], 'hf' : hrv_params[:,1] }, index=data.index))

        else:
            continue
    return hrv_df    



# %%
def get_ecg(subject_name, start_flag=0, end_flag=0 ):
# %%
    # subject_name = "LB3_005_phaseII_D01"
    # start_flag = 90129
    # end_flag = 90729


    temp_list= json.load(open("./../tools/subject_metadata_jp.json"))
    metadata = pd.DataFrame(temp_list).T

    #shift data to realtime
    delta = pd.to_datetime(metadata.loc[subject_name]['Start day/time']).tz_convert(tz)-pd.to_datetime('1970-1-1').tz_localize(tz)


    #Configuring getting  ieeg data
    portal_name = metadata.loc[subject_name]['portal_ID']
    print("Getting data for %s" %(subject_name))
    print("Getting data for %s from IEEG" %(portal_name))

    with open(pwd_bin_path, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(portal_name)


    # start_flag = 0
    # end_flag = 5000
    if start_flag!='0':
        start_time_sec = int(start_flag)
    else:
        start_time_sec = 0

    if end_flag!='0':
        end_time_sec = int(end_flag)
    else:
        end_time_sec = (ds.end_time-ds.start_time)/1e6
    print("time stamps retrived from IEEG")
    print("Getting data for %d to %d" %(start_time_sec, end_time_sec))

    data_pull_min=5
    ecg_channels = metadata.loc[subject_name]['ECG Electrode']
    ecg_mul = int(metadata.loc[subject_name]['ECG Multiplier'])

    clip_duration_sec =  end_time_sec - start_time_sec
    clip_duration_min = clip_duration_sec / 60

    # how many data_pull_min minute data pulls are there?
    n_iter = int(np.ceil(clip_duration_min / data_pull_min))

# %%
    
    hr_win = pd.DataFrame()
    for i in tqdm(range(n_iter)):

        start_usec = start_time_sec * 1e6 + i * (5 * 60 * 1e6)
        if i == n_iter - 1:
            end_usec = end_time_sec * 1e6
        else:
            end_usec = start_usec + (5 * 60 * 1e6)
        data, fs = get_iEEG_data(username, pwd_bin_path, portal_name, start_usec, end_usec, select_electrodes=ecg_channels)

        time = np.linspace(start_usec, end_usec, len(data), endpoint=False)
        data.index = pd.to_datetime(time, unit='us')

        # format resulting data into pandas DataFrame
        signal_filt = pd.DataFrame(columns=ecg_channels, index=data.index)


        hr_df = pd.DataFrame()
        for n,c in enumerate(ecg_channels):
            #remove baseline wander
            temp_ = hp.remove_baseline_wander(data.iloc[:,n]*ecg_mul, fs)


            #bandpas filter
            order = int(0.3 * fs)
            signal_filt[c], _, _ = biosppy.tools.filter_signal(signal=temp_,ftype='FIR',band='bandpass',order=order,frequency=[3,25],sampling_rate=fs)


            # work on 5s samples
            temp_std= pd.DataFrame()
            for i in range(int(len(signal_filt)/int(5*fs))):

                win_data = signal_filt[i*5*int(fs):(i+1)*5*int(fs)][c]

                #remove noise data < 1.5 std

                std_rem = remove_std(win_data, 1.5)
                temp_std = temp_std.append(pd.DataFrame(std_rem))

            try :
            
                out = ecg.ecg(signal=temp_std[c], sampling_rate=fs, show=False)
                # ecg_filtered = out[1]
                # rpeaks = out[2]
                heart_rate_ts = out[-2]
                heart_rate = out[-1]
                idx = (heart_rate_ts*fs).astype(int)

                temp_ = pd.DataFrame(data={c : heart_rate}, index=temp_std.index[idx])
                hr_df = hr_df.append(temp_)
            except:
                    continue  
        
        #shift time to real time
        hr_df.index = pd.to_datetime(hr_df.index).tz_localize(tz)+ delta

    
        # resample to 1Hz
        hr_rs = hr_df.resample("1s").mean()   

        #take the min in each 1s window
        hr_min = hr_rs.min(axis=1)

        hr_win = hr_win.append(pd.DataFrame(hr_min.dropna()))

    # %%
    #smoothen
    hr_sm = pd.DataFrame({'heartRate' : savgol_filter(hr_win[0], 11,3)}, index=hr_win.index)
    hr_sm['heartRate'] = hr_sm['heartRate'].apply(lambda x : round(x,1))

    # %%
    #calc hrv parameters
    hrv_df = calc_hrv_param(hr_sm, 300,0)

    # %%
    hrv_file = subject_name + "_"+str(start_time_sec) + "_"+str(end_time_sec)+ "_hrv.h5"

    # %%
    hrv_df.to_hdf("./../stress_analysis/data/"+hrv_file, key="hr", mode='w')

    # %%

if __name__ == "__main__":
    get_ecg(sys.argv[1],sys.argv[2], sys.argv[3] )
# %%
