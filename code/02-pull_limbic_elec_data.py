'''
This script pulls clips from the patients who have mesolimbic network coverage
and saves them in EDF format

Seizures must be at least 6 hours away from each other. 
All non-ictal clips are 5 minutes.

Interictal clips 1-20:
    TODO:
    1 day after implant
    2 hours after subclinical seizures
    6 hours after focal seizures
    12 hours after generalized seizures
    2 hours before all seizures

    for now, 6 hours after seizures, 2 hours before seizures

    TODO: During wake times

Preictal clip 1: 4 hours before EEC
Preictal clip 2: 2 hours before EEC
Preictal clip 3: 1 hours before EEC
Preictal clip 4: 30 minutes before EEC
Preictal clip 5: 15 minutes before EEC

Ictal clip

Postictal clip 1: 5 minutes after end
'''
# %%
import os
from os.path import join as ospj
import json
import numpy as np
import pandas as pd
import tools

import mne
from tqdm import tqdm

import matplotlib
# matplotlib.use('Qt5Agg')
|#%%
# config
with open("../config.json", "rb") as f:
    config = json.load(f)

#%%
n_preictal = 5
n_interictal = 20

preictal_intervals = np.array([4 * 60 * 60, 2 * 60 * 60, 60 * 60, 30 * 60, 15 * 60])
preictal_interval_names = ["4hr", "2hr", "1hr", "30min", "15min"]

n_preictal = len(preictal_intervals)
# %%
patient_table = pd.read_csv(
    ospj(config['root_path'], "data/metadata/mesolimbic_coverage_patients.csv"),
    index_col=0
    )
sz_table = pd.read_excel(
    ospj(config['root_path'], "data/metadata/seizure_table.xlsx"),
    index_col=0
    )

#%%
rng = np.random.default_rng(2022)

for i_pt, pt in tqdm(patient_table.iterrows(), total=len(patient_table), desc='patients', position=0):
    subject = f"sub-RID{pt['RID']:04d}"

    eeg_fldr = ospj(
        config['root_path'],
        "data",
        subject,
        'eeg'
    )
    if not os.path.exists(eeg_fldr):
        os.makedirs(eeg_fldr)

    electrodes = np.load(
        ospj(
            config['root_path'],
            "data",
            subject,
            "mesolimbic_labels.npy"
        )
    )

    ictal_times = sz_table[sz_table.index == pt['iEEG_filename']].to_numpy()
    ictal_times[:, 1] += (np.ceil(np.diff(ictal_times)) - np.diff(ictal_times))[:, 0]
    ictal_times = ictal_times[ictal_times[:, 0].argsort()] 

    total_duration = tools.get_iEEG_duration(config['username'], config['pwd_bin'], pt['iEEG_filename'])

    interictal_samples = []
    rng = np.random.RandomState(2022)
    while len(interictal_samples) < 20:
        temp = rng.uniform(0, total_duration)

        is_out_of_sz = np.all(~np.logical_and((temp > ictal_times[:, 0] - (2 * 60 * 60)), (temp < ictal_times[:, 1] + (6 * 60 * 60))))

        interictal_so_far = np.array(interictal_samples)
        is_out_of_interictal = np.all(~np.logical_and((temp > interictal_so_far), (temp < interictal_so_far + (5 * 60))))

        if is_out_of_sz and is_out_of_interictal:
            interictal_samples.append(temp)

    interictal_samples.sort()
    interictal_times = np.zeros((n_interictal, 2))
    interictal_times[:, 0] = interictal_samples
    interictal_times[:, 1] = interictal_times[:, 0] + (5 * 60)

    # keep only seizures that are 6 hours apart
    far_apart_sz_mask = np.insert(np.diff(ictal_times[:, 0]) > (6 * 60 * 60), 0, [True])
    ictal_times = ictal_times[far_apart_sz_mask]
    n_seizures = ictal_times.shape[0]


    preictal_times = np.expand_dims(ictal_times[:, 0], 1) - np.expand_dims(preictal_intervals, 1).T
    preictal_times = np.repeat(preictal_times[:, :, np.newaxis], 2, axis=-1)
    preictal_times[:, :, 1] = preictal_times[:, :, 0] + 60 * 5
    assert(preictal_times.shape == (n_seizures, n_preictal, 2))


    postictal_times = np.repeat(ictal_times[:, 1, np.newaxis], 2, axis=1)
    postictal_times[:, 0] += 5 * 60
    postictal_times[:, 1] += 10 * 60
    assert(postictal_times.shape == (n_seizures, 2))


    # put all the times together
    all_times = []
    for ind, times in enumerate(ictal_times):
        all_times.append(list(times) + [f"ictal_{ind}"])

    for i_sz, pre_seizure in enumerate(preictal_times):
        for i_period, times in enumerate(pre_seizure):
            all_times.append(list(times) + [f"preictal_{i_sz}_{preictal_interval_names[i_period]}"])

    for ind, times in enumerate(postictal_times):
        all_times.append(list(times) + [f"postictal_{ind}"])

    for ind, times in enumerate(interictal_times):
        all_times.append(list(times) + [f"interictal_{ind}"])

    all_times = pd.DataFrame(all_times, columns=["Start (sec)", "End (sec)", "Name"])
    all_times.sort_values(by="Start (sec)", inplace=True, ignore_index=True)
    all_times.to_excel(
        ospj(
            config['root_path'],
            "data",
            subject,
            "all_clip_times.xlsx"
        )
    )

    for i_times, times in tqdm(all_times.iterrows(), total=len(all_times), desc='clips', position=1, leave=False):
        eeg_fname = ospj(eeg_fldr, f"{times['Name']}.edf")
        # if os.path.exists(eeg_fname):
        #     continue

        data, fs = tools.get_iEEG_data(
            config['username'],
            config['pwd_bin'],
            pt['iEEG_filename'],
            times['Start (sec)']*1e6,
            times['End (sec)']*1e6,
            select_electrodes=list(electrodes)
        )
        
        # artifact rejection
        artifacts = tools.artifact_removal(data, fs)
        # set channels with more than 20% artifacts to nan
        remv_ch_idx = artifacts.sum(axis=0) / artifacts.shape[0] > 0.2
        remv_ch = data.columns[remv_ch_idx]
        data_clean = data.copy()
        data_clean[remv_ch] = np.nan
        
        # remove 60Hz noise and bandpass
        data_filt = tools.notch_filter(data_clean, fs)
        data_filt = tools.bandpass(data_filt, fs)

        if np.all(np.isnan(data_filt)):
            continue
        
        # save as edf
        info = mne.create_info(
            ch_names=list(data_clean.columns),
            sfreq=fs,
            ch_types="eeg"
        )

        edf_obj = mne.io.RawArray(
            data=data_filt.T / 1e6, # mne needs data in volts
            info=info,
            verbose=False
        )


        mne.export.export_raw(
            fname=eeg_fname,
            raw=edf_obj,
            overwrite=True
        )


# %%
'''
# Pull coordinates, eventually transition to BIDS coordinate format
label_file_path = ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
        "sub-RID0031",
        "imaging_electrode_labels.csv")
coord_file_path = "/gdrive/public/USERS/pattnaik/subjects/sub-RID0031/CT/electrodes_inMRImm.txt"

# get electrode coordinates from imaging pipeline
coords_only = np.loadtxt(coord_file_path, skiprows=1)
labels = pd.read_csv(label_file_path, index_col=0)

coords = labels
coords[['x', 'y', 'z']] = list(coords_only)
coords['label'] = tools.clean_labels(coords['label'])

# we want electrodes that are in iEEG.org and electrode localization
ieeg_elec_labels = data.columns.values
img_elec_labels = coords['label'].values

common_elec_labels = np.intersect1d(ieeg_elec_labels, img_elec_labels)
data = data[list(common_elec_labels)]
coords = coords.loc[coords['label'].isin(common_elec_labels)]
'''
# %%