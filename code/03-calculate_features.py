"""
Features to calculate:
    - Coherence in each band
    - Relative bandpower in each band

"""

# %%
import os
from os.path import join as ospj
import itertools
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import tools

import mne
from tqdm import tqdm

import matplotlib

# %%
regions = ["AMY", "HPC", "OFC", "CIN", "INS"]
def condense_atlas(labels, networks):
    """
    This function searches the electrode region labels and condenses them to 
    5 limbic regions by taking the mean across all connections
    """
    assert len(labels) == networks.shape[-1]

    search_series = pd.Series(labels).str

    is_left = pd.Series(labels).str.contains('Left')
    is_right = pd.Series(labels).str.contains('Right')
 
    if is_left.sum() > is_right.sum():
        mask = is_left
        laterality = 'Left'
    else:
        mask = is_right
        laterality = 'Right'
        
    unilateral_labels = labels[mask]
    unilateral_networks = networks[:, mask][:, :, mask]

    search_series = pd.Series(unilateral_labels).str
    # order AMY, HPC, OFC, CIN, INS
    amy_idx = search_series.contains('Amygdala')
    hpc_idx = search_series.contains('Hippocampus')
    ofc_idx = search_series.contains('Orbito')
    cin_idx = search_series.contains('cingulate')
    ins_idx = search_series.contains('Insula')

    region_idx = [amy_idx, hpc_idx, ofc_idx, cin_idx, ins_idx]
    region_labels = regions.copy()
    region_labels = [f"{laterality} {i}" for i in region_labels]

    # this code only keeps represented regions, but it makes it harder to compare across patients
    # if np.array([i.sum() == 0 for i in region_idx]).any():
    #     del_idx = np.where([i.sum() == 0 for i in region_idx])[0][0]
    #     del region_idx[del_idx]
    #     del region_labels[del_idx]
    
    n_regions = len(region_idx)
    region_networks = np.zeros((6, n_regions, n_regions))
    for reg1, reg2 in itertools.combinations(range(n_regions), 2):
        region_networks[:, reg1, reg2] = unilateral_networks[:, region_idx[reg1]][:, :, region_idx[reg2]].mean(axis=(1, 2))
        region_networks[:, reg2, reg1] = region_networks[:, reg1, reg2]

    return region_networks, region_labels
#%%
# config
with open("../config.json", "rb") as f:
    config = json.load(f)

# %%
patient_table = pd.read_csv(
    ospj(config['root_path'], "data/metadata/mesolimbic_coverage_patients.csv"),
    index_col=0
    )
sz_table = pd.read_excel(
    ospj(config['root_path'], "data/metadata/seizure_table.xlsx"),
    index_col=0
    )

# %%
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

    coords_table = pd.read_csv(
        ospj(
            config['root_path'],
            "data", 
            subject,
            "iEEG_coordAll_VEP.csv"
            ),
            index_col=[0],
            header=[0, 1]
        )
    electrode_coords_table = coords_table[coords_table['label']['label'].isin(electrodes)]
    electrode_atlas_labels = np.squeeze(electrode_coords_table['region_label'].values)

    all_times = pd.read_excel(
        ospj(
            config['root_path'],
            "data",
            subject,
            "all_clip_times.xlsx"
        )
    )

    for i_times, times in tqdm(all_times.iterrows(), total=len(all_times), desc='clips', position=1, leave=False):
        eeg_fname = ospj(eeg_fldr, f"{times['Name']}.edf")

        raw = mne.io.read_raw_edf(eeg_fname, verbose=False)

        data = raw.get_data()
        fs = raw.info['sfreq']

        coherence_bands = tools.coherence_bands(data, fs)
        electrode_networks = np.array([squareform(i) for i in coherence_bands])

        # convert regions into mesolimbic regions
        region_networks, region_labels = condense_atlas(electrode_atlas_labels, electrode_networks)
        break
    break
# %%
