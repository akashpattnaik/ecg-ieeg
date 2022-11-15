# %%
from os.path import join as ospj
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import tools
# %%
seizure_table = pd.read_excel(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/seizure_table.xlsx"
)
patient_table = pd.read_csv(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/mesolimbic_coverage_patients.csv",
    index_col=0
    )
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]

#%%
reg1 = "AMY"; reg2 = "HPC"
band = "beta"
band_idx = band_names.index(band)

# %%
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
ind = 0
for idx, row in tqdm(patient_table.iterrows(), total=patient_table.shape[0]):
    subject = f"sub-RID{row['RID']:04d}"

    # ret = animate_time_var_nets(subject)
    # print(ret)

    load_fname = ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
        subject,
        "limbic_region_networks.npz"
    )
    npzfile = np.load(load_fname)
    time_var_networks = npzfile['time_var_networks']
    t_sec = npzfile['t_sec']
    region_labels = npzfile['region_labels']
    del npzfile

    seizures = seizure_table[seizure_table['iEEG_filename'] == row['iEEG_filename']]

    reg1_idx = np.where(pd.Series(region_labels).str.contains(reg1))[0]
    if len(reg1_idx) == 0:
        continue
    reg1_idx = reg1_idx[0]

    reg2_idx = np.where(pd.Series(region_labels).str.contains(reg2))[0]
    if len(reg2_idx) == 0:
        continue
    reg2_idx = reg2_idx[0]

    if np.isnan(time_var_networks[:, band_idx, reg1_idx, reg2_idx]).all():
        continue
    axes.flat[ind].plot(t_sec / (60 * 60 * 24), time_var_networks[:, band_idx, reg1_idx, reg2_idx])
    for st in seizures['start']:
        axes.flat[ind].axvline(st / (60 * 60 * 24), ls='--', c='r')
    # ax.set_ylim([0, 1])
    axes.flat[ind].set_title(f"{subject}")
    sns.despine(ax=axes.flat[ind])
    # ax.set_xlabel("Time in EMU (days)")
    # ax.set_ylabel(f"{reg1}-{reg2} {band} coherence")
    ind += 1
    # break
plt.tight_layout()
plt.savefig(
    ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
        f"{reg1}-{reg2}_sz_times.pdf"),
    bbox_inches='tight',
    transparent=True)

# %%
