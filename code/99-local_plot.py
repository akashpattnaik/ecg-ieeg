# %%
import os
from os.path import join as ospj
import json
import mne

import matplotlib
matplotlib.use('Qt4Agg')
# %gui qt5
#%%
# config
cwd = os.getcwd()
if cwd[:4] == "/Vol":
    config_file = "../config_local.json"
else:
    config_file = "../config.json"
with open(config_file, "rb") as f:
    config = json.load(f)

#%%
def mne_plot(subj='sub-RID0031', file='ictal_0'):
    edf_file = ospj(
        config['root_path'],
        "data",
        subj,
        "eeg",
        f"{file}.edf"
    )
    raw = mne.io.read_raw_edf(edf_file, verbose=False)
    raw.plot(title=f"{subj} {file}")

    return raw

# %%
raw = mne_plot(
    subj='sub-RID0031',
    file='interictal_0'
)
# %%
