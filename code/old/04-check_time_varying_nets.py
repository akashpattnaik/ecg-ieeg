# %%
from os.path import join as ospj
import numpy as np
import pandas as pd
from tqdm import tqdm

regions = ["AMY", "HPC", "OFC", "CIN", "INS"]
# %%
patient_table = pd.read_csv(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/mesolimbic_coverage_patients.csv",
    index_col=0
    )

region_edges = {}

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

    # all time_var_networks should have 5 nodes
    assert time_var_networks.shape[-1] == len(regions)
    # the number of regions from the file should match actual number of regions
    assert len(region_labels) == len(regions)
    # the number of time samples should be the same as the number of networks
    assert t_sec.shape[0] == time_var_networks.shape[0]

    
# %%
