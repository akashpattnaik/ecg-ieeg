#%%
from os.path import join as ospj
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tools
from tqdm import tqdm
# %%
USR = "pattnaik"
PWD = "pat_ieeglogin.bin"

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
# %%
patient_table = pd.read_csv(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/mesolimbic_coverage_patients.csv",
    index_col=0
    )

for idx, row in tqdm(patient_table.iterrows(), total=patient_table.shape[0]):
    subject = f"sub-RID{row['RID']:04d}"

    electrode_labels = np.load(
            ospj(
                "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
                subject,
                "mesolimbic_labels.npy"
            ),
        )
    coords_table = pd.read_csv(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data", 
            subject,
            "iEEG_coordAll_VEP.csv"
            ),
            index_col=[0],
            header=[0, 1]
        )
    electrode_coords_table = coords_table[coords_table['label']['label'].isin(electrode_labels)]
    electrode_atlas_labels = np.squeeze(electrode_coords_table['region_label'].values)

    total_duration = tools.get_iEEG_duration(USR, PWD, row['iEEG_filename'])
    n_hours = int((total_duration - 600) // (60 * 60))

    n_clips = n_hours - 12

    all_reg_nets = []
    all_starts = []
    for i_hour in range(12, n_hours):
        start_time_sec = i_hour * 60 * 60
        end_time_sec = start_time_sec + 60

        electrode_networks = np.load(
                ospj(
                    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
                    subject,
                    f"coherence_start-{start_time_sec}_end-{end_time_sec}.npy"))

        n_edges = electrode_networks.shape[-1]
        n_channels = len(electrode_atlas_labels)

        # convert into adjacency matrices
        electrode_networks = np.array([squareform(i) for i in electrode_networks])
        # convert regions into mesolimbic regions
        region_networks, region_labels = condense_atlas(electrode_atlas_labels, electrode_networks)

        all_reg_nets.append(region_networks)
        all_starts.append(start_time_sec)

    time_var_networks = np.array(all_reg_nets)
    t = np.array(all_starts)

    save_fname = ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
        subject,
        "limbic_region_networks.npz"
    )
    np.savez(
        save_fname,
        time_var_networks=time_var_networks,
        t_sec=t,
        region_labels=region_labels
    )
    # break
# %%


# For making figures in 20220926_mesolimbic.pptx
# %%
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.matshow(electrode_networks[3])

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel('Beta coherence')

ax.set_xticks(np.arange(electrode_networks.shape[-1]))
ax.set_yticks(np.arange(electrode_networks.shape[-1]))

ax.set_xticklabels(electrode_atlas_labels, rotation=45, ha='left', rotation_mode='anchor')
ax.set_yticklabels(electrode_atlas_labels, rotation=45, ha='right', rotation_mode='anchor')

fig.savefig(
    ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
        subject,
        f"electrode_network_example.pdf"),
    bbox_inches='tight',
    transparent=True)
# %%
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.matshow(region_networks[3])

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel('Mean beta coherence')

ax.set_xticks(np.arange(region_networks.shape[-1]))
ax.set_yticks(np.arange(region_networks.shape[-1]))

ax.set_xticklabels(region_labels, rotation=45, ha='left', rotation_mode='anchor')
ax.set_yticklabels(region_labels, rotation=45, ha='right', rotation_mode='anchor')
fig.savefig(
    ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
        subject,
        f"region_network_example.pdf"),
    bbox_inches='tight',
    transparent=True)

# %%
