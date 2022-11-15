# %%
from os.path import join as ospj
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import tools
import itertools

from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
# %%
def apply_squareform(arr):
    new_arr = np.zeros((arr.shape[0], arr.shape[1], n_edges))
    for i_win, win in enumerate(arr):
        for i_band, band in enumerate(win):
            new_arr[i_win, i_band] = squareform(arr[i_win, i_band], checks=False)
    return new_arr

def normalize_edges(arr):
    new_arr = np.zeros((arr.shape[0], arr.shape[1], n_edges))
    for i_win, win in enumerate(arr):
        for i_band, _ in enumerate(win):
            new_arr[i_win, i_band] = (arr[i_win, i_band] - np.nanmean(arr[i_win, i_band])) / np.nanstd(arr[i_win, i_band])
    return new_arr
# %%
seizure_table = pd.read_excel(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/seizure_table.xlsx"
)
patient_table = pd.read_csv(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/mesolimbic_coverage_patients.csv",
    index_col=0
    )
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
regions = ["AMY", "HPC", "OFC", "CIN", "INS"]

n_bands = len(band_names)
# %%
n_edges = sum(1 for i in itertools.combinations(range(len(regions)), 2))
edge_names = [f"{reg1}-{reg2}" for (reg1, reg2) in itertools.combinations(regions, 2)]

# %%
all_edges = None
all_subjs = []
all_t = []

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

    time_var_edges = apply_squareform(time_var_networks)
    print(time_var_edges.shape)
    time_var_edges = normalize_edges(time_var_edges)
    if all_edges is None:
        all_edges = time_var_edges
    else:
        all_edges = np.vstack((all_edges, time_var_edges))

    all_subjs.extend([subject] * len(t_sec))
    all_t.extend(t_sec)
all_subjs = np.array(all_subjs)
all_t = np.array(all_t)
# %%
band = "beta"
band_idx = band_names.index(band)

fig, ax = plt.subplots()
im = ax.matshow(
    all_edges[:, band_idx, :].T,
    aspect='auto'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel(f"Normalized {band} coherence")

ax.set_yticks(range(n_edges))
ax.set_yticklabels(edge_names)
ax.set_xlabel('Samples across patients')

fig.savefig(
    ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
        f"all_limbic_edges.pdf"),
    bbox_inches='tight',
    transparent=True)

# %%
subset = ['AMY-HPC', 'AMY-CIN', 'AMY-INS', 'HPC-CIN', 'HPC-INS', 'CIN-INS']
subset_idx = [edge_names.index(i) for i in subset]
subset_edges = all_edges[:, :, subset_idx]

no_nans = ~np.isnan(subset_edges).any(axis=(1, 2))
subset_edges = subset_edges[no_nans]
subset_subjects = all_subjs[no_nans]
subset_t = all_t[no_nans]

fig, ax = plt.subplots()
im = ax.matshow(
    subset_edges[:, band_idx, :].T,
    aspect='auto'
)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
cax.set_ylabel(f"Normalized {band} coherence")

ax.set_yticks(range(len(subset)))
ax.set_yticklabels(subset)
ax.set_xlabel('Samples across patients')

fig.savefig(
    ospj(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
        f"subset_limbic_edges.pdf"),
    bbox_inches='tight',
    transparent=True)

# %%
############################### MDS ##########################################
for i_band, band in enumerate(band_names):
    band_edges = subset_edges[:, i_band, :]

    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(band_edges)

    # ax.scatter(embeddings[:, 0], embeddings[:, 1])
    fig, ax = plt.subplots()

    for subj in np.unique(subset_subjects):
        subj_idx = np.where(subset_subjects == subj)[0]
        ax.scatter(embeddings[subj_idx, 0], embeddings[subj_idx, 1], label=subj)

    ax.legend()
    ax.set_title(band)
    sns.despine()
    ax.set_xlabel("Embedding #1")
    ax.set_ylabel("Embedding #2")

    fig.savefig(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
            f"mds_{band}_subset_limbic_edges.pdf"),
        bbox_inches='tight',
        transparent=True)

    # break
# %%
