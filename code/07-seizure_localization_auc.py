'''
This script determines the AUC for resection localization using node strenght
'''
# %%
from os.path import join as ospj
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from ieeg.auth import Session
from bct import strengths_und
from sklearn.metrics import roc_auc_score
import seaborn as sns
import tools
import itertools

# %%
USR = "pattnaik"
PWD = "pat_ieeglogin.bin"
s = Session(USR, open(PWD, 'r').read())

# %%
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
n_bands = len(band_names)

# %%
patient_table = pd.read_csv(
    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/mesolimbic_coverage_patients.csv",
    index_col=0
    )
seizure_table = pd.read_excel(
        "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/seizure_table.xlsx"
    )

def get_resected_channels(hupid):
    patients, labels, _, resect, _, _, _, _ = tools.pull_patient_localization(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/",
            'patient_localization_final.mat'
            ))
    patient_idx = patients.index(f"HUP{hupid:3d}")
    resect_idx = np.squeeze(resect[patient_idx]).astype(bool)
    
    return np.squeeze(labels[patient_idx])[resect_idx].astype(str)

def get_soz_channels(hupid):
    patients, labels, _, _, _, _, _, soz = tools.pull_patient_localization(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/",
            'patient_localization_final.mat'
            ))
    patient_idx = patients.index(f"HUP{hupid:3d}")
    soz_idx = np.squeeze(soz[patient_idx]).astype(bool)
    
    return np.squeeze(labels[patient_idx])[soz_idx].astype(str)

# %%
for idx, row in tqdm(patient_table.iterrows(), total=patient_table.shape[0]):
    subject = f"sub-RID{row['RID']:04d}"

    # get electrodes from imaging pipeline
    coords_table = pd.read_csv(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data", 
            subject,
            "iEEG_coordAll_VEP.csv"
            ),
            index_col=[0],
        header=[0, 1])
    img_elec_labels = tools.clean_labels(coords_table['label']['label'])

    # get electrodes from iEEG.org
    ds = s.open_dataset(row['iEEG_filename'])
    ieeg_elec_labels = ds.get_channel_labels()
    ieeg_elec_labels = tools.clean_labels(ieeg_elec_labels)

    # select electrodes that are in the imaging pipeline and iEEG.org
    common_elec_labels = np.intersect1d(ieeg_elec_labels, img_elec_labels)

    # save dimensions for use in creating data structures
    n_electrodes = len(common_elec_labels)
    n_edges = sum(1 for _ in itertools.combinations(range(n_electrodes), 2))

    # get the total duration of recording and determine number of iterations based on window size
    total_duration = tools.get_iEEG_duration(USR, PWD, row['iEEG_filename'])
    n_iter = int(total_duration // (20 * 60))

    # set up bandpower, coherence, and time data structures
    coherence = []
    t_sec = []

    errors = 0
    for i_iter in range(n_iter):
        start_time_sec = i_iter * 20 * 60
        end_time_sec = start_time_sec + 60

        electrode_networks = np.load(
                ospj(
                    "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data",
                    subject,
                    f"coherence_start-{start_time_sec}_end-{end_time_sec}.npy"))

        if n_edges != electrode_networks.shape[-1]:
            errors += 1
            continue

        # convert into adjacency matrices
        electrode_networks = np.array([squareform(i) for i in electrode_networks])

        coherence.append(electrode_networks)
        t_sec.append(start_time_sec)
    
    coherence = np.array(coherence)
    t_sec = np.array(t_sec)

    # calculate node strength
    network_stat = np.zeros((coherence.shape[0], n_bands, n_electrodes))
    for i_win in range(len(t_sec)):
        for i_band in range(n_bands):
            network = coherence[i_win, i_band].copy()
            network[np.isnan(network)] = 0
            network_stat[i_win, i_band, :] = strengths_und(network)

    # report AUC or ROC using resected nodes as labels
    resected = get_resected_channels(row['HUPID'])
    resected = tools.clean_labels(resected)
    resected_idx = np.argwhere(np.isin(common_elec_labels, resected)).ravel()

    soz_ch = get_soz_channels(row['HUPID'])
    soz_ch = tools.clean_labels(soz_ch)
    soz_idx = np.argwhere(np.isin(common_elec_labels, soz_ch)).ravel()

    ez_labels = np.zeros((n_electrodes), dtype=int)
    # ez_labels[np.intersect1d(resected_idx, soz_idx)] = 1
    ez_labels[resected_idx] = 1

    n_bootstraps = 20
    rng_seed = 42  # control reproducibility

    aucs = np.zeros((coherence.shape[0], n_bands))
    conf_lower = np.zeros((coherence.shape[0], n_bands))
    conf_upper = np.zeros((coherence.shape[0], n_bands))
    for i_win in range(len(t_sec)):
        for i_band in range(n_bands):
            y_true = ez_labels
            y_pred = network_stat[i_win, i_band]
            aucs[i_win, i_band] = roc_auc_score(y_true, y_pred)

            # confidence interval
            bootstrapped_scores = []
            rng = np.random.RandomState(rng_seed)
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(y_pred), len(y_pred))
                if len(np.unique(y_true[indices])) < 2:
                    # We need at least one positive and one negative sample for ROC AUC
                    # to be defined: reject the sample
                    continue

                score = roc_auc_score(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)

            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()

            # Computing the lower and upper bound of the 90% confidence interval
            # You can change the bounds percentiles to 0.025 and 0.975 to get
            # a 95% confidence interval instead.
            conf_lower[i_win, i_band] = sorted_scores[int(0.05 * len(sorted_scores))]
            conf_upper[i_win, i_band] = sorted_scores[int(0.95 * len(sorted_scores))]

    # get seizure times
    seizures = seizure_table[seizure_table['iEEG_filename'] == row['iEEG_filename']]

    fig, axes = plt.subplots(
        nrows=n_bands, 
        sharex=True,
        figsize=(10, 10))
    for i_band, band in enumerate(band_names):
        axes[i_band].plot(t_sec[4:] / (60*60*24), tools.movmean(aucs[:, i_band], 5))
        axes[i_band].fill_between(
            t_sec[4:] / (60*60*24), 
            tools.movmean(conf_lower[:, i_band], 5),
            tools.movmean(conf_upper[:, i_band], 5),
            color='b',
            alpha=.1
            )
        for st in seizures['start']:
            axes[i_band].axvline(st / (60 * 60 * 24), ls='--', c='r')
        axes[i_band].set_title(band)
        axes[i_band].set_ylim([0, 1])
        axes[i_band].set_xlim([0, np.ceil(t_sec[-1] / (60*60*24))])
    sns.despine(fig=fig)
    plt.tight_layout()

    axes[i_band].set_xlabel('Time in EMU (days)')
    plt.savefig(
        ospj(
            "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/figures",
            subject,
            f"node_strength_auc_roc.pdf"),
        bbox_inches='tight',
        transparent=True
    )
    break
# %%
