# %%
import os
from os.path import join as ospj
import json

import pandas as pd
import numpy as np

import tools

with open("../config.json", "rb") as f:
    config = json.load(f)

# %%
def change_iEEG_fname(table, hupid, new_fname):
    """
    This function changes the iEEG file name that's in an existing table
    """
    return_table = table.copy(deep=True)

    idx = return_table[patient_table["iEEG_filename"].str.contains(hupid)].index[0]
    return_table.at[idx, 'iEEG_filename'] = new_fname

    return return_table

# %%
patient_table = pd.read_csv(
    ospj(config['root_path'], "data/metadata/patient_table.csv"),
    index_col=0
    )

vep_lut = tools.read_lut("/mnt/leif/littlab/users/pattnaik/VEP_atlas_shared/data/VepFreeSurferColorLut.txt")
dkt_lut = tools.read_lut("/mnt/leif/littlab/tools/freesurfer/FreeSurferColorLUT.txt")

# %%
# Desired regions in VEP
# IDs are [L, R]
amy_idx = [18, 54]
hpc_ant_idx = [71073, 72073]
hpc_pos_idx = [71074, 72074]
hpc_idx = hpc_ant_idx + hpc_pos_idx
ofc_idx = [71002, 72002]
cin_ant_idx = [71018, 72028]
cin_mid_ant_idx = [71029, 72029]
cin_mid_pos_idx = [71030, 72030]
cin_pos_idx = [71031]
cin_idx = cin_ant_idx + cin_mid_ant_idx + cin_mid_pos_idx + cin_pos_idx
ins_lon_idx = [71034, 72034]
ins_bre_idx = [71033, 72033]
ins_idx = ins_lon_idx + ins_bre_idx

all_idx = amy_idx + hpc_idx + ofc_idx + cin_idx + ins_idx
# %%
keep_inds = []
for ind, row in patient_table.iterrows():
    subject = f"sub-RID{row['RID']:04d}"

    if os.path.exists(ospj(config['root_path'], "data", subject, 'iEEG_coordAll_VEP.csv')):
        keep_inds.append(ind)

new_patient_table = patient_table.loc[keep_inds]

#%%
# new_patient_table.to_excel(
#     "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data/metadata/manual_check_patient_table_new.xlsx"
#     )
# %%
patient_regions = []
patients_with_all_idx = []
for ind, row in new_patient_table.iterrows():
    if row['RID'] in [454, 520, 785]:
        continue
    subject = f"sub-RID{row['RID']:04d}"
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
    has_amy = coords_table['region_id'].isin(amy_idx)
    has_hpc = coords_table['region_id'].isin(hpc_idx)
    has_ofc = coords_table['region_id'].isin(ofc_idx)
    has_cin = coords_table['region_id'].isin(cin_idx)
    has_ins = coords_table['region_id'].isin(ins_idx)

    has_other = (has_hpc.any().astype(int) + has_ofc.any().astype(int) + has_cin.any().astype(int) + has_ins.any().astype(int)).values[0]

    has_all = has_amy.any() * has_hpc.any() * has_ofc.any() * has_cin.any() * has_ins.any()
    has_all = has_amy.any().values[0] * (has_other >= 3)
    
    # if a patient has all region coverage, save the electrodes that do for pulling data
    if has_all:
        patients_with_all_idx.append(ind)
        electrodes = np.logical_or.reduce((has_amy, has_hpc, has_ofc, has_cin, has_ins))
        
        labels_to_save = coords_table.loc[electrodes, :]['label']['label'].values
        # print(labels_to_save)
        # print(subject)
        np.save(
            ospj(
                config['root_path'],
                "data",
                subject,
                "mesolimbic_labels.npy"
            ),
            labels_to_save.astype(str),
        )
        # break
        # all_labels
    patient_regions.extend(list(set(np.squeeze(coords_table['region_id'].values))))


patients_with_all = new_patient_table.loc[patients_with_all_idx]
patients_with_all = change_iEEG_fname(patients_with_all, "HUP179", "HUP179_phaseII_D01")
patients_with_all = change_iEEG_fname(patients_with_all, "HUP181", "HUP181_phaseII_D01")
patients_with_all = change_iEEG_fname(patients_with_all, "HUP195", "HUP195_phaseII_D03")
patients_with_all = change_iEEG_fname(patients_with_all, "HUP214", "HUP214_phaseII_D01")
patients_with_all = change_iEEG_fname(patients_with_all, "HUP216", "HUP216_phaseII_D02")

patients_with_all.to_csv(
    ospj(config['root_path'], "data/metadata/mesolimbic_coverage_patients.csv"),
)
# %%
region_df = []

represented_regions =  pd.DataFrame(patient_regions).value_counts()
region_df.append(represented_regions.values)

sorted_regions = np.array([i[0] for i in represented_regions.index.to_numpy()])
region_df.append(sorted_regions)

region_names = vep_lut.loc[sorted_regions]
region_df.append([i[0] for i in region_names.values])

region_df = pd.DataFrame(region_df).T
region_df.columns = ['region_frequency', 'region_number', 'region_label']

region_df.to_csv(
    ospj(config['root_path'], "data/metadata/region_coverage.csv")
)

# %%
