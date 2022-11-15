"""
This script quantiies the regions and overlap across the iEEG cohort
"""
#%%
import os
from os.path import join as ospj
import json

import pandas as pd
import nibabel as nib
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import scipy.spatial as spatial
import numpy as np
from scipy.stats import mode

from pqdm.processes import pqdm

import tools

# %%
# config
with open("../config.json", "rb") as f:
    config = json.load(f)

# %%
# Paths
fs_dir = "/gdrive/public/USERS/nishants/t3_freesurfer/"

# %%
patient_table = pd.read_csv(
    ospj(config['root_path'], "data/metadata/patient_table.csv"),
    index_col=0
    )
rid = 31

#%%
def get_iEEG_coords(rid):
    subject = f"sub-RID{rid:04d}"

    if not os.path.exists(ospj(fs_dir, subject)):
        return None

    iEEG_coords_fname = ospj(fs_dir, subject, "CT", "iEEG_cordAll.csv")
    df = pd.read_csv(iEEG_coords_fname)

    new_cols = ['label'] + ['Vox'] * 3 + ['mm'] * 3 + ['surf'] * 3
    df.columns = pd.MultiIndex.from_arrays([new_cols, df.columns.values])

    return df

def apply_affine(coords, affine):
    M = affine[:3, :3]
    abc = np.flip(affine[:3, 3])

    return np.add(np.matmul(M, coords.T).T, abc)

#%%
def save_coords_as_nifti(rid, coords):
    subject = f"sub-RID{rid:04d}"

    aff = nib.load(ospj(fs_dir, subject, "mri", "aparc.DKTatlas+aseg.mgz")).header.get_vox2ras()

    arr = np.zeros((256, 256, 256), dtype=np.int32)

    # Using a KDTree to find the points within 5 mm
    all_indices = np.array(np.meshgrid(np.arange(256), np.arange(256), np.arange(256))).reshape(3, -1).T
    all_indices_ras = apply_affine(all_indices, aff)

    point_tree = spatial.cKDTree(all_indices_ras)
    neighbors = point_tree.query_ball_point(
            coords,
            2.5)

    for neighbor in neighbors:
        sphere_coords = np.array(np.unravel_index(neighbor, arr.shape), dtype=np.intp)
        arr[sphere_coords[0], sphere_coords[1], sphere_coords[2]] = 1

    image = nib.Nifti1Image(arr, affine=aff)

    fname = ospj(
        config['root_path'],
        "data",
        subject,
        "coordinates.nii"
        )
    if os.path.exists(fname):
        os.remove(fname)
    nib.save(image, fname)
    return image


# %%
def get_regions(rid, coords_table, atlas='DKT'):
    subject = f"sub-RID{rid:04d}"

    if atlas == "DKT":
        atl_name = "aparc.DKTatlas+aseg.mgz"
    elif atlas == "VEP":
        atl_name = "aparc+aseg.vep.mgz"
    else:
        raise ValueError("Correct atlas argument is not given")

    ctx_data = ospj(fs_dir, subject, "mri", atl_name)
    ctx_data = nib.load(ctx_data)
    ctx_arr = ctx_data.get_fdata()

    # Using a KDTree to find the points within 5 mm
    all_indices = np.array(np.meshgrid(np.arange(256), np.arange(256), np.arange(256))).reshape(3, -1).T
    all_indices_ras = apply_affine(all_indices, ctx_data.header.get_vox2ras())

    point_tree = spatial.cKDTree(all_indices_ras)
    neighbors = point_tree.query_ball_point(
            coords_table['mm'][['cord_mm_3', 'cord_mm_2', 'cord_mm_1']].values,
            2.5)

    regions = np.array([mode(ctx_arr[np.unravel_index(neighbor, ctx_arr.shape)]).mode for neighbor in neighbors], dtype=int)
    regions = np.squeeze(regions)
    coords_table['region_id'] = regions

    return coords_table

# %%
def save_coords_table(rid, coords_table, atlas):
    subject = f"sub-RID{rid:04d}"

    save_dir = ospj(config['root_path'], "data")

    if not os.path.exists(ospj(save_dir, subject)):
        os.makedirs(ospj(save_dir, subject))

    coords_table.to_csv(
        ospj(save_dir, subject, f"iEEG_coordAll_{atlas}.csv")
    )

# %%
vep_lut = tools.read_lut("/mnt/leif/littlab/users/pattnaik/VEP_atlas_shared/data/VepFreeSurferColorLut.txt")
dkt_lut = tools.read_lut("/mnt/leif/littlab/tools/freesurfer/FreeSurferColorLUT.txt")

#%%
def pipeline(rid):
    if not os.path.exists(ospj(fs_dir, f"sub-RID{rid:04d}")):
        return

    subject = f"sub-RID{rid:04d}"

    # make data directory
    data_path = ospj(config['root_path'], 'data', subject)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    coords_table = get_iEEG_coords(rid)
    save_coords_as_nifti(rid, coords_table['mm'][['cord_mm_3', 'cord_mm_2', 'cord_mm_1']].values)
    # check CT and MRI against coords

    # '''
    get_regions(rid, coords_table, atlas="VEP")
    coords_table['region_label'] = vep_lut.loc[coords_table['region_id']].values
    save_coords_table(rid, coords_table, "VEP")
    # '''

    '''
    get_regions(rid, coords_table, atlas="DKT")
    coords_table['region_label'] = dkt_lut.loc[coords_table['region_id']].values
    save_coords_table(rid, coords_table, "DKT")
    '''

    return coords_table
result = pqdm(patient_table['RID'], pipeline, n_jobs=8)

# coords_table = pipeline(31)

# %%
