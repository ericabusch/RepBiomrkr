import nibabel as nib
import numpy as np
import scipy.stats
from nilearn import datasets, surface
from tqdm import tqdm
import os

def download_subjects_data(n_subjects=10, data_dir=None, verbose=0):
    """
    Download surface data for n_subjects from the NKI enhanced dataset.
    
    Parameters
    ----------
    n_subjects : int, default=10
        Number of subjects to download.
    data_dir : str, optional
        Directory where data should be downloaded.
        
    Returns
    -------
    list of dicts
        Each dict contains paths to the surface data of a subject.
    """
    if verbose: print(f"Downloading data for {n_subjects} subjects...")
    
    # Fetch data for all subjects
    nki_data = datasets.fetch_surf_nki_enhanced(n_subjects=n_subjects, data_dir=data_dir)
    
    # Extract only the first n_subjects
    subjects_data = []
    for i in range(n_subjects):
        subject_data = {
            'func_left': np.nan_to_num(nki_data.func_left[i]),
            'func_right': np.nan_to_num(nki_data.func_right[i])
        }
        subjects_data.append(subject_data)
    
    if verbose: print(f"Downloaded {len(subjects_data)} subjects' data.")
    return subjects_data

def fetch_destrieux_atlas(data_dir=None):
    """
    Fetch the Destrieux atlas.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory where data should be downloaded.
        
    Returns
    -------
    dict
        Dictionary containing the atlas data.
    """
    print("Fetching Destrieux atlas...")
    
    # Fetch the fsaverage5 surface and Destrieux parcellation
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=data_dir)
    
    # Load the Destrieux parcellation
    destrieux = datasets.fetch_atlas_surf_destrieux(data_dir=data_dir)
    
    atlas = {
        'parcellation_left': destrieux['map_left'],
        'parcellation_right': destrieux['map_right'],
        'labels': destrieux['labels'],
        'mesh_left': fsaverage.pial_left,
        'mesh_right': fsaverage.pial_right
    }
    
    print(f"Fetched Destrieux atlas with {len(atlas['labels'])} regions.")
    return atlas

def extract_surface_data(subject_data):
    """
    Extract surface functional data from a subject.
    
    Parameters
    ----------
    subject_data : dict
        Dictionary containing paths to subject's data.
        
    Returns
    -------
    tuple
        (left_data, right_data) numpy arrays of shape (n_vertices, n_timepoints)
    """
    # Load left and right hemisphere data
    left_data = surface.load_surf_data(subject_data['func_left'])
    right_data = surface.load_surf_data(subject_data['func_right'])
    
    # Transpose to shape (n_vertices, n_timepoints)
    if left_data.shape[0] < left_data.shape[1]:
        left_data = left_data.T
    if right_data.shape[0] < right_data.shape[1]:
        right_data = right_data.T
    
    return left_data, right_data

def load_atlas_labels(atlas):
    """
    Load atlas labels for left and right hemispheres.
    
    Parameters
    ----------
    atlas : dict
        Atlas data returned.
    mesh_left : str
        Path to left hemisphere mesh.
    mesh_right : str
        Path to right hemisphere mesh.
        
    Returns
    -------
    tuple
        (left_labels, right_labels) containing label arrays for each hemisphere.
    """
    # Load the fsaverage5 surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Map the atlas from MNI volume to fsaverage5 surface
    labels_left = surface.vol_to_surf(
        atlas.maps, fsaverage.pial_left
    )
    labels_right = surface.vol_to_surf(
        atlas.maps, fsaverage.pial_right
    )
    
    return labels_left, labels_right

def parcellate_hemisphere_data(hemisphere_data, hemisphere_labels):
    """
    Parcellate data for a single hemisphere using the provided labels.
    
    Parameters
    ----------
    hemisphere_data : numpy.ndarray
        Hemisphere data with shape (n_vertices, n_timepoints).
    hemisphere_labels : numpy.ndarray
        Labels array for the hemisphere.
        
    Returns
    -------
    tuple
        (parcellated_data, unique_rois) where:
        - parcellated_data has shape (n_timepoints, n_rois)
        - unique_rois is a list of unique region IDs
    """
    # Get number of timepoints
    n_timepoints = hemisphere_data.shape[1]
    
    # Get unique ROIs (exclude 0 which is typically background)
    unique_rois = np.unique(hemisphere_labels)
    unique_rois = unique_rois[unique_rois != 0]
    n_rois = len(unique_rois)
    
    # Initialize parcellated data
    parcellated_data = np.zeros((n_timepoints, n_rois))
    
    # Maps from ROI value to index in parcellated_data
    roi_to_idx = {roi: i for i, roi in enumerate(unique_rois)}
    
    # Parcellate hemisphere
    for roi in unique_rois:
        roi_mask = hemisphere_labels == roi
        if np.any(roi_mask):
            roi_data = hemisphere_data[roi_mask, :]
            idx = roi_to_idx[roi]
            parcellated_data[:, idx] = np.mean(roi_data, axis=0)
    
    return parcellated_data

def parcellate_data_separate_hemispheres(subject_data, atlas):
    """
    Parcellate subject's data using the Destrieux atlas, keeping hemispheres separate.
    
    Parameters
    ----------
    subject_data : dict
        Dictionary containing paths to subject's data.
    atlas : dict
        Atlas data returned by fetch_destrieux_atlas.
        
    Returns
    -------
    dict
        Dictionary containing parcellated data and ROI information for each hemisphere.
    """
    # Extract surface data
    left_data, right_data = extract_surface_data(subject_data)
    
    # Get labels from left and right hemispheres
    labels_left = np.array(atlas['parcellation_left'])
    labels_right = np.array(atlas['parcellation_right'])
    
    # Parcellate left hemisphere
    parcellated_left = parcellate_hemisphere_data(left_data, labels_left)
    
    # Parcellate right hemisphere
    parcellated_right = parcellate_hemisphere_data(right_data, labels_right)
    
    return parcellated_left, parcellated_right
    


def extract_region_data_separate_hemispheres(subject_data, atlas, region_id):
    """
    Extract multivariate data from a specific region, keeping hemispheres separate.
    
    Parameters
    ----------
    subject_data : dict
        Dictionary containing paths to subject's data.
    atlas : dict
        Atlas data returned by fetch_destrieux_atlas.
    region_id : int, default=16
        Region ID to extract.
        
    Returns
    -------
    dict
        Dictionary containing region data for each hemisphere.
    """
    # Extract surface data
    left_data, right_data = extract_surface_data(subject_data)
    
    # Get labels from left and right hemispheres
    labels_left = np.array(atlas['parcellation_left'])
    labels_right = np.array(atlas['parcellation_right'])
    
    # Extract vertices from the specified region for each hemisphere
    region_mask_left = left_data[labels_left == region_id,:]
    region_mask_right = right_data[labels_right == region_id,:]
    
    return region_mask_left.T, region_mask_right.T
    
def get_region_name(atlas, region_id):
    """
    Get the name of a region by its ID.
    
    Parameters
    ----------
    atlas : dict
        Atlas data returned by fetch_destrieux_atlas.
    region_id : int
        Region ID.
        
    Returns
    -------
    str
        Region name.
    """
    label = atlas['labels'][region_id+1].decode() # account for background
    return label

def process_rsfmri_demo_subjects_destrieux(n_subjects=10, region_id=16, data_dir=None, verbose=0):
    """
    Process data for multiple subjects using the Destrieux atlas,
    keeping hemispheres separate.
    
    Parameters
    ----------
    n_subjects : int, default=10
        Number of subjects to process.
    region_id : int, default=16
        Region ID to extract data from.
    data_dir : str, optional
        Directory where data should be downloaded.
        
    Returns
    -------
    dict
        Dictionary containing processed data for each hemisphere.
    """
    # Download data for subjects
    subjects_data = download_subjects_data(n_subjects=n_subjects, data_dir=data_dir)
    
    # Fetch Destrieux atlas
    atlas = fetch_destrieux_atlas(data_dir=data_dir)
    
    # Check if the specified region exists in the atlas
    region_name = get_region_name(atlas, region_id)
    print(f"Target region: {region_name}")
    
    # Initialize dictionaries to store results
    parcellated_all, region_all = [],[]
    
    # Process each subject
    for i, subject_data in enumerate(tqdm(subjects_data, desc="Processing subjects")):
        # Parcellate data
        parcellated_left, parcellated_right = parcellate_data_separate_hemispheres(subject_data, atlas)
        
        # Extract region data
        region_data_left, region_data_right = extract_region_data_separate_hemispheres(subject_data, atlas, region_id=region_id)
        if verbose: print(f'subject {i+1} parcellated data of shape: {parcellated_left.shape}, {parcellated_right.shape}')
        if verbose: print(f'subject {i+1} parcellated data of shape: {region_data_left.shape}, {region_data_right.shape}')
        # Normalize data
        parcellated_left_norm = np.nan_to_num(scipy.stats.zscore(parcellated_left, axis=0))
        parcellated_right_norm = np.nan_to_num(scipy.stats.zscore(parcellated_right, axis=0))
        region_left_norm = np.nan_to_num(scipy.stats.zscore(region_data_left, axis=0))
        region_right_norm = np.nan_to_num(scipy.stats.zscore(region_data_right, axis=0))
        
        # Stack left and right together
        parcellated_bh = np.hstack([parcellated_left_norm, parcellated_right_norm])
        region_bh = np.hstack([region_left_norm, region_right_norm])
        if verbose: print(f'after stacking: {parcellated_bh.shape}, {region_bh.shape}')
        parcellated_all.append(parcellated_bh)
        region_all.append(region_bh)
        
    return parcellated_all, region_all

if __name__ == "__main__":
	data_dir = os.path.expanduser("~")
	n_subjects=8
	region=4
	parcellated_data, region_data = process_rsfmri_demo_subjects_destrieux(n_subjects=n_subjects, region_id=region, data_dir=data_dir, verbose=1)
	