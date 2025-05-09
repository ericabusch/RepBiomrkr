import numpy as np
import os, sys, glob, pickle
import load_demo_rsfmri
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from load_demo_rsfmri import process_rsfmri_demo_subjects_destrieux

def load_rsfmri(n_subjects, save_to="./sample_data", seed_region=16, verbose=1):
    if save_to:
        outpath = f'{save_to}/demo_rsfmri_data_{n_subjects}_subjects_seed_{seed_region}.pkl'
        if os.path.exists(outpath):
            with open(outpath, 'rb') as f:
                loaded_data = pickle.load(f)
            return loaded_data
        
    parcellated, region = process_rsfmri_demo_subjects_destrieux(n_subjects=n_subjects, region_id=seed_region, data_dir=None, verbose=verbose)
    loaded_data = {'parcellated_all': parcellated, 'region_all':region}
    if save_to:
        with open(outpath, 'wb') as f:
            pickle.dump(loaded_data, f)
    return loaded_data

def load_sherlock_movie():
    return np.load(f'./sample_data/sherlock_movie_data_early_visual_roi.npy')

def load_simulated_data_manifold_example():
    loaded_data = {'input_data':np.load('./sample_data/input_data.npy'), 
                  'exogenous_data':np.load('./sample_data/exogenous_features.npy'),
                  'scores':np.load('./sample_data/scores.npy')}
    return loaded_data

def compute_connectivity_matrix(m1, m2):
    cnx = 1-cdist(m1, m2, 'correlation')
    return cnx
    

def plot_3d_datasets(datasets, dataset_names=None, feature_names=None, title='',
                     figsize=(6,4), marker='o', alpha=0.7, s=20, 
                     view_angles=(30, 45), save_path=None):
    """
    Plot multiple datasets in a 3D scatterplot with different colors for each dataset.
    
    Parameters:
    -----------
    datasets : list of numpy arrays
        List containing N datasets, where each dataset is a numpy array of shape (S, 3)
        with S samples and 3 features.
    
    dataset_names : list of str, optional
        Names for each dataset to be shown in the legend. If None, datasets will be labeled
        as "Participant 1", "Participant 2", etc.
        
    feature_names : list of str, optional
        Names for the three features (x, y, z axes). If None, features will be labeled
        as "Feature 1", "Feature 2", "Feature 3".
    
    title : str, optional
        Title for the plot
        
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    
    marker : str, optional
        Marker style for the scatter plot.
    
    alpha : float, optional
        Transparency of the markers (0 to 1).
    
    s : int, optional
        Marker size.
    
    view_angles : tuple, optional
        Initial view angles (elevation, azimuth) in degrees.
    
    save_path : str, optional
        If provided, saves the figure to the specified path.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The figure and axis objects for further customization if needed.
    """
    # Validate inputs
    sns.set_context('paper')

    if not isinstance(datasets, list):
        raise TypeError("datasets must be a list of numpy arrays")
    
    for i, dataset in enumerate(datasets):
        if dataset.shape[1] != 3:
            raise ValueError(f"Participant {i+1} has {dataset.shape[1]} features, expected 3 features")
    
    # Set default names if not provided
    if dataset_names is None:
        dataset_names = [f"Participant {i+1}" for i in range(len(datasets))]
    elif len(dataset_names) != len(datasets):
        raise ValueError("Number of dataset names must match number of datasets")
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(3)]
    elif len(feature_names) != 3:
        raise ValueError("Must provide exactly 3 feature names")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a color cycle for the datasets
    colors = list(mcolors.TABLEAU_COLORS)  # Using Tableau colors
    # If we have more datasets than colors, we'll cycle through the colors
    if len(datasets) > len(colors):
        colors = colors * (len(datasets) // len(colors) + 1)
    
    # Plot each dataset with a different color
    for i, dataset in enumerate(datasets):
        ax.scatter(
            dataset[:, 0], dataset[:, 1], dataset[:, 2],
            c=colors[i], marker=marker, alpha=alpha, s=s,
            label=dataset_names[i]
        )
    
    # Set labels and title
    ax.set_xlabel(feature_names[0])
    ax.set_xticks([])
    ax.set_ylabel(feature_names[1])
    ax.set_yticks([])
    ax.set_zlabel(feature_names[2])
    ax.set_zticks([])
    ax.set_title(f"{title}")
    
    # Add legend
    ax.legend()
    
    # Set the viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Add grid
    ax.grid(False)
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_3d_trajectories(trajectories, trajectory_names=None, feature_names=None, title='',
                         figsize=(6,4), linewidth=2, alpha=0.8, linestyle='-',
                         markers=None, marker_size=20, marker_frequency=None,
                         view_angles=(30, 45), save_path=None):
    """
    Plot multiple trajectories (lines) in 3D space, each with a different color.
    
    Parameters:
    -----------
    trajectories : list of numpy arrays
        List containing N trajectories, where each trajectory is a numpy array of shape (S, 3)
        with S samples and 3 features, representing points along a path in 3D space.
    
    trajectory_names : list of str, optional
        Names for each trajectory to be shown in the legend. If None, trajectories will be labeled
        as "Participant 1", "Participant 2", etc.
        
    feature_names : list of str, optional
        Names for the three features (x, y, z axes). If None, features will be labeled
        as "Feature 1", "Feature 2", "Feature 3".
    
    title : str, optional
        title of plot
    
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    
    linewidth : float, optional
        Width of the trajectory lines.
    
    alpha : float, optional
        Transparency of the lines (0 to 1).
    
    linestyle : str, optional
        Style of the trajectory lines ('-', '--', '-.', ':', etc.).
    
    markers : str or list, optional
        Marker style for points along the trajectory. If None, no markers are shown.
        If a string, the same marker is used for all trajectories.
        If a list, each trajectory can have a different marker.
    
    marker_size : int, optional
        Size of markers if markers are shown.
    
    marker_frequency : int, optional
        If provided, show markers every marker_frequency points along the trajectory.
        If None and markers are provided, markers are shown at every point.
    
    view_angles : tuple, optional
        Initial view angles (elevation, azimuth) in degrees.
    
    save_path : str, optional
        If provided, saves the figure to the specified path.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The figure and axis objects for further customization if needed.
    """
    # Validate inputs
    if not isinstance(trajectories, list):
        raise TypeError("trajectories must be a list of numpy arrays")
    
    for i, trajectory in enumerate(trajectories):
        if trajectory.shape[1] != 3:
            raise ValueError(f"Participant {i+1} has {trajectory.shape[1]} features, expected 3 features")
    
    # Set default names if not provided
    if trajectory_names is None:
        trajectory_names = [f"Participant {i+1}" for i in range(len(trajectories))]
    elif len(trajectory_names) != len(trajectories):
        raise ValueError("Number of trajectory names must match number of trajectories")
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(3)]
    elif len(feature_names) != 3:
        raise ValueError("Must provide exactly 3 feature names")
    
    # Handle markers
    if markers is not None and not isinstance(markers, list):
        markers = [markers] * len(trajectories)
    elif markers is not None and len(markers) != len(trajectories):
        markers = markers * (len(trajectories) // len(markers) + 1)
        markers = markers[:len(trajectories)]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a color cycle for the trajectories
    colors = list(mcolors.TABLEAU_COLORS)  # Using Tableau colors
    # If we have more trajectories than colors, we'll cycle through the colors
    if len(trajectories) > len(colors):
        colors = colors * (len(trajectories) // len(colors) + 1)
    
    # Plot each trajectory with a different color
    for i, trajectory in enumerate(trajectories):
        # Plot the line
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=colors[i], linewidth=linewidth, alpha=alpha, linestyle=linestyle,
            label=trajectory_names[i]
        )
        
        # Add markers if requested
        if markers is not None:
            if marker_frequency is not None:
                # Show markers at specified frequency
                idx = np.arange(0, len(trajectory), marker_frequency)
                marker_data = trajectory[idx]
            else:
                # Show markers at every point
                marker_data = trajectory
                
            ax.scatter(
                marker_data[:, 0], marker_data[:, 1], marker_data[:, 2],
                c=colors[i], marker=markers[i], s=marker_size
            )
    
    # Set labels and title
    ax.set_xlabel(feature_names[0])
    ax.set_xticks([])
    ax.set_ylabel(feature_names[1])
    ax.set_yticks([])
    ax.set_zlabel(feature_names[2])
    ax.set_zticks([])
    ax.set_title(f"{title}")
    
    # Add legend
    ax.legend()
    
    # Set the viewing angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Add grid
    ax.grid(False)
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def load_sherlock_labels():
    return np.load('sample_data/sherlock_indoor_scene_labels.npy')