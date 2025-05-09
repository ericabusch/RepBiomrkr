import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class SimulatedCorrelatedData:
    """
    A class for generating and analyzing datasets with controlled correlation structure.
    
    This class can create multiple datasets with specified intra- and inter-dataset correlations,
    based on latent features that are projected to measured features.
    """
    
    def __init__(self, n_datasets=3, n_samples=500, n_features_measured=10, 
                 n_features_latent=0, noise_sigma=0.1, inter_ds_corr=0.3, 
                 intra_ds_corr=0.7, verbose=0):
        """
        Initialize the SimulatedCorrelatedData object with parameters.
        
        Parameters:
        -----------
        n_datasets : int
            Number of datasets to generate
        n_samples : int
            Number of samples in each dataset
        n_features_measured : int
            Number of measured features in each dataset
        n_features_latent : int
            Number of latent features that generate the measured features
        noise_sigma : float
            Standard deviation of the noise
        inter_ds_corr : float
            Target average correlation between datasets
        intra_ds_corr : float
            Target average correlation between features within a dataset
        verbose : int
            Level of verbosity (0=quiet, 1=verbose)
        """
        self.n_datasets = n_datasets
        self.n_samples = n_samples
        self.n_features_measured = n_features_measured
        self.noise_sigma = noise_sigma
        self.inter_ds_corr = inter_ds_corr
        self.intra_ds_corr = intra_ds_corr
        self.verbose = verbose
        if n_features_latent == 0:
            self.n_features_latent = n_features_measured
        else:
            self.n_features_latent = n_features_latent

        # Datasets will be stored here after generation
        self.datasets = None
        self.latent_features = None
        
        # Correlation statistics
        self.avg_intra_latent_corr = None
        self.avg_inter_latent_corr = None
        self.avg_intra_measured_corr = None
        self.avg_inter_measured_corr = None
        self.intra_measured_corrs = None
        self.inter_measured_corrs = None
    
    def _compute_intra_dataset_correlation(self, data_matrix):
        """
        Calculate the average pairwise correlation between features within a dataset.
        
        Parameters:
        -----------
        data_matrix : numpy.ndarray
            Data matrix of shape [n_samples, n_features]
            
        Returns:
        --------
        float
            Average pairwise correlation between features
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_matrix.T)
        
        # Extract off-diagonal elements (exclude self-correlations)
        off_diagonal_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal_corrs = corr_matrix[off_diagonal_mask]
        
        # Return average correlation
        return np.mean(off_diagonal_corrs)
    
    def _compute_average_intra_dataset_correlations(self, datasets):
        """
        Calculate the average intra-dataset correlation across multiple datasets.
        
        Parameters:
        -----------
        datasets : list of numpy.ndarray
            List of data matrices, each of shape [n_samples, n_features]
            
        Returns:
        --------
        float
            Average intra-dataset correlation across all datasets
        list of float
            Individual intra-dataset correlations for each dataset
        """
        intra_ds_corrs = []
        
        for dataset in datasets:
            intra_ds_corrs.append(self._compute_intra_dataset_correlation(dataset))
        
        return np.mean(intra_ds_corrs), intra_ds_corrs
    
    def _compute_inter_dataset_correlation(self, dataset1, dataset2):
        """
        Calculate the average correlation between features from two different datasets.
        
        Parameters:
        -----------
        dataset1 : numpy.ndarray
            First data matrix of shape [n_samples, n_features1]
        dataset2 : numpy.ndarray
            Second data matrix of shape [n_samples, n_features2]
            
        Returns:
        --------
        float
            Average correlation between features from different datasets
        """
        n_features1 = dataset1.shape[1]
        n_features2 = dataset2.shape[1]
        
        # Compute correlations between each pair of features
        correlations = []
        for i in range(n_features1):
            for j in range(n_features2):
                corr = np.corrcoef(dataset1[:, i], dataset2[:, j])[0, 1]
                correlations.append(corr)
        
        return np.mean(correlations)
    
    def _compute_average_inter_dataset_correlations(self, datasets):
        """
        Calculate the average inter-dataset correlation across multiple datasets.
        
        Parameters:
        -----------
        datasets : list of numpy.ndarray
            List of data matrices, each of shape [n_samples, n_features]
            
        Returns:
        --------
        float
            Average inter-dataset correlation across all dataset pairs
        list of float
            Individual inter-dataset correlations for each dataset pair
        """
        n_datasets = len(datasets)
        inter_ds_corrs = []
        
        for i in range(n_datasets):
            for j in range(i+1, n_datasets):
                inter_ds_corrs.append(self._compute_inter_dataset_correlation(datasets[i], datasets[j]))
        
        return np.mean(inter_ds_corrs), inter_ds_corrs
    
    def _compute_latent_correlations(self):
        """
        Calculate the correlation structure of latent features.
        
        Returns:
        --------
        tuple
            (avg_intra_ds_corr, avg_inter_ds_corr)
        """
        if self.latent_features is None:
            raise ValueError("Latent features have not been generated yet. Call generate() first.")
            
        # Calculate correlation matrix of latent features
        actual_corr = np.corrcoef(self.latent_features.T)
        
        # Compute average intra-dataset correlation
        intra_ds_corrs = []
        for ds in range(self.n_datasets):
            start_idx = ds * self.n_features_latent
            end_idx = start_idx + self.n_features_latent
            ds_corr = actual_corr[start_idx:end_idx, start_idx:end_idx]
            
            # Exclude the diagonal (self-correlation is always 1)
            ds_corr_no_diag = ds_corr[~np.eye(ds_corr.shape[0], dtype=bool)]
            intra_ds_corrs.append(np.mean(ds_corr_no_diag))
        
        avg_intra_ds_corr = np.mean(intra_ds_corrs)
        
        # Compute average inter-dataset correlation
        inter_ds_corrs = []
        for ds1 in range(self.n_datasets):
            start_idx1 = ds1 * self.n_features_latent
            end_idx1 = start_idx1 + self.n_features_latent
            
            for ds2 in range(ds1+1, self.n_datasets):
                start_idx2 = ds2 * self.n_features_latent
                end_idx2 = start_idx2 + self.n_features_latent
                
                inter_corr = actual_corr[start_idx1:end_idx1, start_idx2:end_idx2]
                inter_ds_corrs.append(np.mean(inter_corr))
        
        avg_inter_ds_corr = np.mean(inter_ds_corrs)
        
        return avg_intra_ds_corr, avg_inter_ds_corr
    
    def generate(self):
        """
        Generate datasets with controlled correlation structure.
        
        Returns:
        --------
        list of numpy arrays
            Each array has shape [n_samples, n_features_measured]
        """
        if self.verbose:
            print(f"Generating {self.n_datasets} datasets with {self.n_samples} samples each")
            print(f"Each dataset has {self.n_features_measured} measured features derived from {self.n_features_latent} latent features")
            print(f"Target intra-dataset correlation: {self.intra_ds_corr}, inter-dataset correlation: {self.inter_ds_corr}")
        
        # Step 1: Generate correlated latent factors across all datasets
        # We'll create a total of n_datasets * n_features_latent latent factors
        total_latent_features = self.n_datasets * self.n_features_latent
        
        # Create correlation matrix for all latent factors
        corr_matrix = np.ones((total_latent_features, total_latent_features))
        
        # Fill the correlation matrix: intra_ds_corr for features within the same dataset
        # and inter_ds_corr for features between different datasets
        for i in range(total_latent_features):
            for j in range(total_latent_features):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Diagonal elements are 1
                elif i // self.n_features_latent == j // self.n_features_latent:
                    # Same dataset
                    corr_matrix[i, j] = self.intra_ds_corr
                else:
                    # Different datasets
                    corr_matrix[i, j] = self.inter_ds_corr
        
        # Ensure the correlation matrix is positive semi-definite (avoid numerical issues)
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix += -min_eig * np.eye(total_latent_features) * 1.1
        cov_matrix = corr_matrix
        
        if self.verbose:
            print("Generating latent features with the specified correlation structure...")
            
        # Generate random samples with the desired correlation structure
        self.latent_features = np.random.multivariate_normal(
            mean=np.zeros(total_latent_features),
            cov=cov_matrix,
            size=self.n_samples
        )
        
        # Check the correlations of latent features using our helper method
        self.avg_intra_latent_corr, self.avg_inter_latent_corr = self._compute_latent_correlations()
        
        if self.verbose:
            print(f"Average intra-dataset correlation in latent features: {self.avg_intra_latent_corr:.4f} (target: {self.intra_ds_corr:.4f})")
            print(f"Average inter-dataset correlation in latent features: {self.avg_inter_latent_corr:.4f} (target: {self.inter_ds_corr:.4f})")
        
        # Step 2: Generate the measured features for each dataset
        if self.verbose:
            print("\nProjecting latent features to measured features...")
            
        self.datasets = []
        
        for ds in range(self.n_datasets):
            start_idx = ds * self.n_features_latent
            end_idx = start_idx + self.n_features_latent
            
            # Extract latent features for this dataset
            ds_latent = self.latent_features[:, start_idx:end_idx]
            
            if self.n_features_latent < self.n_features_measured:
                # Add noise
                ds_latent += np.random.normal(0, self.noise_sigma, size=(self.n_samples, self.n_features_latent))
    
                # We need to project from latent space to measured space
                # Generate a random projection matrix
                projection_matrix = np.random.normal(0, self.noise_sigma, size=(self.n_features_latent, self.n_features_measured))
                
                # Normalize the projection matrix columns
                projection_matrix /= np.sqrt(np.sum(projection_matrix**2, axis=0))
                
                # Project the latent features to the measured space
                ds_measured = np.dot(ds_latent, projection_matrix)
                            
                # if self.verbose:
                #     print(f"Dataset {ds+1}: Projected {self.n_features_latent} latent features to {self.n_features_measured} measured features with noise sigma={self.noise_sigma}")
            else:
                # When n_features_latent == n_features_measured, we can just take a subset
                # (or all) of the latent features as our measured features
                ds_measured = ds_latent[:, :self.n_features_measured].copy()
                
                # Add noise
                ds_measured += np.random.normal(0, self.noise_sigma, size=(self.n_samples, self.n_features_measured))
                
                # if self.verbose:
                #     print(f"Dataset {ds+1}: Used {self.n_features_measured} out of {self.n_features_latent} latent features as measured features with noise sigma={self.noise_sigma}")
            
            self.datasets.append(ds_measured)
        
        # Step 3: Verify the correlation structure of the measured features
        if self.verbose:
            print("\nVerifying correlation structure of measured features...")
        
        # Compute average intra-dataset correlation of measured features
        self.avg_intra_measured_corr, self.intra_measured_corrs = self._compute_average_intra_dataset_correlations(self.datasets)
        
        if self.verbose:
            print(f"Average intra-dataset correlation in measured features: {self.avg_intra_measured_corr:.4f} (target: {self.intra_ds_corr:.4f})")
        
        # Compute average inter-dataset correlation of measured features
        self.avg_inter_measured_corr, self.inter_measured_corrs = self._compute_average_inter_dataset_correlations(self.datasets)
        
        if self.verbose:
            print(f"Average inter-dataset correlation in measured features: {self.avg_inter_measured_corr:.4f} (target: {self.inter_ds_corr:.4f})")
        
        return self.datasets
    
    def visualize(self, title=None):
        """
        Visualize correlation structure of the datasets.
        
        Parameters:
        -----------
        title : str, optional
            Title for the visualization. If None, a default title will be used.
        """
        if self.datasets is None:
            raise ValueError("No datasets to visualize. Call generate() first.")
        
        if title is None:
            if self.n_features_latent < self.n_features_measured:
                title = f"N_FEATURES_LATENT ({self.n_features_latent}) < N_FEATURES_MEASURED ({self.n_features_measured})"
            else:
                title = f"N_FEATURES_LATENT ({self.n_features_latent}) = N_FEATURES_MEASURED ({self.n_features_measured})"
        
        sns.set(context="notebook", style='white')
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Intra-dataset correlation heatmaps
        ax = axes[0]
        
        # Create a big correlation matrix of all features within all datasets
        all_features = np.hstack(self.datasets)
        corr_matrix = np.corrcoef(all_features.T)
        
        # Add dataset boundaries
        boundaries = [0]
        for ds in self.datasets:
            boundaries.append(boundaries[-1] + ds.shape[1])
        
        # Create heatmap
        sns.heatmap(corr_matrix, ax=ax, cmap="PiYG_r", vmin=-1, vmax=1, 
                    cbar_kws={"label": "Correlation"}, square=True)
        
        # Add lines to separate datasets
        for b in boundaries[1:-1]:
            ax.axhline(b, color='black', linewidth=1)
            ax.axvline(b, color='black', linewidth=1)
        
        ax.set_title(f"Corrs. - {title}")
        
        # Plot 2: Distribution of correlations
        ax = axes[1]
        
        # Collect all correlation values
        intra_corrs = []
        inter_corrs = []
        
        for i, ds1 in enumerate(self.datasets):
            # Intra-dataset correlations
            corr_matrix = np.corrcoef(ds1.T)
            intra_corrs.extend(corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)])
            
            # Inter-dataset correlations
            for j, ds2 in enumerate(self.datasets[i+1:], i+1):
                # Compute correlation between corresponding features
                for f1 in range(ds1.shape[1]):
                    for f2 in range(ds2.shape[1]):
                        corr = np.corrcoef(ds1[:, f1], ds2[:, f2])[0, 1]
                        inter_corrs.append(corr)
        
        # Create kernel density estimates
        sns.kdeplot(intra_corrs, ax=ax, label=f"Intra-ds corrs (mu: {np.mean(intra_corrs):.3f})", color='magenta', lw=4)
        sns.kdeplot(inter_corrs, ax=ax, label=f"Inter-ds corrs (mu: {np.mean(inter_corrs):.3f})", color='green', lw=4)
        
        ax.set_xlabel("Correlation Value")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of Correlations - {title}")
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return

    def get_datasets(self):
        """
        Get the generated datasets.
        
        Returns:
        --------
        list of numpy arrays
            Each array has shape [n_samples, n_features_measured]
        """
        if self.datasets is None:
            raise ValueError("No datasets available. Call generate() first.")
        return self.datasets
        
    def get_correlation_stats(self):
        """
        Get the correlation statistics for the generated datasets.
        
        Returns:
        --------
        dict
            Dictionary containing various correlation statistics
        """
        if self.datasets is None:
            raise ValueError("No datasets available. Call generate() first.")
            
        return {
            "avg_intra_latent_corr": self.avg_intra_latent_corr,
            "avg_inter_latent_corr": self.avg_inter_latent_corr,
            "avg_intra_measured_corr": self.avg_intra_measured_corr,
            "avg_inter_measured_corr": self.avg_inter_measured_corr,
            "intra_measured_corrs": self.intra_measured_corrs,
            "inter_measured_corrs": self.inter_measured_corrs
        }


# Example usage of the class
if __name__ == "__main__":
    # Case 1: N_FEATURES_LATENT < N_FEATURES_MEASURED
    print("\n===== CASE 1: N_FEATURES_LATENT < N_FEATURES_MEASURED =====")
    sim1 = SimulatedCorrelatedData(
        n_datasets=3,
        n_samples=500,
        n_features_measured=10,
        n_features_latent=5,
        noise_sigma=0.1,
        inter_ds_corr=0.3,
        intra_ds_corr=0.7,
        verbose=1
    )
    datasets1 = sim1.generate()
    sim1.visualize()
    
    # Case 2: N_FEATURES_LATENT = N_FEATURES_MEASURED
    print("\n===== CASE 2: N_FEATURES_LATENT = N_FEATURES_MEASURED =====")
    sim2 = SimulatedCorrelatedData(
        n_datasets=3,
        n_samples=500,
        n_features_measured=8,
        n_features_latent=8,
        noise_sigma=0.1,
        inter_ds_corr=0.3,
        intra_ds_corr=0.7,
        verbose=1
    )
    datasets2 = sim2.generate()
    sim2.visualize()
    
    # Get correlation statistics
    stats1 = sim1.get_correlation_stats()
    print("\nCorrelation Statistics for Case 1:")
    for key, value in stats1.items():
        if isinstance(value, list):
            print(f"{key}: {[round(v, 4) for v in value]}")
        else:
            print(f"{key}: {value:.4f}")