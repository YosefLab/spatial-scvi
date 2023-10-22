from typing import Optional, Literal, NamedTuple

import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from tqdm import tqdm
from anndata import AnnData
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scvi.nearest_neighbors import pynndescent

import pingouin as pg
from scipy.stats import mannwhitneyu, ks_2samp, entropy

from dataclasses import dataclass


def get_values_row(indices, indptr, i):
    """
    Function giving the column index of the non-zero entries in each row

    Inputs:
    i=row index
    indices = csr_matrix indices attribute
    indptr = csr_matrix indptr attribute
    """
    return indices[indptr[i] : indptr[i + 1]]


# ---------------------Spatial latent space--------------------------------------------------------------------------------


def compute_similarity(
    vector1: np.array,
    vectors_list: np.array,
    measure: Literal["euclidean", "pearson", "spearman", "cosine"] = "euclidean",
) -> np.array:
    """Compute similarity between one vector and a set of vectors.

    Args:
        vector1 (np.array): _description_
        vector2 (np.array): _description_
        measure (str): should be one of 'euclidean', 'pearson','spearman', or 'cosine'.

    Raises
    ------
        ValueError: If the measure is not one of 'euclidean', 'pearson','spearman', or 'cosine'.

    Returns
    -------
        float: Similarity value.
    """
    if measure == "euclidean":
        return np.linalg.norm(vector1 - vectors_list, axis=1)
    elif measure == "pearson":
        return np.corrcoef(vector1, vectors_list)[0, 1:]
    elif measure == "spearman":
        spearman_corr, _ = spearmanr(vector1, vectors_list, axis=1)
        return spearman_corr[0, 1:]

    elif measure == "cosine":
        dot_product = np.dot(vector1, vectors_list.T)
        norms = np.linalg.norm(vector1) * np.linalg.norm(vectors_list, axis=1)
        return dot_product / norms
    else:
        raise ValueError(
            "Invalid similarity measure. Choose from 'euclidean', 'pearson', 'spearman', or 'cosine'."
        )


def jaccard_score(neighbors1: np.array, neighbors2: np.array) -> float:
    intersection = np.intersect1d(neighbors1, neighbors2)
    union = np.union1d(neighbors1, neighbors2)

    if len(union) == 0:
        jaccard = 0.0
    else:
        jaccard = len(intersection) / len(union)

    return jaccard


def compute_k_nn(
    adata: AnnData,
    k: int,
    latent_space_key: str,
    method: Literal["sklearn", "pynn"] = "pynn",
    n_jobs: int = -1,
) -> csr_matrix:
    """
    Compute the k nearest neighbors in the latent space.

    Parameters
    ----------
    adata
        Annotated data matrix.
    k
        The number of neighbors to consider in the spatial analysis.
    latent_space_key
        The key in adata.obsm that contains the latent space.

    Returns
    -------

    """
    if method == "sklearn":
        # Create a NearestNeighbors object
        knn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)

        # Fit the kNN model to the points
        knn.fit(adata.obsm[latent_space_key])

        # Find the indices of the kNN for each point
        _, latent_k_nn_indices = knn.kneighbors(adata.obsm[latent_space_key])

    if method == "pynn":
        latent_k_nn_indices = pynndescent(
            X=adata.obsm[latent_space_key], n_neighbors=k, random_state=0, n_jobs=n_jobs
        )

    return latent_k_nn_indices


@dataclass
class ClusterStats:
    mean: pd.DataFrame
    std: pd.DataFrame


class _KEYS_SPATIAL(NamedTuple):
    DISTANCE_KEY: str = "latent_and_phys_corr_"
    SIMILARITY_KEY: str = "neighborhood_similarity_"
    LATENT_OVERLAP_KEY: str = "latent_overlap"
    CLUSTER_KEY: str = "leiden_"


class _METRIC_TITLE(NamedTuple):
    DISTANCE_KEY: str = "distance in micron"
    SIMILARITY_KEY: str = "correlation"
    LATENT_OVERLAP_KEY: str = "Jaccard index"


SET_OF_METRICS = ["distance", "similarity", "latent_overlap", "cluster_stats"]
KEYS_SPATIAL = _KEYS_SPATIAL()
METRIC_TITLE = _METRIC_TITLE()


class SpatialAnalysis:
    def __init__(
        self,
        adata: AnnData,
        label_key: str,
        sample_key: str,
        latent_space_keys: list[str],
        spatial_coord_key: str,
        ct_composition_key: str,
        sample_subset: Optional[list[str]] = None,
        z1_reference: Optional[str] = None,
        z2_comparison: Optional[str] = None,
    ):
        self.label_key = label_key
        self.sample_key = sample_key
        self.latent_space_keys = latent_space_keys
        self.spatial_coord_key = spatial_coord_key
        self.ct_composition_key = ct_composition_key
        self.adata = adata
        self.sample_subset = sample_subset

        self.z1_reference = z1_reference
        self.z2_comparison = z2_comparison

        self.leiden_keys = None

        self.color_plots = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    def compute_metrics(
        self,
        k_nn: int,
        set_of_metrics: list[str] = SET_OF_METRICS,
        similarity_metric: str = "spearman",
        reduction: list[str] = ["median", "mean"],
    ) -> None:
        """
        Compute the metrics for the spatial analysis.

        Parameters
        ----------
        sample_key
            The key in adata.obs that contains the sample names.
        k_nn
            The number of neighbors to consider in the spatial analysis.
        sample_subset
            The subset of samples to consider in the spatial analysis.
        set_of_metrics
            The set of metrics to compute. The options are: "distance", "similarity", "latent_overlap".
        similarity_metric
            The similarity metric to use. The options are: "spearman", "pearson", "jaccard".
        reduction
            The reduction to apply to the similarity metric. The options are: "median", "mean", None.
        """

        self.set_of_metrics = set_of_metrics
        self.similarity_metric = similarity_metric
        self.reduction = reduction

        z2_versus_z1 = [self.z1_reference, self.z2_comparison]
        fov_names = self.adata.obs[self.sample_key].unique().tolist()

        latent_indexes_dict = {}

        # Loop over latent spaces:
        # for latent_space_key in tqdm(
        #     self.latent_space_keys, desc="latent", colour="green"
        # ):
        for latent_space_key in self.latent_space_keys:
            latent_and_phys_corr = []
            neighborhood_similarity = []

            if latent_space_key in z2_versus_z1:
                latent_indexes_dict[latent_space_key] = []

            # Loop over fovs:
            for fov in tqdm(fov_names, desc=latent_space_key, colour="blue"):
                adata_fov = self.adata[self.adata.obs[self.sample_key] == fov].copy()
                n_cells = len(adata_fov)

                # knn in latent space
                cells_in_the_latent_neighborhood = compute_k_nn(
                    adata_fov,
                    k_nn,
                    latent_space_key,
                    method="pynn",
                    n_jobs=-1,
                )

                if latent_space_key in z2_versus_z1:
                    latent_indexes_dict[latent_space_key].append(
                        cells_in_the_latent_neighborhood
                    )

                if "distance" in set_of_metrics:
                    xy = adata_fov.obsm[self.spatial_coord_key].values

                    spatial_coord_of_latent_neighbors_fov = xy[
                        cells_in_the_latent_neighborhood
                    ]

                    # median---------------------------------------------------------------------------
                    # make this compuation Parallel:
                    dists_parallel = Parallel(n_jobs=-1)(
                        delayed(cdist)(
                            xy[i].reshape(1, 2),
                            spatial_coord_of_latent_neighbors_fov[i],
                        )
                        for i in range(n_cells)
                    )

                    dists_parallel = np.squeeze(np.array(dists_parallel))

                    if reduction[0] == "median":
                        reducted_dists = np.median(dists_parallel, axis=-1)

                    if reduction[0] == "mean":
                        reducted_dists = np.mean(dists_parallel, axis=-1)

                    if reduction[0] is None:
                        reducted_dists = dists_parallel

                    latent_and_phys_corr.append(reducted_dists.flatten())

                # similarity between neighborhoods------------------------------
                if "similarity" in set_of_metrics:
                    ct = adata_fov.obsm[self.ct_composition_key].values
                    similarity_parallel = Parallel(n_jobs=-1)(
                        delayed(compute_similarity)(
                            ct[i],
                            ct[cells_in_the_latent_neighborhood[i]],
                            similarity_metric,
                        )
                        for i in range(n_cells)
                    )

                    similarity_parallel = np.squeeze(np.array(similarity_parallel))

                    if reduction[1] == "median":
                        reducted_similarity = np.median(similarity_parallel, axis=-1)

                    if reduction[1] == "mean":
                        reducted_similarity = np.mean(similarity_parallel, axis=-1)

                    if reduction[1] == None:
                        reducted_similarity = similarity_parallel

                    neighborhood_similarity.append(reducted_similarity.flatten())

            if "distance" in set_of_metrics:
                self.adata.obs[
                    KEYS_SPATIAL.DISTANCE_KEY + latent_space_key
                ] = np.concatenate(latent_and_phys_corr)
                rprint(
                    "Saved latent and physical correlation in the adata.obs column latent_and_phys_corr_"
                    + latent_space_key
                    + "."
                )

            if "similarity" in set_of_metrics:
                self.adata.obs[
                    KEYS_SPATIAL.SIMILARITY_KEY + latent_space_key
                ] = np.concatenate(neighborhood_similarity)
                rprint(
                    "Saved compositional neighborhood similarity in the adata.obs column neighborhood_similarity_"
                    + latent_space_key
                    + "."
                )

        if "cluster_stats" in set_of_metrics:
            if self.leiden_keys is None:
                raise ValueError(
                    "Please run the method leiden_clusters before running cluster_stats."
                )

            ct = self.adata.obsm[self.ct_composition_key]

            self.cluster_stats = {}

            for leiden_key in self.leiden_keys:
                leiden_clusters = self.adata.obs[
                    KEYS_SPATIAL.CLUSTER_KEY + leiden_key
                ].unique()

                df_mean = pd.DataFrame(columns=ct.columns, index=leiden_clusters)
                df_std = pd.DataFrame(columns=ct.columns, index=leiden_clusters)

                for cluster in leiden_clusters:
                    ct_cluster = ct[
                        self.adata.obs[KEYS_SPATIAL.CLUSTER_KEY + leiden_key] == cluster
                    ]

                    df_mean.loc[cluster] = ct_cluster.mean(axis=0)
                    df_std.loc[cluster] = ct_cluster.std(axis=0)

                self.cluster_stats[leiden_key] = ClusterStats(df_mean, df_std)

        if "latent_overlap" in set_of_metrics:
            # check if latent_indexes_dict is empty
            if len(latent_indexes_dict) == 0:
                raise ValueError(
                    "latent_indexes_dict is empty. "
                    "Please provide the keys for the 2 latent spaces you want to compare with z1_reference and z2_comparison."
                )
            # compute the jaccard index between the two values of the dictionary latent_indexes_dict
            latent_neighbors_1 = np.concatenate(
                latent_indexes_dict[list(latent_indexes_dict.keys())[0]]
            )
            latent_neighbors_2 = np.concatenate(
                latent_indexes_dict[list(latent_indexes_dict.keys())[1]]
            )
            self.adata.obs[KEYS_SPATIAL.LATENT_OVERLAP_KEY] = [
                jaccard_score(latent_neighbors_1[i], latent_neighbors_2[i])
                for i in range(len(latent_neighbors_1))
            ]
            rprint(
                "The latent spaces overlap is saved in the.obs column: latent_overlap."
            )

        return None

    def get_latent_overlap(
        self,
    ):
        if "latent_overlap" not in self.adata.obs.columns:
            raise ValueError(
                'Please run compute_spatial_metrics with the argument set_of_metrics=["latent_overlap"]'
            )

        # compute the avergae of adata.obs['latent_overlap'] by categories in adata.obs['sample']
        cell_types = self.adata.obs[self.label_key].unique().tolist()
        n_cell_types = len(cell_types)
        int_to_cell_types = {i: cell_types[i] for i in range(n_cell_types)}

        # Create a console object
        console = Console()

        # Create a table
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Cell type")
        table.add_column("Average Jaccard")

        for i in range(n_cell_types):
            type = int_to_cell_types[i]
            mean_jaccard = np.mean(
                self.adata[self.adata.obs[self.label_key] == type].obs[
                    "latent_overlap"
                ],
                axis=0,
            )

            # round the mean jaccard to 2 decimals
            mean_jaccard = round(mean_jaccard, 2)

            # Add rows to the table
            table.add_row(type, str(mean_jaccard))

        # Print the table
        console.print(table)

        return None

    def plot_metrics(
        self,
        metric: Literal[*SET_OF_METRICS],
        plot_type: Literal["kde", "ecdf"] = "ecdf",
    ):
        if metric == "distance":
            metric_key = KEYS_SPATIAL.DISTANCE_KEY
            metric_title = self.reduction[0] + " " + METRIC_TITLE.DISTANCE_KEY
        if metric == "similarity":
            metric_key = KEYS_SPATIAL.SIMILARITY_KEY
            metric_title = (
                self.reduction[1]
                + " "
                + self.similarity_metric
                + " "
                + METRIC_TITLE.SIMILARITY_KEY
            )

        for idx, latent_key in enumerate(self.latent_space_keys):
            if plot_type == "kde":
                sns.kdeplot(
                    data=self.adata.obs[metric_key + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

            if plot_type == "ecdf":
                sns.ecdfplot(
                    data=self.adata.obs[metric_key + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

        if plot_type == "kde":
            plt.title("Kernel density estimation")
        if plot_type == "ecdf":
            plt.title("Empirical cumulative distribution function")
        plt.xlabel(metric_title)
        plt.legend()  # Add a legend to display the labels
        plt.show()

        return None

    def test_distributions(
        self,
        test: Literal["mannwhitneyu", "ks_2samp"] = "mannwhitneyu",
        distribution: Literal["distance", "similarity"] = "distance",
    ):
        if distribution == "distance":
            metric = KEYS_SPATIAL.DISTANCE_KEY
        if distribution == "similarity":
            metric = KEYS_SPATIAL.SIMILARITY_KEY

        x = self.adata.obs[metric + self.z1_reference]
        p_values = []
        mean_values = []
        median_values = []

        for latent_key in self.latent_space_keys:
            y = self.adata.obs[metric + latent_key]
            mean_values.append(np.mean(y).round(2))
            median_values.append(np.median(y).round(2))
            U1, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            p_values.append(p)

        reject, p_values_corr = pg.multicomp(p_values, method="fdr_bh")

        # Create a console instance
        console = Console()

        # Create a table
        table = Table(title=distribution, show_header=True, header_style="bold green")

        table.add_column("Model")
        table.add_column("Mean " + distribution)
        table.add_column("Median " + distribution)
        table.add_column("p-value")
        table.add_column("p-value corrected")

        for idx, latent_key in enumerate(self.latent_space_keys):
            table.add_row(
                latent_key,
                str(mean_values[idx]),
                str(median_values[idx]),
                str(p_values[idx]),
                str(p_values_corr[idx]),
            )

        # Print the table
        console.print(table)

        return None

    def leiden_clusters(
        self,
        resolution: float = 0.5,
        leiden_keys: Optional[str] = None,
        sample_subset: Optional[list[str]] = None,
        plot: bool = True,
    ):
        """
        Show the clusters in the spatial coordinates.

        Parameters
        ----------
        resolution
            The resolution for the clustering.
        sample_subset
            The subset of samples to consider in the spatial analysis.
        """

        if leiden_keys is None:
            leiden_keys = self.latent_space_keys

        self.leiden_keys = leiden_keys

        for key in tqdm(leiden_keys, desc="Leiden", colour="green"):
            if key not in self.adata.obs.columns:
                sc.pp.neighbors(self.adata, use_rep=key)
                key_to_add = KEYS_SPATIAL.CLUSTER_KEY + key
                sc.tl.leiden(self.adata, resolution, key_added=key_to_add)
                rprint("Saved leiden clusters in " + key_to_add)

        sample_names = (
            self.adata.obs[self.sample_key].unique().tolist()
            if not sample_subset
            else sample_subset
        )

        if plot:
            for sample in sample_names:
                sc.pl.spatial(
                    self.adata[self.adata.obs[self.sample_key] == sample],
                    spot_size=40,
                    color=[
                        self.label_key,
                        leiden_keys[0],
                        leiden_keys[1],
                    ],
                    ncols=3,
                    frameon=False,
                    title=[
                        sample + "_" + self.label_key,
                        leiden_keys[0],
                        leiden_keys[1],
                    ],
                )

        return None

    def compare_neighborhoods(
        self,
        comparison_keys: str,
        reference_key: Optional[str] = None,
        save_metric_key: str = "entropy_",
    ):
        if reference_key is None:
            reference_key = self.ct_composition_key

        neighborhood_ref = self.adata.obsm[reference_key]

        for key in comparison_keys:
            neighborhood_pred = pd.DataFrame(
                self.adata.obsm[key], columns=neighborhood_ref.columns
            )
            # then loop over cell types:
            entropy_dict = {}
            for ct in neighborhood_ref.columns:
                # compute the entropy for each cell type:
                true_neighbors_ct = neighborhood_ref[ct]
                pred_neighbors_ct = neighborhood_pred[ct]
                entropy_ct = [
                    entropy(
                        true_neighbors_ct[self.adata.obs.cell_type == i],
                        pred_neighbors_ct[self.adata.obs.cell_type == i],
                    )
                    for i in neighborhood_ref.columns
                ]
                entropy_dict[ct] = entropy_ct
            self.adata.obsm[save_metric_key + key] = pd.DataFrame(
                entropy_dict,
                columns=neighborhood_ref.columns,
                index=neighborhood_ref.columns,
            )
