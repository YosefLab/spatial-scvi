from typing import Optional, Literal

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
from scipy.stats import mannwhitneyu, ks_2samp


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
    vector1: np.array, vectors_list: np.array, measure: str
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
        set_of_metrics: list[str] = ["distance", "similarity", "latent_overlap"],
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
        z2_versus_z1
        """

        z2_versus_z1 = [self.z1_reference, self.z2_comparison]

        self.set_of_metrics = set_of_metrics

        fov_names = self.adata.obs[self.sample_key].unique().tolist()

        latent_indexes_dict = {}

        for latent_space_key in self.latent_space_keys:
            latent_and_phys_corr = []
            neighborhood_similarity = []

            if latent_space_key in z2_versus_z1:
                latent_indexes_dict[latent_space_key] = []

            # for fov in fov_names:
            for fov in tqdm(fov_names, desc=latent_space_key, colour="green"):
                adata_fov = self.adata[self.adata.obs[self.sample_key] == fov].copy()
                n_cells = len(adata_fov)

                # knn in latent space
                cells_in_the_latent_neighborhood = compute_k_nn(
                    adata_fov,
                    k_nn,
                    latent_space_key,
                    method="sklearn",
                    n_jobs=-1,
                )

                if latent_space_key in z2_versus_z1:
                    latent_indexes_dict[latent_space_key].append(
                        cells_in_the_latent_neighborhood
                    )

                if "distance" in set_of_metrics:
                    xy = adata_fov.obsm[self.spatial_coord_key].values

                    # spatial_coord_of_latent_neighbors = [
                    #     xy[cells_in_the_latent_neighborhood[cell], :]
                    #     for cell in range(n_cells)
                    # ]

                    # spatial_coord_of_latent_neighbors_fov = np.stack(
                    #     spatial_coord_of_latent_neighbors,
                    #     axis=0,
                    # )

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
                        for i in range(len(xy))
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
                # if "similarity" in set_of_metrics:
                #     similarity_parallel = Parallel(n_jobs=-1)(
                #         delayed(compute_similarity)(
                #             ct[i].reshape(1, -1),
                #             ct[cells_in_the_latent_neighborhood[i]],
                #             similarity_metric,
                #         )
                #         for i in range(len(xy))
                #     )

                #     similarity_parallel = np.squeeze(np.array(similarity_parallel))

                #     if reduction[1] == "median":
                #         # make this computation Parallel:
                #         reducted_similarity = np.median(similarity_parallel, axis=-1)

                #     if reduction[1] == "mean":
                #         reducted_similarity = np.mean(similarity_parallel, axis=-1)

                #     if reduction[1] == None:
                #         reducted_similarity = similarity_parallel

                #     neighborhood_similarity.append(reducted_similarity.flatten())

                # similarity between neighborhoods------------------------------
                if "similarity" in set_of_metrics:
                    ct = adata_fov.obsm[self.ct_composition_key].values

                    if reduction[1] == "median":
                        reducted_similarity = [
                            np.median(
                                compute_similarity(
                                    ct[i].reshape(1, -1),
                                    ct[cells_in_the_latent_neighborhood[i]],
                                    similarity_metric,
                                )
                            )
                            for i in range(len(xy))
                        ]

                    if reduction[1] == "mean":
                        reducted_similarity = [
                            np.mean(
                                compute_similarity(
                                    ct[i].reshape(1, -1),
                                    ct[cells_in_the_latent_neighborhood[i]],
                                    similarity_metric,
                                )
                            )
                            for i in range(len(xy))
                        ]

                    if reduction[1] is None:
                        reducted_similarity = [
                            compute_similarity(
                                ct[i].reshape(1, -1),
                                ct[cells_in_the_latent_neighborhood[i]],
                                similarity_metric,
                            )
                            for i in range(len(xy))
                        ]

                    neighborhood_similarity.append(
                        np.array(reducted_similarity).flatten()
                    )

                # neighborhood_similarity[fov] = reducted_similarity

            if "distance" in set_of_metrics:
                self.adata.obs[
                    "latent_and_phys_corr_" + latent_space_key
                ] = np.concatenate(latent_and_phys_corr)
                rprint(
                    "Saved latent and physical correlation in the adata.obs column latent_and_phys_corr_"
                    + latent_space_key
                    + "."
                )

            if "similarity" in set_of_metrics:
                self.adata.obs[
                    "neighborhood_similarity_" + latent_space_key
                ] = np.concatenate(neighborhood_similarity)
                rprint(
                    "Saved compositional neighborhood similarity in the adata.obs column neighborhood_similarity_"
                    + latent_space_key
                    + "."
                )

        if "latent_overlap" in set_of_metrics:
            # check if latent_indexes_dict is empty
            if len(latent_indexes_dict) == 0:
                raise ValueError(
                    "latent_indexes_dict is empty. "
                    "Please provide the keys for the 2 latent spaces you want to compare as a list: z2_versus_z1=[z1, z2]"
                )
            # compute the jaccard index between the two values of the dictionary latent_indexes_dict
            latent_neighbors_1 = np.concatenate(
                latent_indexes_dict[list(latent_indexes_dict.keys())[0]]
            )
            latent_neighbors_2 = np.concatenate(
                latent_indexes_dict[list(latent_indexes_dict.keys())[1]]
            )
            self.adata.obs["latent_overlap"] = [
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

    def plot_distance(self, plot_type: Literal["kde", "ecdf"] = "ecdf"):
        for idx, latent_key in enumerate(self.latent_space_keys):
            if plot_type == "kde":
                sns.kdeplot(
                    data=self.adata.obs["latent_and_phys_corr_" + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

            if plot_type == "ecdf":
                sns.ecdfplot(
                    data=self.adata.obs["latent_and_phys_corr_" + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

        if plot_type == "kde":
            plt.title("Kernel density estimation")
        if plot_type == "ecdf":
            plt.title("Empirical cumulative distribution function")
        plt.xlabel("Median distance in micron")
        plt.legend()  # Add a legend to display the labels
        plt.show()

        return None

    def plot_similarity(self, plot_type: Literal["kde", "ecdf"] = "ecdf"):
        for idx, latent_key in enumerate(self.latent_space_keys):
            if plot_type == "kde":
                sns.kdeplot(
                    data=self.adata.obs["neighborhood_similarity_" + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

            if plot_type == "ecdf":
                sns.ecdfplot(
                    data=self.adata.obs["neighborhood_similarity_" + latent_key],
                    label=latent_key,
                    color=self.color_plots[idx],
                    alpha=0.5,
                )

        if plot_type == "kde":
            plt.title("Kernel density estimation")
        if plot_type == "ecdf":
            plt.title("Empirical cumulative distribution function")
        plt.xlabel("Mean Spearman correlation")
        plt.legend()
        plt.show()

        return None

    def test_distributions(
        self,
        test: Literal["mannwhitneyu", "ks_2samp"] = "mannwhitneyu",
        distribution: Literal["distance", "similarity"] = "distance",
    ):
        if distribution == "distance":
            metric = "latent_and_phys_corr_"
        if distribution == "similarity":
            metric = "neighborhood_similarity_"

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

    def show_clusters(
        self, resolution: float = 0.5, sample_subset: Optional[list[str]] = None
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

        sc.pp.neighbors(self.adata, use_rep=self.z1_reference)
        sc.tl.leiden(self.adata, resolution, key_added="leiden_" + self.z1_reference)

        rprint("Computed leiden clusters for latent space: " + self.z1_reference)

        sc.pp.neighbors(self.adata, use_rep=self.z2_comparison)
        sc.tl.leiden(self.adata, resolution, key_added="leiden_" + self.z2_comparison)

        rprint("Computed leiden clusters for latent space: " + self.z2_comparison)

        sample_names = (
            self.adata.obs[self.sample_key].unique().tolist()
            if not sample_subset
            else sample_subset
        )

        for sample in sample_names:
            sc.pl.spatial(
                self.adata[self.adata.obs[self.sample_key] == sample],
                spot_size=40,
                color=[
                    self.label_key,
                    "leiden_" + self.z1_reference,
                    "leiden_" + self.z2_comparison,
                ],
                ncols=3,
                frameon=False,
                title=[
                    sample + "_" + self.label_key,
                    "leiden_" + self.z1_reference,
                    "leiden_" + self.z2_comparison,
                ],
            )

        return None
