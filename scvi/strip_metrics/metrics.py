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
from scipy.stats import mannwhitneyu, ks_2samp, entropy, pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
)

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
    measure: Literal["euclidean", "pearson", "spearman", "cosine", "kl"] = "euclidean",
) -> np.array:
    """
    Compute the similarity between one vector and one set of vectors.

    Parameters
    ----------
    vector1
        The first vector (1D).
    vectors_list
        The second set of vectors (2D).
    measure
        The similarity measure to use.

    Returns
    -------
    np.array
        The similarity between vector1 and each row of vectors_list.
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

    elif measure == "kl":
        return kl_divergence_set(vector1, vectors_list)
    else:
        raise ValueError(
            "Invalid similarity measure. Please choose between 'euclidean', 'pearson', 'spearman', 'cosine' or 'kl'."
        )


def jaccard_score(neighbors1: np.array, neighbors2: np.array) -> float:
    intersection = np.intersect1d(neighbors1, neighbors2)
    union = np.union1d(neighbors1, neighbors2)

    if len(union) == 0:
        jaccard = 0.0
    else:
        jaccard = len(intersection) / len(union)

    return jaccard


def kl_divergence_set(p: np.array, q_set: np.array, epsilon: float = 1e-7) -> np.array:
    """
    Compute the KL divergence between two sets of probabilities.

    Parameters
    ----------
    p
        The first set of probabilities (1D).
    q_set
        The second set of probabilities (2D).

    Returns
    -------
    float
        The KL divergence between p and each row of q_set.
    """

    # Add epsilon to both p and q_set
    p += epsilon
    q_set += epsilon

    return np.sum(p * np.log(p / q_set), axis=1)


def log_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


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
    SIMILARITY_KEY: str = ""
    LATENT_OVERLAP_KEY: str = "Jaccard index"


SET_OF_METRICS = [
    "distance",
    "similarity",
    "latent_overlap",
    "cluster_stats",
]
KEYS_SPATIAL = _KEYS_SPATIAL()
METRIC_TITLE = _METRIC_TITLE()


color_plots = [
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
    "teal",
    "lavender",
    "maroon",
    "gold",
    "indigo",
]


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
        train_indices: Optional[list[int]] = None,
        validation_indices: Optional[list[int]] = None,
        ##########
        set_of_metrics: list[str] = SET_OF_METRICS,
        similarity_metric: str = "spearman",
        reduction: list[str] = ["median", "mean"],
    ):
        self.adata = adata
        self.sample_subset = sample_subset
        self.train_indices = train_indices
        self.validation_indices = validation_indices

        self.label_key = label_key
        self.sample_key = sample_key
        self.latent_space_keys = latent_space_keys
        self.spatial_coord_key = spatial_coord_key
        self.ct_composition_key = ct_composition_key
        self.z1_reference = z1_reference
        self.leiden_keys = None

        set = np.empty(len(self.adata), dtype="object")
        set[self.validation_indices] = "validation"
        set[self.train_indices] = "train"
        self.adata.obs["set"] = set

        self.set_of_metrics = set_of_metrics
        self.similarity_metric = similarity_metric
        self.reduction = reduction

    def leiden_clusters(
        self,
        resolution: float = 0.5,
        leiden_keys: Optional[str] = None,
        sample_subset: Optional[list[str]] = None,
        plot: bool = True,
    ):
        """
        Compute the leiden clusters.

        Parameters
        ----------
        resolution
            The resolution parameter of the leiden algorithm.
        leiden_keys
            The keys in adata.obsm that contain the latent spaces to use for the leiden clustering.
        sample_subset
            The subset of samples to use for plotting the leiden clustering.
        plot
            Whether to plot the leiden clusters.
        """

        if leiden_keys is None:
            leiden_keys = self.latent_space_keys

        self.leiden_keys = leiden_keys

        keys_added = []

        for key in tqdm(leiden_keys, desc="Leiden", colour="green"):
            if key not in self.adata.obs.columns:
                sc.pp.neighbors(self.adata, use_rep=key)
                key_to_add = KEYS_SPATIAL.CLUSTER_KEY + key
                sc.tl.leiden(self.adata, resolution, key_added=key_to_add)
                # rprint("Saved leiden clusters in " + key_to_add)
                keys_added.append(key_to_add)

        sample_names = (
            self.adata.obs[self.sample_key].unique().tolist()
            if not sample_subset
            else sample_subset
        )

        rprint("Leiden clusters saved in: ", keys_added)

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

    def compute_metrics(
        self,
        k_nn_range: list[int] = [50],
    ) -> None:
        """
        Compute the spatial metrics.

        Parameters
        ----------
        k_nn
            The number of latent neighbors to consider in the spatial analysis.
        """

        self.k_nn_range = k_nn_range

        fov_names = self.adata.obs[self.sample_key].unique().tolist()

        latent_indexes_dict = {}

        # Loop over latent spaces:
        for latent_space_key in self.latent_space_keys:
            latent_and_phys_corr = {k: [] for k in k_nn_range}
            neighborhood_similarity = {k: [] for k in k_nn_range}

            # Loop over fovs:
            for fov in tqdm(fov_names, desc=latent_space_key, colour="blue"):
                adata_fov = self.adata[self.adata.obs[self.sample_key] == fov].copy()
                n_cells = len(adata_fov)

                # knn in latent space
                cells_in_the_latent_neighborhood = compute_k_nn(
                    adata_fov,
                    k_nn_range[-1],
                    latent_space_key,
                    method="pynn",
                    n_jobs=-1,
                )

                for k in k_nn_range:
                    cells_in_the_latent_neighborhood_k = (
                        cells_in_the_latent_neighborhood[:, :k]
                    )

                    if "distance" in self.set_of_metrics:
                        xy = adata_fov.obsm[self.spatial_coord_key]  # .values

                        spatial_coord_of_latent_neighbors_fov = xy[
                            cells_in_the_latent_neighborhood_k
                        ]

                        dists_parallel = Parallel(n_jobs=-1)(
                            delayed(cdist)(
                                xy[i].reshape(1, 2),
                                spatial_coord_of_latent_neighbors_fov[i],
                            )
                            for i in range(n_cells)
                        )

                        dists_parallel = np.squeeze(np.array(dists_parallel))

                        if self.reduction[0] == "median":
                            reducted_dists = np.median(dists_parallel, axis=-1)

                        if self.reduction[0] == "mean":
                            reducted_dists = np.mean(dists_parallel, axis=-1)

                        if self.reduction[0] is None:
                            reducted_dists = dists_parallel

                        latent_and_phys_corr[k].append(reducted_dists.flatten())

                    if "similarity" in self.set_of_metrics:
                        # cell types in the neighborhood
                        ct = adata_fov.obsm[self.ct_composition_key].values

                        similarity_parallel = Parallel(n_jobs=-1)(
                            delayed(compute_similarity)(
                                ct[i],
                                ct[cells_in_the_latent_neighborhood_k[i]],
                                self.similarity_metric,
                            )
                            for i in range(n_cells)
                        )

                        similarity_parallel = np.squeeze(np.array(similarity_parallel))

                        if self.reduction[1] == "median":
                            reducted_similarity = np.median(
                                similarity_parallel, axis=-1
                            )

                        if self.reduction[1] == "mean":
                            reducted_similarity = np.mean(similarity_parallel, axis=-1)

                        if self.reduction[1] == None:
                            reducted_similarity = similarity_parallel

                        neighborhood_similarity[k].append(reducted_similarity.flatten())

            if "distance" in self.set_of_metrics:
                for k in k_nn_range:
                    self.adata.obs[
                        KEYS_SPATIAL.DISTANCE_KEY + latent_space_key + "_k" + str(k)
                    ] = np.concatenate(latent_and_phys_corr[k])

            if "similarity" in self.set_of_metrics:
                for k in k_nn_range:
                    self.adata.obs[
                        KEYS_SPATIAL.SIMILARITY_KEY + latent_space_key + "_k" + str(k)
                    ] = np.concatenate(neighborhood_similarity[k])

        return None

        if "cluster_stats" in self.set_of_metrics:
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

                self.cluster_stats[leiden_key] = ClusterStats(
                    df_mean.sort_index(), df_std.sort_index()
                )

        if "latent_overlap" in self.set_of_metrics:
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

    def boxplot_metrics(
        self,
        metric: Literal[*SET_OF_METRICS],
        reference_key: Optional[str] = None,
        comparison_key: Optional[str] = None,
        validation_only: bool = True,
        train_only: bool = False,
        plot_type: Literal["kde", "ecdf", "boxplot", "hist"] = "boxplot",
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

        if reference_key is None:
            reference_key = self.z1_reference

        if validation_only:
            indices = self.validation_indices
        elif train_only:
            indices = self.train_indices
        else:
            indices = [*range(len(self.adata))]

        data = {
            "k="
            + str(k): (
                self.adata.obs[metric_key + reference_key + "_k" + str(k)][indices],
                self.adata.obs[metric_key + comparison_key + "_k" + str(k)][indices],
            )
            for k in self.k_nn_range
        }

        # Create individual subplots for each key
        fig, axes = plt.subplots(1, len(data), figsize=(12, 5), sharey=True)

        if plot_type == "boxplot":
            linewidth = 1.5

            # Plot boxplots for each key
            for (key, (series1, series2)), ax in zip(data.items(), axes):
                boxprops = dict(color="blue", linewidth=linewidth)
                medianprops = dict(color="blue", linewidth=linewidth)
                whiskerprops = dict(color="blue", linewidth=linewidth)
                capprops = dict(color="blue", linewidth=linewidth)

                # Plot boxplots
                ax.boxplot(
                    [series1, series2],
                    boxprops=boxprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    showmeans=True,
                    meanline=True,
                )

                # # Add a dashed line for the mean
                # mean_line = ax.lines[-2]  # The line corresponding to the mean
                # mean_line.set_linestyle("--")

                ax.set_title(key)

                # Add grid
                ax.grid(True, linestyle="--", alpha=0.7)

                # Set xticks
                ax.set_xticks([1, 2])
                ax.set_xticklabels([reference_key, comparison_key], rotation=80)

        if plot_type == "hist":
            for (key, (series1, series2)), ax in zip(data.items(), axes):
                differences = series2 - series1

                # Plot histogram
                ax.hist(differences, bins=50, color="blue", alpha=0.7)

                # Add a title (you may customize this based on your data)
                ax.set_title(f"Differences for {key}")

                # Add labels
                ax.set_xlabel("Differences")
                ax.set_ylabel("Frequency")

                # Add grid
                ax.grid(True, linestyle="--", alpha=0.7)

                # Add a common y-axis label
                axes[0].set_ylabel("Frequency")

                # Adjust layout to prevent clipping of ylabel
                plt.tight_layout(rect=[0, 0, 1, 0.96])

        if plot_type == "entropy":
            ct_entropy = entropy(
                self.adata.obsm[self.ct_composition_key].values[indices], axis=1
            )
            for (key, (series1, series2)), ax in zip(data.items(), axes):
                # Plot series2 metric against entropy
                ax.scatter(series2, ct_entropy, color="blue", alpha=0.5)

                # Add a title (you may customize this based on your data)
                ax.set_title(f"{key}")

                # Add labels
                ax.set_ylabel("Entropy")
                ax.set_xlabel(metric_title)

                # Add grid
                ax.grid(True, linestyle="--", alpha=0.7)

                # Adjust layout to prevent clipping of ylabel
                plt.tight_layout(rect=[0, 0, 1, 0.96])

        if plot_type == "barplot":
            df_ct_pos = {}
            df_ct_neg = {}
            for (key, (series1, series2)), ax in zip(data.items(), axes):
                differences = series2 - series1

                # count the number of values for which the difference is positive, negative
                positive = np.sum(differences > 0)
                negative = np.sum(differences < 0)

                # Count the cell types for which the difference is positive:
                ct = self.adata.obs[self.label_key][indices]

                # for each cell type compute the fraction of positive and negative differences

                ct_positive = ct[differences > 0].value_counts(sort=False)
                ct_negative = ct[differences < 0].value_counts(sort=False)
                df_ct_neg[key] = ct_negative / ct.value_counts(sort=False)
                df_ct_pos[key] = ct_positive / ct.value_counts(sort=False)

                # Plot bar chart
                ax.bar(
                    ["Positive", "Negative"],
                    [positive, negative],
                    color=["green", "red"],
                )

                # Add a title
                ax.set_title(f"Difference for {key}")

            # Add a common y-axis label
            axes[0].set_ylabel("Count")

            rprint("Positive differences:")
            df_ct_pos = pd.DataFrame(df_ct_pos)
            rprint(df_ct_pos.sort_values(by=df_ct_pos.columns[-1], ascending=False))
            rprint("Negative differences:")
            df_ct_neg = pd.DataFrame(df_ct_neg)
            rprint(df_ct_neg.sort_values(by=df_ct_neg.columns[-1], ascending=False))

        # Add labels and title
        fig.suptitle(metric_title)
        plt.show()

        return None

    def plot_metrics(
        self,
        metric: Literal[*SET_OF_METRICS],
        plot_type: Literal["kde", "ecdf", "boxplot"] = "ecdf",
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

        if plot_type == "boxplot":
            columns_to_plot = [
                metric_key + latent_key for latent_key in self.latent_space_keys
            ]
            self.adata.obs.boxplot(
                column=columns_to_plot,
                by="set",
                layout=(len(columns_to_plot), 1),
                rot=45,
                figsize=(5, 11 * len(columns_to_plot)),
            )

        else:
            for idx, latent_key in enumerate(self.latent_space_keys):
                if plot_type == "kde":
                    sns.kdeplot(
                        data=self.adata.obs[metric_key + latent_key],
                        label=latent_key,
                        color=color_plots[idx],
                        alpha=0.5,
                    )

                if plot_type == "ecdf":
                    sns.ecdfplot(
                        data=self.adata.obs[metric_key + latent_key],
                        label=latent_key,
                        color=color_plots[idx],
                        alpha=0.5,
                    )

        if plot_type == "kde":
            plt.title("Kernel density estimation")
        if plot_type == "ecdf":
            plt.title("Empirical cumulative distribution function")
        if plot_type == "boxplot":
            plt.title("Boxplot")
        plt.xlabel(metric_title)
        plt.legend()  # Add a legend to display the labels
        plt.show()

        return None

    def test_distributions(
        self,
        test: Literal["mannwhitneyu", "ks_2samp"] = "mannwhitneyu",
        distribution: Literal["distance", "similarity"] = "distance",
        plot: Optional[int] = None,
    ):
        if distribution == "distance":
            metric = KEYS_SPATIAL.DISTANCE_KEY
        if distribution == "similarity":
            metric = KEYS_SPATIAL.SIMILARITY_KEY

        # initialize a dict
        stat_dict = {
            "Model": self.latent_space_keys,
        }

        indices_dict = {
            "train": self.train_indices,
            "validation": self.validation_indices,
        }

        df_sorted = {k: [] for k in self.k_nn_range}

        for k in self.k_nn_range:
            for indices_key, indices in indices_dict.items():
                x = self.adata.obs[metric + self.z1_reference + "_k" + str(k)][indices]
                p_values = []
                mean_values = []
                median_values = []

                for latent_key in self.latent_space_keys:
                    y = self.adata.obs[metric + latent_key + "_k" + str(k)][indices]
                    mean_values.append(np.mean(y))
                    median_values.append(np.std(y))
                    U1, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                    p_values.append(p)

                reject, p_values_corr = pg.multicomp(p_values, method="fdr_bh")

                stat_dict["Mean " + distribution + " " + indices_key] = mean_values
                stat_dict["Std " + distribution + " " + indices_key] = median_values
                stat_dict["p-value corrected " + indices_key] = p_values_corr

            df = pd.DataFrame(stat_dict)
            df = df.set_index("Model")
            df = df.round(3)
            df_sorted_k = df.sort_values(
                by="Mean " + distribution + " " + indices_key, ascending=True
            )

            df_sorted[k] = df_sorted_k

        if plot:
            df = df_sorted[plot]
            # Set the x-axis locations
            x = range(len(df))
            # Specify the bar width
            bar_width = 0.35
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.bar(
                x,
                df["Mean " + distribution + " train"],
                width=bar_width,
                yerr=df["Std " + distribution + " train"],
                label="Train",
                capsize=5,
            )
            plt.bar(
                [i + bar_width for i in x],
                df["Mean " + distribution + " validation"],
                width=bar_width,
                yerr=df["Std " + distribution + " validation"],
                label="Val",
                capsize=5,
            )
            plt.xlabel("Setup")
            plt.ylabel("Mean " + distribution)
            plt.title("Means with Standard Deviation Error Bars, k=" + str(plot))
            plt.xticks([i + bar_width / 2 for i in x], df.index, rotation=75)
            plt.legend()
            plt.grid()
            plt.show()

        return df_sorted

    def compare_neighborhoods(
        self,
        comparison_keys: str,
        reference_key: Optional[str] = None,
        train_only: bool = False,
        validation_only: bool = False,
        metric: Literal["Pearson", "AUC", "BCE", "CE"] = "Pearson",
    ):
        """
        Compare neighborhoods between two sets of data and compute a summary score for each set.

        Args:
            comparison_keys: The keys of the data to compare.
            reference_key: The key of the reference data. If None, the ct_composition_key is used.
            train_only: If True, only the train_indices are used for comparison.
            validation_only: If True, only the validation_indices are used for comparison.
            metric: The metric to use for comparison. Either "Pearson" or "AUC".

        Returns:
            A pandas DataFrame containing the summary score for each set of data.
        """

        if train_only:
            if self.train_indices is None:
                raise ValueError(
                    "Please provide train_indices when train_only is True."
                )
            else:
                adata = self.adata[self.train_indices].copy()
                save_metric_key = "train_"
                mode = "train"

        if validation_only:
            if self.validation_indices is None:
                raise ValueError(
                    "Please provide validation_indices when validation_only is True."
                )
            else:
                adata = self.adata[self.validation_indices].copy()
                save_metric_key = "val_"
                mode = "validation"
        else:
            adata = self.adata.copy()
            # mode = "all"

        if reference_key is None:
            reference_key = self.ct_composition_key

        neighborhood_ref = adata.obsm[reference_key]

        if metric == "AUC":
            # make sure that the reference is binary
            neighborhood_ref = (neighborhood_ref > 0).astype(np.float32)
            metric_fct = lambda x, y: roc_auc_score(x, y)
            save_metric_key += "auc_"

        elif metric == "Pearson":
            metric_fct = lambda x, y: pearsonr(x, y)[0]
            save_metric_key += "corr_"

        elif metric == "BCE":
            metric_fct = lambda x, y: log_loss(x, y)
            save_metric_key += "bce_"

        elif metric == "CE":
            metric_fct = lambda x, y: entropy(x, y, axis=1)
            save_metric_key += "ce_"

            ce_keys = []
            ce_keys_mean = []

            for key in comparison_keys:
                neighborhood_pred = adata.obsm[key]

                CE = metric_fct(neighborhood_ref, neighborhood_pred)
                ce_keys.append(CE)
                ce_keys_mean.append(np.mean(CE))
                self.adata.uns[save_metric_key + key] = CE  # optional

            # Create a single subplot for all series
            fig, ax = plt.subplots(figsize=(15, 6))

            linewidth = 1.5
            boxprops = dict(color="blue", linewidth=linewidth)
            medianprops = dict(color="blue", linewidth=linewidth)
            whiskerprops = dict(color="blue", linewidth=linewidth)
            capprops = dict(color="blue", linewidth=linewidth)

            # Plot boxplots
            ax.boxplot(
                ce_keys,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                showmeans=True,
                meanline=True,
            )

            # Add titles, labels, and grid
            ax.set_title("Cross-entropy over cells")
            ax.set_xticks(range(1, len(ce_keys) + 1))
            ax.set_xticklabels([key for key in comparison_keys], rotation=80)
            ax.set_xlabel("Model")
            ax.set_ylabel("Cross-entropy")
            ax.grid(True, linestyle="--", alpha=0.7)

            plt.show()

            df_summary = pd.DataFrame(
                {
                    "Model": comparison_keys,
                    mode + " " + metric + " ": ce_keys_mean,
                }
            )

            return df_summary

        proportions_ref = adata.obs[self.label_key].value_counts() / len(adata)
        proportions_ref_series = pd.Series(
            proportions_ref, index=neighborhood_ref.columns
        )  # TODO cell-type specific proportions

        keys_added = []
        summary_score = []

        for key in comparison_keys:
            neighborhood_pred = pd.DataFrame(
                adata.obsm[key],
                columns=neighborhood_ref.columns,
                index=adata.obs_names,
            )
            # then loop over cell types:
            metric_dict = {}
            for ct in neighborhood_ref.columns:
                # compute the entropy for each cell type:
                true_neighbors_ct = neighborhood_ref[ct]
                pred_neighbors_ct = neighborhood_pred[ct]
                metric_ct = [
                    metric_fct(
                        true_neighbors_ct[adata.obs.cell_type == i],
                        pred_neighbors_ct[adata.obs.cell_type == i],
                    )
                    for i in neighborhood_ref.columns
                ]
                metric_dict[ct] = metric_ct

            metric_df = pd.DataFrame(
                metric_dict,
                columns=neighborhood_ref.columns,
                index=neighborhood_ref.columns,
            )

            # add to df a column being the weighted average of each row by the proportion of each cell type in the dataset
            metric_df["weighted_mean"] = metric_df.apply(
                lambda x: np.average(x, weights=proportions_ref_series), axis=1
            )
            # add to df a row, each value is the weighted average of each column by the proportion of each cell type in the dataset
            metric_df.loc["weighted_mean"] = metric_df.apply(
                lambda x: np.average(x, weights=proportions_ref_series), axis=0
            )

            self.adata.uns[save_metric_key + key] = metric_df

            keys_added.append(save_metric_key + key)

            summary_score.append(metric_df["weighted_mean"].values[-1])

        rprint("Saved metric in: ", keys_added)

        df_summary = pd.DataFrame(
            {
                "Model": comparison_keys,
                mode + " " + metric + " ": summary_score,
            }
        )

        df_summary = df_summary.set_index("Model")
        df_summary = df_summary.round(3)
        df_summary_sorted = df_summary.sort_values(
            by=mode + " " + metric + " ", ascending=False
        )

        return df_summary_sorted
