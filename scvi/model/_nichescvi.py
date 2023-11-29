import logging
from typing import List, Literal, Optional, Sequence

import numpy as np, pandas as pd
from anndata import AnnData
import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from scvi.nearest_neighbors import NeighborsOutput, pynndescent

from scvi import REGISTRY_KEYS
from scvi._types import MinifiedDataType
from scvi.data import AnnDataManager
from scvi.data._constants import _ADATA_MINIFY_TYPE_UNS_KEY, ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.model.utils import get_minified_adata_scrna
from scvi.module import nicheVAE
from scvi.utils import setup_anndata_dsp

from .base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin

from rich import print

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"
_SCVI_OBSERVED_LIB_SIZE = "_scvi_observed_lib_size"

logger = logging.getLogger(__name__)


class nicheSCVI(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
):
    """single-cell Variational Inference :cite:p:`Lopez18`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    _module_cls = nicheVAE

    def __init__(
        self,
        adata: AnnData,
        # n_cell_types: int,
        ###########
        # k_nn: int,  # TODO access th obsm keys to infer these parameters from the data!
        # n_latent_z1: int,
        ###########
        # niche_kl_weight: float = 1.0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key and self.minified_data_type is None:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.k_nn = self.summary_stats.n_niche_indexes
        self.n_latent_mean = self.summary_stats.n_latent_mean
        self.n_cell_types = self.summary_stats.n_niche_composition

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            ###########
            n_cell_types=self.n_cell_types,
            k_nn=self.k_nn,
            n_latent_z1=self.n_latent_mean,
            ###########
            # niche_kl_weight=niche_kl_weight,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type
        self._model_summary_string = (
            "nicheVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @torch.inference_mode()
    def predict_neighborhood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        ct_prediction = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            decoder_input = outputs["qz"].loc

            # put batch_index in the same device as decoder_input
            batch_index = batch_index.to(decoder_input.device)

            predicted_ct = self.module.composition_decoder(
                decoder_input,
                batch_index,
            )  # no batch correction here

            if self.module.compo_transform == "none":
                # predicted_ct_temperature = predicted_ct / self.module.compo_temperature
                # predicted_ct_prob = F.softmax(predicted_ct_temperature, dim=-1)
                predicted_ct_prob = predicted_ct.mean

            elif self.module.compo_transform == "log_softmax":
                predicted_ct_prob = torch.exp(predicted_ct)
            elif self.module.compo_transform == "log_compo":
                predicted_ct_prob = torch.exp(predicted_ct)
            # TODO maybe replace elif by else? meaning either you provide
            # raw logits and you softmax it or you provide log_probas from
            # the definition of the model and you just exponentiate it.

            ct_prediction.append(predicted_ct_prob.detach().cpu())

        return torch.cat(ct_prediction).numpy()

    @torch.inference_mode()
    def predict_niche_activation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        activation_prediction = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            decoder_input = outputs["qz"].loc

            # put batch_index in the same device as decoder_input
            batch_index = batch_index.to(decoder_input.device)

            niche_mean, niche_variance = self.module.niche_decoder(
                decoder_input,
                batch_index,
            )

            activation_prediction.append(niche_mean.detach().cpu())

        return torch.cat(activation_prediction).numpy()

    def preprocessing_anndata(
        adata: AnnData,
        niche_composition_key: Optional[str] = None,
        niche_indexes_key: Optional[str] = None,
        niche_distances_key: Optional[str] = None,
        ###########
        niche_type_key: Optional[str] = None,
        niche_treshold: float = 0.2,
        cell_type_for_niches: list[str] = None,
        ###########
        label_key: Optional[str] = None,
        sample_key: Optional[str] = None,
        cell_coordinates_key: Optional[str] = None,
        k_nn: int = 10,
        latent_mean_key: Optional[str] = None,
        latent_var_key: Optional[str] = None,
        latent_mean_niche_keys: Optional[list] = None,
        latent_var_niche_keys: Optional[str] = None,
        zero_prior: bool = False,
        ###########
        latent_mean_knn_key: Optional[str] = "latent_mean_knn",
    ):
        adata.obsm[niche_indexes_key] = np.zeros(
            (adata.n_obs, k_nn)
        )  # for each cell, store the indexes of its k_nn neighbors
        adata.obsm[niche_distances_key] = np.zeros(
            (adata.n_obs, k_nn)
        )  # for each cell, store the distances to its k_nn neighbors
        n_cell_types = len(adata.obs[label_key].unique())  # number of cell types
        adata.obsm[niche_composition_key] = np.zeros(
            (adata.n_obs, n_cell_types)
        )  # for each cell, store the composition of its neighborhood as a convex vector of cell type proportions

        get_niche_indexes(
            adata=adata,
            sample_key=sample_key,
            niche_indexes_key=niche_indexes_key,
            niche_distances_key=niche_distances_key,
            cell_coordinates_key=cell_coordinates_key,
            k_nn=k_nn,
        )

        get_neighborhood_composition(
            adata=adata,
            cell_type_column=label_key,
            indices_key=niche_indexes_key,
            niche_composition_key=niche_composition_key,
        )

        get_cell_niches(
            adata=adata,
            cell_types_to_include=cell_type_for_niches,
            treshold=niche_treshold,
            niche_type_key=niche_type_key,
            niche_composition_key=niche_composition_key,
        )

        adata.obsm[latent_mean_knn_key] = adata.obsm[latent_mean_key][
            adata.obsm[niche_indexes_key]
        ].mean(axis=1)

        get_average_latent_per_celltype(
            adata=adata,
            labels_key=label_key,
            niche_indexes_key=niche_indexes_key,
            latent_mean_key=latent_mean_key,
            latent_var_key=latent_var_key,
            latent_mean_ct_keys=latent_mean_niche_keys,
            latent_var_ct_keys=latent_var_niche_keys,
            zero_prior=zero_prior,
        )

        return None

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        # --specific to nicheVI
        niche_composition_key: str,
        niche_indexes_key: str,
        niche_distances_key: str,
        # ---------------------
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        latent_mean_key: Optional[str] = None,
        latent_var_key: Optional[str] = None,
        latent_mean_ct_key: Optional[str] = None,
        latent_var_ct_key: Optional[str] = None,
        ###########
        latent_mean_knn_key: Optional[str] = "latent_mean_knn",
        # ---------------------
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        cell_index_key="cell_index",
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """

        # adata.obsm[niche_indexes_key] = np.zeros((adata.n_obs, k_nn))
        # adata.obsm[niche_distances_key] = np.zeros((adata.n_obs, k_nn))
        # n_cell_types = len(adata.obs[labels_key].unique())
        # adata.obsm[niche_composition_key] = np.zeros((adata.n_obs, n_cell_types))
        adata.obs[cell_index_key] = adata.obs.reset_index().index.astype(int)

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ObsmField(REGISTRY_KEYS.NICHE_COMPOSITION_KEY, niche_composition_key),
            # ObsmField(REGISTRY_KEYS.NICHE_DISTANCES_KEY, niche_distances_key),
            ObsmField(REGISTRY_KEYS.NICHE_INDEXES_KEY, niche_indexes_key),
            ObsmField(REGISTRY_KEYS.Z1_MEAN_KEY, latent_mean_key),
            ObsmField(REGISTRY_KEYS.Z1_VAR_KEY, latent_var_key),
            ObsmField(REGISTRY_KEYS.Z1_MEAN_CT_KEY, latent_mean_ct_key),
            ObsmField(REGISTRY_KEYS.Z1_VAR_CT_KEY, latent_var_ct_key),
            ObsmField(REGISTRY_KEYS.Z1_MEAN_KNN_KEY, latent_mean_knn_key),
            NumericalObsField(
                REGISTRY_KEYS.INDICES_KEY, cell_index_key, required=False
            ),
        ]

        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _get_fields_for_adata_minification(
        minified_data_type: MinifiedDataType,
    ) -> List[BaseAnnDataField]:
        """Return the anndata fields required for adata minification of the given minified_data_type."""
        if minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            fields = [
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZM_KEY,
                    _SCVI_LATENT_QZM,
                ),
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZV_KEY,
                    _SCVI_LATENT_QZV,
                ),
                NumericalObsField(
                    REGISTRY_KEYS.OBSERVED_LIB_SIZE,
                    _SCVI_OBSERVED_LIB_SIZE,
                ),
            ]
        else:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")
        fields.append(
            StringUnsField(
                REGISTRY_KEYS.MINIFY_TYPE_KEY,
                _ADATA_MINIFY_TYPE_UNS_KEY,
            ),
        )
        return fields

    def minify_adata(
        self,
        minified_data_type: MinifiedDataType = ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ) -> None:
        """Minifies the model's adata.

        Minifies the adata, and registers new anndata fields: latent qzm, latent qzv, adata uns
        containing minified-adata type, and library size.
        This also sets the appropriate property on the module to indicate that the adata is minified.

        Parameters
        ----------
        minified_data_type
            How to minify the data. Currently only supports `latent_posterior_parameters`.
            If minified_data_type == `latent_posterior_parameters`:

            * the original count data is removed (`adata.X`, adata.raw, and any layers)
            * the parameters of the latent representation of the original data is stored
            * everything else is left untouched
        use_latent_qzm_key
            Key to use in `adata.obsm` where the latent qzm params are stored
        use_latent_qzv_key
            Key to use in `adata.obsm` where the latent qzv params are stored

        Notes
        -----
        The modification is not done inplace -- instead the model is assigned a new (minified)
        version of the adata.
        """
        # TODO(adamgayoso): Add support for a scenario where we want to cache the latent posterior
        # without removing the original counts.
        if minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")

        if self.module.use_observed_lib_size is False:
            raise ValueError(
                "Cannot minify the data if `use_observed_lib_size` is False"
            )

        minified_adata = get_minified_adata_scrna(self.adata, minified_data_type)
        minified_adata.obsm[_SCVI_LATENT_QZM] = self.adata.obsm[use_latent_qzm_key]
        minified_adata.obsm[_SCVI_LATENT_QZV] = self.adata.obsm[use_latent_qzv_key]
        counts = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        minified_adata.obs[_SCVI_OBSERVED_LIB_SIZE] = np.squeeze(
            np.asarray(counts.sum(axis=1))
        )
        self._update_adata_and_manager_post_minification(
            minified_adata, minified_data_type
        )
        self.module.minified_data_type = minified_data_type


def get_niche_indexes(
    adata: AnnData,
    sample_key: str,
    niche_indexes_key: str,
    niche_distances_key: Optional[str],
    cell_coordinates_key: str,
    k_nn: int,
):
    adata.obs["index"] = np.arange(adata.shape[0])
    # build a dictionnary giving the index of each 'donor_slice' observation:
    donor_slice_index = {}
    for sample in adata.obs[sample_key].unique():
        donor_slice_index[sample] = adata.obs[adata.obs[sample_key] == sample][
            "index"
        ].values

    for sample in adata.obs[sample_key].unique():
        sample_coord = adata.obsm[cell_coordinates_key][adata.obs[sample_key] == sample]

        # pynndescent is not faster than sklearn.neighbors.NearestNeighbors in such low dimensions (2D space)
        # neigh_output = pynndescent(
        #     X=sample_coord, n_neighbors=k_nn + 1, random_state=0, n_jobs=-1
        # )
        # indices, distances = neigh_output.indices, neigh_output.distances

        # Create a NearestNeighbors object
        knn = NearestNeighbors(n_neighbors=k_nn + 1)

        # Fit the kNN model to the points
        knn.fit(sample_coord)

        # Find the indices of the kNN for each point
        distances, indices = knn.kneighbors(sample_coord)

        # apply an inverse exp transformation to the distances
        # distances = np.exp(-(distances**2)) - or inverse of the distance
        distances[:, 1:] = 1 / distances[:, 1:]

        # Store the indices in the adata object
        sample_global_index = donor_slice_index[sample][indices].astype(int)

        adata.obsm[niche_indexes_key][
            adata.obs[sample_key] == sample
        ] = sample_global_index[:, 1:]

        adata.obsm[niche_indexes_key] = adata.obsm[niche_indexes_key].astype(int)

        adata.obsm[niche_distances_key][adata.obs[sample_key] == sample] = distances[
            :, 1:
        ]

    print(
        "[bold cyan]Saved niche_indexes and niche_distances in adata.obsm[/bold cyan]"
    )

    return None


def get_neighborhood_composition(
    adata: AnnData,
    cell_type_column: str,
    indices_key: str = "niche_indexes",
    niche_composition_key: str = "niche_composition",
):
    indices = adata.obsm[indices_key].astype(int)

    cell_types = adata.obs[cell_type_column].unique().tolist()
    cell_type_to_int = {cell_types[i]: i for i in range(len(cell_types))}

    # Transform the query vector into an integer-valued vector
    integer_vector = np.vectorize(cell_type_to_int.get)(adata.obs[cell_type_column])

    n_cells = adata.n_obs
    # For each cell, get the cell types of its neighbors
    cell_types_in_the_neighborhood = [
        integer_vector[indices[cell, :]] for cell in range(n_cells)
    ]

    # Compute the composition of each neighborhood
    composition = np.array(
        [
            np.bincount(
                cell_types_in_the_neighborhood[cell],
                minlength=len(cell_type_to_int),
            )
            for cell in range(n_cells)
        ]
    )

    # Normalize the composition of each neighborhood
    composition = composition / indices.shape[1]
    composition = np.array(composition)

    neighborhood_composition_df = pd.DataFrame(
        data=composition,
        columns=cell_types,
        index=adata.obs_names,
    )

    adata.obsm[niche_composition_key] = neighborhood_composition_df

    print("[bold green]Saved niche_composition in adata.obsm[/bold green]")

    return None


def get_cell_niches(
    adata: AnnData,
    cell_types_to_include: Optional[list[str]] = None,
    treshold: float = 0.2,
    niche_type_key: str = "niche_type",
    niche_composition_key: str = "niche_composition",
):
    if cell_types_to_include is None:
        pass

    else:
        composition_subet = adata.obsm[niche_composition_key][cell_types_to_include]

        # for each cell, get the cell type with the highest proportion in its neighborhood

        max_ct = composition_subet.max(axis=1)

        # Create a new column with the name of the column containing the maximum value for each row
        composition_subet["niche_assignment"] = composition_subet.idxmax(axis=1)

        # Set 'max_column' to 'unknown' for rows where the maximum value is less than the threshold
        composition_subet.loc[max_ct < treshold, "niche_assignment"] = "unknown"

        adata.obs[niche_type_key] = composition_subet["niche_assignment"]

    return None


def get_average_latent_per_celltype(
    adata: AnnData,
    labels_key: str,
    niche_indexes_key: str,
    latent_mean_key: str,
    latent_var_key: str,
    latent_mean_ct_keys: list[str] = ["qz1_m_niche_ct"],
    latent_var_ct_keys: list[str] = ["qz1_var_niche_ct"],
    zero_prior: bool = False,
):
    # for each cell, take the average of the latent space for each label, namely the label-averaged latent_mean obsm

    if latent_mean_ct_keys is None:
        adata.obsm["qz1_m_niche_ct"] = np.empty(
            (adata.n_obs, adata.obsm[latent_mean_key].shape[1])
        )
        adata.obsm["qz1_var_niche_ct"] = np.empty(
            (adata.n_obs, adata.obsm[latent_var_key].shape[1])
        )

        return None

    n_cells = adata.n_obs
    niche_indexes = adata.obsm[niche_indexes_key]

    z1_mean_niches = adata.obsm[latent_mean_key][niche_indexes]
    z1_var_niches = adata.obsm[latent_var_key][niche_indexes]

    if "qz1_m_niche_knn" in latent_mean_ct_keys:
        adata.obsm["qz1_m_niche_knn"] = z1_mean_niches
        adata.obsm["qz1_var_niche_knn"] = z1_var_niches

        print(
            "[bold green]Saved qz1_m_niche_knn and qz1_var_niche_knn in adata.obsm[/bold green]"
        )

    if "qz1_m_niche_ct" in latent_mean_ct_keys:
        cell_types = adata.obs[labels_key].unique().tolist()

        cell_type_to_int = {cell_types[i]: i for i in range(len(cell_types))}
        integer_vector = np.vectorize(cell_type_to_int.get)(adata.obs[labels_key])

        # For each cell, get the cell types of its neighbors (as integers)
        cell_types_in_the_neighborhood = np.vstack(
            [integer_vector[niche_indexes[cell, :]] for cell in range(n_cells)]
        )

        dict_of_cell_type_indices = {}

        for cell_type, cell_type_idx in cell_type_to_int.items():
            ct_row_indices, ct_col_indices = np.where(
                cell_types_in_the_neighborhood == cell_type_idx
            )  # [1]

            # dict of cells:local index of the cells of cell_type in the neighborhood.
            result_dict = {}
            for row_idx, col_idx in zip(ct_row_indices, ct_col_indices):
                result_dict.setdefault(row_idx, []).append(col_idx)

            dict_of_cell_type_indices[cell_type] = result_dict

        # print(dict_of_cell_type_indices)

        latent_mean_ct_prior, latent_var_ct_prior = get_cell_type_priors(
            adata=adata,
            labels_key=labels_key,
            latent_mean_key=latent_mean_key,
            latent_var_key=latent_var_key,
            # latent_mean_ct_prior=latent_mean_ct_prior,
            # latent_var_ct_prior=latent_var_ct_prior,
            zero_prior=zero_prior,
        )

        z1_mean_niches_ct = np.stack(
            [latent_mean_ct_prior] * n_cells, axis=0
        )  # batch times n_cell_types times n_latent. Initialize your prior with a non-spatial average.
        z1_var_niches_ct = np.stack([latent_var_ct_prior] * n_cells, axis=0)

        # outer loop over cell types
        for cell_type, cell_type_idx in cell_type_to_int.items():
            ct_dict = dict_of_cell_type_indices[cell_type]
            # inner loop over every cell that has this cell type in its neighborhood.
            for cell_idx, neighbor_idxs in ct_dict.items():
                z1_mean_niches_ct[cell_idx, cell_type_idx, :] = np.mean(
                    z1_mean_niches[cell_idx, neighbor_idxs, :], axis=0
                )
                z1_var_niches_ct[cell_idx, cell_type_idx, :] = np.mean(
                    z1_var_niches[cell_idx, neighbor_idxs, :], axis=0
                )

        adata.obsm["qz1_m_niche_ct"] = z1_mean_niches_ct
        adata.obsm["qz1_var_niche_ct"] = z1_var_niches_ct

        print(
            "[bold green]Saved qz1_m_niche_ct and qz1_var_niche_ct in adata.obsm[/bold green]"
        )

    return None


def get_cell_type_priors(
    adata: AnnData,
    labels_key: str,
    latent_mean_key: str,
    latent_var_key: str,
    zero_prior: bool = False,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (non-spatial) prior for each cell type, as the average of the latent space for each cell type.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    labels_key
        Key for cell type annotation stored in `adata.obs`.
    latent_mean_key
        Key for the latent mean stored in `adata.obsm`.
    latent_var_key
        Key for the latent variance stored in `adata.obsm`.

    Returns
    -------
    latent_mean_priors
        The prior for the latent mean.
    latent_var_priors
        The prior for the latent variance.

    """

    cell_types = adata.obs[labels_key].unique().tolist()
    n_cell_types = len(cell_types)

    int_to_cell_types = {i: cell_types[i] for i in range(n_cell_types)}
    n_latent_z1 = adata.obsm[latent_mean_key].shape[1]

    latent_mean_priors = np.zeros((n_cell_types, n_latent_z1))
    latent_var_priors = np.zeros_like(latent_mean_priors) + epsilon

    if zero_prior:
        return latent_mean_priors, latent_var_priors

    for i in range(n_cell_types):
        type = int_to_cell_types[i]
        latent_mean_priors[i] = np.mean(
            adata[adata.obs[labels_key] == type].obsm[latent_mean_key], axis=0
        )
        latent_var_priors[i] = np.mean(
            adata[adata.obs[labels_key] == type].obsm[latent_var_key], axis=0
        )

    return latent_mean_priors, latent_var_priors
