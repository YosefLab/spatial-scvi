import inspect
import os
import pickle
import tarfile
from unittest import mock

import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from flax import linen as nn
from lightning.pytorch.callbacks import LearningRateMonitor
from scipy.sparse import csr_matrix
from torch.nn import Softplus

import scvi
from scvi.data import _constants, synthetic_iid
from scvi.data._compat import LEGACY_REGISTRY_KEY_MAP, registry_from_setup_dict
from scvi.data._download import _download
from scvi.dataloaders import (
    AnnDataLoader,
    DataSplitter,
    DeviceBackedDataSplitter,
    SemiSupervisedDataLoader,
    SemiSupervisedDataSplitter,
)
from scvi.model import (
    AUTOZI,
    MULTIVI,
    PEAKVI,
    SCANVI,
    SCVI,
    TOTALVI,
    CondSCVI,
    DestVI,
    JaxSCVI,
    LinearSCVI,
    nicheSCVI,
    compoSCVI,
)
from scvi.model.utils import mde
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils import attrdict

# from tests.dataset.utils import generic_setup_adata_manager, scanvi_setup_adata_manager

LEGACY_REGISTRY_KEYS = set(LEGACY_REGISTRY_KEY_MAP.values())
LEGACY_SETUP_DICT = {
    "scvi_version": "0.0.0",
    "categorical_mappings": {
        "_scvi_batch": {
            "original_key": "testbatch",
            "mapping": np.array(["batch_0", "batch_1"], dtype=object),
        },
        "_scvi_labels": {
            "original_key": "testlabels",
            "mapping": np.array(["label_0", "label_1", "label_2"], dtype=object),
        },
    },
    "extra_categoricals": {
        "mappings": {
            "cat1": np.array([0, 1, 2, 3, 4]),
            "cat2": np.array([0, 1, 2, 3, 4]),
        },
        "keys": ["cat1", "cat2"],
        "n_cats_per_key": [5, 5],
    },
    "extra_continuous_keys": np.array(["cont1", "cont2"], dtype=object),
    "data_registry": {
        "X": {"attr_name": "X", "attr_key": None},
        "batch_indices": {"attr_name": "obs", "attr_key": "_scvi_batch"},
        "labels": {"attr_name": "obs", "attr_key": "_scvi_labels"},
        "cat_covs": {
            "attr_name": "obsm",
            "attr_key": "_scvi_extra_categoricals",
        },
        "cont_covs": {
            "attr_name": "obsm",
            "attr_key": "_scvi_extra_continuous",
        },
    },
    "summary_stats": {
        "n_batch": 2,
        "n_cells": 400,
        "n_vars": 100,
        "n_labels": 3,
        "n_proteins": 0,
        "n_continuous_covs": 2,
    },
}


N_LAYERS = 1
N_LATENT = 2
LIKELIHOOD = "nb"
K_NN = 5


def test_nichevi():
    adata = synthetic_iid(
        batch_size=256,
        n_genes=100,
        n_proteins=0,
        n_regions=0,
        n_batches=2,
        n_labels=3,
        dropout_ratio=0.5,
        coordinates_key="coordinates",
        sparse_format=None,
        return_mudata=False,
    )

    adata.obsm["qz1_m"] = np.random.normal(size=(adata.shape[0], N_LATENT))
    # positive variance:
    adata.obsm["qz1_var"] = np.random.normal(size=(adata.shape[0], N_LATENT)) ** 2

    nicheSCVI.preprocessing_anndata(
        adata,
        niche_composition_key="neighborhood_composition",
        niche_indexes_key="niche_indexes",
        niche_distances_key="niche_distances",
        label_key="labels",
        sample_key="batch",
        cell_coordinates_key="coordinates",
        k_nn=K_NN,
        latent_mean_key="qz1_m",
        latent_var_key="qz1_var",
        latent_mean_niche_keys=["qz1_m_niche_ct", "qz1_m_niche_knn"],
        latent_var_niche_keys=["qz1_var_niche_ct", "qz1_var_niche_knn"],
        # latent_mean_ct_prior="qz1_m_niche_ct_prior",
        # latent_var_ct_prior="qz1_var_niche_ct_prior",
    )

    nicheSCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
        niche_composition_key="neighborhood_composition",
        niche_indexes_key="niche_indexes",
        niche_distances_key="niche_distances",
        latent_mean_key="qz1_m",
        latent_var_key="qz1_var",
        latent_mean_ct_key="qz1_m_niche_ct",
        latent_var_ct_key="qz1_var_niche_ct",
    )

    niche_setup = {
        "mix_kl0_compo0": {
            "niche_components": "cell_type",
            "niche_combination": "mixture",
            "rec_weight": 1,
            "niche_kl_weight": 0,
            "niche_compo_weight": 0,
        },
        "mix_unif_kl0_compo0": {
            "niche_components": "cell_type_unweighted",
            "niche_combination": "mixture",
            "rec_weight": 1,
            "niche_kl_weight": 0,
            "niche_compo_weight": 0,
        },
        "mix_kl1_compo1": {
            "niche_components": "cell_type",
            "niche_combination": "mixture",
            "rec_weight": 1,
            "niche_kl_weight": 1,
            "niche_compo_weight": 1,
        },
    }

    # setup_dict = niche_setup["knn_unweighted_setup"]
    setup_dict = niche_setup["mix_kl1_compo1"]
    # setup_dict = niche_setup["knn_setup"]

    vae = nicheSCVI(
        adata,
        rec_weight=setup_dict["rec_weight"],
        niche_kl_weight=setup_dict["niche_kl_weight"],
        niche_compo_weight=setup_dict["niche_compo_weight"],
        niche_components=setup_dict["niche_components"],
        niche_combination=setup_dict["niche_combination"],
        gene_likelihood=LIKELIHOOD,
        n_layers=N_LAYERS,
        n_latent=N_LATENT,
        compo_transform="none",
        compo_temperature=1,
        use_batch_norm="both",
        use_layer_norm="none",
    )

    vae.train(1)
    print("I am here")
    vae.train(3)
    vae.get_elbo(indices=vae.validation_indices)
    vae.get_normalized_expression()
    vae.get_latent_representation()
    print("Finished training")
    vae.differential_expression(groupby="labels", group1="label_1")
    vae.differential_expression(groupby="labels", group1="label_1", group2="label_2")


test_nichevi()

print("nicheSCVI test passed")
