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


def test_nichevi():
    adata = synthetic_iid()

    adata.obsm["qz1_m"] = np.random.normal(size=(adata.shape[0], 10))
    # positive variance:
    adata.obsm["qz1_var"] = np.random.normal(size=(adata.shape[0], 10)) ** 2

    nicheSCVI.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
        niche_composition_key="neighborhood_composition",
        niche_indexes_key="niche_indexes",
        niche_distances_key="niche_distances",
        sample_key="batch",
        cell_coordinates_key="coordinates",
        k_nn=10,
        latent_mean_key="qz1_m",
        latent_var_key="qz1_var",
        latent_mean_ct_key="qz1_m_niches",
        latent_var_ct_key="qz1_var_niches",
    )

    vae = nicheSCVI(
        adata,
        z1_mean=adata.obsm["qz1_m_niches"],
        z1_var=adata.obsm["qz1_var_niches"],
        niche_kl_weight=1,
        # niche_components="knn_unweighted",
        niche_components="cell_type",
        n_latent=10,
        gene_likelihood="poisson",
    )

    vae.train(1)
    vae.train(3)
    vae.get_elbo(indices=vae.validation_indices)
    vae.get_normalized_expression()
    vae.get_latent_representation()
    vae.differential_expression(groupby="labels", group1="label_1")
    vae.differential_expression(groupby="labels", group1="label_1", group2="label_2")


test_nichevi()

print("nicheSCVI test passed")
