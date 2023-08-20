from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    MINIFY_TYPE_KEY: str = "minify_type"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    NICHE_COMPOSITION_KEY: str = "niche_composition"
    Z1_MEAN_KEY: str = "latent_mean"
    Z1_VAR_KEY: str = "latent_var"
    NICHE_INDEXES_KEY: str = "niche_indexes"
    NICHE_DISTANCES_KEY: str = "niche_distances"
    Z1_MEAN_CT_KEY: str = "latent_mean_ct_key"
    Z1_VAR_CT_KEY: str = "latent_var_ct_key"


class _METRIC_KEYS_NT(NamedTuple):
    TRAINING_KEY: str = "training"
    VALIDATION_KEY: str = "validation"
    # classification
    ACCURACY_KEY: str = "accuracy"
    F1_SCORE_KEY: str = "f1_score"
    AUROC_KEY: str = "auroc"
    CLASSIFICATION_LOSS_KEY: str = "classification_loss"
    TRUE_LABELS_KEY: str = "true_labels"
    LOGITS_KEY: str = "logits"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
METRIC_KEYS = _METRIC_KEYS_NT()
