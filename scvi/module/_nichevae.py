"""Main module."""
import logging
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import (
    DecoderSCVI,
    Encoder,
    LinearDecoderSCVI,
    one_hot,
    Decoder,
    NicheDecoder,
    CompoDecoder,
    DirichletDecoder,
)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class nicheVAE(BaseMinifiedModeModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the scVI model described in :cite:p:`Lopez18`.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    extra_encoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
    """

    def __init__(
        self,
        n_input: int,
        ###########
        n_output_niche: int,
        n_cell_types: Optional[int],
        k_nn: Optional[int],
        niche_components: Literal[
            "cell_type", "knn", "knn_unweighted", "cell_type_unweighted"
        ] = "cell_type",
        niche_combination: Literal["latent", "observed"] = "latent",
        ###########
        cell_rec_weight: float = 1.0,
        niche_rec_weight: float = 1.0,
        niche_compo_weight: float = 1.0,
        latent_kl_weight: float = 1.0,
        ###########
        compo_transform: Literal["log_softmax", "log_compo", "none"] = "none",
        compo_temperature: float = 1.0,
        ###########
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        ###########
        n_layers_niche: int = 1,
        n_layers_compo: int = 1,
        n_hidden_niche: Tunable[int] = 128,
        n_hidden_compo: Tunable[int] = 128,
        ###########
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        ###########
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[
            Literal["gene", "gene-batch", "gene-label", "gene-cell"]
        ] = "gene",
        log_variational: bool = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        self.latent_kl_weight = latent_kl_weight
        self.cell_rec_weight = cell_rec_weight
        self.niche_rec_weight = niche_rec_weight
        self.niche_compo_weight = niche_compo_weight
        self.niche_components = niche_components
        self.niche_combination = niche_combination
        self.compo_transform = compo_transform
        self.compo_temperature = compo_temperature
        self.n_output_niche = n_output_niche

        self.n_cats_per_cov = n_cats_per_cov
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.n_latent_niche = n_latent // 1
        n_input_decoder_niche = self.n_latent_niche + n_continuous_cov
        n_input_decoder = n_latent + n_continuous_cov
        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

        self.n_niche_components = (
            n_cell_types if niche_components.startswith("cell_type") else k_nn
        )
        self.niche_decoder = NicheDecoder(
            n_input=n_input_decoder_niche,
            n_output=n_output_niche,
            n_niche_components=self.n_niche_components,
            n_cat_list=cat_list,
            n_layers=n_layers_niche,
            n_hidden=n_hidden_niche,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            **_extra_decoder_kwargs,
        )

        self.composition_decoder = DirichletDecoder(
            n_input_decoder_niche,
            n_cell_types,
            n_cat_list=cat_list,
            n_layers=n_layers_compo,
            n_hidden=n_hidden_compo,
            inject_covariates=deeply_inject_covariates,
            temperature=compo_temperature,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            **_extra_decoder_kwargs,
        )

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.minified_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            input_dict = {
                "x": x,
                "batch_index": batch_index,
                "cont_covs": cont_covs,
                "cat_covs": cat_covs,
            }
        else:
            if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                observed_lib_size = tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE]
                input_dict = {
                    "qzm": qzm,
                    "qzv": qzv,
                    "observed_lib_size": observed_lib_size,
                }
            else:
                raise NotImplementedError(
                    f"Unknown minified-data type: {self.minified_data_type}"
                )

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor,
        }
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    @auto_move_data
    def _cached_inference(self, qzm, qzv, observed_lib_size, n_samples=1):
        if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            dist = Normal(qzm, qzv.sqrt())
            # use dist.sample() rather than rsample because we aren't optimizing the z here
            untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = torch.log(observed_lib_size)
            if n_samples > 1:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
        else:
            raise NotImplementedError(
                f"Unknown minified-data type: {self.minified_data_type}"
            )
        outputs = {"z": z, "qz_m": qzm, "qz_v": qzv, "ql": None, "library": library}
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
        z_niche = z[..., : self.n_latent_niche]
        if cont_covs is None:
            decoder_input = z
            decoder_input_niche = z_niche
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
            decoder_input_niche = torch.cat(
                [z_niche, cont_covs.unsqueeze(0).expand(z_niche.size(0), -1, -1)],
                dim=-1,
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
            decoder_input_niche = torch.cat([z_niche, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        niche_mean, niche_variance = self.niche_decoder(
            decoder_input_niche, batch_index, *categorical_input
        )

        # niche_expression = Normal(niche_mean, niche_variance.sqrt())
        niche_expression = torch.distributions.Poisson(niche_variance)

        niche_composition = self.composition_decoder(
            decoder_input_niche, batch_index, *categorical_input
        )  # if DirichletDecoder, niche_composition is a distribution

        return {
            "px": px,
            "pl": pl,
            "pz": pz,
            "niche_mean": niche_mean,
            "niche_variance": niche_variance,
            "niche_composition": niche_composition,
            "niche_expression": niche_expression,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        epsilon: float = 1e-6,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]

        niche_weights_ct = tensors[
            REGISTRY_KEYS.NICHE_COMPOSITION_KEY
        ]  # .unsqueeze(-1)
        # niche_weights_distances = tensors[REGISTRY_KEYS.NICHE_DISTANCES_KEY].unsqueeze(
        #     -1
        # )

        if self.niche_components == "cell_type":
            niche_weights = niche_weights_ct

        elif self.niche_components == "cell_type_unweighted":
            niche_weights = (niche_weights_ct > 0).float()

        # elif self.niche_components == "knn":
        #     niche_weights = niche_weights_distances

        # elif self.niche_components == "knn_unweighted":
        #     niche_weights = torch.ones_like(niche_weights_distances)

        n_batch = niche_weights.shape[0]
        n_cell_types = niche_weights_ct.size(dim=-1)

        # NICHE PRIOR DISTRIBUTION-------------------------------------------------------------------------------

        z1_mean_niche = tensors[
            REGISTRY_KEYS.Z1_MEAN_CT_KEY
        ]  # batch times cell_types times n_latent
        z1_var_niche = tensors[REGISTRY_KEYS.Z1_VAR_CT_KEY]

        # z1_mean_niche_knn = tensors[REGISTRY_KEYS.Z1_MEAN_KNN_KEY]

        # POSTERIOR DISTRIBUTION-------------------------------------------------------------------------------
        niche_mean_mat = generative_outputs["niche_mean"]
        niche_var_mat = generative_outputs["niche_variance"]

        # --------------Niche mixture posterior distribution--------------------------
        if self.niche_combination == "latent":
            weighted_reconst_loss_niche = (
                torch.zeros(n_batch).type(torch.float64).to(x.device)
            )
            for type in range(
                n_cell_types
            ):  # TODO can you replace the for loop with a torch.sum?
                latent_mean_type_prior, latent_var_type_prior = (
                    z1_mean_niche[:, type, :],
                    z1_var_niche[:, type, :],
                )
                niche_weights_type = niche_weights[:, type]
                latent_mean_type_posterior, latent_var_type_posterior = (
                    niche_mean_mat[:, type, :],  # batch_size times n_latent
                    niche_var_mat[:, type, :],
                )

                niche_type_prior_distribution = Normal(
                    latent_mean_type_prior, latent_var_type_prior.sqrt()
                )
                niche_type_posterior_distribution = Normal(
                    latent_mean_type_posterior, latent_var_type_posterior.sqrt()
                )

                weighted_reconst_loss_niche += niche_weights_type * kl(
                    niche_type_posterior_distribution, niche_type_prior_distribution
                ).sum(dim=-1)

        elif self.niche_combination == "observed":
            reconst_loss_niche = (
                -generative_outputs["niche_expression"]
                .log_prob(z1_mean_niche)
                .sum(dim=(-1))
            )

            weighted_reconst_loss_niche = (reconst_loss_niche * niche_weights).sum(
                dim=-1
            )

        # COMPOSITION LOSS----------------------------------------------------------------
        true_niche_composition = tensors[REGISTRY_KEYS.NICHE_COMPOSITION_KEY] + epsilon
        true_niche_composition = true_niche_composition / true_niche_composition.sum(
            dim=-1,
            keepdim=True,
        )

        reconstructed_niche_composition = generative_outputs["niche_composition"]

        composition_loss = -reconstructed_niche_composition.log_prob(
            true_niche_composition
        )

        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=-1
        )

        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss_cell = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        # weighted_kl_local = (
        #     kl_local_for_warmup + kl_local_no_warmup
        # )  # means that we ignore kl_weight warmup

        _weighted_reconst_loss_cell = self.cell_rec_weight * reconst_loss_cell
        _weighted_reconst_loss_niche = (
            self.niche_rec_weight * weighted_reconst_loss_niche
        )
        _weighted_composition_loss = self.niche_compo_weight * composition_loss
        _weighted_kl_local = self.latent_kl_weight * weighted_kl_local

        loss = torch.mean(
            _weighted_reconst_loss_cell
            + _weighted_reconst_loss_niche
            + _weighted_kl_local
            + _weighted_composition_loss
        )

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss_cell,
            kl_local=kl_local,
            # classification_loss=composition_loss,
            # true_labels=true_niche_composition,
            # logits=reconstructed_niche_composition,
            extra_metrics={
                "niche_compo": torch.mean(composition_loss),
                "niche_reconst": torch.mean(weighted_reconst_loss_niche),
            },
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale samples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = {"n_samples": n_samples}
        (
            _,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        dist = generative_outputs["px"]
        if self.gene_likelihood == "poisson":
            l_train = generative_outputs["px"].rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(
        self,
        tensors,
        n_mc_samples,
        return_mean=False,
        n_mc_samples_per_pass=1,
    ):
        """Computes the marginal log likelihood of the model.

        Parameters
        ----------
        tensors
            Dict of input tensors, typically corresponding to the items of the data loader.
        n_mc_samples
            Number of Monte Carlo samples to use for the estimation of the marginal log likelihood.
        return_mean
            Whether to return the mean of marginal likelihoods over cells.
        n_mc_samples_per_pass
            Number of Monte Carlo samples to use per pass. This is useful to avoid memory issues.
        """
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            logger.warn(
                "Number of chunks is larger than the total number of samples, setting it to the number of samples"
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(
                tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass}
            )
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl


class LDVAE(nicheVAE):
    """Linear-decoded Variational auto-encoder model.

    Implementation of :cite:p:`Svensson20`.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset. It may also be used
    for all scVI tasks, like differential expression, batch correction, imputation, etc.
    However, batch correction may be less powerful as it assumes a linear model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        **vae_kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=False,
            **vae_kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
        )

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.use_batch_norm is True:
            w = self.decoder.factor_regressor.fc_layers[0][0].weight
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings
