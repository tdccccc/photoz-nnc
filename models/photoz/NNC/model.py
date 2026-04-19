"""Photo-z bin-classifier neural network.

Users wanting to try a different architecture should edit this file;
training code in :mod:`core`, losses in :mod:`losses`, and the data
pipeline in :mod:`dataset` stay untouched.

The single model exported here, :class:`PhotozBinningClassifier`,
takes a feature vector and outputs one logit per redshift bin. The
softmax over logits is the predicted PDF; downstream code combines
that with the bin centres to obtain point estimates and with the bin
edges to compute CRPS/PIT.

Example:
    >>> from models.photoz.NNC.model import PhotozBinningClassifier
    >>> model = PhotozBinningClassifier(
    ...     input_dim=10,
    ...     num_bins=400,
    ...     hidden_dims=[512, 256, 128, 64],
    ...     dropout_rate=0.1,
    ...     activation_function="relu",
    ...     use_batch_norm=True,
    ...     use_residual=True,
    ... )
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn
from torch import Tensor

__all__ = ["ACTIVATION_FUNCTIONS", "PhotozBinningClassifier"]


#: String → ``nn.Module`` factory for the activation functions the
#: classifier supports out of the box. Extend this dict to add more
#: (the factory must take no required args).
ACTIVATION_FUNCTIONS: dict[str, type] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class PhotozBinningClassifier(nn.Module):
    """MLP that outputs one logit per redshift bin.

    Architecture: a stack of ``Linear → Norm → Activation → Dropout``
    blocks, optionally with residual skips, followed by a final
    linear classifier head. Normalisation is either BatchNorm or
    LayerNorm; if both flags are set BatchNorm wins and a warning is
    logged.

    Args:
        input_dim: Number of input features.
        num_bins: Number of output redshift bins.
        hidden_dims: Widths of the hidden layers, read left-to-right.
            ``None`` falls back to ``[512, 256, 128, 64]``.
        dropout_rate: Dropout probability after the activation. Pass
            ``0`` to disable.
        activation_function: Key into :data:`ACTIVATION_FUNCTIONS`.
        use_batch_norm: Insert :class:`torch.nn.BatchNorm1d` after
            each hidden linear layer.
        use_layer_norm: Insert :class:`torch.nn.LayerNorm` instead.
            Ignored (with a warning) when ``use_batch_norm`` is also
            ``True``.
        use_residual: Add a residual skip (with a learned projection
            when input/output widths differ) around every hidden
            block.

    Raises:
        ValueError: ``activation_function`` is not in
            :data:`ACTIVATION_FUNCTIONS`.
    """

    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        hidden_dims: Optional[list[int]] = None,
        dropout_rate: float = 0.1,
        activation_function: str = "relu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_residual: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        if use_batch_norm and use_layer_norm:
            logging.warning(
                "Both BatchNorm and LayerNorm are enabled; using "
                "BatchNorm only."
            )
            use_layer_norm = False

        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unsupported activation function: {activation_function!r}. "
                f"Choose from {sorted(ACTIVATION_FUNCTIONS)}."
            )
        activation_fn = ACTIVATION_FUNCTIONS[activation_function]

        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        in_dim = input_dim
        for h in hidden_dims:
            blocks: list[nn.Module] = [nn.Linear(in_dim, h)]
            if use_batch_norm:
                blocks.append(nn.BatchNorm1d(h))
            elif use_layer_norm:
                blocks.append(nn.LayerNorm(h))
            blocks.append(activation_fn())
            if dropout_rate > 0:
                blocks.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Sequential(*blocks))

            if use_residual:
                # Identity skip when widths match, learned projection
                # otherwise so the residual add is well-defined.
                self.residual_layers.append(
                    nn.Linear(in_dim, h) if in_dim != h else nn.Identity()
                )
            else:
                self.residual_layers.append(None)

            in_dim = h

        self.classifier = nn.Linear(in_dim, num_bins)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming-normal init for linear layers; 1/0 for norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Return per-bin logits of shape ``(batch, num_bins)``."""
        for layer, res in zip(self.layers, self.residual_layers):
            identity = x
            out = layer(x)
            if self.use_residual and res is not None:
                if isinstance(res, nn.Identity):
                    out = out + identity
                else:
                    out = out + res(identity)
            x = out
        return self.classifier(x)
