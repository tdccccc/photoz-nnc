"""Regression MLP for photo-z point estimation.

The model has the same hidden-block architecture as
:class:`models.photoz.NNC.model.PhotozBinningClassifier`
(``Linear → Norm → Activation → Dropout`` with optional residual
skips) but ends with a single-output linear head instead of an
N-bin classifier.

Example:
    >>> from models.photoz.ANN.model import PhotozRegressor
    >>> model = PhotozRegressor(input_dim=12)
    >>> model(torch.randn(4, 12)).shape
    torch.Size([4, 1])
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn
from torch import Tensor

__all__ = ["ACTIVATION_FUNCTIONS", "PhotozRegressor"]


ACTIVATION_FUNCTIONS: dict[str, type] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class PhotozRegressor(nn.Module):
    """MLP that outputs a single scalar redshift prediction.

    Args:
        input_dim: Number of input features.
        hidden_dims: Widths of the hidden layers. ``None`` falls
            back to ``[512, 256, 128, 64]``.
        dropout_rate: Dropout probability after the activation.
        activation_function: Key into :data:`ACTIVATION_FUNCTIONS`.
        use_batch_norm: Insert BatchNorm after each hidden linear.
        use_layer_norm: Insert LayerNorm instead (ignored if
            ``use_batch_norm`` is also ``True``).
        use_residual: Add a residual skip around every hidden block.

    Raises:
        ValueError: ``activation_function`` is not in
            :data:`ACTIVATION_FUNCTIONS`.
    """

    def __init__(
        self,
        input_dim: int,
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
                self.residual_layers.append(
                    nn.Linear(in_dim, h) if in_dim != h else nn.Identity()
                )
            else:
                self.residual_layers.append(None)

            in_dim = h

        self.output = nn.Linear(in_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
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
        """Return a scalar redshift prediction of shape ``(batch, 1)``."""
        for layer, res in zip(self.layers, self.residual_layers):
            identity = x
            out = layer(x)
            if self.use_residual and res is not None:
                if isinstance(res, nn.Identity):
                    out = out + identity
                else:
                    out = out + res(identity)
            x = out
        return self.output(x)
