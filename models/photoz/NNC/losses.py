"""Loss functions for binned-PDF photo-z training.

This module is where users pick / swap / combine training objectives.
Every loss takes the model output as **logits** (softmax is applied
inside the loss) so the model stays free of softmax and downstream
code can still access the raw logits when needed.

Two families are provided:

1. **Likelihood-style** (compare predicted probability to the soft
   target distribution):
   :func:`kl_divergence_with_softmax_logits`,
   :func:`weighted_kl_divergence_loss`, :func:`focal_loss`,
   :func:`anchor_loss`.
2. **CDF-style** (compare predicted CDF to the step-function true
   CDF): :func:`crps_loss`, :func:`emd_loss`.

:func:`combined_loss` mixes components from both families with
per-component weights, plus optional entropy regularisation and
redshift-range sample weighting.

The numpy helpers :func:`compute_bin_weights` and
:func:`compute_sample_weights` produce the weight tensors that the
weighted KL / combined loss variants consume.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "kl_divergence_with_softmax_logits",
    "weighted_kl_divergence_loss",
    "focal_loss",
    "anchor_loss",
    "crps_loss",
    "emd_loss",
    "combined_loss",
    "compute_bin_weights",
    "compute_sample_weights",
]


Reduction = Literal["mean", "sum", "none"]


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    """Apply the ``mean`` / ``sum`` / ``none`` reduction to a loss."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _true_cdf_step(true_z: Tensor, bin_edges: Tensor) -> Tensor:
    """Step-function CDF at ``true_z`` evaluated at every right-edge.

    Returns a ``(batch_size, num_bins)`` tensor where entry ``[i, j]``
    is 1 if ``bin_edges[j + 1] >= true_z[i]`` else 0. Shared by
    :func:`crps_loss` and :func:`emd_loss`.
    """
    if true_z.dim() == 2:
        true_z = true_z.squeeze(1)
    bin_rights = bin_edges[1:]
    return (bin_rights.unsqueeze(0) >= true_z.unsqueeze(1)).float()


# ---------------------------------------------------------------------------
# Likelihood-style losses
# ---------------------------------------------------------------------------


def kl_divergence_with_softmax_logits(
    logits: Tensor,
    target_probs: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Soft cross-entropy between model logits and target probabilities.

    Equivalent to KL divergence for optimisation purposes (the target
    distribution is fixed, so its entropy is a constant additive
    term that does not affect gradients). Uses
    :func:`torch.nn.functional.log_softmax` internally for numerical
    stability.

    Args:
        logits: ``(batch, num_bins)`` unnormalised model outputs.
        target_probs: ``(batch, num_bins)`` soft target distribution.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (``mean`` / ``sum``) or ``(batch,)`` vector
        (``none``).
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)
    return _reduce(loss, reduction)


def weighted_kl_divergence_loss(
    logits: Tensor,
    target_probs: Tensor,
    sample_weights: Tensor | None = None,
    bin_weights: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    """KL divergence with optional bin- and sample-level weights.

    Useful for counteracting class imbalance across redshift bins,
    or up-weighting a scarce spectroscopic sub-sample.

    Args:
        logits: ``(batch, num_bins)`` logits.
        target_probs: ``(batch, num_bins)`` soft target.
        sample_weights: ``(batch,)`` weight per row, or ``None``.
        bin_weights: ``(num_bins,)`` weight per bin, or ``None``.
        reduction: Reduction mode.

    Returns:
        Weighted loss, reduced according to ``reduction``.
    """
    log_probs = F.log_softmax(logits, dim=1)
    if bin_weights is not None:
        loss = -(target_probs * log_probs * bin_weights.unsqueeze(0)).sum(dim=1)
    else:
        loss = -(target_probs * log_probs).sum(dim=1)

    if sample_weights is not None:
        loss = loss * sample_weights

    return _reduce(loss, reduction)


def focal_loss(
    logits: Tensor,
    target_probs: Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Focal loss adapted to soft labels.

    Re-weights the cross-entropy by ``(1 - p_t) ** gamma`` where
    ``p_t = sum_k target_probs[k] * softmax(logits)[k]`` is the
    model's expected probability under the target distribution.
    ``gamma > 0`` down-weights easy examples; ``gamma == 0`` reduces
    to plain cross-entropy scaled by ``alpha``.

    Args:
        logits: ``(batch, num_bins)`` logits.
        target_probs: ``(batch, num_bins)`` soft target.
        alpha: Global scale. Defaults to 1.0.
        gamma: Focusing exponent. Defaults to 2.0.
        reduction: Reduction mode.

    Returns:
        Focal loss.
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    pt = (target_probs * probs).sum(dim=1)
    focal_weight = (1 - pt) ** gamma
    ce_loss = -(target_probs * log_probs).sum(dim=1)
    loss = alpha * focal_weight * ce_loss

    return _reduce(loss, reduction)


def anchor_loss(
    logits: Tensor,
    target_probs: Tensor,
    gamma: float = 0.5,
    slack: float = 0.05,
    anchor: str = "neg",
    reduction: Reduction = "mean",
) -> Tensor:
    """Anchor loss — modulates loss scale by prediction difficulty.

    Reference: Ryou et al., "Anchor Loss: Modulating Loss Scale
    based on Prediction Difficulty" (ICCV 2019). For photo-z, Lee &
    Shin, "Estimation of Photometric Redshifts. I." (AJ, 2021).

    For soft labels the "target bin" is taken as ``argmax`` of
    ``target_probs``. With ``anchor='neg'`` (default), the anchor is
    the max probability among the non-target bins; with
    ``anchor='pos'``, it's the target-bin probability. The per-bin
    weight is ``clamp(1 + gamma * (q - p + slack), min=0)``, so
    harder examples (anchor closer to / above the target prob) get a
    larger weight.

    Args:
        logits: ``(batch, num_bins)`` logits.
        target_probs: ``(batch, num_bins)`` soft target.
        gamma: Modulation strength. 0 recovers plain cross-entropy;
            typical values 0.2 / 0.5 / 1.0.
        slack: Margin term to avoid vanishing gradients.
        anchor: ``"neg"`` or ``"pos"``.
        reduction: Reduction mode.

    Returns:
        Anchor loss.
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    target_bins = target_probs.argmax(dim=1)
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    p_target = probs[idx, target_bins]

    if anchor == "neg":
        mask = torch.ones_like(probs)
        mask[idx, target_bins] = 0
        q = (probs * mask).max(dim=1)[0]
    else:  # anchor == "pos"
        q = p_target

    # Larger when anchor is "close to or above" the target probability.
    anchor_weight = torch.clamp(
        1 + gamma * (q.unsqueeze(1) - probs + slack), min=0
    )
    loss_per_bin = -target_probs * anchor_weight * log_probs
    loss = loss_per_bin.sum(dim=1)

    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# CDF-style losses
# ---------------------------------------------------------------------------


def crps_loss(
    logits: Tensor,
    true_z: Tensor,
    bin_edges: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Continuous Ranked Probability Score loss.

    CRPS integrates ``(F_pred(z) - F_true(z)) ** 2`` over z, where
    ``F_true`` is the step function at ``true_z``. It rewards both
    accuracy and calibration simultaneously.

    Args:
        logits: ``(batch, num_bins)`` logits.
        true_z: ``(batch,)`` or ``(batch, 1)`` true redshifts.
        bin_edges: ``(num_bins + 1,)`` bin edges.
        reduction: Reduction mode.

    Returns:
        CRPS loss.
    """
    probs = F.softmax(logits, dim=1)
    pred_cdf = torch.cumsum(probs, dim=1)
    true_cdf = _true_cdf_step(true_z, bin_edges)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    crps = torch.sum(
        (pred_cdf - true_cdf) ** 2 * bin_widths.unsqueeze(0), dim=1
    )
    return _reduce(crps, reduction)


def emd_loss(
    logits: Tensor,
    true_z: Tensor,
    bin_edges: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Earth Mover's Distance / 1-Wasserstein loss.

    Same as :func:`crps_loss` but with L1 instead of L2:
    ``EMD = integral |F_pred(z) - F_true(z)| dz``. More robust to
    outliers, typically produces less sharp PDFs.

    Args:
        logits: ``(batch, num_bins)`` logits.
        true_z: ``(batch,)`` or ``(batch, 1)`` true redshifts.
        bin_edges: ``(num_bins + 1,)`` bin edges.
        reduction: Reduction mode.

    Returns:
        EMD loss.
    """
    probs = F.softmax(logits, dim=1)
    pred_cdf = torch.cumsum(probs, dim=1)
    true_cdf = _true_cdf_step(true_z, bin_edges)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    emd = torch.sum(
        torch.abs(pred_cdf - true_cdf) * bin_widths.unsqueeze(0), dim=1
    )
    return _reduce(emd, reduction)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def combined_loss(
    logits: Tensor,
    target_probs: Tensor,
    true_z: Tensor,
    bin_edges: Tensor,
    config: dict[str, Any],
    reduction: Reduction = "mean",
) -> Tensor:
    """Linear combination of named loss components + optional extras.

    ``config`` structure::

        {
          "components": [
              {"type": "kl",     "weight": 1.0},
              {"type": "crps",   "weight": 0.5},
              {"type": "focal",  "weight": 0.1,
               "params": {"alpha": 1.0, "gamma": 2.0}},
              {"type": "anchor", "weight": 0.1,
               "params": {"gamma": 0.5, "slack": 0.05, "anchor": "neg"}},
              {"type": "emd",    "weight": 0.0},
              {"type": "brier",  "weight": 0.0},
          ],
          "entropy_reg": 0.0,
          "z_weights": {
              "enabled": true,
              "ranges": [{"z_min": 0.0, "z_max": 0.3, "weight": 1.0}, ...]
          }
        }

    ``entropy_reg > 0`` subtracts ``entropy_reg * H(softmax(logits))``
    from the loss, encouraging smoother (higher-entropy) PDFs.
    ``z_weights`` applies a per-sample multiplier based on which
    range ``true_z`` falls into.

    Args:
        logits: ``(batch, num_bins)`` logits.
        target_probs: ``(batch, num_bins)`` soft target.
        true_z: ``(batch,)`` or ``(batch, 1)`` true redshifts.
        bin_edges: ``(num_bins + 1,)`` bin edges.
        config: See structure above.
        reduction: Reduction mode.

    Returns:
        Combined loss reduced according to ``reduction``.

    Raises:
        ValueError: An unknown component ``type`` appears in
            ``config["components"]``.
    """
    if true_z.dim() == 2:
        true_z = true_z.squeeze(1)

    batch_size = logits.size(0)
    device = logits.device

    # Per-sample weights from redshift-range config.
    sample_weights = torch.ones(batch_size, device=device)
    z_weights_config = config.get("z_weights")
    if z_weights_config is not None and z_weights_config.get("enabled", False):
        for range_cfg in z_weights_config.get("ranges", []):
            mask = (
                (true_z >= range_cfg["z_min"])
                & (true_z < range_cfg["z_max"])
            )
            sample_weights[mask] = range_cfg["weight"]

    # Sum components with per-component weight.
    total_loss = torch.zeros(batch_size, device=device)
    components = config.get("components", [{"type": "kl", "weight": 1.0}])

    for comp in components:
        loss_type = comp["type"]
        weight = comp.get("weight", 1.0)
        if weight == 0:
            continue

        if loss_type == "kl":
            log_probs = F.log_softmax(logits, dim=1)
            comp_loss = -(target_probs * log_probs).sum(dim=1)
        elif loss_type == "crps":
            comp_loss = crps_loss(logits, true_z, bin_edges, reduction="none")
        elif loss_type == "emd":
            comp_loss = emd_loss(logits, true_z, bin_edges, reduction="none")
        elif loss_type == "focal":
            p = comp.get("params", {})
            comp_loss = focal_loss(
                logits, target_probs,
                alpha=p.get("alpha", 1.0),
                gamma=p.get("gamma", 2.0),
                reduction="none",
            )
        elif loss_type == "anchor":
            p = comp.get("params", {})
            comp_loss = anchor_loss(
                logits, target_probs,
                gamma=p.get("gamma", 0.5),
                slack=p.get("slack", 0.05),
                anchor=p.get("anchor", "neg"),
                reduction="none",
            )
        elif loss_type == "brier":
            probs = F.softmax(logits, dim=1)
            comp_loss = ((probs - target_probs) ** 2).sum(dim=1)
        else:
            raise ValueError(f"Unknown loss component type: {loss_type!r}")

        total_loss = total_loss + weight * comp_loss

    # Entropy regularisation (subtracted → encourages higher entropy).
    entropy_reg = config.get("entropy_reg", 0.0)
    if entropy_reg > 0:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)
        total_loss = total_loss - entropy_reg * entropy

    total_loss = total_loss * sample_weights
    return _reduce(total_loss, reduction)


# ---------------------------------------------------------------------------
# Bin / sample weight helpers (numpy)
# ---------------------------------------------------------------------------


def compute_bin_weights(
    bin_centers: np.ndarray,
    labels: np.ndarray,
    method: str = "inverse_frequency",
) -> np.ndarray:
    """Per-bin weight from the empirical label histogram.

    Args:
        bin_centers: ``(num_bins,)`` centres of the redshift bins.
        labels: ``(n_samples,)`` true redshift values to build the
            histogram from.
        method: One of:

            * ``"inverse_frequency"`` — ``1 / count`` (default).
            * ``"balanced"`` — ``len(labels) / (num_bins * count)``.
            * ``"sqrt_inverse"`` — ``1 / sqrt(count)``.

    Returns:
        ``(num_bins,)`` ``float32`` array, normalised so the mean
        weight equals 1.

    Raises:
        ValueError: Unknown ``method``.
    """
    # Convert bin centres to edges (midpoints + extrapolated outer edges).
    bin_edges = np.concatenate([
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2],
    ])

    counts, _ = np.histogram(labels, bins=bin_edges)
    counts = np.maximum(counts, 1)  # Avoid division by zero.

    if method == "inverse_frequency":
        weights = 1.0 / counts
    elif method == "balanced":
        weights = len(labels) / (len(bin_centers) * counts)
    elif method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_sample_weights(
    labels: np.ndarray,
    bin_centers: np.ndarray,
    bin_weights: np.ndarray,
) -> np.ndarray:
    """Broadcast :func:`compute_bin_weights` output to per-sample.

    Each sample is assigned the weight of its closest bin centre.

    Args:
        labels: ``(n_samples,)`` true redshift values.
        bin_centers: ``(num_bins,)`` bin centres.
        bin_weights: ``(num_bins,)`` from :func:`compute_bin_weights`.

    Returns:
        ``(n_samples,)`` ``float32`` array.
    """
    bin_indices = np.searchsorted(bin_centers, labels, side="left")
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    return bin_weights[bin_indices].astype(np.float32)
