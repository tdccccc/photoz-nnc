"""Evaluation metrics for photo-z predictions and classifiers.

This module collects the metric helpers used to evaluate the models
trained in this repository.

Current scope
-------------
Redshift **point-estimation** quality, via the normalized residual

.. math::

    \\Delta z_\\mathrm{norm} = \\frac{z_\\mathrm{phot} - z_\\mathrm{spec}}{1 + z_\\mathrm{spec}}

which cancels the natural ``(1 + z)`` growth of the absolute residual.
The conventional 0.15 outlier threshold on
``|delta_z_norm|`` follows Ilbert et al. (2006, A&A, 457, 841) and is
the default for :func:`redshift_outlier_fraction`,
:func:`redshift_bias`, :func:`redshift_scatter`, and
:func:`redshift_quality_metrics`.

Exposed helpers:

* :func:`redshift_residual`
* :func:`redshift_outlier_fraction`
* :func:`redshift_bias`
* :func:`redshift_scatter`
* :func:`redshift_nmad`
* :func:`redshift_quality_metrics` — convenience bundle.

Future additions
----------------
Reserved slots in this module, to be added as needed. Keep the
``<topic>_<metric>`` naming convention so the flat namespace stays
scannable:

* Probabilistic photo-z diagnostics — PIT, zconf, CRPS, ... exposed
  as ``redshift_pit``, ``redshift_zconf``, ``redshift_crps``, ...
* Galaxy / star classifier metrics — accuracy, precision, recall,
  ROC AUC, PR AUC, ... exposed as ``classification_accuracy``,
  ``classification_roc_auc``, ...

Once the file grows past ~15 public helpers, split into
``lib/metrics/redshift.py`` and ``lib/metrics/classification.py``
sub-modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "redshift_residual",
    "redshift_outlier_fraction",
    "redshift_bias",
    "redshift_scatter",
    "redshift_nmad",
    "redshift_quality_metrics",
]


# ---------------------------------------------------------------------------
# Redshift point-estimation metrics
# ---------------------------------------------------------------------------


def redshift_residual(
    z_phot: ArrayLike, z_spec: ArrayLike
) -> np.ndarray:
    """Normalized redshift residual ``(z_phot - z_spec) / (1 + z_spec)``.

    The ``(1 + z)`` denominator flattens the residual scale across
    redshift and is the standard quantity downstream summary metrics
    operate on.

    Args:
        z_phot: Photometric redshift estimates.
        z_spec: Spectroscopic redshifts (same length as ``z_phot``).

    Returns:
        ``np.ndarray`` of normalized residuals, same length as the
        inputs.
    """
    z_phot = np.asarray(z_phot)
    z_spec = np.asarray(z_spec)
    return (z_phot - z_spec) / (1.0 + z_spec)


def redshift_outlier_fraction(
    delta_z_norm: ArrayLike, *, threshold: float = 0.15
) -> float:
    """Fraction of sources with ``|delta_z_norm| > threshold``.

    The canonical 0.15 threshold follows Ilbert et al. (2006).

    Args:
        delta_z_norm: Normalized residuals, typically the output of
            :func:`redshift_residual`.
        threshold: Absolute threshold on the normalized residual.

    Returns:
        Outlier fraction in ``[0, 1]``.
    """
    delta_z_norm = np.asarray(delta_z_norm)
    return float(np.mean(np.abs(delta_z_norm) > threshold))


def redshift_bias(
    delta_z_norm: ArrayLike, *, threshold: float = 0.15
) -> float:
    """Mean of ``delta_z_norm`` over the non-outlier subset.

    Sources with ``|delta_z_norm| > threshold`` are excluded before
    averaging.

    Args:
        delta_z_norm: Normalized residuals.
        threshold: Absolute threshold used to define outliers.

    Returns:
        Mean bias on the retained subset.
    """
    delta_z_norm = np.asarray(delta_z_norm)
    mask = np.abs(delta_z_norm) <= threshold
    return float(np.mean(delta_z_norm[mask]))


def redshift_scatter(
    delta_z_norm: ArrayLike, *, threshold: float = 0.15
) -> float:
    """Standard deviation of ``delta_z_norm`` over the non-outlier subset.

    Commonly written as ``sigma`` or ``sigma_{Delta z / (1 + z)}`` in
    the photo-z literature.

    Args:
        delta_z_norm: Normalized residuals.
        threshold: Absolute threshold used to define outliers.

    Returns:
        Standard deviation on the retained subset.
    """
    delta_z_norm = np.asarray(delta_z_norm)
    mask = np.abs(delta_z_norm) <= threshold
    return float(np.std(delta_z_norm[mask]))


def redshift_nmad(delta_z_norm: ArrayLike) -> float:
    """Robust scatter estimate ``1.4826 * median(|x - median(x)|)``.

    The 1.4826 factor rescales MAD to the Gaussian standard
    deviation for a Gaussian-distributed sample; the result is the
    quantity usually reported as ``sigma_NMAD`` in photo-z papers.
    Unlike :func:`redshift_scatter`, this helper uses the full
    sample without outlier rejection — NMAD is itself robust to
    outliers.

    Args:
        delta_z_norm: Normalized residuals.

    Returns:
        Gaussian-equivalent robust sigma.
    """
    delta_z_norm = np.asarray(delta_z_norm)
    median_dz = np.median(delta_z_norm)
    return float(1.4826 * np.median(np.abs(delta_z_norm - median_dz)))


def redshift_quality_metrics(
    z_phot: ArrayLike,
    z_spec: ArrayLike,
    *,
    threshold: float = 0.15,
) -> dict[str, float]:
    """Compute the four standard point-estimation metrics in one call.

    Convenience wrapper that routes ``(z_phot, z_spec)`` through
    :func:`redshift_residual` and then feeds the result into
    :func:`redshift_bias`, :func:`redshift_scatter`,
    :func:`redshift_nmad`, and :func:`redshift_outlier_fraction`.

    Args:
        z_phot: Photometric redshift estimates.
        z_spec: Spectroscopic redshifts.
        threshold: Absolute threshold on ``|delta_z_norm|`` used by
            the outlier-sensitive metrics.

    Returns:
        Dict with keys ``"bias"``, ``"scatter"``, ``"nmad"``,
        ``"outlier_fraction"``.

    Example:
        >>> stats = redshift_quality_metrics(z_phot, z_spec)
        >>> print(f"sigma_NMAD = {stats['nmad']:.4f}, "
        ...       f"eta = {stats['outlier_fraction']:.2%}")
    """
    delta_z_norm = redshift_residual(z_phot, z_spec)
    return {
        "bias": redshift_bias(delta_z_norm, threshold=threshold),
        "scatter": redshift_scatter(delta_z_norm, threshold=threshold),
        "nmad": redshift_nmad(delta_z_norm),
        "outlier_fraction": redshift_outlier_fraction(
            delta_z_norm, threshold=threshold
        ),
    }
