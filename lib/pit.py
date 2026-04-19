"""Probability Integral Transform (PIT) diagnostics for binned PDF photo-z.

The PIT value of a predicted PDF at the true spectroscopic redshift
is the CDF of that PDF evaluated at the true redshift. If the PDFs
are well calibrated, PIT values across the sample are uniformly
distributed on ``[0, 1]``; any departure from uniformity points at a
specific calibration pathology:

* **U-shape** (peaks at 0 and 1) — PDFs are too narrow → overconfident.
* **Inverse-U** (peak in the middle) — PDFs are too wide →
  underconfident.
* **Left-heavy** — model systematically over-estimates z.
* **Right-heavy** — model systematically under-estimates z.

This module exposes two helpers:

* :func:`compute_pit` — vectorised PIT computation **plus** a full
  distribution-shape diagnosis, returned as a single dict.
* :func:`plot_pit` — standard 3-panel PIT figure (histogram +
  Q-Q + CDF comparison).

The current API assumes a **binned** PDF — ``probs`` of shape
``(n_sources, n_bins)`` and ``bin_edges`` of shape
``(n_bins + 1,)``. That matches the NNC and LSTM-classifier flavors
of our photo-z models. Continuous-PDF variants (e.g. MDN output)
would need their own helper.

Example:
    >>> from lib.pit import compute_pit, plot_pit
    >>> result = compute_pit(probs, true_z, bin_edges)
    >>> result["is_calibrated"], result["problem"]
    (True, 'none')
    >>> plot_pit(result["pit_values"], save_path="pit.png")
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["compute_pit", "plot_pit"]


# KS-statistic thresholds for the binary "is this calibrated?" call.
# We key off the statistic rather than the p-value because the p-value
# is unreliable at large N (millions of sources).
_KS_EXCELLENT = 0.01
_KS_GOOD = 0.02
_KS_ACCEPTABLE = 0.05


def compute_pit(
    probs: ArrayLike,
    true_z: ArrayLike,
    bin_edges: ArrayLike,
    *,
    n_hist_bins: int = 20,
) -> dict[str, Any]:
    """Compute PIT values and diagnose the calibration shape.

    Returns a single dict holding both the raw PIT values and a full
    calibration diagnosis (KS stat, shape ratios, a human-readable
    ``problem`` label and a suggested remedy).

    Args:
        probs: ``(n_sources, n_bins)`` per-bin probabilities. Each
            row should sum to 1.
        true_z: ``(n_sources,)`` true redshifts.
        bin_edges: ``(n_bins + 1,)`` bin edges, sorted ascending.
        n_hist_bins: Number of histogram bins used by the shape-ratio
            analysis. Defaults to 20.

    Returns:
        Dict with keys:

        * ``"pit_values"`` — ``(n_sources,)`` PIT values in ``[0, 1]``.
        * ``"ks_stat"``, ``"ks_pval"`` — Kolmogorov-Smirnov test vs a
          uniform distribution.
        * ``"is_calibrated"`` — ``True`` iff ``ks_stat < 0.02``.
        * ``"pit_mean"``, ``"pit_std"`` — sample mean / std of the PIT
          values.
        * ``"left_ratio"``, ``"right_ratio"``, ``"center_ratio"`` —
          ratio of observed histogram density to the expected uniform
          density in the left / right quarter and the central half of
          ``[0, 1]``. Values > 1 mean over-populated.
        * ``"problem"`` — one of ``"overconfident"``,
          ``"underconfident"``, ``"high_bias"``, ``"low_bias"``,
          ``"minor"``, ``"needs_calibration"``, ``"none"``.
        * ``"suggestion"`` — short human-readable remedy.

    Raises:
        ImportError: ``scipy`` is not installed.
    """
    from scipy import stats

    probs = np.asarray(probs)
    true_z = np.asarray(true_z)
    bin_edges = np.asarray(bin_edges)

    n_samples, n_bins = probs.shape

    # Vectorised PIT: the CDF at the left edge of the true-z bin plus
    # the fraction of that bin lying below true_z weighted by the bin
    # probability. This replaces the per-row Python loop in the
    # original implementation for a 10-100x speed-up.
    cumsum = np.cumsum(probs, axis=1)
    bin_idx = np.clip(np.digitize(true_z, bin_edges) - 1, 0, n_bins - 1)

    rows = np.arange(n_samples)
    cdf_left = np.where(bin_idx > 0, cumsum[rows, bin_idx - 1], 0.0)
    bin_lo = bin_edges[bin_idx]
    bin_hi = bin_edges[bin_idx + 1]
    frac = np.clip((true_z - bin_lo) / (bin_hi - bin_lo), 0, 1)
    pit_values = cdf_left + probs[rows, bin_idx] * frac

    # KS test against the uniform distribution.
    ks_stat, ks_pval = stats.kstest(pit_values, "uniform")

    # Shape-ratio diagnostics from a coarse histogram.
    hist, _ = np.histogram(pit_values, bins=n_hist_bins, range=(0, 1))
    expected = n_samples / n_hist_bins
    quarter = n_hist_bins // 4
    half = n_hist_bins // 2

    left_ratio = hist[:quarter].sum() / (expected * quarter)
    right_ratio = hist[-quarter:].sum() / (expected * quarter)
    center_ratio = hist[quarter:-quarter].sum() / (expected * half)

    problem, suggestion = _classify_pit_shape(
        ks_stat=float(ks_stat),
        left_ratio=float(left_ratio),
        right_ratio=float(right_ratio),
        center_ratio=float(center_ratio),
    )

    return {
        "pit_values": pit_values,
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "is_calibrated": bool(ks_stat < _KS_GOOD),
        "pit_mean": float(np.mean(pit_values)),
        "pit_std": float(np.std(pit_values)),
        "left_ratio": float(left_ratio),
        "right_ratio": float(right_ratio),
        "center_ratio": float(center_ratio),
        "problem": problem,
        "suggestion": suggestion,
    }


def _classify_pit_shape(
    *,
    ks_stat: float,
    left_ratio: float,
    right_ratio: float,
    center_ratio: float,
) -> tuple[str, str]:
    """Map a PIT-histogram shape to a calibration-problem label."""
    if left_ratio > 1.3 and right_ratio > 1.3:
        return (
            "overconfident",
            "PDFs are too narrow. Use temperature > 1 to soften.",
        )
    if center_ratio > 1.3:
        return (
            "underconfident",
            "PDFs are too wide. Use temperature < 1 to sharpen.",
        )
    if left_ratio > 1.5:
        return (
            "high_bias",
            "Model systematically overestimates redshift.",
        )
    if right_ratio > 1.5:
        return (
            "low_bias",
            "Model systematically underestimates redshift.",
        )
    if ks_stat < _KS_EXCELLENT:
        return ("none", "Excellent calibration.")
    if ks_stat < _KS_GOOD:
        return ("none", "Good calibration.")
    if ks_stat < _KS_ACCEPTABLE:
        return ("minor", "Acceptable; temperature scaling may help.")
    return ("needs_calibration", "Consider temperature scaling.")


def plot_pit(
    pit_values: ArrayLike,
    *,
    save_path: Optional[str] = None,
    title: str = "PIT Diagnosis",
    n_hist_bins: int = 20,
    show: bool = False,
):
    """Render the standard 3-panel PIT diagnostic figure.

    The three panels are:

    1. PIT histogram vs the uniform reference density.
    2. Q-Q plot: sorted PIT values vs expected uniform quantiles.
    3. Observed CDF vs the uniform CDF, with the gap shaded.

    Args:
        pit_values: 1-D PIT values from :func:`compute_pit`, expected
            to lie in ``[0, 1]``.
        save_path: If given, save the figure here (dpi=150) and close
            it. Parent directory must already exist.
        title: Figure-level title.
        n_hist_bins: Number of histogram bins in the left panel.
        show: If ``True`` and ``save_path is None``, call
            :func:`matplotlib.pyplot.show`. Ignored when ``save_path``
            is given.

    Returns:
        The :class:`matplotlib.figure.Figure` when neither
        ``save_path`` nor ``show`` is set; otherwise ``None``.

    Raises:
        ImportError: ``matplotlib`` is not installed.
    """
    import matplotlib.pyplot as plt

    pit_values = np.asarray(pit_values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) PIT histogram vs uniform density.
    axes[0].hist(
        pit_values,
        bins=n_hist_bins,
        range=(0, 1),
        density=True,
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    axes[0].axhline(
        y=1, color="r", linestyle="--", linewidth=2, label="Ideal (uniform)"
    )
    axes[0].set_xlabel("PIT value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("PIT histogram")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # (2) Q-Q plot: sorted PIT vs expected quantile.
    sorted_pit = np.sort(pit_values)
    expected_q = np.linspace(0, 1, len(pit_values))
    axes[1].plot(expected_q, sorted_pit, "b.", alpha=0.1, markersize=1)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Ideal")
    axes[1].set_xlabel("Expected quantile")
    axes[1].set_ylabel("Observed PIT")
    axes[1].set_title("PIT Q-Q plot")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    # (3) Empirical vs ideal CDF.
    axes[2].plot(
        sorted_pit, expected_q, "b-", linewidth=2, label="Observed CDF"
    )
    axes[2].plot(
        [0, 1], [0, 1], "r--", linewidth=2, label="Ideal (uniform)"
    )
    axes[2].fill_between(
        sorted_pit, expected_q, np.linspace(0, 1, len(pit_values)),
        alpha=0.3, color="blue",
    )
    axes[2].set_xlabel("PIT value")
    axes[2].set_ylabel("CDF")
    axes[2].set_title("PIT CDF comparison")
    axes[2].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    if show:
        plt.show()
        return None
    return fig
