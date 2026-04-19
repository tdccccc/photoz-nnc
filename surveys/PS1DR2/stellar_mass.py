"""PS1DR2 photometric log-stellar-mass estimator (empirical template).

Implements the ``calculate_mstar`` shortcut used in the legacy
``cosmic.panstarrs_dr2.dataProcess`` module for Pan-STARRS DR2 Kron
photometry.

The estimator interpolates over two precomputed redshift grids:

* ``char_mag_zbins_center`` — 105 fine redshift bins whose
  corresponding z-band characteristic magnitude ``z_char_mag`` sets
  the magnitude reference at each redshift.
* ``zbins_center`` — 8 coarse redshift bins providing the slope
  ``z_a(z)`` and intercept ``z_b(z)`` of the linear ``r − z`` colour
  term.

For each source with redshift ``z`` and dereddened PS1 Kron
magnitudes ``r_Kron_dered`` / ``z_Kron_dered`` the log stellar mass
is computed as::

    logL   = -0.4 * (z_Kron_dered - z_char_mag(z))
    f      = z_a(z) + z_b(z) * (r_Kron_dered - z_Kron_dered)
    log M* = z_gamma * logL + f             (capped at MSTAR_CAP)

The relation is PS1-band specific (tied to the PS1 Kron grizy
calibration and a Wen-style cluster-member template), which is why
this helper lives under the PS1 survey module rather than in the
shared ``lib`` layer.

Example:
    >>> import numpy as np
    >>> from surveys.PS1DR2.stellar_mass import calculate_mstar
    >>> z = np.array([0.1, 0.3, 0.5])
    >>> r = np.array([19.0, 20.0, 21.0])
    >>> zmag = np.array([18.0, 19.0, 20.0])
    >>> calculate_mstar(z, r, zmag)
    array([...])
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["MSTAR_CAP", "get_mstar_params", "calculate_mstar"]


#: Upper cap applied to log10(M*/Msun). Empirical ceiling from the
#: template's validity range; sources above this are clipped.
MSTAR_CAP: float = 12.7


def get_mstar_params() -> dict[str, Any]:
    """Return the precomputed stellar-mass lookup tables.

    The returned dictionary holds two redshift grids and the
    associated calibration coefficients used by
    :func:`calculate_mstar`:

    * ``char_mag_zbins_center`` (105,) — fine redshift bin centers.
    * ``z_char_mag``            (105,) — z-band characteristic
      magnitude at each bin center.
    * ``zbins_center``            (8,) — coarse redshift bin centers
      for the r-z colour term.
    * ``z_a``, ``z_b``            (8,) — slope and intercept of the
      linear ``r - z`` colour term in each coarse bin.
    * ``z_gamma``                scalar — global luminosity prefactor.

    The numeric values are fixed calibration outputs derived from
    the PS1 cluster-member template. Do not edit in place; callers
    that need a tweak should copy the dict first.

    Returns:
        Dict with the six keys listed above.
    """
    return {
        "char_mag_zbins_center": np.array([
            0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
            0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09,
            0.095, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
            0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5,
            0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725,
            0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95,
            0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175,
            1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4,
            1.425, 1.45, 1.475, 1.5, 1.525, 1.55, 1.575, 1.6, 1.625,
            1.65, 1.675, 1.7, 1.725, 1.75, 1.775, 1.8, 1.825, 1.85,
            1.875, 1.9, 1.925, 1.95, 1.975, 2.0, 2.1, 2.2, 2.3,
            2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.988,
        ]),
        "z_char_mag": np.array([
            9.6927, 11.1998, 12.0838, 12.7126, 13.2012, 13.6006, 13.9392,
            14.2334, 14.494, 14.7281, 14.9408, 15.1355, 15.3133, 15.4772,
            15.6296, 15.771, 15.9027, 16.0272, 16.146, 16.26, 16.7682,
            17.2026, 17.5643, 17.8812, 18.1703, 18.4311, 18.6559, 18.8542,
            19.0498, 19.2235, 19.3845, 19.542, 19.6855, 19.8289, 19.9639,
            20.0889, 20.2123, 20.3315, 20.4534, 20.5811, 20.6956, 20.8026,
            20.9028, 20.9989, 21.088, 21.1793, 21.2506, 21.3149, 21.3892,
            21.4683, 21.559, 21.6519, 21.7532, 21.8516, 21.9447, 22.0315,
            22.1192, 22.2063, 22.3179, 22.4295, 22.5397, 22.6457, 22.7458,
            22.8256, 22.897, 22.966, 23.0521, 23.1529, 23.2346, 23.2771,
            23.318, 23.347, 23.3692, 23.4011, 23.4254, 23.4564, 23.4991,
            23.5449, 23.5807, 23.6114, 23.648, 23.6768, 23.7073, 23.7408,
            23.7757, 23.8052, 23.8404, 23.8775, 23.9056, 23.9356, 23.9678,
            24.0039, 24.0452, 24.0992, 24.1506, 24.1984, 24.2934, 24.3757,
            24.2617, 24.0119, 23.7251, 23.258, 22.5805, 21.8331, 20.6888,
            17.019,
        ]),
        "zbins_center": np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
        "z_a": np.array([
            11.13861784, 10.49609211, 10.66235534, 10.26808033,
            10.13127691, 9.83012158, 11.32148531, 8.24798616,
        ]),
        "z_b": np.array([
            -0.36245079, 0.28450657, 0.12026569, 0.30011592,
            0.23783054, 0.06002614, -0.96828277, 0.47226136,
        ]),
        "z_gamma": 1.5681417103650086,
    }


def calculate_mstar(
    z: ArrayLike,
    rKronMag_dered: ArrayLike,
    zKronMag_dered: ArrayLike,
    params: Optional[dict[str, Any]] = None,
    *,
    mstar_cap: float = MSTAR_CAP,
) -> np.ndarray:
    """Compute log stellar mass from PS1 Kron dereddened magnitudes.

    For each source, the redshift ``z`` is mapped to the nearest
    fine bin (for the z-band characteristic magnitude) and the
    nearest coarse bin (for the r-z colour term coefficients), and
    the following relation is evaluated::

        logL   = -0.4 * (zKronMag_dered - z_char_mag(z))
        f      = z_a(z) + z_b(z) * (rKronMag_dered - zKronMag_dered)
        log M* = z_gamma * logL + f

    Sources with ``log M* > mstar_cap`` are clipped to ``mstar_cap``.

    Args:
        z: Source redshift(s) (spectroscopic or photometric).
        rKronMag_dered: Dereddened PS1 r-band Kron magnitude(s).
            Must be broadcast-compatible with ``z``.
        zKronMag_dered: Dereddened PS1 z-band Kron magnitude(s).
            Must be broadcast-compatible with ``z``.
        params: Calibration dictionary as returned by
            :func:`get_mstar_params`. If ``None`` (default), the
            table is allocated inside the function.
        mstar_cap: Upper bound applied to the returned log M*.
            Defaults to :data:`MSTAR_CAP` (``12.7``).

    Returns:
        ``np.ndarray`` of log10(M*/Msun) with the same length as the
        input photometry.
    """
    if params is None:
        params = get_mstar_params()

    z = np.asarray(z)
    rKron = np.asarray(rKronMag_dered)
    zKron = np.asarray(zKronMag_dered)

    # Nearest-neighbour bin lookup per source (one pass each grid).
    bin_indices_char = np.array([
        (abs(params["char_mag_zbins_center"] - z_val)).argmin()
        for z_val in z
    ])
    bin_indices = np.array([
        (abs(params["zbins_center"] - z_val)).argmin()
        for z_val in z
    ])

    # Luminosity relative to the z-band characteristic magnitude.
    logL_z = -0.4 * (zKron - params["z_char_mag"][bin_indices_char])

    # Linear r-z colour term, per coarse redshift bin.
    f_z = (
        params["z_a"][bin_indices]
        + params["z_b"][bin_indices] * (rKron - zKron)
    )

    mstar = params["z_gamma"] * logL_z + f_z
    return np.clip(mstar, a_min=None, a_max=mstar_cap)
