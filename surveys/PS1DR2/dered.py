"""PS1DR2 Galactic extinction correction via the SFD dust map.

Two building blocks:

* :func:`query_ebv` — look up SFD E(B-V) per source from a DataFrame's
  sky-position columns.
* :func:`apply_extinction` — given per-source E(B-V), write dereddened
  PS1 grizy magnitude columns back onto the DataFrame.

Typical usage::

    ebv = query_ebv(df)                 # df has Galactic l, b columns
    df = apply_extinction(df, ebv)      # adds ``<band><mag_type>_dered``

Splitting the query from the application lets callers cache E(B-V)
once and reuse it, e.g. apply the same E(B-V) to PSF and Kron
photometry, or write E(B-V) back to disk alongside the catalog.

The extinction coefficients come from Schlafly & Finkbeiner (2011),
ApJ, 737, 103, Table 6, assuming R_V = 3.1 and the SFD E(B-V) map.

Dependencies and setup
----------------------
Requires the ``dustmaps`` Python package and the SFD data files.
One-time setup::

    pip install dustmaps

    python -c "
    from dustmaps.config import config
    config['data_dir'] = '/path/to/store/dust_maps'
    import dustmaps.sfd
    dustmaps.sfd.fetch()
    "

The ``data_dir`` can either be hard-coded as above, or resolved via
the ``DUSTMAPS_DATA_DIR`` environment variable. See
`<https://dustmaps.readthedocs.io/>`_ for details.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

__all__ = ["PS1_SFD_COEFFS", "query_ebv", "apply_extinction"]


#: Pan-STARRS1 grizy extinction coefficients A_band / E(B-V) for SFD
#: E(B-V), assuming R_V = 3.1. Schlafly & Finkbeiner (2011), ApJ, 737,
#: 103, Table 6.
PS1_SFD_COEFFS: dict[str, float] = {
    "g": 3.172,
    "r": 2.271,
    "i": 1.682,
    "z": 1.322,
    "y": 1.087,
}


def query_ebv(
    df: pd.DataFrame,
    *,
    lon_col: str = "l",
    lat_col: str = "b",
    frame: str = "galactic",
) -> np.ndarray:
    """Query SFD E(B-V) for every row in a PS1 catalog.

    Args:
        df: PS1 catalog. Must contain the position columns
            ``(lon_col, lat_col)``.
        lon_col: Longitude column name. Defaults to ``"l"`` (Galactic
            longitude). Switch to ``"ra"`` together with
            ``frame="icrs"`` for equatorial inputs.
        lat_col: Latitude column name. Defaults to ``"b"``. Switch to
            ``"dec"`` for equatorial inputs.
        frame: Coordinate frame accepted by
            :class:`astropy.coordinates.SkyCoord` (``"galactic"`` or
            ``"icrs"``). ``SkyCoord`` converts to whatever frame the
            SFD map expects internally.

    Returns:
        1-D ``np.ndarray`` of E(B-V) with the same length as ``df``.

    Raises:
        KeyError: ``lon_col`` or ``lat_col`` is missing from ``df``.
        ImportError: ``dustmaps`` is not installed, or the SFD data
            files have not been fetched.

    Example:
        >>> ebv = query_ebv(df)                          # Galactic l, b
        >>> ebv = query_ebv(df, lon_col="ra",            # equatorial
        ...                 lat_col="dec", frame="icrs")
    """
    from dustmaps.sfd import SFDQuery

    for pos_col in (lon_col, lat_col):
        if pos_col not in df.columns:
            raise KeyError(
                f"Column {pos_col!r} is missing from the DataFrame."
            )

    coords = SkyCoord(
        df[lon_col].values,
        df[lat_col].values,
        unit="deg",
        frame=frame,
    )
    return np.asarray(SFDQuery()(coords))


def apply_extinction(
    df: pd.DataFrame,
    ebv: ArrayLike,
    *,
    mag_types: Sequence[str] = ("PSFMag", "KronMag", "ApMag"),
    suffix: str = "_dered",
    inplace: bool = False,
) -> pd.DataFrame:
    """Deredden PS1 grizy magnitudes with a pre-computed E(B-V).

    For every ``band`` in ``grizy`` and every ``mag_type`` in
    ``mag_types``, writes::

        df[f"{band}{mag_type}{suffix}"] = df[f"{band}{mag_type}"] \\
                                          - PS1_SFD_COEFFS[band] * ebv

    Args:
        df: PS1 catalog. Must contain every magnitude column
            ``f"{band}{mag_type}"`` for ``band`` in ``grizy``.
        ebv: Per-row E(B-V), typically the output of :func:`query_ebv`.
            Must have the same length as ``df``.
        mag_types: PS1 magnitude-type suffixes to deredden. Defaults
            to the three PSF / Kron / aperture flavors used by the
            paper pipeline.
        suffix: Output column suffix. Defaults to ``"_dered"``.
        inplace: If ``True``, mutate ``df`` directly; otherwise update
            a shallow copy.

    Returns:
        The DataFrame with ``<band><mag_type><suffix>`` columns added.

    Raises:
        ValueError: ``len(ebv) != len(df)``.
        KeyError: A required magnitude column is missing from ``df``.

    Example:
        >>> ebv = query_ebv(df)
        >>> df = apply_extinction(df, ebv)
        >>> df = apply_extinction(df, ebv, mag_types=("KronMag",))
    """
    ebv = np.asarray(ebv)
    if len(ebv) != len(df):
        raise ValueError(
            f"len(ebv)={len(ebv)} does not match len(df)={len(df)}."
        )

    out = df if inplace else df.copy()
    for band, coeff in PS1_SFD_COEFFS.items():
        a_band = coeff * ebv
        for mag_type in mag_types:
            col = f"{band}{mag_type}"
            if col not in out.columns:
                raise KeyError(
                    f"Column {col!r} is missing from the DataFrame."
                )
            out[f"{col}{suffix}"] = out[col] - a_band
    return out
