"""NNC photo-z inference entry point.

Provides programmatic helpers and a CLI:

* :func:`predict_photoz` — ``DataFrame + model_dir → DataFrame`` with
  the three point-estimate columns ``z_mean``, ``z_std``, ``z_mode``.
* :func:`predict_photoz_pdf` — return (or stream to disk) the full
  per-source PDF plus its bin grid. Two modes: in-memory dict or
  streaming to HDF5 for catalogs too big to fit in RAM.
* :func:`rebin_stacked_prob` — helper to re-bin a fine stacked PDF
  onto a coarser grid (useful for plotting N(z)).
* :func:`main` — CLI that reads a catalog, runs
  :func:`predict_photoz`, and optionally also streams the full PDF
  to HDF5.

Usage (CLI):
    # point estimates only
    python -m models.photoz.NNC.predict \\
        --input catalog.fits \\
        --model-dir runs/experiment_01/ \\
        --output catalog_with_photoz.fits

    # also stream the full per-source PDF to HDF5
    python -m models.photoz.NNC.predict \\
        --input catalog.fits \\
        --model-dir runs/experiment_01/ \\
        --output catalog_with_photoz.fits \\
        --output-pdf catalog_pdf.h5
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import numpy as np
import pandas as pd

from lib.io import readfile, savefile

from .inference import PhotozBinInference

__all__ = [
    "predict_photoz",
    "predict_photoz_pdf",
    "rebin_stacked_prob",
    "main",
]


def predict_photoz(
    df: pd.DataFrame,
    model_dir: str,
    calibrator_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Run NNC photo-z inference on a catalog DataFrame.

    Instantiates a :class:`~.inference.PhotozBinInference` from
    ``model_dir``, optionally loads a temperature-scaling
    calibrator, and returns three summary columns.

    Args:
        df: Input catalog with every feature column referenced by the
            model's training config.
        model_dir: Path to a directory produced by training
            (``config.yaml`` + ``scaler.pkl`` + ``best_model.pkl``).
        calibrator_path: Optional path to a ``calibrator.pkl`` from
            :meth:`PhotozBinInference.calibrate`. If ``None``, no
            temperature scaling is applied.
        device: Torch device spec (``"cuda"``, ``"cuda:0"``,
            ``"cpu"``).
        batch_size: Inference batch size.

    Returns:
        DataFrame with three columns, aligned with ``df`` row order:

        * ``z_mean`` — expected redshift
          ``sum_i p_i * bin_center_i``.
        * ``z_std`` — standard deviation of the predicted PDF
          ``sqrt(sum_i p_i * (bin_center_i - z_mean) ** 2)``.
        * ``z_mode`` — centre of the highest-probability bin.

    Example:
        >>> result = predict_photoz(df, "runs/experiment_01/",
        ...                         calibrator_path="runs/experiment_01/calibrator.pkl")
        >>> df[["z_mean", "z_std", "z_mode"]] = result
    """
    infer = PhotozBinInference(
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
    )
    if calibrator_path is not None:
        infer.load_calibrator(calibrator_path)

    result = infer.predict_probabilities(
        data_source=df,
        batch_size=batch_size,
        apply_calibration=(calibrator_path is not None),
    )
    probs = result["probabilities"]          # (n, n_bins)
    bin_centers = result["bin_centers"]      # (n_bins,)
    z_mean = result["expectation"]           # (n,)

    # Standard deviation of the predicted PDF.
    z_diff = bin_centers[np.newaxis, :] - z_mean[:, np.newaxis]
    z_std = np.sqrt(np.sum(probs * z_diff ** 2, axis=1))

    # Mode = centre of the highest-probability bin.
    z_mode = bin_centers[np.argmax(probs, axis=1)]

    return pd.DataFrame({
        "z_mean": z_mean,
        "z_std": z_std,
        "z_mode": z_mode,
    })


def predict_photoz_pdf(
    df: pd.DataFrame,
    model_dir: str,
    calibrator_path: Optional[str] = None,
    save_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 4096,
    chunk_size: int = 100_000,
    compression: str = "gzip",
) -> Optional[dict[str, Any]]:
    """Predict the full per-source PDF.

    Two mutually exclusive output modes, keyed on ``save_path``:

    * ``save_path=None`` — run in memory and **return** a dict with
      the probabilities, bin grid, and point estimate. Requires
      ``n_sources * n_bins * 4`` bytes of RAM to hold the result
      (e.g. 7M sources x 400 bins ≈ 11 GB).
    * ``save_path="<path>.h5"`` — stream the prediction to HDF5 in
      ``chunk_size`` chunks and return ``None``. Use this for large
      catalogs that don't fit in memory.

    Args:
        df: Input catalog with every feature column referenced by
            the model's training config.
        model_dir: Path to a trained model directory (``config.yaml``
            + ``scaler.pkl`` + ``best_model.pkl``).
        calibrator_path: Optional ``calibrator.pkl`` from
            :meth:`PhotozBinInference.calibrate`. If ``None``, no
            temperature scaling is applied.
        save_path: Output HDF5 path (must end in ``.h5``). Enables
            the streaming mode.
        device: Torch device spec (``"cuda"``, ``"cuda:0"``,
            ``"cpu"``).
        batch_size: Inference batch size inside each chunk.
        chunk_size: Streaming mode only — number of sources
            processed per flush to HDF5.
        compression: Streaming mode only — HDF5 compression
            (``"gzip"`` / ``"lzf"`` / ``None``).

    Returns:
        In-memory mode: dict with ``"probabilities"``
        ``(n, n_bins)``, ``"bin_centers"``, ``"bin_edges"``,
        ``"expectation"``, ``"calibrated"``, ``"temperature"``.

        Streaming mode: ``None``. The HDF5 file contains the same
        arrays plus metadata attrs; see
        :meth:`PhotozBinInference.predict_probabilities_chunked`.

    Example:
        >>> # In-memory
        >>> pdf = predict_photoz_pdf(df, "runs/experiment_01/")
        >>> pdf["probabilities"].shape, pdf["bin_centers"].shape
        ((1000, 400), (400,))
        >>>
        >>> # Streaming (no return value)
        >>> predict_photoz_pdf(df, "runs/experiment_01/",
        ...                    save_path="pdf.h5", chunk_size=50_000)
    """
    infer = PhotozBinInference(
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
    )
    if calibrator_path is not None:
        infer.load_calibrator(calibrator_path)
    apply_calibration = calibrator_path is not None

    if save_path is None:
        return infer.predict_probabilities(
            data_source=df,
            batch_size=batch_size,
            apply_calibration=apply_calibration,
        )

    infer.predict_probabilities_chunked(
        data_source=df,
        save_path=save_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        compression=compression,
        apply_calibration=apply_calibration,
    )
    return None


def rebin_stacked_prob(
    stack_prob: np.ndarray,
    fine_centers: np.ndarray,
    coarse_edges: np.ndarray,
) -> np.ndarray:
    """Merge a fine-bin stacked PDF onto a coarser bin grid.

    Used when plotting an N(z) reconstruction at a resolution lower
    than the training bin grid — e.g. the classifier has 400 bins
    but the diagnostic plot only needs 40.

    Args:
        stack_prob: 1-D array of summed probabilities on the fine
            grid, shape ``(n_fine_bins,)``.
        fine_centers: Fine-bin centres, shape ``(n_fine_bins,)``.
        coarse_edges: Coarse-bin edges, shape
            ``(n_coarse_bins + 1,)``.

    Returns:
        1-D array of coarse-bin probabilities, shape
        ``(n_coarse_bins,)``.
    """
    n_coarse = len(coarse_edges) - 1
    coarse_prob = np.zeros(n_coarse)
    for i in range(n_coarse):
        mask = (
            (fine_centers >= coarse_edges[i])
            & (fine_centers < coarse_edges[i + 1])
        )
        coarse_prob[i] = stack_prob[mask].sum()
    return coarse_prob


def main(argv: list[str] | None = None) -> int:
    """CLI: read a catalog, run :func:`predict_photoz`, write out the result."""
    parser = argparse.ArgumentParser(
        description="Run NNC photo-z inference on a catalog."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input catalog (FITS / HDF5 / CSV / DAT).",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Trained model directory (config.yaml + scaler.pkl + *.pkl).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the catalog + point estimates "
             "(z_mean / z_std / z_mode).",
    )
    parser.add_argument(
        "--output-pdf", default=None,
        help="Optional HDF5 path (.h5). If given, the full per-source "
             "PDF is also streamed out via predict_photoz_pdf().",
    )
    parser.add_argument(
        "--calibrator", default=None,
        help="Optional temperature-scaling calibrator .pkl.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (cuda, cuda:0, cpu). Defaults to cuda.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Rows per HDF5 flush when --output-pdf is given.",
    )
    args = parser.parse_args(argv)

    df = readfile(args.input)
    result = predict_photoz(
        df,
        model_dir=args.model_dir,
        calibrator_path=args.calibrator,
        device=args.device,
        batch_size=args.batch_size,
    )
    out = pd.concat([df.reset_index(drop=True), result], axis=1)
    savefile(out, args.output)
    print(f"Wrote {len(out):,} rows to {args.output}")

    if args.output_pdf is not None:
        predict_photoz_pdf(
            df,
            model_dir=args.model_dir,
            calibrator_path=args.calibrator,
            save_path=args.output_pdf,
            device=args.device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        print(f"Wrote full per-source PDF to {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
