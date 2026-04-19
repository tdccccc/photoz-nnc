"""ANN photo-z inference entry point.

Usage:
    python -m models.photoz.ANN.predict \\
        --input catalog.fits \\
        --model-dir runs/experiment_01/ \\
        --output catalog_with_photoz.fits
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from lib.io import readfile, savefile

from .inference import PhotozRegressionInference

__all__ = ["predict_photoz", "main"]


def predict_photoz(
    df: pd.DataFrame,
    model_dir: str,
    device: str = "cuda",
    batch_size: int = 4096,
    return_uncertainty: bool = False,
    n_mc_forward: int = 100,
) -> pd.DataFrame:
    """Run ANN photo-z inference on a catalog DataFrame.

    Args:
        df: Input catalog with the feature columns expected by the
            model.
        model_dir: Path to a trained model directory.
        device: Torch device spec.
        batch_size: Inference batch size.
        return_uncertainty: If ``True``, enable MC-dropout uncertainty.
        n_mc_forward: Number of MC forward passes when
            ``return_uncertainty=True``.

    Returns:
        DataFrame with ``z_pred`` and optionally ``z_uncertainty``.

    Example:
        >>> result = predict_photoz(df, "runs/ann_exp01/")
        >>> df["z_pred"] = result["z_pred"]
    """
    infer = PhotozRegressionInference(
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
    )
    result = infer.predict(
        data_source=df,
        batch_size=batch_size,
        return_uncertainty=return_uncertainty,
        n_mc_forward=n_mc_forward,
    )

    out = {"z_pred": result["predictions"]}
    if return_uncertainty:
        out["z_uncertainty"] = result["uncertainties"]
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ANN photo-z inference on a catalog."
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
        help="Output path. Original columns + z_pred.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device. Defaults to cuda.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--uncertainty", action="store_true",
        help="Enable MC-dropout uncertainty estimation.",
    )
    args = parser.parse_args(argv)

    df = readfile(args.input)
    result = predict_photoz(
        df,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        return_uncertainty=args.uncertainty,
    )
    out = pd.concat([df.reset_index(drop=True), result], axis=1)
    savefile(out, args.output)
    print(f"Wrote {len(out):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
