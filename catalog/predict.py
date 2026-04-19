"""Photo-z prediction stage.

Two surveys flow through the same machinery but with different
priority strategies and input schemas:

* **LSDR10**: P1 (native ``i``-band), P2 (borrows PS1 ``i`` when
  LSDR10 is missing it), P3 (no ``i``-band at all).
* **PS1DR2**: P4 (has unWISE ``W1/W2`` cross-match), P5 (no unWISE).

A galaxy-probability head runs alongside photo-z when a classifier
directory is configured. Outputs land in the per-survey directory
declared in ``catalog/config.yaml`` as one HDF5 file per input shard.

Entry points:

* :class:`PhotozPredictor` — reusable predictor holding all lazily
  loaded models, bin centres, and resample matrices.
* :func:`predict_lsdr10_batch` / :func:`predict_ps1dr2_batch` —
  batch drivers (single-process, multi-process pool, or one fresh
  subprocess per file).
* :func:`run_lsdr10` / :func:`run_ps1dr2` — zero-argument wrappers
  that read paths and runtime knobs from ``config.yaml``.
"""

from __future__ import annotations

import gc
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import torch

from models.galaxy_clf.ANN.galaxyClf_ann import GalaxyClassificationInference
from models.photoz.NNC.inference import PhotozBinInference

from .common import (
    PDF_BINS,
    build_resample_matrix,
    compute_photoz_stats_torch,
    encode_pdf_to_uint16,
    is_valid_prediction_hdf5,
    load_config,
    read_table,
    save_table,
)

__all__ = [
    "PhotozPredictor",
    "predict_lsdr10_batch",
    "predict_ps1dr2_batch",
    "run_lsdr10",
    "run_ps1dr2",
]


# Column groups reused across both surveys; kept at module scope to
# avoid re-allocating on every prediction call.
_PHOTOZ_SCALAR_COLS = [
    "z_phot_mean",
    "z_phot_std",
    "z_phot_mode",
    "z_phot_median",
    "z_phot_l68",
    "z_phot_u68",
    "z_phot_l95",
    "z_phot_u95",
]


# ===========================================================================
# Predictor
# ===========================================================================


class PhotozPredictor:
    """Reusable photo-z predictor with lazy per-model loading.

    Each model id (``p1``..``p5``) lives in its own sub-directory of
    ``model_base_dir`` and is paired with a
    ``TempScalingCalib/calibrator.pkl`` pickle. The first call that
    needs a model triggers loading, computes its bin centres, and
    caches a pre-built resample matrix that maps the model's native
    binning onto :data:`PDF_BINS`.

    When ``galaxy_clf_dir`` is given, :meth:`predict_galaxy_prob`
    also becomes available and is invoked automatically by
    :meth:`predict_lsdr10` / :meth:`predict_ps1dr2`.
    """

    def __init__(
        self,
        model_base_dir: str,
        device: str = "cuda:0",
        galaxy_clf_dir: Optional[str] = None,
    ):
        self.model_base_dir = model_base_dir
        self.device = device
        self._infers: dict[str, PhotozBinInference] = {}
        self._bin_centers_torch_cache: dict[str, torch.Tensor] = {}
        self._resample_matrix_cache: dict[str, torch.Tensor] = {}

        self.galaxy_clf_dir = galaxy_clf_dir
        self._galaxy_clf: Optional[GalaxyClassificationInference] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_infer(self, model_name: str) -> PhotozBinInference:
        """Load (or return cached) inference object for one priority model."""
        if model_name not in self._infers:
            model_dir = os.path.join(self.model_base_dir, model_name)
            calibrator_path = os.path.join(
                model_dir, "TempScalingCalib", "calibrator.pkl"
            )

            infer = PhotozBinInference(model_dir=model_dir, device=self.device)
            infer.load_calibrator(calibrator_path)
            self._infers[model_name] = infer

            binning_config = infer.config.dataset_params["binning_config"]
            z_min = binning_config.get("z_min", 0.0)
            z_max = binning_config.get("z_max", 2.0)
            num_bins = binning_config.get("num_bins", 400)
            bin_edges = np.linspace(z_min, z_max, num_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            self._bin_centers_torch_cache[model_name] = torch.from_numpy(
                bin_centers.astype(np.float32)
            ).to(self.device)
            self._resample_matrix_cache[model_name] = torch.from_numpy(
                build_resample_matrix(bin_edges, target_n=PDF_BINS)
            ).to(self.device)

        return self._infers[model_name]

    def preload_lsdr10_models(self) -> None:
        for name in ("p1", "p2", "p3"):
            self._get_infer(name)

    def preload_ps1dr2_models(self) -> None:
        for name in ("p4", "p5"):
            self._get_infer(name)

    def _get_galaxy_clf(self) -> GalaxyClassificationInference:
        if self._galaxy_clf is None:
            if self.galaxy_clf_dir is None:
                raise ValueError("galaxy_clf_dir not specified")
            self._galaxy_clf = GalaxyClassificationInference(
                model_dir=self.galaxy_clf_dir, device=self.device
            )
        return self._galaxy_clf

    # ------------------------------------------------------------------
    # Galaxy-probability head
    # ------------------------------------------------------------------

    def predict_galaxy_prob(
        self,
        df: pd.DataFrame,
        batch_size: int = 4096,
        chunk_size: int = 1_000_000,
    ) -> np.ndarray:
        """Run the galaxy classifier on ``df`` and return ``P(galaxy) in [0, 1]``.

        Features come from ``clf.config.dataset_params['feature_columns']``;
        rows are streamed through the classifier in chunks to keep
        scaler + dataframe slicing costs bounded.
        """
        clf = self._get_galaxy_clf()
        n_samples = len(df)
        probs = np.empty(n_samples, dtype=np.float32)
        feature_columns = clf.config.dataset_params.get("feature_columns")

        with torch.no_grad():
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_df = df.iloc[chunk_start:chunk_end]
                chunk_features = chunk_df[feature_columns].values.astype(np.float32)
                chunk_features = clf.scaler.transform(chunk_features)

                for local_start in range(0, len(chunk_features), batch_size):
                    local_end = min(local_start + batch_size, len(chunk_features))
                    start = chunk_start + local_start
                    end = chunk_start + local_end

                    inputs = torch.from_numpy(
                        chunk_features[local_start:local_end]
                    ).to(clf.device)
                    outputs = (
                        torch.sigmoid(clf.model(inputs))
                        .squeeze(1)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    probs[start:end] = outputs

                del chunk_df, chunk_features

        return probs

    # ------------------------------------------------------------------
    # Photo-z inference core
    # ------------------------------------------------------------------

    def _predict_subset_efficient(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        model_name: str,
        batch_size: int,
        chunk_size: int,
    ) -> dict:
        """Run a single priority model over the selected row indices.

        The hot path bypasses the high-level ``predict_*`` methods on
        :class:`PhotozBinInference` and drives ``infer.model`` and
        temperature scaling directly, then runs statistics and the
        PDF resample on GPU. This avoids the per-batch ``Dataset``
        construction overhead and the logits -> CPU -> stats
        round-trip that the high-level API would incur.
        """
        infer = self._get_infer(model_name)
        bin_centers_torch = self._bin_centers_torch_cache[model_name]
        resample_matrix = self._resample_matrix_cache[model_name]

        n_samples = len(indices)
        results = {
            col: np.empty(n_samples, dtype=np.float32)
            for col in _PHOTOZ_SCALAR_COLS
        }
        pdf_40 = np.empty((n_samples, PDF_BINS), dtype=np.float32)

        feature_columns = infer.config.dataset_params.get("feature_columns")
        temperature = infer.calibrator.temperature if infer.calibrator else 1.0

        with torch.no_grad():
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_indices = indices[chunk_start:chunk_end]
                chunk_df = df.iloc[chunk_indices]
                chunk_features = chunk_df[feature_columns].values.astype(np.float32)
                chunk_features = infer.scaler.transform(chunk_features)

                for local_start in range(0, len(chunk_features), batch_size):
                    local_end = min(local_start + batch_size, len(chunk_features))
                    start = chunk_start + local_start
                    end = chunk_start + local_end

                    batch_features = torch.from_numpy(
                        chunk_features[local_start:local_end]
                    ).to(infer.device)
                    if infer.device.type == "cuda":
                        torch.cuda.synchronize(infer.device)

                    logits = infer.model(batch_features)
                    if temperature != 1.0:
                        logits = logits / temperature
                    probs_torch = torch.softmax(logits, dim=1)
                    stats_torch = compute_photoz_stats_torch(
                        probs_torch, bin_centers_torch, to_cpu=False
                    )
                    resampled_probs_torch = torch.matmul(
                        probs_torch, resample_matrix
                    )
                    if infer.device.type == "cuda":
                        torch.cuda.synchronize(infer.device)

                    stats = {
                        k: v.detach().cpu().numpy().astype(np.float32)
                        for k, v in stats_torch.items()
                    }
                    resampled_probs = resampled_probs_torch.cpu().numpy()
                    if infer.device.type == "cuda":
                        torch.cuda.synchronize(infer.device)

                    for col in _PHOTOZ_SCALAR_COLS:
                        results[col][start:end] = stats[col]
                    pdf_40[start:end] = resampled_probs

                del chunk_df, chunk_features

        torch.cuda.empty_cache()
        return {
            **results,
            "z_phot_pdf": encode_pdf_to_uint16(pdf_40),
        }

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------

    def predict_lsdr10(
        self,
        df: pd.DataFrame,
        batch_size: int = 4096,
        chunk_size: int = 1_000_000,
    ) -> pd.DataFrame:
        """Predict photo-z (and optionally galaxy_prob) for an LSDR10 shard.

        Priority routing:

        * **P1** — native LSDR10 ``i``-band available (``grizW1W2``).
        * **P2** — no native ``i`` but PS1 ``i`` is cross-matched
          (``grzW1W2 + PS1 i``).
        * **P3** — no ``i``-band from either source (``grzW1W2``).

        The returned DataFrame keeps the input ``uid`` under
        ``uid_ls`` and carries timing attributes
        (``timing_photoz``, ``timing_clf``) on ``result.attrs``.
        """
        ps1_i_col = "iKronMag_dered"
        n_samples = len(df)
        df = df.reset_index(drop=True)

        # Galaxy classifier expects magnitude errors derived from SNR.
        snr_to_err = {
            "snr_g": "mag_g_Err",
            "snr_r": "mag_r_Err",
            "snr_i": "mag_i_Err",
            "snr_z": "mag_z_Err",
            "snr_w1": "mag_w1_Err",
            "snr_w2": "mag_w2_Err",
        }
        for snr_col, err_col in snr_to_err.items():
            if snr_col in df.columns:
                df[err_col] = (1.0857 / df[snr_col]).astype(np.float32)

        result_arrays = {
            col: np.full(n_samples, np.nan, dtype=np.float32)
            for col in _PHOTOZ_SCALAR_COLS
        }
        pdf_array = np.zeros((n_samples, PDF_BINS), dtype=np.uint16)
        priority_array = np.zeros(n_samples, dtype=np.int16)
        survey_array = np.zeros(n_samples, dtype=np.int16)  # LSDR10 = 0

        has_native_i = (
            (df["dered_mag_i"].notna() & (df["dered_mag_i"] > 0))
            .values.astype(bool)
        )
        if ps1_i_col in df.columns:
            has_ps1_i = (
                (df[ps1_i_col].notna() & (df[ps1_i_col] > 0))
                .values.astype(bool)
            )
        else:
            has_ps1_i = np.zeros(n_samples, dtype=bool)

        model_configs = [
            ("p1", has_native_i, 1),
            ("p2", ~has_native_i & has_ps1_i, 2),
            ("p3", ~has_native_i & ~has_ps1_i, 3),
        ]

        t_photoz = time.time()
        for model_name, mask, priority in model_configs:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            stats = self._predict_subset_efficient(
                df, indices, model_name, batch_size, chunk_size
            )
            for col in _PHOTOZ_SCALAR_COLS:
                result_arrays[col][indices] = stats[col]
            pdf_array[indices] = stats["z_phot_pdf"]
            priority_array[indices] = priority
        photoz_time = time.time() - t_photoz

        galaxy_prob_array: Optional[np.ndarray] = None
        clf_time = 0.0
        if self.galaxy_clf_dir is not None:
            t_clf = time.time()
            galaxy_prob_array = self.predict_galaxy_prob(
                df, batch_size=batch_size, chunk_size=chunk_size
            )
            clf_time = time.time() - t_clf

        result = pd.DataFrame({
            "uid_ls": df["uid"].values,
            "ra": df["ra"].values,
            "dec": df["dec"].values,
            **result_arrays,
            "priority": priority_array,
            "survey": survey_array,
        })
        result["z_phot_pdf"] = list(pdf_array)
        if galaxy_prob_array is not None:
            result["galaxy_prob"] = galaxy_prob_array
        result.attrs["timing_photoz"] = photoz_time
        result.attrs["timing_clf"] = clf_time
        return result

    def predict_ps1dr2(
        self,
        df: pd.DataFrame,
        batch_size: int = 4096,
        chunk_size: int = 1_000_000,
    ) -> pd.DataFrame:
        """Predict photo-z (and optionally galaxy_prob) for a PS1DR2 shard.

        Priority routing:

        * **P4** — cross-matched to unWISE (``grizy + W1W2``).
        * **P5** — no unWISE match (``grizy`` only).

        Mirrors :meth:`predict_lsdr10`: the input ``uid`` is kept as
        ``uid_ps`` and timing attributes are attached to
        ``result.attrs``.
        """
        w1_col = "mag_w1"
        w2_col = "mag_w2"
        n_samples = len(df)
        df = df.reset_index(drop=True)

        result_arrays = {
            col: np.full(n_samples, np.nan, dtype=np.float32)
            for col in _PHOTOZ_SCALAR_COLS
        }
        pdf_array = np.zeros((n_samples, PDF_BINS), dtype=np.uint16)
        priority_array = np.zeros(n_samples, dtype=np.int16)
        survey_array = np.ones(n_samples, dtype=np.int16)  # PS1DR2 = 1

        has_unwise = (
            (
                df[w1_col].notna() & (df[w1_col] > 0)
                & df[w2_col].notna() & (df[w2_col] > 0)
            )
            .values.astype(bool)
        )

        model_configs = [
            ("p4", has_unwise, 4),
            ("p5", ~has_unwise, 5),
        ]

        t_photoz = time.time()
        for model_name, mask, priority in model_configs:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            stats = self._predict_subset_efficient(
                df, indices, model_name, batch_size, chunk_size
            )
            for col in _PHOTOZ_SCALAR_COLS:
                result_arrays[col][indices] = stats[col]
            pdf_array[indices] = stats["z_phot_pdf"]
            priority_array[indices] = priority
        photoz_time = time.time() - t_photoz

        galaxy_prob_array: Optional[np.ndarray] = None
        clf_time = 0.0
        if self.galaxy_clf_dir is not None:
            t_clf = time.time()
            galaxy_prob_array = self.predict_galaxy_prob(
                df, batch_size=batch_size, chunk_size=chunk_size
            )
            clf_time = time.time() - t_clf

        result = pd.DataFrame({
            "uid_ps": df["uid"].values,
            "ra": df["ra"].values,
            "dec": df["dec"].values,
            **result_arrays,
            "priority": priority_array,
            "survey": survey_array,
        })
        result["z_phot_pdf"] = list(pdf_array)
        if galaxy_prob_array is not None:
            result["galaxy_prob"] = galaxy_prob_array
        result.attrs["timing_photoz"] = photoz_time
        result.attrs["timing_clf"] = clf_time
        return result


# ===========================================================================
# Batch drivers
# ===========================================================================


def _required_columns(uid_col: str, with_galaxy: bool) -> list[str]:
    cols = [
        uid_col,
        "ra",
        "dec",
        *_PHOTOZ_SCALAR_COLS,
        "priority",
        "survey",
        "z_phot_pdf",
    ]
    if with_galaxy:
        cols.append("galaxy_prob")
    return cols


def _process_single_file_lsdr10(args, predictor: Optional[PhotozPredictor] = None):
    """Worker wrapping one LSDR10 input shard -> output HDF5.

    Returns ``(fid, status, info)`` where ``status`` is
    ``'success'`` / ``'skipped'`` / ``'error'``; ``info`` is either
    the row count on success or the exception message on error.
    """
    (
        fid,
        filepath,
        output_path,
        model_base_dir,
        device,
        batch_size,
        chunk_size,
        galaxy_clf_dir,
    ) = args

    try:
        t0 = time.time()
        required_columns = _required_columns("uid_ls", galaxy_clf_dir is not None)

        # Skip intact outputs; wipe and reprocess anything malformed.
        if is_valid_prediction_hdf5(output_path, required_columns):
            return fid, "skipped", None
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"  [{fid:02d}] Begin")

        # Each subprocess loads its own predictor; single-process mode
        # reuses the one built by the caller.
        if predictor is None:
            predictor = PhotozPredictor(
                model_base_dir=model_base_dir,
                device=device,
                galaxy_clf_dir=galaxy_clf_dir,
            )
            predictor.preload_lsdr10_models()

        t_read = time.time()
        df = read_table(filepath)
        n_rows = len(df)
        t_read = time.time() - t_read

        result = predictor.predict_lsdr10(
            df, batch_size=batch_size, chunk_size=chunk_size
        )
        t_photoz = float(result.attrs.get("timing_photoz", 0.0))
        t_clf = float(result.attrs.get("timing_clf", 0.0))

        t_save = time.time()
        save_table(result, output_path)
        t_save = time.time() - t_save
        t_total = time.time() - t0
        print(
            f"  [{fid:02d}] Done, rows={n_rows:,} "
            f"read={t_read:.1f}s photoz={t_photoz:.1f}s clf={t_clf:.1f}s "
            f"write={t_save:.1f}s total={t_total:.1f}s"
        )

        del df, result
        gc.collect()
        torch.cuda.empty_cache()
        return fid, "success", n_rows
    except Exception as e:
        return fid, "error", str(e)


def _process_single_file_ps1dr2(args, predictor: Optional[PhotozPredictor] = None):
    """PS1DR2 counterpart to :func:`_process_single_file_lsdr10`."""
    (
        fid,
        filepath,
        output_path,
        model_base_dir,
        device,
        batch_size,
        chunk_size,
        galaxy_clf_dir,
    ) = args

    try:
        t0 = time.time()
        required_columns = _required_columns("uid_ps", galaxy_clf_dir is not None)

        if is_valid_prediction_hdf5(output_path, required_columns):
            return fid, "skipped", None
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"  [{fid:02d}] Begin")

        if predictor is None:
            predictor = PhotozPredictor(
                model_base_dir=model_base_dir,
                device=device,
                galaxy_clf_dir=galaxy_clf_dir,
            )
            predictor.preload_ps1dr2_models()

        t_read = time.time()
        df = read_table(filepath)
        n_rows = len(df)
        t_read = time.time() - t_read

        result = predictor.predict_ps1dr2(
            df, batch_size=batch_size, chunk_size=chunk_size
        )
        t_photoz = float(result.attrs.get("timing_photoz", 0.0))
        t_clf = float(result.attrs.get("timing_clf", 0.0))

        t_save = time.time()
        save_table(result, output_path)
        t_save = time.time() - t_save
        t_total = time.time() - t0
        print(
            f"  [{fid:02d}] Done, rows={n_rows:,} "
            f"read={t_read:.1f}s photoz={t_photoz:.1f}s clf={t_clf:.1f}s "
            f"write={t_save:.1f}s total={t_total:.1f}s"
        )

        del df, result
        gc.collect()
        torch.cuda.empty_cache()
        return fid, "success", n_rows
    except Exception as e:
        return fid, "error", str(e)


def _print_worker_result(fid: int, status: str, info) -> None:
    if status == "skipped":
        print(f"  [{fid:02d}] Skipped (exists)")
    elif status == "error":
        print(f"  [{fid:02d}] Error: {info}")


def predict_lsdr10_batch(
    data_dir: str,
    output_dir: str,
    model_base_dir: str,
    devices: list,
    batch_size: int = 4096,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
    galaxy_clf_dir: Optional[str] = None,
    isolate_per_file: bool = False,
) -> None:
    """Batch-predict every ``*.fits`` file in ``data_dir`` for LSDR10.

    Three execution modes:

    * ``isolate_per_file=True`` — one fresh subprocess per file.
      Slowest but the most robust to GPU / CUDA leaks between files.
    * ``num_workers == 1`` and ``isolate_per_file=False`` — reuse a
      single in-process predictor; fastest for a small file count.
    * ``num_workers > 1`` — multi-process pool, GPUs assigned
      round-robin via ``devices``.

    The template used to name output files is
    ``templates.lsdr10_output`` in ``config.yaml``.
    """
    cfg = load_config()
    output_template = cfg["templates"]["lsdr10_output"]
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    print(f"Found {len(file_list)} files in {data_dir}")
    if len(file_list) == 0:
        print("No files found. Exiting.")
        return

    tasks = []
    for fid, filepath in enumerate(file_list):
        output_path = os.path.join(output_dir, output_template.format(fid=fid))
        device = devices[fid % len(devices)]  # round-robin GPU assignment
        tasks.append((
            fid, filepath, output_path, model_base_dir, device,
            batch_size, chunk_size, galaxy_clf_dir,
        ))

    if isolate_per_file:
        print("Using one fresh subprocess per LSDR10 file")
        for task in tasks:
            fid = task[0]
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_process_single_file_lsdr10, task)
                try:
                    fid, status, info = future.result()
                    _print_worker_result(fid, status, info)
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")
    elif num_workers == 1:
        predictor = PhotozPredictor(
            model_base_dir=model_base_dir,
            device=devices[0],
            galaxy_clf_dir=galaxy_clf_dir,
        )
        predictor.preload_lsdr10_models()
        for task in tasks:
            fid, status, info = _process_single_file_lsdr10(task, predictor=predictor)
            _print_worker_result(fid, status, info)
    else:
        print(f"Using {num_workers} workers with devices: {devices}")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_file_lsdr10, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                fid = futures[future]
                try:
                    fid, status, info = future.result()
                    _print_worker_result(fid, status, info)
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")


def predict_ps1dr2_batch(
    data_dir: str,
    output_dir: str,
    model_base_dir: str,
    devices: list,
    batch_size: int = 4096,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
    galaxy_clf_dir: Optional[str] = None,
    isolate_per_file: bool = False,
) -> None:
    """PS1DR2 counterpart to :func:`predict_lsdr10_batch`."""
    cfg = load_config()
    output_template = cfg["templates"]["ps1dr2_output"]
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    print(f"Found {len(file_list)} files in {data_dir}")
    if len(file_list) == 0:
        print("No files found. Exiting.")
        return

    tasks = []
    for fid, filepath in enumerate(file_list):
        output_path = os.path.join(output_dir, output_template.format(fid=fid))
        device = devices[fid % len(devices)]
        tasks.append((
            fid, filepath, output_path, model_base_dir, device,
            batch_size, chunk_size, galaxy_clf_dir,
        ))

    if isolate_per_file:
        print("Using one fresh subprocess per PS1DR2 file")
        for task in tasks:
            fid = task[0]
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_process_single_file_ps1dr2, task)
                try:
                    fid, status, info = future.result()
                    _print_worker_result(fid, status, info)
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")
    elif num_workers == 1:
        predictor = PhotozPredictor(
            model_base_dir=model_base_dir,
            device=devices[0],
            galaxy_clf_dir=galaxy_clf_dir,
        )
        predictor.preload_ps1dr2_models()
        for task in tasks:
            fid, status, info = _process_single_file_ps1dr2(task, predictor=predictor)
            _print_worker_result(fid, status, info)
    else:
        print(f"Using {num_workers} workers with devices: {devices}")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_file_ps1dr2, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                fid = futures[future]
                try:
                    fid, status, info = future.result()
                    _print_worker_result(fid, status, info)
                except Exception as e:
                    print(f"  [{fid:02d}] Exception: {e}")


# ===========================================================================
# Config-driven run entry points
# ===========================================================================


def run_lsdr10() -> None:
    """Run the full LSDR10 prediction stage using ``config.yaml``."""
    cfg = load_config()
    rt = cfg["runtime"]["lsdr10"]
    paths = cfg["paths"]
    galaxy_clf_dir = paths.get("galaxy_clf_ls")

    print("=" * 60)
    print("LSDR10 Photo-z Prediction")
    if galaxy_clf_dir:
        print(f"Galaxy classification enabled: {galaxy_clf_dir}")
    print("=" * 60)

    predict_lsdr10_batch(
        data_dir=paths["lsdr10"]["input"],
        output_dir=paths["lsdr10"]["output"],
        model_base_dir=paths["model_base"],
        devices=list(rt["devices"]),
        batch_size=int(rt["batch_size"]),
        chunk_size=int(rt["chunk_size"]),
        num_workers=int(rt["num_workers"]),
        galaxy_clf_dir=galaxy_clf_dir,
        isolate_per_file=bool(rt.get("isolate_per_file", False)),
    )
    print("\nLSDR10 prediction completed!")


def run_ps1dr2() -> None:
    """Run the full PS1DR2 prediction stage using ``config.yaml``."""
    cfg = load_config()
    rt = cfg["runtime"]["ps1dr2"]
    paths = cfg["paths"]
    galaxy_clf_dir = paths.get("galaxy_clf_ps")

    print("=" * 60)
    print("PS1DR2 Photo-z Prediction")
    if galaxy_clf_dir:
        print(f"Galaxy classification enabled: {galaxy_clf_dir}")
    print("=" * 60)

    predict_ps1dr2_batch(
        data_dir=paths["ps1dr2"]["input"],
        output_dir=paths["ps1dr2"]["output"],
        model_base_dir=paths["model_base"],
        devices=list(rt["devices"]),
        batch_size=int(rt["batch_size"]),
        chunk_size=int(rt["chunk_size"]),
        num_workers=int(rt["num_workers"]),
        galaxy_clf_dir=galaxy_clf_dir,
        isolate_per_file=bool(rt.get("isolate_per_file", False)),
    )
    print("\nPS1DR2 prediction completed!")
