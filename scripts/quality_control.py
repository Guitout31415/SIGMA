"""
quality_control.py
------------------
Quality control pipeline for scRNA-seq data.

Performs filtering, outlier removal, and doublet detection on h5ad files.
"""

import time
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import scrublet as scr
from scipy.stats import median_abs_deviation
from scipy import sparse
from joblib import Parallel, delayed

from constants import (
    DEFAULT_DOUBLET_THRESHOLD,
    MIN_PCA_COMPONENTS,
    MAX_PCA_COMPONENTS,
    BATCH_SIZE_THRESHOLD,
)
from adata_utils import (
    set_thread_environment,
    prepare_adata_qc,
    identify_special_genes,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality control for scRNA-seq data"
    )
    parser.add_argument(
        "--h5ad_file",
        type=str,
        required=True,
        help="Path to the h5ad file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use",
    )
    parser.add_argument(
        "--percent_top",
        type=int,
        default=20,
        help="Percent of top genes to consider",
    )
    parser.add_argument(
        "--nmads",
        type=int,
        default=5,
        help="Number of median absolute deviations to consider as outliers",
    )
    parser.add_argument(
        "--do_QC",
        type=str,
        default="True",
        help="Whether to perform quality control",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="hsapiens",
        help="Species of the data",
    )
    return parser.parse_args()


def calculate_outlier_vector(values: np.ndarray, nmads: int) -> np.ndarray:
    """Calculate outlier mask using median absolute deviation.

    Args:
        values: Array of values to check
        nmads: Number of MADs for outlier threshold

    Returns:
        Boolean array indicating outliers
    """
    med = np.median(values)
    mad = median_abs_deviation(values)
    lower = med - nmads * mad
    upper = med + nmads * mad
    return (values < lower) | (values > upper)


def _run_scrublet_on_matrix(
    matrix: np.ndarray,
    n_prin_comps: int,
) -> tuple:
    """Run Scrublet doublet detection on a count matrix.

    Args:
        matrix: Count matrix
        n_prin_comps: Number of PCA components

    Returns:
        Tuple of (doublet_scores, doublet_mask)
    """
    if matrix is None:
        return np.array([], dtype=float), np.array([], dtype=bool)

    n_obs, n_vars = matrix.shape
    if n_obs == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    # Scrublet (via PCA) cannot run with 0 variables.
    if n_vars == 0:
        print("Scrublet skipped: matrix has 0 features (genes).")
        return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)

    # If the matrix contains no non-zero entries, Scrublet's gene filtering may
    # drop all genes, leading to PCA errors. Skip in that case.
    if sparse.issparse(matrix):
        if matrix.nnz == 0:
            print("Scrublet skipped: count matrix is all zeros.")
            return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)
    else:
        # Avoid expensive full scans for large dense matrices; only do a quick
        # summary for reasonably sized inputs.
        try:
            if getattr(matrix, "size", 0) and matrix.size <= 50_000_000:
                if np.nanmax(matrix) == 0:
                    print("Scrublet skipped: count matrix max is 0.")
                    return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)
        except Exception:
            # If we cannot summarize safely, proceed and let Scrublet decide.
            pass

    local_components = min(
        n_prin_comps,
        max(MIN_PCA_COMPONENTS, min(n_obs - 1, n_vars - 1)),
    )
    scrub = scr.Scrublet(matrix)
    try:
        scores, _ = scrub.scrub_doublets(verbose=False, n_prin_comps=local_components)
    except ValueError as e:
        msg = str(e)
        # Common failure mode: Scrublet internal gene filtering drops all genes.
        # This can happen if the input isn't raw UMI counts or is extremely sparse.
        if "0 feature" in msg or "0 features" in msg:
            try:
                print(
                    "Scrublet hit 0 features after gene filtering. "
                    "Retrying with permissive gene filters (min_counts=1, min_cells=1, min_gene_variability_pctl=0)..."
                )
                scrub_retry = scr.Scrublet(matrix)
                scores, _ = scrub_retry.scrub_doublets(
                    verbose=False,
                    n_prin_comps=min(local_components, 10),
                    min_counts=1,
                    min_cells=1,
                    min_gene_variability_pctl=0,
                )
                scrub = scrub_retry
            except Exception as e2:
                print(f"Scrublet skipped: {e} (retry failed: {type(e2).__name__}: {e2})")
                return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)

        # Another frequent PCA failure mode: n_components too large after internal
        # filtering. Retry with a conservative number of components.
        if "n_components" in msg or "between" in msg or "min(n_samples" in msg:
            try:
                print(f"Scrublet PCA issue ({e}). Retrying with n_prin_comps=1...")
                scrub_retry = scr.Scrublet(matrix)
                scores, _ = scrub_retry.scrub_doublets(verbose=False, n_prin_comps=1)
                scrub = scrub_retry
            except Exception as e2:
                print(
                    f"Scrublet retry failed ({type(e2).__name__}: {e2}). "
                    "Skipping doublet detection."
                )
                return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)
        else:
            raise
    except Exception as e:
        print(f"Scrublet failed ({type(e).__name__}: {e}). Skipping doublet detection.")
        return np.zeros(n_obs, dtype=float), np.zeros(n_obs, dtype=bool)

    try:
        mask = scrub.call_doublets()
        threshold_info = getattr(scrub, "threshold_", None)
        if threshold_info is not None:
            print(f"Automatically identified doublet score threshold: {threshold_info}")
    except Exception:
        mask = scrub.call_doublets(threshold=DEFAULT_DOUBLET_THRESHOLD)
        print(f"Using manual doublet score threshold: {DEFAULT_DOUBLET_THRESHOLD}")

    return scores, mask


def run_scrublet(adata: AnnData, n_jobs: int = 1) -> AnnData:
    """Run Scrublet doublet detection on AnnData.

    Args:
        adata: AnnData object
        n_jobs: Number of parallel jobs

    Returns:
        AnnData with doublets removed
    """
    # Prefer raw counts if available (Scrublet expects raw counts)
    matrix = adata.layers["raw"] if "raw" in adata.layers else adata.X

    # Remove cells with zero total counts
    if "total_counts" in adata.obs:
        zero_total = adata.obs["total_counts"].to_numpy() == 0
        if zero_total.any():
            print(
                f"Removing {zero_total.sum()} cells with zero total counts "
                "before doublet detection."
            )
            adata = adata[~zero_total].copy()
            matrix = adata.layers["raw"] if "raw" in adata.layers else adata.X

    # If no genes, skip immediately
    if matrix is None or matrix.shape[1] == 0:
        print("No genes available for Scrublet (0 features). Skipping doublet detection.")
        adata.obs["doublet_score"] = 0.0
        adata.obs["doublet_class"] = pd.Categorical(["False"] * adata.n_obs)
        return adata

    # Check if dataset is large enough for PCA
    max_components = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_components < MIN_PCA_COMPONENTS:
        print("Dataset too small for Scrublet PCA. Skipping doublet detection.")
        adata.obs["doublet_score"] = 0.0
        adata.obs["doublet_class"] = pd.Categorical(["False"] * adata.n_obs)
        return adata

    n_prin_comps = min(MAX_PCA_COMPONENTS, max(MIN_PCA_COMPONENTS, max_components))

    # Run Scrublet (batched for large datasets)
    if adata.n_obs > BATCH_SIZE_THRESHOLD:
        print("Large dataset, running Scrublet in batches...")
        batch_size = int(np.ceil(adata.n_obs / n_jobs))
        batches = [
            adata[i : i + batch_size]
            for i in range(0, adata.n_obs, batch_size)
        ]

        def process_batch(batch: AnnData) -> tuple:
            batch_matrix = batch.layers["raw"] if "raw" in batch.layers else batch.X
            return _run_scrublet_on_matrix(
                batch_matrix, n_prin_comps
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(b) for b in batches
        )
        scores = np.concatenate([r[0] for r in results])
        masks = np.concatenate([r[1] for r in results])
    else:
        scores, masks = _run_scrublet_on_matrix(
            matrix, n_prin_comps
        )

    adata.obs["doublet_score"] = scores
    adata.obs["doublet_class"] = pd.Categorical(masks.astype(str))
    adata = adata[adata.obs["doublet_class"] == "False"].copy()

    return adata


def detect_outliers(
    adata: AnnData,
    percent_top: int,
    nmads: int,
    threads: int,
) -> AnnData:
    """Detect and filter outlier cells.

    Args:
        adata: AnnData object with QC metrics calculated
        percent_top: Percent of top genes for metric calculation
        nmads: Number of MADs for outlier detection
        threads: Number of parallel threads

    Returns:
        AnnData with outliers removed
    """
    print("Detecting outliers...")
    metrics = [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        f"pct_counts_in_top_{percent_top}_genes",
    ]

    outlier_results = Parallel(n_jobs=threads)(
        delayed(calculate_outlier_vector)(adata.obs[m], nmads) for m in metrics
    )

    adata.obs["outlier"] = np.any(outlier_results, axis=0)
    adata.obs["mt_outlier"] = (
        calculate_outlier_vector(adata.obs["pct_counts_mt"], 3)
        | (adata.obs["pct_counts_mt"] > 8)
    )

    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    print(f" - Number of cells after filtering low quality cells: {adata.n_obs}")

    return adata


def run_quality_control(
    adata: AnnData,
    percent_top: int,
    nmads: int,
    threads: int,
) -> AnnData:
    """Run the full quality control pipeline.

    Args:
        adata: Input AnnData object
        percent_top: Percent of top genes to consider
        nmads: Number of MADs for outlier detection
        threads: Number of parallel threads

    Returns:
        Quality-controlled AnnData object
    """
    print("===============================")
    print("Quality control...")
    print(f"Initial number of cells: {adata.n_obs}")

    adata = identify_special_genes(adata)

    print("Computing quality control metrics...")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[percent_top],
        log1p=True,
    )

    adata = detect_outliers(adata, percent_top, nmads, threads)

    print("Detecting doublets...")
    adata = run_scrublet(adata, n_jobs=threads)

    print(f"Final number of cells: {adata.n_obs}")

    return adata


def main() -> None:
    """Main entry point for quality control."""
    args = parse_args()

    if args.do_QC == "False":
        print("Quality control is disabled. Exiting...")
        adata = sc.read_h5ad(args.h5ad_file)
        adata = prepare_adata_qc(adata, args.species)
        adata.write(args.output_file)
        return

    start_time = time.time()
    set_thread_environment(args.threads)

    try:
        adata = sc.read_h5ad(args.h5ad_file)
    except OSError as e:
        if "wrong B-tree signature" in str(e):
            print(
                f"Error: The h5ad file '{args.h5ad_file}' appears to be "
                "corrupted or truncated."
            )
            print(
                "This often happens if the file was not properly written "
                "or the write process was interrupted."
            )
            print("Please regenerate the h5ad file from the source data.")
            exit(1)
        raise

    adata = prepare_adata_qc(adata, args.species)
    adata = run_quality_control(adata, args.percent_top, args.nmads, args.threads)

    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.4f} seconds")

    print(f"Saving to {args.output_file}")
    adata.write(args.output_file)


if __name__ == "__main__":
    main()