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
    n_obs: int,
    n_vars: int,
    n_prin_comps: int,
) -> tuple:
    """Run Scrublet doublet detection on a count matrix.

    Args:
        matrix: Count matrix
        n_obs: Number of observations
        n_vars: Number of variables
        n_prin_comps: Number of PCA components

    Returns:
        Tuple of (doublet_scores, doublet_mask)
    """
    local_components = min(
        n_prin_comps,
        max(MIN_PCA_COMPONENTS, min(n_obs - 1, n_vars - 1)),
    )
    scrub = scr.Scrublet(matrix)
    scores, _ = scrub.scrub_doublets(verbose=False, n_prin_comps=local_components)

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
    # Remove cells with zero total counts
    if "total_counts" in adata.obs:
        zero_total = adata.obs["total_counts"].to_numpy() == 0
        if zero_total.any():
            print(
                f"Removing {zero_total.sum()} cells with zero total counts "
                "before doublet detection."
            )
            adata = adata[~zero_total].copy()

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
            return _run_scrublet_on_matrix(
                batch.X, batch.n_obs, batch.n_vars, n_prin_comps
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(b) for b in batches
        )
        scores = np.concatenate([r[0] for r in results])
        masks = np.concatenate([r[1] for r in results])
    else:
        scores, masks = _run_scrublet_on_matrix(
            adata.X, adata.n_obs, adata.n_vars, n_prin_comps
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