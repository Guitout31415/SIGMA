import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
import scrublet as scr
from pybiomart import Dataset
from scipy.stats import median_abs_deviation
import os
import argparse
from rename_genes import rename_genes
from joblib import Parallel, delayed
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Quality control for scRNA-seq data")
    parser.add_argument("--h5ad_file", type=str, required=True, help="Path to the h5ad file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--percent_top", type=lambda x: int(x) if isinstance(x, str) else x, default=20, help="Percent of top genes to consider")
    parser.add_argument("--nmads", type=lambda x: int(x) if isinstance(x, str) else x, default=5, help="Number of median absolute deviations to consider as outliers")
    parser.add_argument("--do_QC", type=str, default="True", help="Whether to perform quality control")
    parser.add_argument("--species", type=str, default="hsapiens", help="Species of the data")
    return parser.parse_args()

def prepare_adata(adata, species):
    count_matrix = adata.X
    adata = sc.AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())
    genes = rename_genes(adata.var_names.to_list(), species)
    # Convert gene names to uppercase
    adata.var_names = genes
    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
    adata.layers["raw"] = adata.X.copy()
    return adata

def identify_special_genes(adata):
    adata.var["mt"] = adata.var_names.str.match(r"^MT-|^mt-")
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[LS]\d+")
    adata.var["hb"] = adata.var_names.str.match(r"^HB[^P]|^HB[AB]")
    print(f"- # mt genes : {adata.var.mt.sum()}")
    print(f"- # ribo genes : {adata.var.ribo.sum()}")
    print(f"- # hb genes : {adata.var.hb.sum()}")
    return adata.copy()

def calculate_outlier_vector(M, nmads):
    med = np.median(M)
    mad = median_abs_deviation(M)
    lower = med - nmads * mad
    upper = med + nmads * mad
    return (M < lower) | (M > upper)

def run_scrublet(adata, n_jobs=1):
    if "total_counts" in adata.obs:
        zero_total = adata.obs["total_counts"].to_numpy() == 0
        if zero_total.any():
            print(f"Removing {zero_total.sum()} cells with zero total counts before doublet detection.")
            adata = adata[~zero_total].copy()

    max_components = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_components < 1:
        print("Dataset too small for Scrublet PCA. Skipping doublet detection.")
        adata.obs["doublet_score"] = 0.0
        adata.obs["doublet_class"] = pd.Categorical(["False"] * adata.n_obs)
        return adata

    n_prin_comps = min(30, max_components)
    n_prin_comps = max(1, n_prin_comps)

    def _run_scrublet_on_matrix(matrix, obs_count, var_count):
        local_components = min(n_prin_comps, max(1, min(obs_count - 1, var_count - 1)))
        scrub = scr.Scrublet(matrix)
        scores, _ = scrub.scrub_doublets(verbose=False, n_prin_comps=local_components)
        try:
            mask = scrub.call_doublets()
            threshold_info = getattr(scrub, "threshold_", None)
            if threshold_info is not None:
                print(f"Automatically identified doublet score threshold: {threshold_info}")
        except Exception:
            mask = scrub.call_doublets(threshold=0.25)
            print("Using manual doublet score threshold: 0.25")
        return scores, mask

    if adata.n_obs > 10_000:
        print("Large dataset, running Scrublet in batches...")
        batch_size = int(np.ceil(adata.n_obs / n_jobs))
        batches = [adata[i:i+batch_size] for i in range(0, adata.n_obs, batch_size)]
        def process_batch(batch):
            scores, mask = _run_scrublet_on_matrix(batch.X, batch.n_obs, batch.n_vars)
            return scores, mask
        results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(b) for b in batches)
        scores = np.concatenate([r[0] for r in results])
        masks = np.concatenate([r[1] for r in results])
    else:
        scores, masks = _run_scrublet_on_matrix(adata.X, adata.n_obs, adata.n_vars)
    adata.obs["doublet_score"] = scores
    adata.obs["doublet_class"] = pd.Categorical(masks.astype(str))
    adata = adata[adata.obs["doublet_class"] == "False"].copy()
    return adata

if __name__ == "__main__":
    args = parse_args()

    # Check if quality control is enabled
    if args.do_QC == "False":
        print("Quality control is disabled. Exiting...")
        adata = sc.read_h5ad(args.h5ad_file)
        adata = prepare_adata(adata, args.species)
        adata.write(args.output_file)
    else:
        # Start timer
        start_time = time.time()

        # Parallel thread control
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

        try:
            adata = sc.read_h5ad(args.h5ad_file)
        except OSError as e:
            if "wrong B-tree signature" in str(e):
                print(f"Error: The h5ad file '{args.h5ad_file}' appears to be corrupted or truncated.")
                print("This often happens if the file was not properly written or the write process was interrupted.")
                print("Please regenerate the h5ad file from the source data.")
                exit(1)
            else:
                raise

        print("===============================")
        print("Quality control...")

        adata = prepare_adata(adata, args.species)
        print(f"Initial number of cells: {adata.n_obs}")
        adata = identify_special_genes(adata)

        print("Computing quality control metrics...")
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "ribo", "hb"],
            inplace=True,
            percent_top=[args.percent_top],
            log1p=True
        )

        print("Detecting outliers...")
        metrics = [
            "log1p_total_counts",
            "log1p_n_genes_by_counts",
            f"pct_counts_in_top_{args.percent_top}_genes"
        ]

        outlier_results = Parallel(n_jobs=args.threads)(
            delayed(calculate_outlier_vector)(adata.obs[m], args.nmads) for m in metrics
        )

        adata.obs["outlier"] = np.any(outlier_results, axis=0)
        adata.obs["mt_outlier"] = calculate_outlier_vector(
            adata.obs["pct_counts_mt"], 3
        ) | (adata.obs["pct_counts_mt"] > 8)

        adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
        print(f" - Number of cells after filtering low quality cells: {adata.n_obs}")

        print("Detecting doublets...")
        adata = run_scrublet(adata, n_jobs=args.threads)

        print(f"Final number of cells: {adata.n_obs}")

        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.4f} seconds")

        print(f"Saving to {args.output_file}")
        adata.write(args.output_file)