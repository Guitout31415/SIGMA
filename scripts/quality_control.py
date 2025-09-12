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
    parser.add_argument("--percent_top", type=lambda x: int(x) if isinstance(x, str) else x, default=20, help="Percent of top genes to consider")
    parser.add_argument("--nmads", type=lambda x: int(x) if isinstance(x, str) else x, default=5, help="Number of median absolute deviations to consider as outliers")
    parser.add_argument("--do_QC", type=str, default="True", help="Whether to perform quality control")
    parser.add_argument("--species", type=str, default="hsapiens", help="Species of the data")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for parallel processing")
    parser.add_argument("--force_parallel", default=False, action="store_true", help="Force parallel processing even for small datasets")
    return parser.parse_args()

def prepare_adata(adata, species):
    count_matrix = adata.X
    genes = rename_genes(adata.var_names, species)
    # Convert gene names to uppercase
    genes = pd.Series(genes).str.upper()
    adata = sc.AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())
    adata.var_names = genes
    adata.var_names_make_unique()
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
    # Use parallel processing for datasets > 20,000 cells (reduced threshold)
    if adata.n_obs > 20_000 and n_jobs > 1:
        # Optimize batch size based on dataset size and available cores
        min_batch_size = max(5_000, int(adata.n_obs / (n_jobs * 2)))  # Smaller batches for better parallelization
        batch_size = max(min_batch_size, int(np.ceil(adata.n_obs / n_jobs)))
        batches = [adata[i:i+batch_size] for i in range(0, adata.n_obs, batch_size)]
        
        def process_batch(batch):
            # Convert sparse matrix to dense for better performance with small batches
            X_batch = batch.X.toarray() if hasattr(batch.X, 'toarray') else batch.X
            scrub = scr.Scrublet(X_batch)
            scores, _ = scrub.scrub_doublets(verbose=False)
            try:
                mask = scrub.call_doublets()
            except:
                mask = scrub.call_doublets(threshold=0.25)
            return scores, mask
        
        results = Parallel(n_jobs=min(n_jobs, len(batches)), backend='threading')(delayed(process_batch)(b) for b in batches)
        scores = np.concatenate([r[0] for r in results])
        masks = np.concatenate([r[1] for r in results])
    else:
        # For sequential processing, also convert to dense if beneficial
        X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') and adata.n_obs < 100_000 else adata.X
        scrub = scr.Scrublet(X_data)
        scores, _ = scrub.scrub_doublets(verbose=False)
        try:
            masks = scrub.call_doublets()
            print(f"Automatically identified doublet score threshold: {scrub.threshold_}")
        except:
            masks = scrub.call_doublets(threshold=0.25)
            print("Using manual doublet score threshold: 0.25")
    
    adata.obs["doublet_score"] = scores
    adata.obs["doublet_class"] = pd.Categorical(masks.astype(str))
    adata = adata[adata.obs["doublet_class"] == "False"].copy()
    return adata

if __name__ == "__main__":
    args = parse_args()

    # Check if quality control is enabled
    if args.do_QC == "False":
        print("Quality control is disabled. Loading and saving data...")
        
        adata = sc.read_h5ad(args.h5ad_file)
        print(f"Dataset size: {adata.n_obs} cells x {adata.n_vars} genes")
        
        adata = prepare_adata(adata, args.species)
        adata.write(args.output_file)
    else:
        # Start timer
        start_time = time.time()

        # Optimize thread usage: dedicate more threads to computational libraries
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

        adata = sc.read_h5ad(args.h5ad_file)
        print("===============================")
        print("Quality control...")
        print(f"Dataset size: {adata.n_obs} cells x {adata.n_vars} genes")
        print(f"Memory usage: {adata.X.data.nbytes / 1e9:.2f} GB" if hasattr(adata.X, 'data') else f"Memory usage: {adata.X.nbytes / 1e9:.2f} GB")

        adata = prepare_adata(adata, args.species)
        print(f"\nInitial number of cells: {adata.n_obs}")
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

        outlier_results = []
        for metric in metrics:
            outlier_results.append(calculate_outlier_vector(adata.obs[metric], args.nmads))

        adata.obs["outlier"] = np.any(outlier_results, axis=0)
        adata.obs["mt_outlier"] = calculate_outlier_vector(
            adata.obs["pct_counts_mt"], 3
        ) | (adata.obs["pct_counts_mt"] > 8)

        adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
        print(f" - Number of cells after filtering low quality cells: {adata.n_obs}")

        print("Detecting doublets...")
        # Use parallel processing more aggressively for large datasets
        use_parallel = args.force_parallel or (adata.n_obs > 20_000 and args.threads > 1)
        adata = run_scrublet(adata, n_jobs=args.threads if use_parallel else 1)

        print(f"Final number of cells: {adata.n_obs}")

        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.4f} seconds")

        print(f"Saving to {args.output_file}")
        adata.write(args.output_file)