import os
import shutup; shutup.please()
import argparse
import scanpy as sc
import anndata as ad

from scripts.read_config import read_config
from scripts.rename_genes import rename_genes
from scripts.quality_control import *


def parse_args():
    parser = argparse.ArgumentParser(description="CellExtractor pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)

    input_folder = config["Folder"]["input_folder"]
    studies_path = [path for path in os.listdir(input_folder) if path.endswith(".h5ad")]

    for h5ad_file in studies_path:
        print(f"Processing {h5ad_file}...")
        adata = sc.read_h5ad(os.path.join(input_folder, h5ad_file))

        print("\n===============================")
        print("Quality control...")

        adata = prepare_adata(adata)
        
        print(f"Initial number of cells: {adata.n_obs}")
        adata = identify_special_genes(adata)

        print("Computing quality control metrics...")
        sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=["mt", "ribo", "hb"],
                inplace=True,
                percent_top=[int(config["Optional"]["percent_top"])],
                log1p=True
            )

        print("Detecting outliers...")
        metrics = [
            "log1p_total_counts",
            "log1p_n_genes_by_counts",
            "pct_counts_in_top_20_genes",
        ]
        nmads = int(config["Optional"]["nmads"])
        outlier_results = [calculate_outlier(adata, m, nmads) for m in metrics]
        adata.obs["outlier"] = np.any(outlier_results, axis=0)
        adata.obs["mt_outlier"] = calculate_outlier(adata, "pct_counts_mt", 3) | (
            adata.obs["pct_counts_mt"] > 8
        )

        # Filter low-quality cells
        adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
        print(f" - Number of cells after filtering of low quality cells: {adata.n_obs}")

        print("Detecting doublets...")
        adata = run_scrublet(adata)

        print(f"Final number of cells: {adata.n_obs}")
