import os
import shutup; shutup.please()
import argparse
import scanpy as sc
import anndata as ad

import sys
full_path = os.path.abspath(__file__)


from read_config import read_config
from rename_genes import rename_genes
from quality_control import *


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
        adata = identify_special_genes(adata)
        adata = calculate_outlier(adata, "log1p_total_counts", 5)
        adata = run_scrublet(adata)

