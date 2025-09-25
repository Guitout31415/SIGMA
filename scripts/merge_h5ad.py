import scanpy as sc
import argparse
import anndata as ad
import os
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple h5ad study files into a single dataset with common genes."
    )
    parser.add_argument(
        "--study_folder", "-i", type=str, required=True,
        help="Absolute path to folder containing h5ad study files to merge"
    )
    parser.add_argument(
        "--output_file", "-o", type=str, required=True,
        help="Absolute path to save the merged h5ad file"
    )
    return parser.parse_args()


def sanitize_obs(adata: ad.AnnData) -> None:
    """Convert all columns in .obs to string to prevent HDF5 writing errors."""
    print("[Sanitize] Converting all .obs columns to strings...")
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str)

if __name__ == "__main__":
    args = parse_arguments()
    study_files = [
        f for f in os.listdir(args.study_folder) if f.endswith(".h5ad")
    ]
    studies_dict = {}
    gene_sets = []
    for study in study_files:
        study_name = os.path.splitext(study)[0]
        adata = sc.read_h5ad(os.path.join(args.study_folder, study))
        if adata.n_obs == 0:
            print(f"[Warning] Study '{study_name}' is empty, skipping.")
            continue
        # Remove duplicate gene names, keep first occurrence
        if not adata.var_names.is_unique:
            adata = adata[:, ~adata.var_names.duplicated(keep='first')]
        studies_dict[study_name] = adata
        gene_sets.append(set(adata.var_names))

    # Find intersection of all gene sets
    common_genes = list(set.intersection(*gene_sets))

    # Subset each AnnData to common genes, preserving order
    for name in studies_dict:
        adata = studies_dict[name]
        genes_in_order = [gene for gene in adata.var_names if gene in common_genes]
        studies_dict[name] = adata[:, genes_in_order]

    # Concatenate all studies
    adatas = list(studies_dict.values())
    study_names = list(studies_dict.keys())

    adata_merged = ad.concat(adatas, label="study", keys=study_names, index_unique="-")

    # Reset index name to avoid conflicts with column names
    adata_merged.obs.index.name = None

    # Try writing merged file with fallback on TypeError
    try:
        adata_merged.write(args.output_file)
    except TypeError as e:
        if "Can't implicitly convert non-string objects to string" in str(e):
            print("[Error] TypeError during write: converting .obs columns to string.")
            sanitize_obs(adata_merged)
            print("[Info] Retrying write after sanitization...")
            adata_merged.write(args.output_file)
        else:
            raise
