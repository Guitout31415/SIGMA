"""
merge_h5ad.py
-------------
Merge multiple h5ad study files into a single dataset with common genes.
"""

import os
import argparse
from typing import List, Dict

import scanpy as sc
import anndata as ad
import pandas as pd


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple h5ad study files into a single dataset."
    )
    parser.add_argument(
        "--study_folder",
        "-i",
        type=str,
        required=True,
        help="Absolute path to folder containing h5ad study files to merge",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="Absolute path to save the merged h5ad file",
    )
    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def get_study_files(folder: str) -> List[str]:
    """Get list of h5ad files in a folder.

    Args:
        folder: Path to folder containing h5ad files

    Returns:
        List of h5ad filenames
    """
    return [f for f in os.listdir(folder) if f.endswith(".h5ad")]


def load_studies(
    study_folder: str,
    study_files: List[str],
) -> tuple:
    """Load all study files and collect metadata.

    Args:
        study_folder: Path to folder containing studies
        study_files: List of h5ad filenames

    Returns:
        Tuple of (studies_dict, gene_sets, obs_columns_union)
    """
    studies_dict: Dict[str, ad.AnnData] = {}
    gene_sets: List[set] = []
    obs_columns_union: List[str] = []

    for study in study_files:
        study_name = os.path.splitext(study)[0]
        adata = sc.read_h5ad(os.path.join(study_folder, study))

        if adata.n_obs == 0:
            print(f"[Warning] Study '{study_name}' is empty, skipping.")
            continue

        # Remove duplicate gene names
        if not adata.var_names.is_unique:
            adata = adata[:, ~adata.var_names.duplicated(keep="first")]

        studies_dict[study_name] = adata
        gene_sets.append(set(adata.var_names))

        # Collect all observation columns
        for col in adata.obs.columns:
            if col not in obs_columns_union:
                obs_columns_union.append(col)

    return studies_dict, gene_sets, obs_columns_union


def find_common_genes(gene_sets: List[set]) -> List[str]:
    """Find intersection of all gene sets.

    Args:
        gene_sets: List of gene name sets

    Returns:
        List of common gene names
    """
    return list(set.intersection(*gene_sets))


def subset_to_common_genes(
    studies_dict: Dict[str, ad.AnnData],
    common_genes: List[str],
    obs_columns_union: List[str],
) -> Dict[str, ad.AnnData]:
    """Subset each study to common genes and unify obs columns.

    Args:
        studies_dict: Dictionary of study name to AnnData
        common_genes: List of common gene names
        obs_columns_union: Union of all observation columns

    Returns:
        Updated studies dictionary
    """
    for name in studies_dict:
        adata = studies_dict[name]

        # Reindex obs columns to include all columns
        if obs_columns_union:
            adata.obs = adata.obs.reindex(columns=obs_columns_union)

        # Subset to common genes (preserving order)
        genes_in_order = [g for g in adata.var_names if g in common_genes]
        studies_dict[name] = adata[:, genes_in_order]

    return studies_dict


def sanitize_obs(adata: ad.AnnData) -> None:
    """Convert all obs columns to string to prevent HDF5 errors.

    Args:
        adata: AnnData object to sanitize (modified in place)
    """
    print("[Sanitize] Converting all .obs columns to strings...")
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str)


def merge_studies(studies_dict: Dict[str, ad.AnnData]) -> ad.AnnData:
    """Merge all studies into a single AnnData object.

    Args:
        studies_dict: Dictionary of study name to AnnData

    Returns:
        Merged AnnData object
    """
    adatas = list(studies_dict.values())
    study_names = list(studies_dict.keys())

    adata_merged = ad.concat(
        adatas,
        label="study",
        keys=study_names,
        index_unique="-",
    )

    # Reset index name to avoid conflicts
    adata_merged.obs.index.name = None

    return adata_merged


def save_merged_file(adata: ad.AnnData, output_file: str) -> None:
    """Save merged AnnData with fallback on TypeError.

    Args:
        adata: AnnData object to save
        output_file: Output file path
    """
    try:
        adata.write(output_file)
    except TypeError as e:
        if "Can't implicitly convert non-string objects to string" in str(e):
            print("[Error] TypeError during write: converting .obs columns to string.")
            sanitize_obs(adata)
            print("[Info] Retrying write after sanitization...")
            adata.write(output_file)
        else:
            raise


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main entry point for merging studies."""
    args = parse_arguments()

    study_files = get_study_files(args.study_folder)
    print(f"Found {len(study_files)} h5ad files to merge.")

    studies_dict, gene_sets, obs_columns_union = load_studies(
        args.study_folder, study_files
    )

    if not studies_dict:
        print("[Error] No valid studies found. Exiting.")
        return

    common_genes = find_common_genes(gene_sets)
    print(f"Found {len(common_genes)} common genes across all studies.")

    studies_dict = subset_to_common_genes(studies_dict, common_genes, obs_columns_union)

    adata_merged = merge_studies(studies_dict)
    print(f"Merged dataset contains {adata_merged.n_obs} cells.")

    save_merged_file(adata_merged, args.output_file)
    print(f"Merged file saved to: {args.output_file}")


if __name__ == "__main__":
    main()
