import shutup; shutup.please()
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
import scrublet as scr
from pybiomart import Dataset
from scipy.stats import median_abs_deviation
import pandas as pd
from typing import List
from pybiomart import Dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Quality control")
    parser.add_argument("--", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

def rename_genes(genes: List[str] | pd.Index, 
                 species: str = "hsapiens", 
                 host: str = "http://www.ensembl.org") -> List[str]:
    """Rename genes using Ensembl gene names.

    Args:
        genes (List[str] | pd.Index): List of gene names
        species (str): Species (default: hsapiens)
        host (str): Ensembl host (default: http://www.ensembl.org)
    Returns:
        List[str]: List of renamed gene names
    Raises:
        ValueError: If unable to connect to Ensembl host
    """
    assert isinstance(genes, (list, pd.Index)), "genes must be a list or pandas index"
    assert all(isinstance(gene, str) for gene in genes), "all genes must be strings"
    assert isinstance(species, str), "species must be a string"
    assert isinstance(host, str), "host must be a string"

    try:
        dataset = Dataset(name=species+"_gene_ensembl", host=host)
    except Exception as e:
        raise ValueError(f"Unable to connect to Ensembl host: {e}")
    
    try:
        gene_mapping = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    except Exception as e:
        raise ValueError(f"Unable to retrieve Ensembl data: {e}")
    
    gene_mapping.index = gene_mapping["Gene stable ID"]
    gene_mapping = gene_mapping[~pd.isna(gene_mapping["Gene name"])]

    genes = [gene_mapping.loc[gene, "Gene name"] if gene in gene_mapping.index else gene for gene in genes]

    return genes

def prepare_adata(adata):
    if adata.raw is not None:
        raw_adata = adata.raw.to_adata()
        count_matrix = raw_adata[:, adata.var_names].X
    else:
        count_matrix = adata.X

    adata = sc.AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())
    adata.var_names = rename_genes(adata.var_names)
    return adata

def identify_special_genes(adata):
    """Identify special genes (e.g., mitochondrial, ribosomal, hemoglobin) in the AnnData object.

    :param adata: (sc.AnnData) AnnData object

    :return: (sc.AnnData) AnnData object with special genes identified

    Notes:
    - The function identifies mitochondrial genes, ribosomal genes, and hemoglobin genes.
    - The function adds columns to the AnnData object for each special gene type.

    Examples:
    >>> adata = identify_special_genes(adata)
    # mt genes : 100
    # ribo genes : 20
    # hb genes : 5
    """
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[LS]\d+")
    adata.var["hb"] = adata.var_names.str.match(r"^HB[^P]")
    print(f"- # mt genes : {adata.var.mt.sum()}")
    print(f"- # ribo genes : {adata.var.ribo.sum()}")
    print(f"- # hb genes : {adata.var.hb.sum()}")
    return adata.copy() 

def calculate_outlier(adata, metric, nmads):
    """Calculate outliers for a given metric in the AnnData object.

    :param adata: (sc.AnnData) AnnData object
    :param metric: (str) Metric to calculate outliers
    :param nmads: (int) Number of median absolute deviations to consider as outliers

    :return: (np.ndarray) Boolean array indicating outliers

    Notes:
    - The function calculates outliers based on the median and median absolute deviation.
    - The function returns a boolean array indicating the outliers for the given metric.

    Examples:
    >>> calculate_outlier(adata, "log1p_total_counts", 5)
    array([False, False, False, ..., False, False, False])
    """
    M = adata.obs[metric]
    med = np.median(M)
    mad = median_abs_deviation(M)
    lower_bound = med - nmads * mad
    upper_bound = med + nmads * mad
    return (M < lower_bound) | (M > upper_bound)

def run_scrublet(adata):
    """Run Scrublet to identify doublets in the AnnData object.

    :param adata: (sc.AnnData) AnnData object

    :return: (sc.AnnData) AnnData object with doublet scores
    """
    scrub = scr.Scrublet(adata.X)
    doublet_scores, _ = scrub.scrub_doublets(verbose=False)
    adata.obs["doublet_score"] = doublet_scores
    try:
        # Try automatic threshold detection first
        doublet_mask = scrub.call_doublets()
        print(f"Automatically identified doublet score threshold: {scrub.threshold_}")
    except Exception as e:
        print(f"Warning: {str(e)}")
        # Set a conservative manual threshold if automatic detection fails
        threshold = 0.25  # Conservative default threshold
        doublet_mask = scrub.call_doublets(threshold=threshold)
        print(f"Using manual doublet score threshold: {threshold}")

    adata.obs["doublet_class"] = doublet_mask
    adata.obs["doublet_class"] = (
        adata.obs["doublet_class"].astype(str).astype("category")
    )
    # Filter doublets
    adata = adata[adata.obs["doublet_class"] == "False"].copy()
    return adata.copy()

if __name__ == "__main__":