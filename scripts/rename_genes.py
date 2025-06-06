#!/usr/bin/env python3
"""
Function to rename genes stored in a list using standard Ensembl gene names.

Author: Guillaume Lemaire
License: MIT

Examples:
    >>> rename_genes(["ENSG00000100000", "ENSG00000100001"])
    ['gene1', 'gene2']
"""
import pandas as pd
from typing import List
from pybiomart import Dataset

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
