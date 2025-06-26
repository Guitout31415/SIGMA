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
        dataset = Dataset(name=species+"_gene_ensembl", host="http://www.ensembl.org")
    except Exception as e:
        raise ValueError(f"Unable to connect to Ensembl host: {e}")
    
    try:
        gene_mapping = dataset.query(attributes=['ensembl_gene_id',
                                                 'external_gene_name',
                                                 'hgnc_symbol',
                                                 'external_synonym'])
    except Exception as e:
        raise ValueError(f"Unable to retrieve Ensembl data: {e}")
    
    df_converter = pd.DataFrame(columns=results.columns)
    for gene in gene_list:
        mask = results.eq(gene).any(axis=1)
        df_converter = pd.concat([df_converter, results[mask]], ignore_index=True)
    df_converter

    gene_names = []
    for gene in gene_list:
        mask = results.eq(gene).any(axis=1)
        gene_names += results[mask]["Gene name"].dropna().unique().tolist()
    gene_names

    return genes
