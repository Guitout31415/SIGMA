#!/usr/bin/env python3
"""
Function to rename genes stored in a list using standard Ensembl gene names.

Author: Guillaume Lemaire
License: MIT

Examples:
    >>> rename_genes(["ENSG00000100000", "ENSG00000100001"])
    ['gene1', 'gene2']
"""
from typing import List
from pybiomart import Dataset

def rename_genes(gene_list: List[str], 
                 species: str = "hsapiens", 
                 host: str = "http://www.ensembl.org") -> List[str]:
    """Rename genes using Ensembl gene names.

    Args:
        genes (List[str]): List of gene names
        species (str): Species (default: hsapiens)
    Returns:
        List[str]: List of renamed gene names
    Raises:
        ValueError: If unable to connect to Ensembl host
    """
    assert isinstance(gene_list, list), "genes must be a list"
    assert all(isinstance(gene, str) for gene in gene_list), "all genes must be strings"
    assert isinstance(species, str), "species must be a string"
    assert isinstance(host, str), "host must be a string"

    try:
        dataset = Dataset(name=f"{species}_gene_ensembl", host=host)
        genes_df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'hgnc_symbol', 'external_synonym'])
    except Exception as e:
        raise ValueError(f"Unable to connect to Ensembl host: {e}")

    # Harmonise les colonnes et majuscules
    genes_upper = [g.strip().upper() for g in gene_list]

    # Recherche vectorisée
    lookup = set(genes_upper)
    genes_map = {}
    for _, row in genes_df.iterrows():
        for val in row:
            if val in lookup and val not in genes_map:
                genes_map[val] = row['Gene name']

    # Remplacement
    renamed_genes = [genes_map.get(g, g) for g in genes_upper]
    return renamed_genes
