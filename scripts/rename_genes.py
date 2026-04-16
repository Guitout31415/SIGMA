#!/usr/bin/env python3
"""
rename_genes.py
---------------
Function to rename genes stored in a list using standard Ensembl gene names.

Author: Guillaume Lemaire
License: MIT

Examples:
    >>> rename_genes(["ENSG00000100000", "ENSG00000100001"])
    ['gene1', 'gene2']
"""

from functools import lru_cache
from typing import List, Dict
import warnings

import pandas as pd
from pybiomart import Dataset

from constants import (
    DEFAULT_SPECIES,
    DEFAULT_HOST,
    ENSEMBL_FALLBACK_HOSTS,
    ENSEMBL_ATTRIBUTES,
    GENE_NAME_COLUMN,
)


def _get_host_candidates(host: str) -> List[str]:
    """Build an ordered list of Ensembl hosts to try."""
    ordered_hosts = [host.strip()]

    if host.startswith("http://"):
        ordered_hosts.append("https://" + host[len("http://"):])
    elif host.startswith("https://"):
        ordered_hosts.append("http://" + host[len("https://"):])
    elif host:
        ordered_hosts.extend([f"https://{host}", f"http://{host}"])

    ordered_hosts.extend(ENSEMBL_FALLBACK_HOSTS)

    deduped_hosts = []
    seen = set()
    for h in ordered_hosts:
        clean_h = h.strip()
        if clean_h and clean_h not in seen:
            deduped_hosts.append(clean_h)
            seen.add(clean_h)
    return deduped_hosts


@lru_cache(maxsize=32)
def _query_ensembl(species: str, host: str) -> pd.DataFrame:
    """Run a single BioMart query for one species/host pair."""
    dataset = Dataset(name=f"{species}_gene_ensembl", host=host)
    return dataset.query(attributes=ENSEMBL_ATTRIBUTES)


def _fetch_ensembl_data(species: str, host: str) -> pd.DataFrame:
    """Fetch gene data from Ensembl BioMart.

    Args:
        species: Species identifier (e.g., 'hsapiens')
        host: Ensembl host URL

    Returns:
        DataFrame with gene annotation data

    Raises:
        ValueError: If unable to connect to Ensembl
    """
    errors = []
    for candidate_host in _get_host_candidates(host):
        try:
            return _query_ensembl(species, candidate_host)
        except Exception as e:
            errors.append(f"{candidate_host} -> {type(e).__name__}: {e}")

    raise ValueError(
        "Unable to connect to Ensembl BioMart. Tried hosts: "
        + " | ".join(errors)
    )


def _build_gene_lookup(genes_df: pd.DataFrame) -> Dict[str, int]:
    """Build a lookup dictionary from gene identifiers to DataFrame row indices.

    Args:
        genes_df: DataFrame containing gene annotation data

    Returns:
        Dictionary mapping gene identifiers to their row index
    """
    lookup = {}
    for row_idx, row in genes_df.iterrows():
        for identifier in row:
            if pd.notna(identifier) and identifier != "":
                lookup[str(identifier).strip().upper()] = row_idx
    return lookup


def _map_single_gene(
    gene: str, lookup: Dict[str, int], genes_df: pd.DataFrame
) -> str:
    """Map a single gene identifier to its standard name.

    Args:
        gene: Gene identifier (uppercase)
        lookup: Gene lookup dictionary
        genes_df: DataFrame with gene annotations

    Returns:
        Mapped gene name or original identifier if not found
    """
    if gene not in lookup:
        return gene

    row_idx = lookup[gene]
    gene_name = genes_df.loc[row_idx, GENE_NAME_COLUMN]

    if pd.isna(gene_name) or gene_name == "":
        return gene

    return gene_name


def rename_genes(
    gene_list: List[str],
    species: str = DEFAULT_SPECIES,
    host: str = DEFAULT_HOST,
) -> List[str]:
    """Rename genes using Ensembl gene names.

    Args:
        gene_list: List of gene names/identifiers
        species: Species identifier (default: 'hsapiens')
        host: Ensembl host URL

    Returns:
        List of renamed gene names

    Raises:
        ValueError: If unable to connect to Ensembl host
        AssertionError: If input types are invalid
    """
    assert isinstance(gene_list, list), "genes must be a list"
    assert all(isinstance(gene, str) for gene in gene_list), "all genes must be strings"
    assert isinstance(species, str), "species must be a string"
    assert isinstance(host, str), "host must be a string"

    genes_upper = [g.strip().upper() for g in gene_list]

    try:
        genes_df = _fetch_ensembl_data(species, host)
    except ValueError as e:
        warnings.warn(
            "Ensembl BioMart is unavailable. "
            "Proceeding without gene alias remapping and keeping input identifiers. "
            f"Details: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return genes_upper

    if GENE_NAME_COLUMN not in genes_df.columns:
        warnings.warn(
            f"Expected column '{GENE_NAME_COLUMN}' not found in Ensembl response. "
            "Proceeding without gene alias remapping.",
            RuntimeWarning,
            stacklevel=2,
        )
        return genes_upper

    lookup = _build_gene_lookup(genes_df)
    return [_map_single_gene(g, lookup, genes_df) for g in genes_upper]
