"""
constants.py
------------
Centralized constants for the SIGMA pipeline.

Contains all shared constants, thresholds, and default values used across
multiple scripts in the pipeline.
"""

import matplotlib.colors as mcolors


# =============================================================================
# Gene Pattern Constants
# =============================================================================

MT_GENE_PATTERN = r"^MT-|^mt-"
"""Regex pattern for mitochondrial genes."""

RIBO_GENE_PATTERN = r"^RP[LS]\d+"
"""Regex pattern for ribosomal genes."""

HB_GENE_PATTERN = r"^HB[^P]|^HB[AB]"
"""Regex pattern for hemoglobin genes."""


# =============================================================================
# Normalization Constants
# =============================================================================

TARGET_SUM = 1e6
"""Target sum for CPM normalization."""


# =============================================================================
# Quality Control Constants
# =============================================================================

DEFAULT_DOUBLET_THRESHOLD = 0.25
"""Default threshold for Scrublet doublet detection."""

MIN_PCA_COMPONENTS = 1
"""Minimum number of PCA components for doublet detection."""

MAX_PCA_COMPONENTS = 30
"""Maximum number of PCA components for doublet detection."""

BATCH_SIZE_THRESHOLD = 10_000
"""Cell count threshold for batched Scrublet processing."""


# =============================================================================
# GMM Clustering Constants
# =============================================================================

MIN_CELLS_FOR_UMAP = 10
"""Minimum cells required for UMAP computation."""

KDE_GRID_POINTS = 1_500
"""Number of points in KDE grid for peak detection."""

PEAK_PROMINENCE = 0.01
"""Minimum prominence for KDE peak detection."""

SECOND_DERIVATIVE_THRESHOLD = -0.001
"""Second derivative threshold for significant peak filtering."""

ASHMANN_DISTANCE_THRESHOLD = 2
"""Ashmann distance threshold for component merging."""

CV_FOLDS = 5
"""Number of cross-validation folds for KDE bandwidth selection."""

CV_TEST_SIZE = 1/3
"""Test set proportion for cross-validation."""


# =============================================================================
# Ensembl / Gene Renaming Constants
# =============================================================================

DEFAULT_SPECIES = "hsapiens"
"""Default species for Ensembl queries."""

DEFAULT_HOST = "http://www.ensembl.org"
"""Default Ensembl BioMart host URL."""

ENSEMBL_ATTRIBUTES = [
    "ensembl_gene_id",
    "external_gene_name",
    "hgnc_symbol",
    "external_synonym",
]
"""Ensembl BioMart attributes to query for gene mapping."""

GENE_NAME_COLUMN = "Gene name"
"""Column name for gene names in Ensembl DataFrame."""


# =============================================================================
# Custom Colormaps
# =============================================================================

GRAY_TO_RED = mcolors.LinearSegmentedColormap.from_list(
    "gray_to_red", [(0.5, 0.5, 0.5), (1, 0, 0)]
)
"""Colormap from gray to red for probability visualization."""

BLUE_GRAY_RED = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", [(0, 0, 1), (0.5, 0.5, 0.5), (1, 0, 0)]
)
"""Diverging colormap from blue through gray to red for score visualization."""


# =============================================================================
# Thread Environment Variables
# =============================================================================

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
"""Environment variables to set for controlling parallel thread count."""
