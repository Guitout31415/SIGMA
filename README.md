# CellExtractor

CellExtractor is a bioinformatics pipeline for automated extraction of targeted cell types from multiple heterogeneous single-cell transcriptomic datasets (formats supported: `.h5ad`, `.rds`, `.csv`). It is designed to simplify and accelerate integrative large-scale single-cell analyses.

## Features

- Quality control, filtering, and doublet removal
- Extraction of target cell types based on user-defined gene markers
- Harmonization and merging of multiple studies
- Fully configurable via a single config file
- Supports AnnData (`.h5ad`), RDS, and CSV input formats

## Requirements

- Linux or macOS
- Python 3.8+
- [Snakemake](https://snakemake.readthedocs.io/en/stable/) >=7.0
- Python packages: `scanpy`, `anndata`, `numpy`, `matplotlib`, `scrublet`, `pybiomart`, `pandas`
- R (optional, for `.rds` input)

Install Python dependencies (example with pip):

```bash
pip install scanpy anndata numpy matplotlib scrublet pybiomart pandas
```

Install Snakemake (if not already):

```bash
pip install snakemake
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Guitout31415/CellExtractor.git
cd CellExtractor
```

## Configuration

Edit the `config_mk.conf` file to specify:

- **[Metadata]**: Metadata fields to extract
- **[Candidate]**: Candidate genes for initial filtering
- **[Markers]**: Marker genes for final assignment
- **[Thresholds]**: Filtering and assignment thresholds
- **[Folder]**: Input/output folder paths
- **[Options]**: Species, QC parameters, plotting options

Example (excerpt):

```
[Folder]
input_folder = /path/to/data/raw
output_folder = /path/to/data/results

[Thresholds]
min_genes_detected = 1
assign_threshold = auto2
```

## Data Organization

- Place your raw single-cell files (`.h5ad`, `.rds`, `.csv`) in `data/raw/` (or the folder specified in config).
- Results will be written to `data/results/` (or as specified).

## Usage

Run the pipeline with Snakemake:

```bash
snakemake \
  --config file=config_mk.conf threads=8 mem_mb=200000 \
  --cores 50 \
  --resources mem_mb=700000 \
  --latency-wait 60 \
  -p
```

- Adjust `--cores` as needed.
- The main workflow is defined in `Snakefile`.

### Main Steps

1. **Quality control**: Filtering, outlier removal, doublet detection (`scripts/quality_control.py`)
2. **Target extraction**: Select cells expressing candidate/marker genes (`scripts/extract_target.py`)
3. **Harmonization**: Standardize metadata
4. **Merging**: Combine all studies into a single `.h5ad`

## Output

- Filtered and processed `.h5ad` files per study: `data/results/qc/`, `data/results/extracted/`, `data/results/harmonized/`
- Merged dataset: `data/results/merged.h5ad`
- Logs: `data/results/logs/`
- Plots (if enabled): as specified in config

## Troubleshooting

- Check log files in `data/results/logs/` for errors.
- Ensure all dependencies are installed and input files are correctly formatted.
- For `.rds` or `.csv` support, you may need to adapt or extend the scripts.

## License

MIT License. See [LICENSE](LICENSE).
