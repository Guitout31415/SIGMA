# SIGMA : Single-cell Identifier using Gaussian Mixture Approach

SIGMA is a bioinformatics pipeline for automated extraction of targeted cell types from multiple heterogeneous single-cell transcriptomic datasets (formats supported: `.h5ad`). It is designed to simplify and accelerate integrative large-scale single-cell analyses.

## Features

- Quality control, filtering, and doublet removal
- Extraction of target cell types based on user-defined gene markers
- Harmonization and merging of multiple studies
- Fully configurable via a single config file

## Installation

Clone the repository:

```bash
git clone git@github.com:Guitout31415/SIGMA.git
cd SIGMA
conda env create -f environment.yml
conda activate SIGMA
pip install snakemake
```

## Configuration

Edit the `config_mk.conf` file to specify:

- **[Metadata]**: Metadata fields to extract/conserve
- **[Candidate]**: Candidate genes for initial filtering
- **[Markers]**: Marker genes for final assignment
- **[Exclude]**: Marker genes 
- **[Thresholds]**: Filtering and assignment thresholds
- **[Folder]**: Input/output folder paths
- **[Options]**: Species, QC parameters, plotting options

## Usage

Run the pipeline with Snakemake:

```bash
snakemake \
  --config file=config_mk.conf threads=1 mem_mb=20000 \
  --cores 10 \
  --resources mem_mb=200000
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
- Merged dataset: `data/results/merge.h5ad`
- Logs: `data/results/logs/`
- Plots (if enabled): as specified in config

## Troubleshooting

- Check log files in `data/results/logs/` for errors.
- Ensure all dependencies are installed and input files are correctly formatted.
- For `.rds` file, please open and use `scripts/convert_to_h5ad.R`
## License

GPL-3.0 License. See [LICENSE](LICENSE).
