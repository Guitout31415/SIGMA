# SIGMA : Single-cell Identifier using Gaussian Mixture Approach

SIGMA is a bioinformatics pipeline for automated identification of targeted cell types from multiple heterogeneous single-cell transcriptomic datasets (formats supported: `.h5ad`). It is designed to simplify and accelerate integrative large-scale single-cell analyses.

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

Edit `config_template.conf` to specify:

- **[Metadata]**: Metadata fields to extract/conserve
- **[Candidate]**: Candidate genes for initial filtering
- **[Target]**: Positive marker genes for target cell identification
- **[Exclude1]**: Negative or exclusion marker genes used to exclude cell types 1
- **[Exclude2]**: Negative or exclusion marker genes used to exclude cell types 2
- **[Thresholds]**: Filtering and assignment thresholds
- **[Folder]**: Input/output folder paths
- **[Options]**: Species, QC parameters, plotting options

## Usage

Run the pipeline with Snakemake:

```bash
snakemake \
  --config file=ConfigFile.conf \
  --cores 10 \
  --jobs 5 \
  --resources mem_mb=100000
```

- Adjust `--cores` and `--jobs` as needed.
- The main workflow is defined in `Snakefile`.

### Main Steps

1. **Quality control**: Filtering, outlier removal, doublet detection (`scripts/quality_control.py`)
2. **Target extraction**: Select cells expressing candidate/marker genes (`scripts/extract_target.py`)
3. **Harmonization**: Standardize metadata interface (`scripts/harmonize_metadata.py`)
4. **Merging**: Combine all studies into a single `.h5ad` (`scripts/merge_h5ad.py`)

## Output

- Filtered and processed `.h5ad` files per study: `output_folder/qc/`, `output_folder/find/`, `output_folder/harmonized/`
- Merged dataset: `output_folder/merge.h5ad`
- Logs: `output_folder/logs/`
- Plots: `output_folder/plots/`

## Troubleshooting

- Check log files in `output_folder/logs/` for errors.
- Ensure all dependencies are installed and input files are correctly formatted.

## License

GPL-3.0 License. See [LICENSE](LICENSE).
