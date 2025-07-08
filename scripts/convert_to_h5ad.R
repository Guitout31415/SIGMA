library(Seurat)
library(dplyr)
library(tibble)
library(ggplot2)
library(future)
library(edgeR)
library(SeuratDisk)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript convert_to_h5ad.R <input_rds> <output_h5ad>")
}

input_rds <- args[1]
output_h5ad <- args[2]
output_h5seurat <- sub("\\.h5ad$", ".h5Seurat", output_h5ad)

if (!file.exists(input_rds)) {
  stop(paste0("Input RDS file not found: ", input_rds))
}


kidn <- readRDS(input_rds)

kidn@meta.data <- kidn@meta.data %>%
  mutate(across(everything(), as.character))

SaveH5Seurat(
  object = kidn,
  filename = output_h5seurat,
  overwrite = TRUE
)

Convert(
  source = output_h5seurat,
  dest = output_h5ad,
  assay = "RNA",
  overwrite = TRUE
)