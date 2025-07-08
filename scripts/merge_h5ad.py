import shutup; shutup.please()
import scanpy as sc
import argparse
import anndata as ad
import os

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple h5ad study files into a single dataset with common genes.")
    parser.add_argument("--study_folder", "-i", type=str, help="Absolute path to folder containing h5ad study files to merge", required=True)
    parser.add_argument("--output_file", "-o", type=str, help="Absolute path to save the merged h5ad file", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    studies_path = [
        path for path in os.listdir(args.study_folder) if path.endswith(".h5ad")
    ]

    studies_dict = dict()
    total_cells = 0
    gene_sets = []

    for study in studies_path:
        study_name = study.split(".")[0]
        adata = sc.read_h5ad(os.path.join(args.study_folder, study))
        adata = adata[adata.obs['is_target']]
        if not adata.var_names.is_unique:
            duplicates = adata.var_names[adata.var_names.duplicated()].tolist()
            adata = adata[:, ~adata.var_names.duplicated(keep='first')]
        studies_dict[study_name] = adata
        total_cells += adata.shape[0]
        gene_sets.append(set(adata.var_names))

    # Ensure all elements in gene_sets are sets
    gene_sets = [set(genes) for genes in gene_sets]
    common_genes = list(set.intersection(*gene_sets))

    studies_dict = {
        name: data[:, common_genes]
        for name, data in studies_dict.items()
    }

    adatas = list(studies_dict.values())
    study_names = list(studies_dict.keys())

    # Concatenate the AnnData objects along the observation axis (rows)
    adata = ad.concat(adatas, label="study", keys=study_names, index_unique="-")

    adata.write(args.output_file)