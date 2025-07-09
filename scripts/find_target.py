import shutup; shutup.please()
import scanpy as sc
import pandas as pd
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import argparse
from pybiomart import Dataset

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--h5ad_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--candidate_genes", nargs='+', type=str, required=True)
    parser.add_argument("--assign_genes", nargs='+', type=str, required=True)
    parser.add_argument("--min_genes_detected", type=float, required=True)
    parser.add_argument("--gene_detection_threshold", type=float, required=True)
    parser.add_argument("--n_components", type=int, required=True)
    parser.add_argument("--plot_extracted", type=bool, default=False)
    parser.add_argument("--plot_folder", type=str, default=None)
    parser.add_argument("--species", type=str, default="hsapiens")
    return parser.parse_args()

def compute_expression(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=4)
    sc.tl.umap(adata)
    return adata.copy()

def get_all_gene_aliases(gene_list, species="hsapiens"):
    dataset = Dataset(name=f'{species}_gene_ensembl', host='http://www.ensembl.org')
    results = dataset.query(attributes=['ensembl_gene_id',
                                        'external_gene_name',
                                        'hgnc_symbol',
                                        'external_synonym'])
    results_lower = results.applymap(lambda x: str(x).lower() if pd.notnull(x) else x)
    all_aliases = set()
    for gene in gene_list:
        gene_lower = gene.lower()
        mask = results_lower.eq(gene_lower).any(axis=1)
        all_aliases |= set(results[mask].values.flatten().tolist())
    return all_aliases

def extract_target(adata, study_name, candidate_genes, assign_genes,
                   min_genes_detected, gene_detection_threshold, n_components=2,
                   plot_extracted=False, plot_folder=None, species="hsapiens"):

    candidate_aliases = get_all_gene_aliases(candidate_genes, species=species)
    assign_aliases = get_all_gene_aliases(assign_genes, species=species)

    candidate_genes_avail = candidate_aliases & set(adata.var_names)
    assign_genes_avail = assign_aliases & set(adata.var_names)

    print(f"Candidate genes available : {candidate_genes_avail}")
    print(f"Assign genes available : {assign_genes_avail}")
    print(f"Keep cells that express at least {int(min_genes_detected)} candidate genes, each with occurrence >= {int(gene_detection_threshold)}.")

    adata_copy = adata.copy()
    candidate_df = adata_copy[:, list(candidate_genes_avail)].to_df()

    genes_detected = (candidate_df >= gene_detection_threshold).sum(axis=1)

    is_expressed = genes_detected >= min_genes_detected
    candidate = adata_copy[is_expressed].copy()

    print(f"Number (%) of candidate cells : {candidate.shape[0]} ({candidate.shape[0]/adata_copy.shape[0]*100:.2f}%)")

    candidate_expressed = compute_expression(candidate)

    assign_df = candidate_expressed[:, list(assign_genes_avail)].to_df()
    candidate_expressed.obs['mean_expr'] = assign_df.mean(axis=1)

    os.environ['OPENBLAS_NUM_THREADS'] = '64'

    data = np.array(candidate_expressed.obs['mean_expr']).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components).fit(data)
    probas = gmm.predict_proba(data)

    means = gmm.means_.flatten()
    mk_component = np.argmax(means)

    proba_mk = probas[:, mk_component]

    candidate_expressed.obs["proba_target"] = proba_mk

    if plot_extracted:
        plot_path = plot_folder + f"/{study_name}_extracted.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        print("Plot UMAP...")
            
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # UMAP plots
        sc.pl.umap(
            candidate_expressed,
            color="mean_expr",
            cmap="viridis",
            size=50,
            ax=ax[0],
            show=False,
            title="UMAP colored by mean expression of candidate genes among candidate cells"
        )
        sc.pl.umap(
            candidate_expressed,
            color="proba_target",
            size=50,
            ax=ax[1],
            show=False,
            title=f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {len(gmm.means_)} components)"
        )
        ax[1].legend(title="is target ?")

        print(f"Save UMAP plot to {plot_path}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Retourner les candidats avec la colonne is_target
    return candidate_expressed

if __name__ == "__main__":
    args = parse_arguments()
    adata_qc = sc.read_h5ad(args.h5ad_file)
    if not adata_qc.var_names.is_unique:
        adata_qc = adata_qc[:, ~adata_qc.var_names.duplicated(keep='first')]
    print("\n===============================")
    print("Extracting target cells...")
    adata_target = extract_target(
        adata=adata_qc,
        study_name=args.study_name,
        candidate_genes=args.candidate_genes,
        assign_genes=args.assign_genes,
        min_genes_detected=args.min_genes_detected,
        gene_detection_threshold=args.gene_detection_threshold,
        n_components=args.n_components,
        plot_extracted=args.plot_extracted,
        plot_folder=args.plot_folder,
        species=args.species
    )
    print("-------------------------------")
    print(f"Saving in {args.output_file}")
    adata_target.write(args.output_file)
