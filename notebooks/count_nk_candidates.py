"""Author labels of the candidate cells in the NK-absent runs (reviewer #2, major concern).

Reads the find_target.py outputs written by figure_S3_nk_absent.ipynb and counts the
author labels of every candidate cell, to check the wording "mostly effector memory CD8
and γδ T cells" used for the 117 candidates of no_NK_t1.

Put this script in the notebook folder (the one holding results/nk_absent), or pass the
results folder as argument:
    python count_nk_candidates.py [path/to/results/nk_absent]
"""
import os
import re
import sys

import pandas as pd
import scanpy as sc

INPUT = ("/mnt/projects_tn03/AD_EB_singlecell/articleSIGMA_results-2/Dominguez-Data/forSIGMA/"
         "Dominguez_science_science-abl5197_BLD_data.h5ad")
# Default: results/nk_absent next to this script (the notebook folder), whatever the cwd
WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "nk_absent")
RUNS = ["no_NK_t1", "no_NK", "all_NK_t1", "all_NK"]
# Same label column choice as the notebook
LABEL_COLUMNS = ["Manually_curated_celltype", "cell_type", "celltype", "Majority_voting_CellTypist"]
SCORE_THRESHOLD = 0.95

# Broad groups for the manuscript wording; labels not matched fall into "Other"
GROUPS = [
    ("Doublets", r"doublets"),
    ("Effector memory CD8 T", r"^Tem/emra_CD8|^Trm/em_CD8"),
    ("γδ T", r"^Tgd"),
    ("Other T", r"^T|MAIT"),
    ("NK", r"^NK"),
    ("ILC", r"^ILC"),
]


def group_of(label):
    for name, pattern in GROUPS:
        if re.search(pattern, label):
            return name
    return "Other"


# Backed mode: only obs is needed
obs_in = sc.read_h5ad(INPUT, backed="r").obs
label_col = next(c for c in LABEL_COLUMNS if c in obs_in.columns)
labels = obs_in[label_col].astype(str)
print(f"Label column: {label_col}")

tables = []
for name in RUNS:
    path = os.path.join(WORK_DIR, f"{name}.h5ad")
    if not os.path.exists(path):
        print(f"\n{name}: {path} not found, skipped")
        continue
    obs = sc.read_h5ad(path, backed="r").obs
    lab = labels.reindex(obs.index)
    if lab.isna().any():
        print(f"{name}: {lab.isna().sum()} candidates not found in the input labels")
    lab = lab.fillna("unmatched")
    sel = obs["score"] >= SCORE_THRESHOLD if "score" in obs else pd.Series(False, index=obs.index)

    counts = pd.DataFrame({"candidates": lab.value_counts(),
                           "selected": lab[sel].value_counts()}).fillna(0).astype(int)
    counts["pct_candidates"] = (100 * counts["candidates"] / max(len(obs), 1)).round(1)
    counts = counts.sort_values("candidates", ascending=False)
    groups = counts.groupby(counts.index.map(group_of))[["candidates", "selected"]].sum()
    groups["pct_candidates"] = (100 * groups["candidates"] / max(len(obs), 1)).round(1)
    groups = groups.sort_values("candidates", ascending=False)

    print(f"\n=== {name}: {len(obs):,} candidates, {int(sel.sum()):,} selected (score >= {SCORE_THRESHOLD})")
    print(counts.to_string())
    print("\nBy group:")
    print(groups.to_string())
    tables.append(counts.assign(run=name))

if tables:
    out = os.path.join(WORK_DIR, "nk_absent_candidate_labels.csv")
    pd.concat(tables).rename_axis("label").reset_index().to_csv(out, index=False)
    print(f"\nWritten: {out}")
