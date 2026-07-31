# Vendored ScType R functions

`gene_sets_prepare.R` and `sctype_score_.R` are copied verbatim from
https://github.com/IanevskiAleksandr/sc-type (GNU GPL v3.0), commit at the
time of vendoring (`master` branch). They are sourced directly by
`sigma_sctype_dominguez_comparison.ipynb` rather than fetched from GitHub at
run time, since ScType has no Bioconductor/CRAN package.

Only `sctype_score_.R` (the `sctype_score()` function) is used by the SIGMA
comparison notebook; `gene_sets_prepare.R` is kept for reference but not
called, since we build the gene set list directly from the SIGMA target /
exclusion gene lists instead of the standard ScType xlsx marker database.
