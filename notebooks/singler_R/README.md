# SingleR reference cache

`sigma_singler_dominguez_comparison.ipynb` expects a local file
`monaco_immune.rds` in this directory. It is not vendored here because it must
be downloaded from `celldex`'s remote store (`gypsum.artifactdb.com`), and this
machine's network intercepts outbound HTTPS with a self-signed certificate,
which blocks `celldex::fetchReference()`.

To produce the file, on a machine with working internet access to
`gypsum.artifactdb.com` (e.g. your laptop, or Positron with a normal network):

```r
# install.packages("BiocManager")
# BiocManager::install(c("celldex", "SingleR"))
ref <- celldex::MonacoImmuneData()
saveRDS(ref, "monaco_immune.rds")
```

Then copy `monaco_immune.rds` into this directory
(`SIGMA/notebooks/singler_R/monaco_immune.rds`).
