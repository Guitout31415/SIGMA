import os

workdir: "/mnt/projects_tn03/AD_EB_singlecell/extract_MK"

STUDIES = [f.name for f in os.scandir("data") if f.is_file()]
THREADS_MERGE = 40 # default 40

rule all:
    input:
        "merged.h5ad"
    threads: THREADS_MERGE

rule merge_studies:
    input:
        expand("data/mk/{study}.h5ad", study=STUDIES)
    output:
        merged.h5ad
