import os
import re
import sys
import shutup; shutup.please()

sys.path.append("scripts")
from read_config import read_config

# Load configuration
CONFIG = read_config(config.get("file", ""))

# Extract studies from input folder
STUDIES_FILES = [
    f.name for f in os.scandir(CONFIG["Folder"]["input_folder"])
    if re.search(r"\.(h5ad|csv|rds)$", f.name)
]
STUDIES_NAMES = [".".join(f.split(".")[:-1]) for f in STUDIES_FILES]

# Resources
THREADS = int(config.get("threads", 4))
MEM_MB = int(config.get("mem_mb", 16000))

# Rules
rule all:
    input:
        CONFIG["Folder"]["output_folder"] + "/merged.h5ad",
        expand(os.path.join(CONFIG["Folder"]["output_folder"], "logs/{study}.log"), study=STUDIES_NAMES)
    threads: THREADS

rule merge_studies:
    input:
        CONFIG["Folder"]["output_folder"] + "/harmonized"
    output:
        CONFIG["Folder"]["output_folder"] + "/merged.h5ad"
    threads: THREADS
    resources:
        mem_mb=MEM_MB
    shell:
        """
        python scripts/merge_h5ad.py \
            --study_folder "{input[0]}" \
            --output_file "{output}"
        """

rule harmonize_metadata:
    input:
        expand(
            os.path.join(CONFIG["Folder"]["output_folder"], "extracted/{study}.h5ad"),
            study=STUDIES_NAMES
        )
    output:
        directory(CONFIG["Folder"]["output_folder"] + "/harmonized")
    threads: THREADS
    resources:
        mem_mb=MEM_MB
    params: 
        input_folder=CONFIG["Folder"]["output_folder"] + "/extracted",
        columns_list=','.join(CONFIG["Metadata"])
    shell:
        """
        streamlit run scripts/harmonize_metada.py -- \
            --input_folder {params.input_folder} \
            --columns_list {params.columns_list} \
            --outdir {output}
        """

rule extract_target:
    input:
        os.path.join(CONFIG["Folder"]["output_folder"], "qc/{study}.h5ad")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "extracted/{study}.h5ad")
    threads: THREADS
    resources:
        mem_mb=MEM_MB,
        mem_mib=MEM_MB
    params:
        candidate_genes=CONFIG['Candidate'],
        assign_genes=CONFIG['Markers'],
        min_genes_detected=float(CONFIG['Thresholds']['min_genes_detected']),
        gene_detection_threshold=float(CONFIG['Thresholds']['gene_detection_threshold']),
        assign_threshold=CONFIG['Thresholds']['assign_threshold'],
        species=CONFIG['Options']['species'],
        plot_extracted=CONFIG['Options']['plot_extracted'],
        plot_folder=os.path.join(CONFIG["Folder"]["output_folder"], "plots")
    log:
        stderr=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stderr"),
        stdout=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout")
    shell:
        """
        python scripts/extract_target.py \
            --h5ad_file "{input}" \
            --output_file "{output}" \
            --study_name {wildcards.study} \
            --candidate_genes {params.candidate_genes} \
            --assign_genes {params.assign_genes} \
            --min_genes_detected {params.min_genes_detected} \
            --gene_detection_threshold {params.gene_detection_threshold} \
            --assign_threshold {params.assign_threshold} \
            --species {params.species} \
            --plot_extracted {params.plot_extracted} \
            --plot_folder {params.plot_folder} >> "{log.stdout}" 2>> "{log.stderr}"
        """

rule quality_control:
    input:
        os.path.join(CONFIG["Folder"]["input_folder"], "{study}.h5ad")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "qc/{study}.h5ad")
    threads: THREADS
    resources:
        mem_mb=MEM_MB,
        mem_mib=MEM_MB
    params:
        percent_top=CONFIG['Options']['percent_top'],
        nmads=CONFIG['Options']['nmads'],
        species=CONFIG['Options']['species']
    log:
        stderr=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stderr"),
        stdout=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout")
    shell:
        """
        OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} \
        MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} \
        python scripts/quality_control.py \
            --h5ad_file "{input}" \
            --output_file "{output}" \
            --percent_top {params.percent_top} \
            --nmads {params.nmads} \
            --species {params.species} >> "{log.stdout}" 2>> "{log.stderr}"
        """

rule merge_logs:
    input:
        merged_file=CONFIG["Folder"]["output_folder"] + "/merged.h5ad",
        i1=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout"),
        i2=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout"),
        i3=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stderr"),
        i4=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stderr")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "logs/{study}.log")
    threads: 1
    shell:
        """
        cat {input.i1} {input.i2} > {output} && rm {input.i1} {input.i2} {input.i3} {input.i4}
        """