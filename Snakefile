import os
import re
import sys
import json

sys.path.append("scripts")
from read_config import read_config

# Load configuration
CONFIG = read_config(config.get("file", ""))

# Extract studies from input folder
def get_studies(input_folder):
    return [
        f.name for f in os.scandir(input_folder)
        if re.search(r"\.h5ad$", f.name)
    ]

STUDIES_FILES = get_studies(CONFIG["Folder"]["input_folder"])
STUDIES_NAMES = [".".join(f.split(".")[:-1]) for f in STUDIES_FILES]

# Default resources
DEFAULT_THREADS = int(config.get("threads", 1))
DEFAULT_MEM_MB = int(config.get("mem_mb", 32000))

if CONFIG['Options']['do_QC'] == 'True':
    qc_folder = os.path.join(CONFIG["Folder"]["output_folder"], "qc")
else:
    qc_folder = CONFIG["Folder"]["input_folder"]
    for study in STUDIES_FILES:
        stdout_log = os.path.join(CONFIG["Folder"]["output_folder"], "logs", "std", f"QC_{study}.stdout")
        stderr_log = os.path.join(CONFIG["Folder"]["output_folder"], "logs", "std", f"QC_{study}.stderr")
        os.makedirs(os.path.dirname(stdout_log), exist_ok=True)
        with open(stdout_log, 'w') as out_f:
            out_f.write("Quality control is disabled. Using input file directly.")
        with open(stderr_log, 'w') as err_f:
            err_f.write("")


# ---------------- Rules ----------------

rule all:
    input:
        CONFIG["Folder"]["output_folder"] + "/merge.h5ad",
        expand(os.path.join(CONFIG["Folder"]["output_folder"], "logs/{study}.log"), study=STUDIES_NAMES)

rule merge_studies:
    input:
        CONFIG["Folder"]["output_folder"] + "/harmonized"
    output:
        CONFIG["Folder"]["output_folder"] + "/merge.h5ad"
    resources:
        mem_mb=DEFAULT_MEM_MB
    shell:
        "python scripts/merge_h5ad.py --study_folder '{input[0]}' --output_file '{output}'"

rule harmonize_metadata:
    input:
        expand(
            os.path.join(CONFIG["Folder"]["output_folder"], "find/{study}.h5ad"),
            study=STUDIES_NAMES
        )
    output:
        directory(CONFIG["Folder"]["output_folder"] + "/harmonized")
    resources:
        mem_mb=DEFAULT_MEM_MB
    params: 
        input_folder=CONFIG["Folder"]["output_folder"] + "/find",
        columns_list=','.join(CONFIG["Metadata"])
    shell:
        "streamlit run scripts/harmonize_metada.py -- "
        "--input_folder {params.input_folder} "
        "--columns_list {params.columns_list} "
        "--outdir {output}"

rule find_target:
    input:
        os.path.join(qc_folder, "{study}.h5ad")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "find/{study}.h5ad")
    threads: DEFAULT_THREADS
    resources:
        mem_mb=DEFAULT_MEM_MB
    params:
        candidate_genes=CONFIG['Candidate'],
        target_genes=CONFIG['Target'],
        exclude_genes=json.dumps(CONFIG['Exclude']),
        min_genes_detected=float(CONFIG['Thresholds']['min_genes_detected']),
        gene_detection_threshold=float(CONFIG['Thresholds']['gene_detection_threshold']),
        n_components_target=CONFIG['Thresholds']['n_components_target'],
        n_components_exclu=CONFIG['Thresholds']['n_components_exclu'],
        min_mean_expression=float(CONFIG['Thresholds']['min_mean_expression']),
        species=CONFIG['Options']['species'],
        do_QC=CONFIG['Options']['do_QC'],
        plot_folder=os.path.join(CONFIG["Folder"]["output_folder"], "plots"),
        exclude_celltypes=CONFIG['Options']['exclude_celltypes']
    log:
        stderr=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stderr"),
        stdout=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout")
    shell:
        "python scripts/find_target.py "
        "--h5ad_file '{input}' "
        "--output_file '{output}' "
        "--study_name {wildcards.study} "
        "--candidate_genes {params.candidate_genes} "
        "--target_genes {params.target_genes} "
        "--exclude_genes '{params.exclude_genes}' "
        "--min_genes_detected {params.min_genes_detected} "
        "--gene_detection_threshold {params.gene_detection_threshold} "
        "--n_components_target {params.n_components_target} "
        "--n_components_exclu {params.n_components_exclu} "
        "--min_mean_expression {params.min_mean_expression} "
        "--do_QC {params.do_QC} "
        "--species {params.species} "
        "--plot_folder {params.plot_folder} "
        "--exclude_celltypes {params.exclude_celltypes} >> '{log.stdout}' 2>> '{log.stderr}'"

rule quality_control:
    input:
        os.path.join(CONFIG["Folder"]["input_folder"], "{study}.h5ad")
    output:
        os.path.join(qc_folder, "{study}.h5ad")
    threads: DEFAULT_THREADS
    resources:
        mem_mb=DEFAULT_MEM_MB
    params:
        percent_top=CONFIG['Options']['percent_top'],
        nmads=CONFIG['Options']['nmads'],
        species=CONFIG['Options']['species'],
        do_QC=CONFIG['Options']['do_QC'],
        threads=DEFAULT_THREADS
    log:
        stderr=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stderr"),
        stdout=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout")
    shell:
        "python scripts/quality_control.py "
        "--h5ad_file '{input}' "
        "--output_file '{output}' "
        "--percent_top {params.percent_top} "
        "--nmads {params.nmads} "
        "--do_QC {params.do_QC} "
        "--species {params.species} "
        "--threads '{threads}' >> '{log.stdout}' 2>> '{log.stderr}'"

rule merge_logs:
    input:
        merged_file=CONFIG["Folder"]["output_folder"] + "/merge.h5ad",
        i1=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout"),
        i2=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "logs/{study}.log")
    threads: 1
    shell:
        "cat {input.i1} {input.i2} > {output}"

rule create_qc_std_logs:
    input:
        i1=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout"),
        i2=os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout")
    output:
        os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/QC_{study}.stdout"),
        os.path.join(CONFIG["Folder"]["output_folder"], "logs/std/EXTRACT_{study}.stdout")
    shell:
        "cat '' > {input.i1} && "
        "cat '' > {input.i2}"
