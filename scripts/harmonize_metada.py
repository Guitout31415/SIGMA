import streamlit as st
import shutup; shutup.please()
import scanpy as sc
import pandas as pd
import difflib
import os
import argparse
import time

# Adding command line arguments management
parser = argparse.ArgumentParser(description='AnnData metadata harmonization (multi-files)')
parser.add_argument('--input_folder', type=str, help='Path to folder containing h5ad files')
parser.add_argument('--outdir', type=str, help='Path to output folder')
parser.add_argument('--columns_list', type=str, help='List of standard columns to harmonize')
args = parser.parse_args()

columns_list = args.columns_list.split(',') if args.columns_list else None
outdir = args.outdir

st.title("AnnData Metadata Harmonization (multi-files)")

# Using argument to load files
if args.input_folder:
    uploaded_files = [open(os.path.join(args.input_folder, f), 'rb') for f in os.listdir(args.input_folder) if f.endswith('.h5ad')]
else:
    uploaded_files = st.file_uploader("Load one or more .h5ad files", type=["h5ad"], accept_multiple_files=True)

def harmonize_interface(data_file, columns_list=None, idx=None):
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    obs = adata.obs.copy()

    # 2. Definition of standard names
    st.markdown("**Standard names to harmonize** (editable)")
    if columns_list:
        default_standards = columns_list
    else:
        default_standards = ["age", "sex", "tissue"]
    base_name = os.path.basename(data_file.name) if not isinstance(data_file, str) else os.path.basename(data_file)
    standards = st.data_editor(pd.DataFrame({"Standard name": default_standards}), num_rows="dynamic", key=f"standards_{base_name}_{idx}")
    standards_list = standards["Standard name"].dropna().tolist()

    # 3. Mapping: assignment of existing columns
    mapping = {}
    st.markdown("### Column mapping")
    for std in standards_list:
        suggestion = difflib.get_close_matches(std, obs.columns, n=1, cutoff=0.6)
        options = ["(none)"] + list(obs.columns)
        
        # Calculate default index based on suggestion
        if suggestion:
            suggested_col = suggestion[0]
            default_index = options.index(suggested_col) if suggested_col in options else 0
        else:
            default_index = 0
            
        col = st.selectbox(f"{std}", options, index=default_index, key=f"{std}_{data_file}_{idx}")
        mapping[std] = col if col != "(none)" else None

    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # 5bis. Save harmonized .h5ad
    base_name = os.path.splitext(os.path.basename(data_file.name))[0] if not isinstance(data_file, str) else os.path.splitext(os.path.basename(data_file))[0]
    default_h5ad = os.path.join(outdir, f"{base_name}.h5ad")

    export_name = st.text_input("Output file name (.h5ad)", value=default_h5ad, key=f"export_name_{data_file}_{idx}")
    export_path = os.path.join(outdir, export_name)

    if st.button("Export harmonized .h5ad", key=f"export_{data_file}_{idx}"):
        selected = {std: col for std, col in mapping.items() if col is not None}
        if not selected:
            st.error("No column selected for export.")
        else:
            selected["is_target"] = "is_target"
            new_obs = obs[[col for col in selected.values()]].copy()
            new_obs.columns = list(selected.keys())
            new_adata = adata.copy()
            new_adata.obs = new_obs
            new_adata.write(export_path)
            st.success(f"File {export_path} saved successfully.")

    # 6. Preview of obs table (full width)
    st.markdown("### Preview of adata.obs")
    st.dataframe(obs.head(10).reset_index(drop=True), use_container_width=True, width=800)

if uploaded_files:
    tabs = st.tabs([os.path.basename(f.name) for f in uploaded_files])
    for idx, (tab, file) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            harmonize_interface(file, columns_list, idx)
else:
    st.info("Please load one or more .h5ad files to start.")

# Add a button to close the entire interface
if st.button("Close interface"): 
    os._exit(0)