import streamlit as st
import scanpy as sc
import pandas as pd
import difflib
import os
import argparse

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

    base_name = os.path.basename(data_file.name)

    # 2. Definition of standard names
    st.markdown("**Standard names to harmonize** (editable)")
    if columns_list:
        default_standards = columns_list
    else:
        default_standards = ["age", "sex", "tissue"]
    standards = st.data_editor(pd.DataFrame({"Standard name": default_standards}), num_rows="dynamic", key=f"standards_{base_name}_{idx}")
    standards_list = standards["Standard name"].dropna().tolist()

    # 3. Mapping: assignment of existing columns
    mapping = {}
    st.markdown("### Column mapping")
    for std in standards_list:
        suggestion = difflib.get_close_matches(std, obs.columns, n=1, cutoff=0.8)
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
    base_name_no_ext = os.path.splitext(base_name)[0]
    default_h5ad = os.path.join(outdir, f"{base_name_no_ext}.h5ad")

    export_name = st.text_input("Output file name (.h5ad)", value=default_h5ad, key=f"export_name_{data_file}_{idx}")
    export_path = os.path.join(outdir, export_name)

    if st.button("Export harmonized .h5ad", key=f"export_{data_file}_{idx}"):
        selected = {std: col for std, col in mapping.items() if col is not None}
        if not selected:
            new_adata = adata.copy()
            new_adata.write(export_path)
            st.success(f"File {export_path} saved successfully.")
            st.session_state.exported_files.add(base_name)
        else:
            proba_cols = [col for col in obs.columns if col.startswith("proba_")]
            mean_cols = [col for col in obs.columns if col.endswith("_mean_expr")]
            if len(proba_cols) >= 2:
                selected["score"] = "score"
            for col in proba_cols:
                selected[col] = col
            for col in mean_cols:
                selected[col] = col
            new_obs = obs[[col for col in selected.values()]].copy()
            new_obs.columns = list(selected.keys())
            new_adata = adata.copy()
            new_adata.obs = new_obs
            new_adata.write(export_path)
            st.success(f"File {export_path} saved successfully.")
            st.session_state.exported_files.add(base_name)

    # 6. Preview of obs table (full width)
    st.markdown("### Preview of adata.obs")
    st.dataframe(obs.head(10).reset_index(drop=True), use_container_width=True, width=800)

if uploaded_files:
    if 'active_tab_idx' not in st.session_state:
        st.session_state.active_tab_idx = 0
    if 'exported_files' not in st.session_state:
        st.session_state.exported_files = set()

    tab_names = [os.path.basename(f.name) for f in uploaded_files]

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("◀", key="prev_tab"):
            st.session_state.active_tab_idx = (st.session_state.active_tab_idx - 1) % len(uploaded_files)

    with col3:
        if st.button("▶", key="next_tab"):
            st.session_state.active_tab_idx = (st.session_state.active_tab_idx + 1) % len(uploaded_files)

    active = st.session_state.active_tab_idx

    with col2:
        # Carousel logic to center the active tab
        target_position = 2
        start = active - target_position
        padding_left = max(0, -start)
        start = max(0, start)
        end = min(len(tab_names), start + 5 - padding_left)
        cols = st.columns(5)
        current_j = 0
        for j in range(5):
            if j < padding_left:
                cols[j].markdown("")
            elif current_j < (end - start):
                idx = start + current_j
                name = tab_names[idx]
                is_active = (idx == active)
                is_exported = name in st.session_state.exported_files
                if is_active and is_exported:
                    cols[j].markdown(f"<span style='color:green; font-weight:bold'>{name}</span>", unsafe_allow_html=True)
                elif is_active:
                    cols[j].markdown(f"**{name}**")
                elif is_exported:
                    cols[j].markdown(f"<span style='color:green'>{name}</span>", unsafe_allow_html=True)
                else:
                    cols[j].markdown(name)
                current_j += 1
            else:
                cols[j].markdown("")

    idx = active
    file = uploaded_files[idx]
    harmonize_interface(file, columns_list, idx)
else:
    st.info("Please load one or more .h5ad files to start.")

# Add a button to close the entire interface
if st.button("Close interface"): 
    os._exit(0)