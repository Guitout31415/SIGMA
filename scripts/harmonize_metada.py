"""
harmonize_metada.py
-------------------
Streamlit application for AnnData metadata harmonization across multiple files.

Provides an interactive interface for mapping column names to standard names
and exporting harmonized .h5ad files.
"""

import os
import argparse
import difflib

import streamlit as st
import scanpy as sc
import pandas as pd

# --- Constants ---
DEFAULT_STANDARDS = ["age", "sex", "tissue"]


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AnnData metadata harmonization (multi-files)"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to folder containing h5ad files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path to output folder",
    )
    parser.add_argument(
        "--columns_list",
        type=str,
        help="Comma-separated list of standard columns to harmonize",
    )
    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def get_uploaded_files(input_folder: str) -> list:
    """Get list of h5ad files from input folder or file uploader.

    Args:
        input_folder: Path to folder containing h5ad files

    Returns:
        List of file objects
    """
    if input_folder:
        return [
            open(os.path.join(input_folder, f), "rb")
            for f in os.listdir(input_folder)
            if f.endswith(".h5ad")
        ]
    return st.file_uploader(
        "Load one or more .h5ad files",
        type=["h5ad"],
        accept_multiple_files=True,
    )


def get_column_suggestion(std_name: str, columns: list) -> int:
    """Get suggested column index for a standard name.

    Args:
        std_name: Standard column name to match
        columns: List of available columns

    Returns:
        Index of best match in options list (0 if none found)
    """
    options = ["(none)"] + list(columns)
    suggestion = difflib.get_close_matches(std_name, columns, n=1, cutoff=0.5)

    if suggestion:
        suggested_col = suggestion[0]
        return options.index(suggested_col) if suggested_col in options else 0
    return 0


def get_probability_columns(obs: pd.DataFrame) -> tuple:
    """Extract probability and mean expression columns.

    Args:
        obs: DataFrame of observations

    Returns:
        Tuple of (proba_cols, mean_cols)
    """
    proba_cols = [col for col in obs.columns if col.startswith("proba_")]
    mean_cols = [col for col in obs.columns if col.startswith("exclude_mean_expr")] + ["target_mean_expr"]

    return proba_cols, mean_cols


def build_export_mapping(
    mapping: dict,
    obs: pd.DataFrame,
) -> dict:
    """Build the final column mapping for export.

    Args:
        mapping: User-defined column mappings
        obs: DataFrame of observations

    Returns:
        Dictionary of columns to export
    """
    selected = {std: col for std, col in mapping.items() if col is not None}

    if not selected:
        return {}

    proba_cols, mean_cols = get_probability_columns(obs)

    # Add score column if multiple probability columns exist
    if len(proba_cols) >= 2:
        selected["score"] = "score"

    # Add all probability and mean expression columns
    for col in proba_cols + mean_cols:
        selected[col] = col

    return selected


# =============================================================================
# Harmonization Interface
# =============================================================================


def harmonize_interface(
    data_file,
    columns_list: list,
    idx: int,
    outdir: str,
) -> None:
    """Main harmonization interface for a single file.

    Args:
        data_file: File object for the h5ad file
        columns_list: List of standard column names
        idx: Index of current file (for unique widget keys)
        outdir: Output directory path
    """
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    obs = adata.obs.copy()

    os.makedirs(outdir, exist_ok=True)

    base_name = os.path.basename(data_file.name)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Handle empty files
    if adata.n_obs == 0:
        st.info("This file is empty")
        export_path = os.path.join(outdir, f"{base_name_no_ext}.h5ad")
        adata.write(export_path)
        st.success(f"File {export_path} saved successfully.")
        st.session_state.exported_files.add(base_name)
        return

    # Standard names editor
    st.markdown("**Standard names to harmonize** (editable)")
    default_standards = columns_list if columns_list else DEFAULT_STANDARDS
    standards = st.data_editor(
        pd.DataFrame({"Standard name": default_standards}),
        num_rows="dynamic",
        key=f"standards_{base_name}_{idx}",
    )
    standards_list = standards["Standard name"].dropna().tolist()

    # Column mapping
    st.markdown("### Column mapping")
    mapping = {}
    options = ["(none)"] + list(obs.columns)

    for std in standards_list:
        default_index = get_column_suggestion(std, obs.columns)
        col = st.selectbox(
            f"{std}",
            options,
            index=default_index,
            key=f"{std}_{data_file}_{idx}",
        )
        mapping[std] = col if col != "(none)" else None

    # Export section
    default_h5ad = os.path.join(outdir, f"{base_name_no_ext}.h5ad")
    export_name = st.text_input(
        "Output file name (.h5ad)",
        value=default_h5ad,
        key=f"export_name_{data_file}_{idx}",
    )
    export_path = os.path.join(outdir, export_name)

    if st.button("Export harmonized .h5ad", key=f"export_{data_file}_{idx}"):
        _export_harmonized_file(adata, obs, mapping, export_path, base_name)

    # Preview
    st.markdown("### Preview of adata.obs")
    st.dataframe(
        obs.head(10).reset_index(drop=True),
        use_container_width=True,
        width=800,
    )


def _export_harmonized_file(
    adata: sc.AnnData,
    obs: pd.DataFrame,
    mapping: dict,
    export_path: str,
    base_name: str,
) -> None:
    """Export the harmonized file.

    Args:
        adata: AnnData object
        obs: Original observations DataFrame
        mapping: Column mapping dictionary
        export_path: Output file path
        base_name: Original file name (for session state)
    """
    selected = build_export_mapping(mapping, obs)

    if not selected:
        new_adata = adata.copy()
    else:
        new_obs = obs[[col for col in selected.values()]].copy()
        new_obs.columns = list(selected.keys())
        new_adata = adata.copy()
        new_adata.obs = new_obs

    new_adata.write(export_path)
    st.success(f"File {export_path} saved successfully.")
    st.session_state.exported_files.add(base_name)


# =============================================================================
# Navigation UI
# =============================================================================


def render_navigation(uploaded_files: list) -> int:
    """Render navigation controls and return active file index.

    Args:
        uploaded_files: List of uploaded file objects

    Returns:
        Index of currently active file
    """
    tab_names = [os.path.basename(f.name) for f in uploaded_files]
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("◀", key="prev_tab"):
            st.session_state.active_tab_idx = (
                st.session_state.active_tab_idx - 1
            ) % len(uploaded_files)

    with col3:
        if st.button("▶", key="next_tab"):
            st.session_state.active_tab_idx = (
                st.session_state.active_tab_idx + 1
            ) % len(uploaded_files)

    active = st.session_state.active_tab_idx

    with col2:
        _render_tab_carousel(tab_names, active)

    return active


def _render_tab_carousel(tab_names: list, active: int) -> None:
    """Render the tab carousel display.

    Args:
        tab_names: List of tab names
        active: Index of active tab
    """
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
            is_active = idx == active
            is_exported = name in st.session_state.exported_files

            if is_active and is_exported:
                cols[j].markdown(
                    f"<span style='color:green; font-weight:bold'>{name}</span>",
                    unsafe_allow_html=True,
                )
            elif is_active:
                cols[j].markdown(f"**{name}**")
            elif is_exported:
                cols[j].markdown(
                    f"<span style='color:green'>{name}</span>",
                    unsafe_allow_html=True,
                )
            else:
                cols[j].markdown(name)
            current_j += 1
        else:
            cols[j].markdown("")


# =============================================================================
# Main Application
# =============================================================================


def main() -> None:
    """Main Streamlit application entry point."""
    args = parse_args()
    columns_list = args.columns_list.split(",") if args.columns_list else None
    outdir = args.outdir

    st.title("AnnData Metadata Harmonization (multi-files)")

    uploaded_files = get_uploaded_files(args.input_folder)

    if uploaded_files:
        # Initialize session state
        if "active_tab_idx" not in st.session_state:
            st.session_state.active_tab_idx = 0
        if "exported_files" not in st.session_state:
            st.session_state.exported_files = set()

        # Filter out files already present in output directory (once at startup)
        if "initial_filter_done" not in st.session_state:
            st.session_state.initial_filter_done = True
            if outdir and os.path.isdir(outdir):
                existing_outputs = set(os.listdir(outdir))
                already_done = [
                    f for f in uploaded_files
                    if os.path.basename(f.name) in existing_outputs
                ]
                uploaded_files = [
                    f for f in uploaded_files
                    if os.path.basename(f.name) not in existing_outputs
                ]
                st.session_state.skipped_files = {
                    os.path.basename(f.name) for f in already_done
                }
                if already_done:
                    st.info(
                        f"{len(already_done)} file(s) already in output "
                        f"directory, skipped: "
                        f"{', '.join(os.path.basename(f.name) for f in already_done)}"
                    )
            else:
                st.session_state.skipped_files = set()
        else:
            skipped = st.session_state.skipped_files
            if skipped:
                uploaded_files = [
                    f for f in uploaded_files
                    if os.path.basename(f.name) not in skipped
                ]

        if uploaded_files:
            active = render_navigation(uploaded_files)
            harmonize_interface(uploaded_files[active], columns_list, active, outdir)
        else:
            st.success("All files have already been harmonized.")
    else:
        st.info("Please load one or more .h5ad files to start.")

    # Close button
    if st.button("Close interface"):
        os._exit(0)


if __name__ == "__main__":
    main()