import streamlit as st
import shutup; shutup.please()
import scanpy as sc
import pandas as pd
import difflib
import os
import argparse
import time

# Ajout de la gestion des arguments de ligne de commande
parser = argparse.ArgumentParser(description='Harmonisation des métadonnées AnnData (multi-fichiers)')
parser.add_argument('--input_folder', type=str, help='Chemin vers le dossier contenant les fichiers h5ad')
parser.add_argument('--outdir', type=str, help='Chemin vers le dossier de sortie')
parser.add_argument('--columns_list', type=str, help='Liste des colonnes standards à harmoniser')
args = parser.parse_args()

columns_list = args.columns_list.split(',') if args.columns_list else None
outdir = args.outdir

st.title("Harmonisation des métadonnées AnnData (multi-fichiers)")

# Utilisation de l'argument pour charger les fichiers
if args.input_folder:
    uploaded_files = [open(os.path.join(args.input_folder, f), 'rb') for f in os.listdir(args.input_folder) if f.endswith('.h5ad')]
else:
    uploaded_files = st.file_uploader("Charger un ou plusieurs fichiers .h5ad", type=["h5ad"], accept_multiple_files=True)

def harmonize_interface(data_file, columns_list=None, idx=None):
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    obs = adata.obs.copy()

    # 2. Définition des noms standards
    st.markdown("**Noms standards à harmoniser** (modifiable)")
    if columns_list:
        default_standards = columns_list
    else:
        default_standards = ["age", "sex", "tissue"]
    base_name = os.path.basename(data_file.name) if not isinstance(data_file, str) else os.path.basename(data_file)
    standards = st.data_editor(pd.DataFrame({"Nom standard": default_standards}), num_rows="dynamic", key=f"standards_{base_name}_{idx}")
    standards_list = standards["Nom standard"].dropna().tolist()

    # 3. Mapping : assignation des colonnes existantes
    mapping = {}
    st.markdown("### Mapping des colonnes")
    for std in standards_list:
        suggestion = difflib.get_close_matches(std, obs.columns, n=1, cutoff=0.6)
        col = st.selectbox(f"{std}", ["(aucune)"] + list(obs.columns), index=1 if suggestion else 0, key=f"{std}_{data_file}_{idx}")
        mapping[std] = col if col != "(aucune)" else None

    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # 5bis. Sauvegarde du .h5ad harmonisé
    base_name = os.path.splitext(os.path.basename(data_file.name))[0] if not isinstance(data_file, str) else os.path.splitext(os.path.basename(data_file))[0]
    default_h5ad = os.path.join(outdir, f"{base_name}.h5ad")

    export_name = st.text_input("Nom du fichier de sortie (.h5ad)", value=default_h5ad, key=f"export_name_{data_file}_{idx}")
    export_path = os.path.join(outdir, export_name)

    if st.button("Exporter le .h5ad harmonisé", key=f"export_{data_file}_{idx}"):
        selected = {std: col for std, col in mapping.items() if col is not None}
        if not selected:
            st.error("Aucune colonne sélectionnée pour l'export.")
        else:
            new_obs = obs[[col for col in selected.values()]].copy()
            new_obs.columns = list(selected.keys())
            new_adata = adata.copy()
            new_adata.obs = new_obs
            new_adata.write(export_path)
            st.success(f"Fichier {export_path} sauvegardé avec succès.")

    # 6. Aperçu du tableau obs (pleine largeur)
    st.markdown("### Aperçu de adata.obs")
    st.dataframe(obs.head(10).reset_index(drop=True), use_container_width=True, width=800)

if uploaded_files:
    tabs = st.tabs([os.path.basename(f.name) for f in uploaded_files])
    for idx, (tab, file) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            harmonize_interface(file, columns_list, idx)
else:
    st.info("Veuillez charger un ou plusieurs fichiers .h5ad pour commencer.")

# Add a button to close the entire interface
if st.button("Fermer l'interface"):
    with open("stop_signal.txt", "w") as f:
        f.write("stop")
    os._exit(0)

# Periodically check for the stop signal file
while True:
    if os.path.exists("stop_signal.txt"):
        os.remove("stop_signal.txt")
        os._exit(0)
    time.sleep(1)