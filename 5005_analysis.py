#!/usr/bin/env python

"""
PHM5005 Final Project: Single-Cell Analysis of Pulmonary Fibrosis
(Version 3 - Enhanced with verbose logging, progress bars, and updated HVG plotting)

This script performs a full single-cell analysis workflow on the GSE122960 dataset.

The primary goal is to:
1.  Load and process 17 individual 10x Genomics datasets.
2.  Perform quality control, normalization, and clustering (Scanpy).
3.  Annotate major cell types based on canonical marker genes.
4.  Investigate macrophage populations, specifically scoring for "homeostatic"
    vs. "profibrotic" gene signatures.
5.  Aggregate cell-level metrics to the subject (patient) level.
6.  Build a predictive model (ElasticNet Logistic Regression) using subject-level
    features to classify subjects as 'Donor' or 'Fibrosis'.
7.  Generate and save all necessary plots and data outputs to an
    'analysis_results' directory.

To run this script:
1.  Ensure you have a Python environment with all required libraries installed
    (os, glob, numpy, pandas, scanpy, matplotlib, seaborn, sklearn, tqdm, tabulate).
2.  Download the GSE122960 data and place the .h5 files in a directory.
3.  Update the `DATA_DIR` variable below to point to that directory.
4.  Run the script from your terminal: python this_script_name.py
"""

import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

# Import tqdm for progress bars
from tqdm import tqdm

from sklearn.model_selection import (
    train_test_split,
    LeaveOneOut,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.utils import resample

# --- Global Settings ---
print("Setting global parameters and random seeds...")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# --- SUPPRESS WARNINGS ---
print("Suppressing common UserWarnings and FutureWarnings...")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# -------------------------

# Set Scanpy plotting parameters
sc.set_figure_params(dpi=100, facecolor="white", dpi_save=150)
sns.set_context("notebook")
# Set scanpy verbosity (0=errors, 1=warnings, 2=info, 3=hints)
# We set to 1 (warnings) to hide routine 'info' messages
sc.settings.verbosity = 1

# --- Helper Functions ---


def robust_min_max(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple:
    """Gets the 2nd and 98th percentile for robust color scaling."""
    return np.nanpercentile(a, lo), np.nanpercentile(a, hi)


def plot_mac_scores(
    adata_subset: sc.AnnData,
    score_keys: list,
    color_limits: dict,
    results_dir: str,
):
    """
    Plots UMAPs for macrophage scores (continuous z-score and top 10% discrete).
    Saves plots to the specified results directory.
    """
    print("Plotting macrophage signature scores...")

    # Plot continuous Z-scores with a symmetric colormap
    print("  -> Plotting continuous z-scores...")
    fig, axes = plt.subplots(
        ncols=len(score_keys), figsize=(5 * len(score_keys), 4.5)
    )
    if len(score_keys) == 1:
        axes = [axes]  # Make it iterable

    for i, k in enumerate(score_keys):
        vmin, vmax = color_limits[k]
        sc.pl.umap(
            adata_subset,
            color=k,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            colorbar_loc="right",
            size=5,
            frameon=False,
            title=f"{k.replace('_z', ' (z)')}",
            ax=axes[i],
            show=False,
        )

    plot_path = os.path.join(results_dir, "umap_myeloid_scores_continuous.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved continuous score UMAP to {plot_path}")

    # Plot top-decile highlight maps
    print("  -> Plotting top 10% highlight maps...")
    fig, axes = plt.subplots(
        ncols=len(score_keys), figsize=(5 * len(score_keys), 4.5)
    )
    if len(score_keys) == 1:
        axes = [axes]

    for i, k in enumerate(score_keys):
        tag = k.split("_")[1]  # e.g., 'homeo' or 'profib'
        zcol = k
        t90 = np.nanpercentile(adata_subset.obs[zcol], 90)
        adata_subset.obs[f"{tag}_top10"] = (adata_subset.obs[zcol] >= t90).astype(str)

        sc.pl.umap(
            adata_subset,
            color=f"{tag}_top10",
            palette=["lightgray", "crimson"], # 'False' (90%) = lightgray, 'True' (10%) = crimson
            size=5,
            frameon=False,
            title=f"Top 10% {tag} score",
            ax=axes[i],
            show=False,
        )

    plot_path = os.path.join(results_dir, "umap_myeloid_scores_top10.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved top 10% score UMAP to {plot_path}")


def plot_cv_roc(
    clf, X: pd.DataFrame, y: pd.Series, cv_splitter, title: str, save_path: str
):
    """
    Plots the Cross-Validated ROC curve for a given classifier.
    """
    print(f"Plotting ROC curve: {title}")
    # Get CV predictions
    if isinstance(cv_splitter, LeaveOneOut):
        # LeaveOneOut doesn't have a `predict_proba` method in cross_val_predict
        # We must iterate manually
        y_probas = np.zeros_like(y, dtype=float)
        for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_train, y_train)
            y_probas[test_idx] = clf.predict_proba(X_test)[:, 1]
    else:
        y_probas = cross_val_predict(
            clf, X, y, cv=cv_splitter, method="predict_proba"
        )[:, 1]

    # Calculate ROC
    fpr, tpr, _ = roc_curve(y, y_probas)
    auroc_cv = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"Cross-Validated AUROC = {auroc_cv:.3f}",
        color="darkorange",
        lw=2,
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUROC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC plot to {save_path}")


def plot_marginal_effect(
    clf, X: pd.DataFrame, y: pd.Series, feature: str, title: str, save_path: str
):
    """
    Plots the marginal effect of a single feature on the model's prediction.
    """
    print(f"Plotting marginal effect: {feature}")
    # Fit the model on all data to get the final relationship
    clf.fit(X, y)

    # Get probabilities
    y_probas = clf.predict_proba(X)[:, 1]

    # Plot
    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=X[feature],
        y=y_probas,
        logistic=True,
        ci=95,
        scatter_kws={"s": 60, "color": "black"},
        line_kws={"color": "red", "lw": 2},
    )
    # Overlay the actual data points (0s and 1s) with jitter
    sns.stripplot(
        x=X[feature],
        y=y,
        color="black",
        jitter=0.05,
        alpha=0.5,
        size=5,
        label="Actual Outcome (0=Donor, 1=Fibrosis)",
    )

    plt.ylabel("Predicted P(Fibrosis)")
    plt.xlabel(f"Feature: {feature}")
    plt.title(f"Marginal Effect Plot: {title}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved marginal effect plot to {save_path}")


def load_and_annotate_data(data_dir: str) -> list:
    """
    Loads all .h5 files from the data directory, annotates them with
    sample and disease info based on their filenames, and returns a list
    of AnnData objects.
    """
    print(f"Searching for data in: {data_dir}")
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*filtered_gene_bc_matrices_h5.h5")))
    if not h5_files:
        print(f"Error: No .h5 files found in {data_dir}.")
        print("Please download the data for GSE122960 and update the DATA_DIR variable.")
        return []
    
    print(f"Found {len(h5_files)} .h5 files. Starting read...")
    adatas = []

    disease_map = {
        "Donor": "Donor",
        "IPF": "IPF",
        "HP": "HP",
        "SSc-ILD": "SSc-ILD",
        "Myositis-ILD": "Myositis-ILD",
        "Cryobiopsy": "Cryobiopsy",
    }
    fibrotic_labels = {"IPF", "HP", "SSc-ILD", "Myositis-ILD", "Cryobiopsy"}

    # Use tqdm to create a progress bar for the loop
    for fp in tqdm(h5_files, desc="Loading .h5 files"):
        fname = os.path.basename(fp)
        # print(f"  [{i}/{len(h5_files)}] Reading {fname}") # tqdm replaces this

        try:
            ad = sc.read_10x_h5(fp)
            ad.var_names_make_unique()
        except Exception as e:
            print(f"\n    Skipping file {fname} due to read error: {e}")
            continue

        # Parse sample ID and disease from file name
        sample_id = fname.replace("_filtered_gene_bc_matrices_h5.h5", "")
        parts = sample_id.split("_")
        
        disease_raw = "Unknown"
        if len(parts) >= 2:
            disease_raw = parts[1]

        disease_status = disease_map.get(disease_raw, disease_raw)
        disease_binary = "Fibrosis" if disease_status in fibrotic_labels else "Donor"

        # Write annotations to .obs
        ad.obs["sample_id"] = sample_id
        ad.obs["disease_status"] = disease_status
        ad.obs["disease_binary"] = disease_binary
        
        # print(f"    Matrix loaded: {ad.n_obs} cells, {ad.n_vars} genes")
        # print(f"    Annotation: sample_id={sample_id}, status={disease_status}, group={disease_binary}")

        adatas.append(ad)

    print(f"\nDone. Loaded {len(adatas)} AnnData objects.")
    return adatas


# --- Main Analysis Workflow ---

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    
    # --- Parameters ---
    # !!! IMPORTANT: Update this path to your local data directory
    DATA_DIR = "./GSE122960_data"
    RESULTS_DIR = "./analysis_results"
    
    # --- Setup ---
    print("===============================================")
    print("--- Starting scRNA-seq Analysis Pipeline ---")
    print("===============================================")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # =========================================================================
    print("\n--- SECTION 1: Data Loading & QC ---")
    # =========================================================================
    
    # Load all individual .h5 files and annotate them
    adatas = load_and_annotate_data(DATA_DIR)
    if not adatas:
        return  # Stop if no data was loaded

    # Concatenate all samples into one AnnData object
    print("Concatenating all samples into one large AnnData object...")
    # We join on 'outer' to keep all genes, filling missing ones with 0
    # 'batch' key stores the original sample_id
    adata = ad.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        fill_value=0,
    )
    adata.obs_names_make_unique()
    
    print(f"\nFull dataset shape (before QC): {adata.n_obs} cells × {adata.n_vars} genes")
    print("Example metadata (random 5 cells):")
    print(adata.obs.sample(5, random_state=42)[["batch", "sample_id", "disease_status", "disease_binary"]])

    # --- Quality Control (QC) ---
    print("\nCalculating QC metrics...")
    # Calculate mitochondrial gene percentage
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Plot QC metrics before filtering
    print("Plotting pre-filter QC violin plots...")
    fig = sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        groupby="sample_id",
        rotation=90,
        multi_panel=True,
        show=False,
    )
    plot_path = os.path.join(RESULTS_DIR, "qc_violins_pre_filter.png")
    # --- FIX for list object ---
    # We grab the figure from the first axis in the returned list
    fig[0].figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig[0].figure)
    print(f"Saved QC violin plot to {plot_path}")
    
    # Apply filters
    print("\nApplying filters...")
    print(f"Cells before filtering: {adata.n_obs}")
    
    # Filter 1: Remove cells with < 200 genes
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"Cells after min_genes=200 filter: {adata.n_obs}")
    
    # Filter 2: Remove genes present in < 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"Genes after min_cells=3 filter: {adata.n_vars}")
    
    # Filter 3: Remove cells with high mitochondrial content (indicates stress/damage)
    adata = adata[adata.obs["pct_counts_mt"] < 15, :].copy()
    print(f"Cells after pct_counts_mt < 15 filter: {adata.n_obs}")
    print(f"\nFull dataset shape (after QC): {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Print summary stats after QC
    print("\nQC Metrics (Post-Filter):")
    # --- FIX for KeyError ---
    # Changed 'n_counts' to 'total_counts'
    print(adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe().to_markdown(floatfmt=".2f"))

    # =========================================================================
    print("\n--- SECTION 2: Preprocessing & Dimensionality Reduction ---")
    # =========================================================================
    
    # Save a copy of the raw (but filtered) counts before normalization
    adata.layers["counts"] = adata.X.copy()
    
    # Normalize total counts per cell to 10,000 (standard practice)
    print("Normalizing and log-transforming data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Log-transform the data
    sc.pp.log1p(adata)

    # Identify highly variable genes (HVGs)
    print(f"Finding 5,000 highly variable genes...")
    # --- UPDATED HVG LOGIC ---
    # We set subset=False to keep all genes for the plot
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False, flavor="seurat_v3")
    
    # Plot HVGs (now shows black and grey dots)
    print("Plotting highly variable genes (black) vs. other genes (grey)...")
    # --- FIX for 'ax' error ---
    # We let scanpy create the plot, then grab the figure
    sc.pl.highly_variable_genes(adata, show=False)
    fig = plt.gcf() # Get current figure
    plot_path = os.path.join(RESULTS_DIR, "hvg_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved HVG plot to {plot_path}")

    # --- NEW SUBSETTING STEP ---
    # Now, we manually filter the data to keep *only* the HVGs for downstream steps
    print(f"Subsetting data to the {adata.var.highly_variable.sum()} highly variable genes...")
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"Data shape after HVG subsetting: {adata.n_obs} cells × {adata.n_vars} genes")

    # Regress out confounding variables (total counts and mitochondrial percent)
    print("Regressing out confounders (total_counts, pct_counts_mt)...")
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])

    # Scale data to unit variance (z-score)
    print("Scaling data...")
    sc.pp.scale(adata, max_value=10)

    # --- Dimensionality Reduction ---
    print("Running PCA (Principal Component Analysis)...")
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=42)
    
    # Plot PCA elbow plot to decide on number of PCs
    print("Plotting PCA elbow plot...")
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False) # No ax=ax

    # Now, get the current figure and save it
    fig = plt.gcf() 
    plot_path = os.path.join(RESULTS_DIR, "pca_elbow_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PCA elbow plot to {plot_path}")
    
    # Based on the elbow plot, ~30 PCs seem appropriate
    N_PCS = 30
    print(f"Using {N_PCS} PCs for downstream analysis (neighborhood graph, UMAP).")

    # =========================================================================
    print("\n--- SECTION 3: Clustering & Cell Type Annotation ---")
    # =========================================================================

    # Compute neighborhood graph
    print("Computing neighborhood graph (calculating cell-cell distances)...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=N_PCS, random_state=42)
    
    # Compute UMAP
    print("Computing UMAP (Uniform Manifold Approximation and Projection)...")
    sc.tl.umap(adata, random_state=42)
    
    # Cluster cells using Leiden algorithm
    print("Clustering cells (Leiden algorithm, resolution=1.0)...")
    sc.tl.leiden(adata, resolution=1.0, random_state=42)
    print(f"Found {len(adata.obs.leiden.unique())} clusters.")

    # Plot UMAP by cluster and disease status
    print("Plotting UMAPs by cluster and disease status...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sc.pl.umap(adata, color="leiden", ax=ax1, show=False, legend_loc="on data")
    sc.pl.umap(adata, color="disease_binary", palette={"Donor": "blue", "Fibrosis": "red"}, ax=ax2, show=False)
    plot_path = os.path.join(RESULTS_DIR, "umap_cluster_disease.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP plots to {plot_path}")

    # --- Marker Gene Identification ---
    print("\nFinding marker genes for each cluster (Wilcoxon rank-sum test)...")
    # Find markers for each cluster vs. all other cells
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

    # Plot marker gene heatmap (top 5 genes per cluster)
    print("Plotting marker gene heatmap...")
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, show_gene_labels=True, show=False)
    fig = plt.gcf()
    # Adjust figure size for better readability
    fig.set_size_inches(12, 10)
    plot_path = os.path.join(RESULTS_DIR, "marker_heatmap.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marker heatmap to {plot_path}")

    # Plot marker gene dotplot
    print("Plotting marker gene dotplot...")
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    plot_path = os.path.join(RESULTS_DIR, "marker_dotplot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marker dotplot to {plot_path}")

    # --- Plot specific marker genes on UMAP ---
    # These plots help confirm cell type identity for each cluster
    print("\nPlotting specific marker genes on UMAPs for manual annotation...")
    marker_genes_to_plot = [
        # Epithelial
        "AGER", "PODOPLANIN", "SFTPC", "SCGB1A1", "KRT5", "KRT17",
        # Endothelial
        "CLDN5", "PECAM1", "VWF",
        # Fibroblasts
        "COL1A1", "LUM", "ACTA2",
        # Immune
        "PTPRC", # All immune
        # Myeloid
        "CD68", "LYZ", "MARCO", "MRC1", "SPP1", "TREM2",
        # T-cells
        "CD3D", "CD3E", "CD4", "CD8A",
        # B-cells
        "MS4A1", "CD79A",
        # Other
        "NKTR", "GNLY" # NK
    ]
    
    # Create a directory for gene plots
    gene_plot_dir = os.path.join(RESULTS_DIR, "umap_gene_plots")
    os.makedirs(gene_plot_dir, exist_ok=True)
    print(f"Saving individual gene plots to {gene_plot_dir}...")

    # Use tqdm for progress bar
    for gene in tqdm(marker_genes_to_plot, desc="Plotting marker genes"):
        if gene in adata.var_names:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc.pl.umap(
                adata,
                color=gene,
                ax=ax,
                show=False,
                cmap="viridis",
                title=gene,
                frameon=False,
                size=5,
            )
            plot_path = os.path.join(gene_plot_dir, f"umap_gene_{gene}.png")
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            # Use \r to overwrite the tqdm line temporarily
            print(f"\n  Skipping gene: {gene} (not in dataset's 5000 HVGs)")
    print(f"\nFinished saving individual gene plots.")

    # --- Manual Cell Type Annotation ---
    # Based on the marker genes plotted above, we assign cell types
    # This is an interpretive step based on the notebook's logic
    print("\nAnnotating cell types based on markers...")
    cell_type_map = {
        "0": "Macrophage",
        "1": "Macrophage",
        "2": "T-cell",
        "3": "Macrophage",
        "4": "Endothelial",
        "5": "Macrophage",
        "6": "Fibroblast",
        "7": "Epithelial (AT2)",
        "8": "Macrophage",
        "9": "Epithelial (AT1)",
        "10": "B-cell",
        "11": "T-cell",
        "12": "Epithelial (Ciliated)",
        "13": "Fibroblast",
        "14": "Endothelial",
        "15": "Macrophage",
        "16": "NK-cell",
        "17": "Epithelial (Club)",
        "18": "T-cell",
        "19": "Macrophage",
        "20": "Epithelial (Basal)",
        "21": "T-cell",
        "22": "Fibroblast",
        "23": "Endothelial",
        "24": "Plasma-cell",
        "25": "Mast-cell",
    }
    
    adata.obs["cell_type"] = adata.obs["leiden"].map(cell_type_map).astype("category")
    # Check for any unmapped clusters (if resolution was changed)
    if adata.obs["cell_type"].isnull().any():
        print("Warning: Some clusters were not in the mapping dictionary!")
        adata.obs["cell_type"] = adata.obs["cell_type"].cat.add_categories("Unmapped")
        adata.obs["cell_type"] = adata.obs["cell_type"].fillna("Unmapped")

    # Plot final annotated UMAP
    print("Plotting final annotated UMAP...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata,
        color="cell_type",
        ax=ax,
        show=False,
        legend_loc="on data",
        frameon=False,
        title="Annotated Cell Types",
    )
    plot_path = os.path.join(RESULTS_DIR, "umap_celltype_annotated.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved annotated UMAP to {plot_path}")

    # Plot cell type fractions
    print("Plotting cell type fractions...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=adata.obs, y="cell_type", ax=ax, order=adata.obs['cell_type'].value_counts().index)
    ax.set_title("Cell Counts by Type")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "celltype_fractions_barplot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cell type fractions plot to {plot_path}")

    # =========================================================================
    print("\n--- SECTION 4: Macrophage Sub-analysis ---")
    # =========================================================================
    
    # The goal is to compare "homeostatic" vs. "profibrotic" macrophages
    # We define gene sets for these states and score all cells.
    
    # Note: We must score on the *full* gene set, before HVG selection.
    # We re-load the concatenated data for this.
    print("Reloading full data (all genes) for gene scoring...")
    # This concat MUST be identical to the first one to ensure matching cell IDs
    adata_allgenes = ad.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        fill_value=0,
    )
    adata_allgenes.obs_names_make_unique()
    adata_allgenes.var_names_make_unique()
    print(f"Full data shape for scoring: {adata_allgenes.n_obs} cells x {adata_allgenes.n_vars} genes")
    
    # Normalize for scoring
    print("Normalizing full data for gene scoring...")
    sc.pp.normalize_total(adata_allgenes, target_sum=1e4)
    sc.pp.log1p(adata_allgenes)
    
    def keep_present(g, ad): 
        present = [x for x in g if x in ad.var_names]
        missing = [x for x in g if x not in ad.var_names]
        if missing:
            print(f"    Missing genes from list: {', '.join(missing)}")
        return present

    print("\nDefining gene lists for scoring...")
    print("  -> Myeloid genes:")
    myeloid_genes = keep_present(["LYZ", "LST1", "CTSS", "MS4A7", "CSF1R"], adata_allgenes)
    print("  -> Homeostatic Mac genes:")
    mac_homeo_genes = keep_present(
        ["FABP4", "PPARG", "MRC1", "MARCO", "C1QA", "C1QB", "C1QC", "SIGLEC1", "ITGAM"], 
        adata_allgenes
    )
    print("  -> Profibrotic Mac genes:")
    mac_profib_genes = keep_present(
        ["SPP1", "TREM2", "GPNMB", "LGALS3", "MMP9", "CHI3L1", "CTSB", "CTSD",
         "APOE", "LPL", "FABP5", "ITGAX", "MERTK", "FN1"],
        adata_allgenes
    )
    
    print("\nCalculating gene scores (Myeloid, Homeostatic, Profibrotic)...")
    sc.tl.score_genes(adata_allgenes, myeloid_genes, score_name="score_myeloid")
    sc.tl.score_genes(adata_allgenes, mac_homeo_genes, score_name="score_mac_homeo")
    sc.tl.score_genes(adata_allgenes, mac_profib_genes, score_name="score_mac_profib")
    print("Gene scoring complete.")

    # Add these scores back to our main filtered/processed object `adata`
    print("Adding scores back to main processed AnnData object...")
    # We must be careful to match barcodes (obs_names)
    adata.obs["score_myeloid"] = adata_allgenes.obs.loc[adata.obs_names, "score_myeloid"]
    adata.obs["score_mac_homeo"] = adata_allgenes.obs.loc[adata.obs_names, "score_mac_homeo"]
    adata.obs["score_mac_profib"] = adata_allgenes.obs.loc[adata.obs_names, "score_mac_profib"]
    
    # Calculate Z-scores for plotting
    print("Calculating Z-scores for scores...")
    lims_for_plotting = {}
    score_keys = ["score_myeloid", "score_mac_homeo", "score_mac_profib"]
    
    for k in score_keys:
        s = adata.obs[k].astype(float)
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-8)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        adata.obs[f"{k}_z"] = z
        # Store robust color limits (2nd-98th percentile) for plotting
        lims_for_plotting[f"{k}_z"] = robust_min_max(z.values)

    # Focus on myeloid cells for better contrast
    # (defined as top 70th percentile of myeloid score)
    print("Subsetting to myeloid cells for visualization...")
    thr70 = np.nanpercentile(adata.obs["score_myeloid_z"], 70)
    adata.obs["is_myeloid"] = adata.obs["score_myeloid_z"] > thr70
    adata_my = adata[adata.obs["is_myeloid"]].copy()
    print(f"Created Myeloid subset: {adata_my.n_obs} cells")
    
    # Plot the macrophage score UMAPs
    plot_mac_scores(
        adata_my,
        ["score_mac_homeo_z", "score_mac_profib_z"],
        lims_for_plotting,
        RESULTS_DIR,
    )

    # =========================================================================
    print("\n--- SECTION 5: Subject-Level Aggregation & Modeling ---")
    # =========================================================================
    
    # --- Aggregate data to subject level ---
    # This is the core hypothesis: the *proportion* of profibrotic macrophages
    # in a patient's sample is indicative of their disease state.
    
    print("Aggregating cell metrics to subject level...")
    rows = []
    
    # We must use the *full* adata (post-QC) for aggregation
    
    # Define thresholds from the notebook
    myeloid_thresh = np.nanpercentile(adata.obs["score_myeloid"], 30)
    mac_thresh = np.nanpercentile(adata.obs["score_myeloid"], 30)
    
    subject_ids = adata.obs["sample_id"].unique()
    
    # Use tqdm for progress bar
    for sid in tqdm(subject_ids, desc="Aggregating by subject"):
        this = adata[adata.obs["sample_id"] == sid].copy()
        tot = this.n_obs
        
        # Define cell populations based on scores
        myeloid_mask = (this.obs["score_myeloid"] > myeloid_thresh)
        my_cnt = int(myeloid_mask.sum())
        
        mac_mask = (
            (this.obs["cell_type"] == "Macrophage") | 
            (this.obs["score_myeloid"] > mac_thresh) |
            (this.obs["score_mac_homeo"] > 0.25) |
            (this.obs["score_mac_profib"] > 0.25)
        )
        mac_cnt = int(mac_mask.sum())

        profib_mask = (
            (this.obs["score_mac_profib"] > 0.35) &
            (this.obs["score_mac_profib"] > this.obs["score_mac_homeo"]) &
            (this.obs["score_myeloid"] > 0.3)
        )
        profib_cnt = int(profib_mask.sum())
        
        if tot == 0:
            continue

        row = {
            "sample_id": sid,
            "disease_status": this.obs["disease_status"].iloc[0],
            "disease_binary": this.obs["disease_binary"].iloc[0],
            "total_cells": tot,
            "myeloid_cells": my_cnt,
            "mac_cells": mac_cnt,
            "profib_mac_cells": profib_cnt,
            "frac_myeloid": my_cnt / tot,
            "frac_mac": mac_cnt / tot,
            "frac_profib_mac": profib_cnt / tot,
            "profib_among_mac": profib_cnt / mac_cnt if mac_cnt else 0.0,
            "mean_profib_score": float(this.obs.loc[profib_mask, "score_mac_profib"].mean()) if profib_cnt else 0.0,
            "mean_homeo_score": float(this.obs.loc[mac_mask, "score_mac_homeo"].mean()) if mac_cnt else 0.0,
        }
        rows.append(row)

    # Create the final subject-level DataFrame
    subj = pd.DataFrame(rows).set_index("sample_id")
    
    # Create the binary target variable 'p_fibrosis' (1 for Fibrosis, 0 for Donor)
    subj["p_fibrosis"] = (subj["disease_binary"] == "Fibrosis").astype(int)
    
    # Save this subject-level data
    csv_path = os.path.join(RESULTS_DIR, "subject_level_features.csv")
    subj.to_csv(csv_path)
    print(f"\nSaved subject-level feature table to {csv_path}")
    
    print("\nSubject-Level Feature Table (Head):")
    print(subj.head().to_markdown(floatfmt=".3f"))

    # --- Visualize Subject-Level Data ---
    print("\nPlotting subject-level features...")
    
    # Violin Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_cols = ["frac_myeloid", "frac_mac", "profib_among_mac"]
    for i, col in enumerate(plot_cols):
        sns.violinplot(data=subj, x="disease_binary", y=col, ax=axes[i], order=["Donor", "Fibrosis"])
        sns.stripplot(data=subj, x="disease_binary", y=col, ax=axes[i], color="black", size=5, order=["Donor", "Fibrosis"])
        axes[i].set_title(col)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "subject_level_violins.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subject-level violin plots to {plot_path}")

    # Pairplot
    print("Plotting subject-level pairplot...")
    pairplot_cols = [
        "p_fibrosis",
        "frac_myeloid",
        "frac_mac",
        "frac_profib_mac",
        "profib_among_mac",
        "mean_profib_score",
    ]
    g = sns.pairplot(subj[pairplot_cols], hue="p_fibrosis", corner=True, diag_kind="kde")
    g.fig.suptitle("Subject-Level Feature Pairplot", y=1.03)
    plot_path = os.path.join(RESULTS_DIR, "subject_level_pairplot.png")
    g.savefig(plot_path, dpi=150)
    plt.close(g.fig)
    print(f"Saved subject-level pairplot to {plot_path}")

    # --- Predictive Modeling ---
    print("\n--- Model Training & Evaluation ---")
    
    # Define features (X) and target (y)
    feature_cols = [
        "frac_myeloid",
        "frac_mac",
        "frac_profib_mac",
        "profib_among_mac",
        "mean_profib_score",
        "mean_homeo_score",
    ]
    X = subj[feature_cols].copy()
    y = subj["p_fibrosis"].copy()
    
    # Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # --- Model 1: ElasticNet Logistic Regression (from Cell 57) ---
    # This model uses L1/L2 regularization (ElasticNet) to select the
    # most important features and prevent overfitting.
    # We use LeaveOneOut Cross-Validation (LOOCV) because the N (17 subjects) is very small.
    
    print("\n--- Model 1: ElasticNetCV (LOOCV) ---")
    
    # C=1/alpha. l1_ratio=1 is Lasso, 0 is Ridge, 0.5 is ElasticNet.
    clf_elastic = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=0.1,
        l1_ratio=0.5,
        random_state=42,
        max_iter=1000,
    )
    cv = LeaveOneOut()

    # Get CV predictions and accuracy
    print("Running Leave-One-Out Cross-Validation...")
    y_pred_cv = cross_val_predict(clf_elastic, X_scaled, y, cv=cv)
    
    # We must run predict_proba manually for LOOCV
    y_proba_cv = np.zeros_like(y, dtype=float)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf_elastic.fit(X_train, y_train)
        y_proba_cv[test_idx] = clf_elastic.predict_proba(X_test)[:, 1]

    # Fit final model on all data to get coefficients
    print("Fitting final model on all data to get coefficients...")
    clf_elastic_final = clf_elastic.fit(X_scaled, y)
    coefs = pd.Series(
        clf_elastic_final.coef_[0], index=X.columns, name="ElasticNet_Coefficient"
    ).sort_values(ascending=False)
    
    print("\nElasticNet Model Coefficients (Z-scaled features):")
    print(coefs.to_markdown(floatfmt=".3f"))
    
    print("\nElasticNet CV Classification Report (N=17):")
    print(classification_report(y, y_pred_cv, target_names=["Donor", "Fibrosis"]))
    
    auroc_cv_elastic = roc_auc_score(y, y_proba_cv)
    print(f"ElasticNet CV AUROC = {auroc_cv_elastic:.4f}")

    # Plot CV ROC Curve
    plot_cv_roc(
        clf_elastic,
        X_scaled,
        y,
        cv,
        "ElasticNet (LOOCV)",
        os.path.join(RESULTS_DIR, "model_elasticnet_roc.png"),
    )
    
    # Plot Marginal Effect of top feature
    top_feature = coefs.abs().idxmax()
    plot_marginal_effect(
        clf_elastic_final,
        X_scaled,
        y,
        top_feature,
        f"ElasticNet (All Data) - {top_feature}",
        os.path.join(RESULTS_DIR, f"model_elasticnet_marginal_effect_{top_feature}.png"),
    )

    # --- Model 2: Simple Logistic Regression (from Cell 59) ---
    # This is a simpler model using only the 'profib_among_mac' feature
    
    print("\n--- Model 2: Simple Logistic Regression (LOOCV) ---")
    
    X_simple = X_scaled[["profib_among_mac"]]
    clf_simple = LogisticRegression(random_state=42)
    cv_simple = LeaveOneOut()

    print("Running Leave-One-Out Cross-Validation for simple model...")
    y_pred_simple = cross_val_predict(clf_simple, X_simple, y, cv=cv_simple)
    
    # Manual predict_proba for LOOCV
    y_proba_simple = np.zeros_like(y, dtype=float)
    for i, (train_idx, test_idx) in enumerate(cv_simple.split(X_simple, y)):
        X_train, X_test = X_simple.iloc[train_idx], X_simple.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf_simple.fit(X_train, y_train)
        y_proba_simple[test_idx] = clf_simple.predict_proba(X_test)[:, 1]

    auroc_cv_simple = roc_auc_score(y, y_proba_simple)
    print(f"Simple LogReg CV AUROC = {auroc_cv_simple:.4f}")
    
    print("\nSimple LogReg CV Classification Report (N=17):")
    print(classification_report(y, y_pred_simple, target_names=["Donor", "Fibrosis"]))

    plot_cv_roc(
        clf_simple,
        X_simple,
        y,
        cv_simple,
        "Simple LogReg (profib_among_mac)",
        os.path.join(RESULTS_DIR, "model_logreg_roc.png"),
    )
    
    print("Fitting final simple model on all data...")
    clf_simple_final = clf_simple.fit(X_simple, y)
    plot_marginal_effect(
        clf_simple_final,
        X_simple,
        y,
        "profib_among_mac",
        "Simple LogReg (All Data) - profib_among_mac",
        os.path.join(RESULTS_DIR, "model_logreg_marginal_effect.png"),
    )
    
    print("\n==========================================")
    print("--- Analysis Pipeline Complete ---")
    print("==========================================")


# =========================================================================
# SCRIPT EXECUTION
# =========================================================================

if __name__ == "__main__":
    main()