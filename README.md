# Pulmonary Fibrosis scRNA-seq and Deep Learning Analysis

This is the final project for the PHM5005 course at NUS. This repository contains the complete analysis pipeline, from raw single-cell data processing to the training of a deep learning model to predict pulmonary fibrosis.

## 📄 Final Project Report

The complete, detailed report for this project, including all figures and analysis, is available here:
**[View the Full Project Report (PDF)](final_phm5005_report.pdf)**

---

## 🔬 Project Overview

This project analyzes the GSE122960 dataset, which contains 17 scRNA-seq samples from healthy donors and patients with pulmonary fibrosis.

**The full analysis pipeline includes:**
* Loading 17 individual `.h5` files.
* Standard QC, normalization, and filtering.
* Dimensionality reduction (PCA) and clustering (Leiden).
* Manual cell type annotation using canonical markers.
* **Hypothesis Testing:** Scoring of a distinct "profibrotic macrophage" population.
* **Predictive Modeling:** Aggregating cell-level data to the subject-level to train and compare three models.

---

## 🚀 Key Results

The analysis confirmed that the fraction of "profibrotic macrophages" in a patient's sample is a powerful biomarker for disease.

A comparison of three models revealed that a non-linear model achieved the best performance:

| Model | Key Feature(s) | Accuracy | AUROC |
| :--- | :--- | :--- | :--- |
| ElasticNet | All 6 Features | 41% | 0.000 |
| Logistic Regression | `profib_among_mac` | 82% | 0.764 |
| **Deep Learning (MLP)** | **All 6 Features** | **94%** | **0.944** |

The custom-built, heavily regularized deep learning model successfully captured the non-linear relationships between all 6 subject-level features to achieve 94% accuracy.

---

## ⚙️ How to Run

This repository contains the two scripts needed to reproduce the analysis.

**1. Main Analysis (QC, Clustering, Macrophage Scoring, Linear Models):**
* `5005_analysis.py`

**2. Deep Learning Model (MLP Training & Evaluation):**
* `run_dl_model.py`

### Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Pulmonary-Fibrosis-scRNA-Analysis.git](https://github.com/your-username/Pulmonary-Fibrosis-scRNA-Analysis.git)
    cd Pulmonary-Fibrosis-scRNA-Analysis
    ```

2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download the Data:**
    * This repository **does not** include the raw data.
    * You must download the 17 `_filtered_gene_bc_matrices_h5.h5` files from [GSE122960 on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122960).
    * Create a folder named `GSE122960_data` in the project's root directory.
    * Place all 17 `.h5` files inside this `GSE122960_data` folder.

4.  **Run the analysis:**
    ```bash
    # Run the main analysis first to generate the subject-level features
    python 5005_analysis.py
    
    # Run the deep learning model (which depends on the files from the first script)
    python run_dl_model.py
    ```
