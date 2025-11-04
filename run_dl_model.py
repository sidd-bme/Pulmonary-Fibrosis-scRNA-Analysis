#!/usr/bin/env python

"""
PHM5005 Final Project: Deep Learning Model (V3 - Final)

This script trains a custom deep learning (MLP) model on the
subject-level features to predict fibrosis.

This version includes:
- Aggressive 3-layer suppression of all TensorFlow logging.
- Plotting of a final Confusion Matrix.
- Plotting of the cross-validated predicted probabilities.
"""

# --- Import Libraries ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import logging # Added for advanced logging control

# --- AGGRESSIVE SUPPRESSION BLOCK (MUST BE FIRST) ---
# 1. Suppress C++ level logging (0=all, 1=no INFO, 2=no INFO/WARNING, 3=no all)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 2. Suppress standard Python warnings
warnings.simplefilter(action='ignore', category=Warning)
# -----------------------------------------------------

import tensorflow as tf

# 3. Suppress TensorFlow's specific Python logger (must be after import)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.FATAL) # Only log FATAL errors
# -----------------------------------------------------

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# --- Global Settings ---
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Plotting Functions ---

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plots a heatmap of the confusion matrix."""
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Donor", "Fibrosis"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels
    )
    plt.title("Cross-Validated Confusion Matrix (N=17)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_cv_probabilities(y_true, y_proba, save_path):
    """Plots the distribution of predicted probabilities."""
    print("Plotting predicted probability distribution...")
    
    plot_df = pd.DataFrame({
        "True Label": y_true,
        "Predicted Probability": y_proba
    }).replace({"True Label": {0: "Donor", 1: "Fibrosis"}})
    
    plt.figure(figsize=(7, 6))
    sns.boxplot(
        data=plot_df, 
        x="True Label", 
        y="Predicted Probability", 
        order=["Donor", "Fibrosis"],
        palette=["#1f77b4", "#ff7f0e"]
    )
    sns.stripplot(
        data=plot_df, 
        x="True Label", 
        y="Predicted Probability", 
        order=["Donor", "Fibrosis"],
        color="black",
        jitter=0.1,
        alpha=0.7,
        size=8
    )
    plt.axhline(0.5, ls="--", color="gray", label="Classification Threshold (0.5)")
    plt.title("Cross-Validated Predicted Probabilities (N=17)")
    plt.ylabel("P(Fibrosis)")
    plt.xlabel(None)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved probability plot to {save_path}")

# --- Data and Model Functions ---

def load_data(csv_path: str):
    """Loads the subject-level feature CSV."""
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        print("Please run the main 5005_analysis.py script first.")
        return None, None, None
    
    subj = pd.read_csv(csv_path, index_col="sample_id")
    
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
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    print(f"Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y, scaler


def build_model(input_shape: int):
    """
    Builds a small, heavily regularized Keras MLP model.
    """
    model = Sequential()
    
    model.add(Dense(
        8, 
        input_dim=input_shape, 
        activation='relu', 
        kernel_regularizer=l2(0.01)
    ))
    model.add(Dropout(0.5))

    model.add(Dense(
        4, 
        activation='relu', 
        kernel_regularizer=l2(0.01)
    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Main function to run the LOOCV training and evaluation."""
    
    print("==========================================")
    print("--- Starting Deep Learning (MLP) Model ---")
    print("==========================================")
    
    DATA_PATH = "./analysis_results/subject_level_features.csv"
    RESULTS_DIR = "./analysis_results"
    
    # Load and scale the data
    X_scaled, y, scaler = load_data(DATA_PATH)
    if X_scaled is None:
        return

    # --- Leave-One-Out Cross-Validation (LOOCV) ---
    cv = LeaveOneOut()
    
    print("\nStarting Deep Learning LOOCV (N=17 folds)...")
    
    # We will store the "out-of-sample" predictions here
    all_y_true = []
    all_y_proba = []
    all_y_pred = []
    
    # Manually loop through each of the 17 folds
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        # Use print() with \r and end="" to create a single updating line
        print(f"  > Running Fold {i+1}/17", end="\r")
        
        # Get the single "hold-out" sample and the 16 training samples
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Build a fresh, untrained model
        model = build_model(input_shape=X_train.shape[1])
        
        # 2. Define Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )

        # 3. Train the model
        model.fit(
            X_train, 
            y_train,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            batch_size=X_train.shape[0], # Train on all 16 samples at once
            verbose=0  # Suppress training log for each fold
        )

        # 4. Get the prediction for the one "held-out" sample
        proba = model.predict(X_test, verbose=0)[0][0]
        pred_class = (proba > 0.5).astype(int)
        
        # 5. Store the results
        all_y_true.append(y_test.iloc[0])
        all_y_proba.append(proba)
        all_y_pred.append(pred_class)

    print("\nLOOCV Complete.                    ") # Spaces to overwrite the \r

    # --- Evaluate the Results ---
    print("\n==========================================")
    print("--- Model Evaluation Results ---")
    print("==========================================")
    
    print("\nDeep Learning (MLP) CV Classification Report (N=17):")
    print(classification_report(all_y_true, all_y_pred, target_names=["Donor", "Fibrosis"]))
    
    auroc_cv = roc_auc_score(all_y_true, all_y_proba)
    print(f"Deep Learning (MLP) CV AUROC = {auroc_cv:.4f}")

    # --- Plotting Results ---
    print("\nGenerating model plots...")
    
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
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
    plt.title("ROC Curve: Deep Learning (MLP) Model (LOOCV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    save_path_roc = os.path.join(RESULTS_DIR, "model_deep_learning_roc.png")
    plt.savefig(save_path_roc, dpi=150)
    plt.close()
    print(f"Saved ROC plot to {save_path_roc}")
    
    # Plot 2: Confusion Matrix
    save_path_cm = os.path.join(RESULTS_DIR, "model_deep_learning_cm.png")
    plot_confusion_matrix(all_y_true, all_y_pred, save_path_cm)
    
    # Plot 3: Probability Distribution
    save_path_proba = os.path.join(RESULTS_DIR, "model_deep_learning_proba.png")
    plot_cv_probabilities(all_y_true, all_y_proba, save_path_proba)
    
    print("\n==========================================")
    print("--- Deep Learning Script Complete ---")
    print("==========================================")


if __name__ == "__main__":
    main()