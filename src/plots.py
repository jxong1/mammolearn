import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from metrics import bootstrap_ci
import numpy as np

def plot_val_loss(history: pd.DataFrame, name: str) -> None:
    """
    Plot the validation loss curve from a training history.

    Parameters
    ----------
    history : pd.DataFrame
        Training history DataFrame containing 'val_loss'
    name : str
        Name of the model

    Returns
    -------
    None
    """
    plt.figure(figsize=(8,6))
    plt.plot(history["val_loss"])
    plt.title(f"{name} - Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.show()

def plot_roc_pathology(mappings: dict, y_true: np.ndarray, y_score: np.ndarray, name: str) -> pd.DataFrame:
    """
    Plot one-vs-rest ROC curves per pathology.

    Parameters
    ----------
    mappings : dict
        Dictionary mapping integer labels to pathology names
    y_true : np.ndarray
        True class labels
    y_score : np.ndarray
        Predicted class probabilities
    name : str
        Name of the model

    Returns
    -------
    results : pd.DataFrame
        Dataframe containing AUCs per pathology
    """
    num_classes = len(mappings)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        # Get labels and plot
        y_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_binary, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))

    for i in range(num_classes):
        plt.plot(fpr[i],tpr[i],label=f"{mappings[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.show()

    aucs = {mappings[i].capitalize() + " AUC": roc_auc[i] for i in range(num_classes)}
    aucs["Macro AUC"] = np.mean(list(roc_auc.values()))
    return aucs

def create_radar_plots(df: pd.DataFrame, metrics: list[str], models: list[str], ncols: int, title: str) -> pd.DataFrame:
    """
    Create radar plots for each model in subplots.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing one row per model and columns for each metric
    metrics : list[str]
        List of metrics to include on the radar plots
    models : list[str]
        List of models to visualise
    ncols: int
        Number of columns
    title : str
        Overall figure title

    Returns
    -------
    areas : pd.DataFrame
        DataFrame containing the radar polygon area for each model
    """
    # Normalise metrics
    norm_table = df.copy()
    norm_table[metrics] = (norm_table[metrics] - norm_table[metrics].min(axis=0)) / (norm_table[metrics].max(axis=0) - norm_table[metrics].min(axis=0))

    # Filter models if provided
    if models is not None:
        norm_table = norm_table[norm_table["Model"].isin(models)]

    num_models = len(norm_table)
    # Calculate rows needed to fit all models
    nrows = int(np.ceil(num_models / ncols))

    # Use polar coords
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), subplot_kw=dict(polar=True))
    plt.suptitle(title)
    axes = axes.flatten()
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.append(angles, angles[0])  # Close loop
    areas = {}
    
    for i, (_, row) in enumerate(norm_table.iterrows()):
        ax = axes[i]
        values = np.array([row[m] for m in metrics])
        
        # Convert to cartesian coords
        x = values * np.cos(angles[:-1])
        y = values * np.sin(angles[:-1])
        # Calculate area with shoelace method
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas[row["Model"]] = area
        
        values = np.append(values, values[0])  # Close loop
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1) # Force scale
        ax.set_title(row["Model"])
    
    # Remove empty subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    return pd.DataFrame.from_dict(areas, orient="index", columns=["Radar Area"])

def plot_fairness_calibration(test_df: pd.DataFrame, 
                              y_score: np.ndarray, 
                              n_bins: int = 10, 
                              n_bootstrap: int = 1000):
    """
    Evaluate calibration fairness across ethnicities for the malignant class.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test set with 'label_int' and 'ethnicity_grouped'
    y_score : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve
    n_bootstrap : int
        Number of bootstrap resamples

    Returns
    -------
    pd.DataFrame
        DataFrame with ethnicity, Brier score, CI, n_samples
    """
    results = []
    ethnicities = test_df["ethnicity_grouped"].unique()

    for eth in ethnicities:
        subset = test_df[test_df["ethnicity_grouped"] == eth]
        
        y_true_bin = (subset["label_int"].values == 2).astype(int)
        y_prob_bin = y_score[subset.index, 2]

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true_bin, y_prob_bin, n_bins=n_bins)

        # Bootstrap Brier score
        mean, lower, upper = bootstrap_ci(y_true_bin, y_prob_bin, threshold=None, n_bootstrap=n_bootstrap, metric="bse")
        
        # Save results
        results.append({"ethnicity_grouped": eth,
                        "Brier Score": mean,
                        "CI Lower": lower,
                        "CI Upper": upper,
                        "n_samples": len(subset)})
        
        # Plot calibration curve
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label=f"Malignant - {eth}")
        plt.plot([0,1],[0,1], linestyle="--", label="Perfect calibration")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.title(f"Calibration Curve - Malignant Class ({eth})")
        plt.legend()
        plt.show()
    return pd.DataFrame(results)

def create_box_plot(df: pd.DataFrame, col: str, palette: str):
    """
    Create a bar plot for a metric for each model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model evaluation results
    col : str
        Metric column to visualise
    palette : str
        Colour palette

    Returns
    -------
    None
    """
    plt.figure(figsize=(10,5))
    sns.barplot(x="Model", y=col, data=df, palette=palette)
    plt.xticks(rotation=45)
    plt.title(f"{col} per Model")
    plt.ylabel(f"{col}")
    plt.xlabel("Model")
    plt.ylim(0,1)
    plt.show()

def plot_confusion_matrices_by_group(df: pd.DataFrame,
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     col: str,
                                     n_cols: int,
                                     labels: list) -> None:
    """
    Create confusion matrices for each subgroup in subplots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame
    y_true : np.ndarray
        True class labels
    y_probs : np.ndarray
        Predicted probabilities from the model
    col : str
        Column name to sort by
    ncols: int
        Number of columns
    labels : list
        Labels in test dataframe
    """
    groups = df[col].unique()
    n_groups = len(groups)
    n_rows = int(np.ceil(n_groups / 2))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for ax, group in zip(axes, groups):
        i = df[df[col] == group].index
        y_true_grp = y_true[i]
        y_pred_grp = y_pred[i]
        
        cm = confusion_matrix(y_true_grp, y_pred_grp, labels=range(len(labels)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{group} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    
    # Hide unused subplots
    for i in range(len(groups), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()