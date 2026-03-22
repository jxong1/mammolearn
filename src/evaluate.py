from dataset import create_ds_from_df
from plots import plot_val_loss, plot_roc_pathology
from models import predict_with_threshold
from metrics import *

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display, Markdown
from sklearn.metrics import f1_score

def evaluate_by_column(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       test_df: pd.DataFrame,
                       col: str) -> pd.DataFrame:
    """
    Evaluate model performance across subgroups defined by a column.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted labels
    test_df : pd.DataFrame
        Test dataframe
    col : str
        Column name to group by

    Returns
    -------
    pd.DataFrame
        Metrics computed per subgroup
    """
    results = []

    for group, group_df in test_df.groupby(col):
        i = group_df.index
        y_true_grp = y_true[i]
        y_pred_grp = y_pred[i]

        # Compute metrics
        metrics = compute_metrics(y_true_grp, y_pred_grp)
        metrics[col] = group
        results.append(metrics)

    return pd.DataFrame(results)

def evaluate_fairness(cols_df: pd.DataFrame) -> dict:
    """
    Compute fairness metrics from subgroup results.

    Parameters
    ----------
    cols_df : pd.DataFrame
        Output from evaluate_by_column()

    Returns
    -------
    dict
        Fairness metrics
    """
    f1_scores = cols_df["F1"].values
    malignant_recalls = cols_df["Malignant Recall"].values

    min_f1 = np.min(f1_scores)
    std_f1 = np.std(f1_scores)
    macro_malignant_recall = np.mean(malignant_recalls)

    return {"Macro Malignant Recall": macro_malignant_recall,
            "Min F1": min_f1,
            "Std (F1)": std_f1}

def overview_eval(model: tf.keras.Model, 
                  history: pd.DataFrame, 
                  test_df: pd.DataFrame,
                  test_ds: pd.DataFrame,
                  name: str,
                  threshold: float = None,
                  fairness_col: str = "ethnicity_grouped"
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run plot loss and plot ROCs

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    history : pd.DataFrame
        Training history DataFrame containing 'val_loss'
    test_df : pd.DataFrame
        Test dataset dataframe
    name : str
        Name of the model

    Returns
    -------
    auc_df : pd.DataFrame
        Overall and per-pathology AUCs
    subgroup_metrics : pd.DataFrame
        Metrics broken down by specified column
    """
    if history is not None:
        # Plot validation loss
        plot_val_loss(history, name)
    
    # Gather all true labels
    mappings = dict(test_df[["label_int", "pathology"]].drop_duplicates().values)
    y_true = test_df["label_int"].values

    # Predict probabilities
    y_score = model.predict(test_ds, verbose=0)

    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = predict_with_threshold(y_score, 2, threshold)
    # Plot ROC curve
    aucs = plot_roc_pathology(mappings, y_true, y_score, name)
    
    # Evaluate metrics
    metrics = compute_metrics(y_true, y_pred)
    
    cols_df = evaluate_by_column(y_true, y_pred, test_df, fairness_col)
    display(cols_df)
    # Evaluate fairness
    fairness = evaluate_fairness(cols_df)
    return pd.DataFrame([aucs | metrics | fairness])

def eval_all_models(models: dict[tf.keras.Model], test_df) -> pd.DataFrame:
    """
    Evaluate multiple models and combine results into a single DataFrame.

    Parameters
    ----------
    models : dict
        Dictionary where keys are the model names and the values are the model info
    test_df : pd.DataFrame
        Test set

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all models' metrics
    """
    model_dfs = {}

    # Convert to dataset
    test_ds = create_ds_from_df(test_df)

    for short_name, info in models.items():
        model = info["model"]
        history = info["history"]
        long_name = info["long_name"]
        
        display(Markdown(f"## {long_name}"))
        # Evalaute and add to dict
        model_dfs[short_name] = overview_eval(model, history, test_df, test_ds, long_name)

    # Combine into a single table
    combined_df = pd.concat(model_dfs.values(),
                            keys=model_dfs.keys(), 
                            names=["Model", "Row"]).reset_index(level=1, drop=True).reset_index()
    # Rename columns
    return combined_df.rename(columns={"index": "Model"})

def find_threshold_for_best_f1(y_true: np.ndarray, 
                               y_prob: np.ndarray, 
                               positive_class: int = 2, 
                               thresholds: np.ndarray = np.linspace(0,1,200)) -> float:
    """
    Find the threshold that maximises F1-score for the positive class.
    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_prob : np.ndarray
        Predicted probabilities
    positive_class : int, default=2
        The class to optimise F1-score for
    thresholds : np.ndarray
        Thresholds to test
    
    Returns
    -------
    best_t : float
        Threshold that produces the highest F1-score
    """
    y_true_binary = (y_true == positive_class).astype(int)
    y_prob_pos = y_prob[:, positive_class]
    
    best_t = 0.5
    best_f1 = 0.0
    
    for t in thresholds:
        y_pred = (y_prob_pos >= t).astype(int)
        f1 = f1_score(y_true_binary, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    return best_t