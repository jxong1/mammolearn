from models import predict_with_threshold

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score, recall_score, precision_score, f1_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes accuracy, precision, recall and f1 score.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted class labels
    
    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, malignant recall and F1
    """
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0), 
            "Malignant Recall": compute_class_recall(y_true, y_pred, 2),
            "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}

def compute_class_recall(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    """
    Computes recall for a class.

    Parameters
    ----------
    y_true: np.ndarray
        True labels
    y_pred: np.ndarray
        Model's predicted labels
    label: int
        Integer label of the class

    Returns
    -------
    class_recall: float
        Recall of the specified class
    """
    # Explicitly convert to integers
    y_true_class = (y_true == label).astype(int)
    y_pred_class = (y_pred == label).astype(int)
    return recall_score(y_true_class, y_pred_class)

def bootstrap_metrics_df(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for key metrics and return as DataFrame.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_score : np.ndarray
        Predicted probabilities
    threshold : float
        Probability threshold for assigning the malignant class

    Returns
    -------
    pd.DataFrame
        DataFrame with bootstrap confidence intervals for Macro F1 and Malignant Recall
    """
    mean_f1, low_f1, high_f1 = bootstrap_ci(y_true, y_score, threshold, metric="f1")
    mean_mrec, low_mrec, high_mrec = bootstrap_ci(y_true, y_score, threshold, metric="malignant_recall")

    results_df = pd.DataFrame({"Metric": ["Macro F1", "Malignant Recall"],
                               "Mean": [mean_f1, mean_mrec],
                               "CI Lower": [low_f1, low_mrec],
                               "CI Upper": [high_f1, high_mrec]})
    return results_df

def bootstrap_ci(y_true: np.ndarray,
                y_probs: np.ndarray, 
                threshold: float = None, 
                n_bootstrap: int = 1000, 
                metric: str = "f1") -> tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a given metric.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_probs : np.ndarray
        Predicted probabilities from the model
    threshold : float or None
        Threshold for prediction
    n_bootstrap : int
        Number of bootstrap resamples.
    metric : str
        Metric to compute

    Returns
    -------
    mean : float
        Mean metric value
    lower : float
        Lower bound of the 95% confidence interval
    upper : float
        Upper bound of the 95% confidence interval
    """
    scores = []
    for _ in range(n_bootstrap):
        i = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_bs = y_true[i] # Bootstrap sample of true labels
        y_probs_bs = y_probs[i] # Bootstrap sample of predicted probabilities

        # Compute metric
        if metric == "bse": # Brier score
            score = brier_score_loss(y_true_bs, y_probs_bs)
        else:
            # Convert to class labels
            if threshold is None:
                y_pred_bs = np.argmax(y_probs_bs, axis=1)
            else:
                y_pred_bs = predict_with_threshold(y_probs_bs, 2, threshold)
            
            if metric == "f1":
                score = f1_score(y_true_bs, y_pred_bs, average="macro")
            elif metric == "malignant_recall":
                y_true_bin = (y_true_bs == 2).astype(int)
                y_pred_bin = (y_pred_bs == 2).astype(int)
                score = recall_score(y_true_bin, y_pred_bin)
        scores.append(score)

    scores = np.array(scores)
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean, lower, upper