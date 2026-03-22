import cv2
import numpy as np
import pandas as pd
from metrics import compute_metrics
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

def extract_radiomics_features(img_path: str, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract radiomics features from an image.

    Parameters
    ----------
    img_path : str
        Image file path
    img_size : tuple[int, int]
        Target image size

    Returns
    -------
    np.ndarray
        Feature vector
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0

    features = []

    # First-order intensity features
    features.extend([np.mean(img),
                     np.std(img),
                     np.min(img),
                     np.max(img),
                     skew(img.flatten()),
                     kurtosis(img.flatten())])

    # GLCM features
    img_uint8 = (img * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8,
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256,
                        symmetric=True,
                        normed=True)

    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        features.append(graycoprops(glcm, prop).mean())

    return np.array(features)

def build_radiomics_features(df: pd.DataFrame, img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix (X) and labels (y) from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - 'processed_path': path to preprocessed image
        - 'label_int': label of the image
    img_size : tuple[int, int]
        Target image size

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (num_samples, num_features)
    y : np.ndarray
        Labels corresponding to each row in X
    """
    X, y = [], []

    # Check if required cols present
    for col in ["processed_path", "label_int"]:
        assert col in df.columns, f"df missing column: {col}"

    for _, row in df.iterrows():
        img_path = row['processed_path']
        features = extract_radiomics_features(img_path, img_size)

        X.append(features)
        y.append(row['label_int'])
    return np.array(X), np.array(y)

def evaluate_by_group_radiomics(clf, 
                                df: pd.DataFrame,
                                col: str,
                                img_size: tuple[int, int],
                                scaler: StandardScaler
                                ) -> pd.DataFrame:
    """
    Evaluate a classifier on subgroups of a dataframe and compute performance metrics on radiomics features.

    Parameters
    ----------
    clf : classifier
        Classifier trained on build_radiomics_features() with .predict() method
    df : pd.DataFrame
        DataFrame containing image paths and labels
    col : str
        Subgroup column name in 'df'
    img_size : tuple[int, int]
        Target image size
    scaler : StandardScaler
        Fitted scaler

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per subgroup containing accuracy, precision, recall, malignant recall, f1 and col
    """
    results = []

    for group, group_df in df.groupby(col):
        X_group, y_group = build_radiomics_features(group_df, img_size)
        X_group_scaled = scaler.transform(X_group)
        
        y_pred = clf.predict(X_group_scaled)
        
        metrics = compute_metrics(y_group, y_pred)
        metrics[col] = group
        results.append(metrics)
    return pd.DataFrame(results)