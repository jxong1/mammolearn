import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from tensorflow.keras.applications.vgg16 import preprocess_input

def map_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps dataset origins to ethnicity labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'dataset' column. Case insensitive. Supported datasets:
        Datasets supported: VinDr, CMMD, CBIS_DDSM, DDSM, DMID, KAU_BCMD, InBreast, Mini_MIAS
        

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with additional 'ethnicity' column.
        Ethnicity labels are assigned as follows:
            - "vindr"      -> "vietnamese"
            - "cmmd"       -> "chinese"
            - "cbis_ddsm"  -> "us"
            - "ddsm"       -> "us"
            - "dmid"       -> "us"
            - "kau_bcmd"   -> "saudi"
            - "inbreast"   -> "portuguese"
            - "mini_mias"  -> "uk"
            - Unrecognized -> "unknown".
    """
    assert "dataset" in df.columns, "DataFrame must contain a 'dataset' column"

    # Create mapping
    mapping = {"vindr": "vietnamese",
               "cmmd": "chinese",
               "cbis_ddsm": "us",
               "ddsm": "us",
               "dmid": "us",
               "kau_bcmd": "saudi",
               "inbreast": "portuguese",
               "mini_mias": "uk"}
    
    df = df.copy()
    # Assign mapping to df
    df["ethnicity"] = df["dataset"].str.lower().map(mapping).fillna("unknown")
    return df
    
def collapse_labels(label_list: list[str]) -> str:
    """
    Collapse a list of pathologies into a single label.
    Priority: malignant > benign > normal.

    Parameters
    ----------
    label_list : list of str
        List of pathology labels associated with patient

    Returns
    -------
    str
        Single label based on priority. Returns 'unknown' if no match
    """
    if "malignant" in label_list:
        return "malignant"
    if "benign" in label_list:
        return "benign"
    if "normal" in label_list:
        return "normal"
    return "unknown"

def assign_patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a single pathology label per patient and creates strata column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        - 'pathologies' : list of pathology labels per patient
        - 'dataset'   : dataset name
    
    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with:
        - 'patient_label' : Collapsed label per patient (malignant > benign > normal)
        - 'strata'        : Combination of dataset and patient_label (eg. VinDr_malignant)
    """
    # Check if required cols present
    for col in ["pathologies", "dataset"]:
        assert col in df.columns, f"df missing column: {col}"

    df = df.copy()
    df["patient_label"] = df["pathologies"].apply(collapse_labels)
    df["strata"] = df["dataset"] + "_" + df["patient_label"]
    return df

def aggregate_patient_pathologies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates unique pathologies per patient.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog containing 'patient_id', 'dataset', and 'pathology'

    Returns
    -------
    patient_pathologies : pd.DataFrame
        Each row is a patient with:
        - 'dataset': first dataset name
        - 'pathology': list of unique pathologies
    """
    patient_pathologies = df.groupby("patient_id").agg({"dataset": "first",
                                                        "pathology": lambda x: list(x.unique())}).reset_index()

    # Rename for clarity
    return patient_pathologies.rename(columns={"pathology": "pathologies"})

def prepare_splits(catalog: pd.DataFrame,
                   patient_groups: pd.DataFrame,
                   train_frac: float = 0.7,
                   val_frac: float = 0.15,
                   random_state: int = 42,
                   save_csv: bool = True,
                   data_path: str = "",
                   weight_col: str = ""
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    Split patients into train, validation, and test sets by patient ID.
    Ensures no patient appears in more than one set.
    
    Parameters
    ----------
    catalog: pd.DataFrame
        Dataframe containing:
        - 'patient_id'     : unique identifier of patient
        - 'processed_path' : path to image
        - 'pathology'      : pathology of patient
    patient_groups : pd.DataFrame
        Dataframe with one row per patient and a 'strata' column for stratification
    train_frac : float
        Fraction of patients to use for training
    val_frac : float
        Fraction of patients to use for validation
    random_state : int
        Random seed for reproducibility
    save_csv : bool
        If True, save split CSVs to data_path/splits
    weight_col : str
        If given, computes weights for training dataset before saving

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Dataframes for each split
    """
    assert 0 < train_frac < 1, "train_frac must be between 0 and 1"
    assert 0 <= val_frac < 1, "val_frac must be between 0 and 1"
    assert train_frac + val_frac < 1, "train_frac + val_frac must be less than 1"

    # Check if required cols present
    for col in ["patient_id", "pathology", "processed_path"]:
        assert col in catalog.columns, f"catalog missing column: {col}"
    
    for col in ["patient_id", "strata"]:
        assert col in patient_groups.columns, f"patient_groups missing column: {col}"

    # Train / temp split
    train_patients, temp_patients = train_test_split(patient_groups,
                                                     test_size=1 - train_frac,
                                                     random_state=random_state,
                                                     stratify=patient_groups["strata"])
    
    # Adjust val/test fractions relative to temp_patients
    temp_frac = val_frac / (1 - train_frac)
    val_patients, test_patients = train_test_split(temp_patients,
                                                   test_size= 1 - temp_frac,
                                                   random_state=random_state,
                                                   stratify=temp_patients["strata"])
    
    # Ensure no patient leakage
    train_ids, val_ids, test_ids = set(train_patients["patient_id"]), set(val_patients["patient_id"]), set(test_patients["patient_id"])
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    print("No patient leakage detected!")
    
    # Expand back into rows
    train_df = catalog[catalog["patient_id"].isin(train_ids)].copy()
    val_df = catalog[catalog["patient_id"].isin(val_ids)].copy()
    test_df = catalog[catalog["patient_id"].isin(test_ids)].copy()

    # Assign sample weights to train_df
    if weight_col:
        train_df["weights"] = compute_sample_weight(class_weight="balanced",y=train_df[weight_col])
    
    # Convert path into relative path to model
    # Model located in 'models/' folder
    # Images located in "MammoNet32k/processed/"
    for df in [train_df, val_df, test_df]:
        df["processed_path"] = df["processed_path"].apply(lambda x: os.path.join(data_path, "processed", x))

    # Save for reuse
    if save_csv:
        os.makedirs(os.path.join(data_path, "splits"), exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "splits", "train.csv"), index=False)
        val_df.to_csv(os.path.join(data_path, "splits", "val.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "splits", "test.csv"), index=False)

    return train_df, val_df, test_df

def load_image(path: str, label: int, img_size: tuple[int, int] = (224, 224)) -> tuple[tf.Tensor, int]:
    """
    Load an image from file path and preprocess it.

    Parameters
    ----------
    path : str
        Path to the image file
    label : int
        Image label
    img_size : tuple[int, int]
        Target image size

    Returns
    -------
    img : tf.Tensor
        Preprocessed image tensor
    label : int
        Image label
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # Ensure 3 color channels
    img = tf.image.resize(img, img_size)
    img = preprocess_input(img)
    return img, label

def augment_image(img: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
    """
    Apply simple augmentations

    Parameters
    ----------
    img : tf.Tensor
        Input image tensor
    label : int
        Image label

    Returns
    -------
    img : tf.Tensor
        Image tensor
    label : int
        Image label
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def prepare(ds: tf.data.Dataset,
            shuffle: bool = False,
            augment: bool = False,
            batch_size: int = 16,
            use_autotune: bool = True,
            img_size: tuple[int, int] = (224, 224),
            weighted: bool = False
            ) -> tf.data.Dataset:
    """
    Prepare a TensorFlow dataset for training or evaluation.

    Parameters
    ----------
    ds : tf.data.Dataset
        Dataset of (path, label) or (path, label, weights)
    shuffle : bool
        Whether to shuffle the dataset
    augment : bool
        Whether to apply augmentation
    batch_size : int
        Training batch size
    use_autotune : bool
        Whether to use tf.data.AUTOTUNE
    img_size : tuple[int, int]
        Target image size
    weighted : bool
        Whether the dataset should be returned with weights

    Returns
    -------
    tf.data.Dataset
        Final dataset
    """
    autotune = tf.data.AUTOTUNE if use_autotune else 1

    if shuffle:
        size = tf.data.experimental.cardinality(ds).numpy()
        buffer_size = min(1000, size) # Fallback size
        ds = ds.shuffle(buffer_size=buffer_size)
    
    # Load images
    if weighted:
        # Take (path, label, weight) and calls load_image with just load_image(path, label, img_size)
        ds = ds.map(lambda path, label, weight: (*load_image(path, label, img_size), weight),
                    num_parallel_calls=autotune)
    else:
        # Takes (path, label) and calls load_image(path, label, img_size)
        ds = ds.map(lambda path, label: load_image(path, label, img_size),
                    num_parallel_calls=autotune)

    if augment:
        if weighted:
            ds = ds.map(lambda img, label, weight: (*augment_image(img, label), weight), num_parallel_calls=autotune)
        else:
            ds = ds.map(augment_image, num_parallel_calls=autotune)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds

def create_ds_from_df(df: pd.DataFrame,
                      shuffle: bool = False, 
                      augment: bool = False, 
                      batch_size: int = 16,
                      use_autotune: bool = True,
                      img_size: tuple[int, int] = (224, 224),
                      weighted: bool = False
                      ) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a dataframe with image paths and labels.
    Applies preprocessing, optional shuffling, and augmentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - 'processed_path' : path to procesed image
        - 'label_int'      : integer label
        - 'weights'        : sample weights (Required if weighted = True)
    shuffle : bool
        Whether to shuffle the dataset
    augment : bool
        Whether to apply data augmentation
    batch_size : int
        Training batch size
    use_autotune : bool
        Whether to use tf.data.AUTOTUNE
    img_size : tuple[int, int]
        Target image size
    weighted : bool
        Whether the dataset should be returned with weights

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset ready for training or evaluation
    """
    # Check if required cols present
    for col in ["processed_path", "label_int"]:
        assert col in df.columns, f"df missing column: {col}"
    
    if weighted:
        assert "weights" in df.columns, f"weights not in dataframe"
        ds = tf.data.Dataset.from_tensor_slices((df["processed_path"].values, df["label_int"].values, df["weights"].values))
    else:
        # Create tf.data.Dataset from (path, label) pairs
        ds = tf.data.Dataset.from_tensor_slices((df["processed_path"].values, df["label_int"].values))

    # Prepare dataset
    ds = prepare(ds, shuffle, augment, batch_size, use_autotune, img_size, weighted)
    
    # Quick validation
    assert isinstance(ds, tf.data.Dataset), f"Dataset not tf.data.Dataset type"
    return ds

def load_splits(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-saved train, validation, and test splits as DataFrames.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the split CSVs
    
    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Dataframes for each split
    """

    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "val.csv")
    test_path = os.path.join(data_path, "test.csv")

    # Ensure files exist
    for path in [train_path, val_path, test_path]:
        assert os.path.exists(path), f"Split file not found {path}"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df

def load_datasets(data_path: str, 
                  batch_size: int = 16,
                  use_autotune: bool = True,
                  img_size: tuple[int, int] = (224, 224),
                  weighted: bool = False
                  ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load pre-saved train, validation, and test datasets as tf.data.Datasets.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the split CSVs
    batch_size : int
        Training batch size
    use_autotune : bool
        Whether to use tf.data.AUTOTUNE
    img_size : tuple[int, int]
        Target image size
    weighted : bool
        Whether to include sample weights in train_ds
    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
        Datasets for each split
    """
    assert(os.path.exists(data_path)), f"{data_path} does not exist"
    train_df, val_df, test_df = load_splits(data_path)
    # Only training data is weighted
    train_ds = create_ds_from_df(train_df, True, True, batch_size, use_autotune, img_size, weighted)
    val_ds = create_ds_from_df(val_df, False, False, batch_size, use_autotune, img_size, False)
    test_ds = create_ds_from_df(test_df, False, False, batch_size, use_autotune, img_size, False)

    assert (len(next(iter(train_ds.take(1)))) == 3) == weighted
    assert len(next(iter(val_ds.take(1)))) == 2
    assert len(next(iter(test_ds.take(1)))) == 2
    return train_ds, val_ds, test_ds

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Computes class weights from y_train.

    Parameters
    ----------
    y_train : np.ndarray
        Numpy array of labels

    Returns
    -------
    class_weights : dict
        Dictionary mapping class index to weight
    """
    classes = np.unique(y_train)

    # Compute balanced weights
    weights_array = compute_class_weight(class_weight="balanced",
                                         classes=np.array(classes),
                                         y=y_train)

    class_weights = {i: w for i, w in enumerate(weights_array)}    
    return class_weights