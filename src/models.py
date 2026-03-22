import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply

def unfreeze_block(model: tf.keras.Model, block_name: str) -> None:
    """
    Unfreeze layers belonging to a specified convolutional block.

    Parameters
    ----------
    model : tf.keras.Model
        The model whose layers will be unfrozen
    block_name : str
        Name of the block to unfreeze.

    Returns
    -------
    None
    """
    for layer in model.layers:
        # Unfreeze only layers in block5
        if block_name in layer.name:
            layer.trainable = True
    print(f"All layers in {block_name} unfrozen.")

def create_se_block(input_tensor: tf.Tensor, reduction: int = 16) -> tf.Tensor:
    """
    Creates a Squeeze-and-Excitation (SE) block.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Output of previous CNN layer
    reduction : int
        Reduction ratio

    Returns
    -------
    tf.Tensor
        SE block configured with reduction parameter
    """
    channels = input_tensor.shape[-1] # Get channels

    # Squeeze (Convert channel to single value)
    x = GlobalAveragePooling2D()(input_tensor)
    # Excitation
    x = Dense(channels // reduction, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)

    # Scale and combine
    x = Reshape((1, 1, channels))(x)
    return Multiply()([input_tensor, x])

def save_model_and_history(model: tf.keras.Model,
                           history: tf.keras.callbacks.History,
                           model_path: str,
                           name: str
                           ) -> None:
    """
    Save model and training history.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model to save
    history : keras.callbacks.History
        History returned by model.fit()
    model_path : str
        Folder where model + history CSV should be saved
    name: str
        Name of model and history
    """
    os.makedirs(model_path, exist_ok=True)

    # Save model
    model.save(os.path.join(model_path, f"{name}.h5"))
    # Save training history
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(os.path.join(model_path, f"{name}_history.csv"), index=False)

def load_model_and_history(model_path: str,
                           name: str
                           ) -> tuple[tf.keras.Model, pd.DataFrame]:
    """
    Load model from {name}.h5 and training history from {name}_history.csv.

    Parameters
    ----------
    model_path : str
        Folder where model + history CSV is be saved
    name: str
        Name of model and history

    Returns
    -------
    model : tf.keras.Model
        Loaded model
    history : pd.DataFrame
        Training history loaded from CSV.
    """

    model_file = os.path.join(model_path, f"{name}.h5")
    history_file = os.path.join(model_path, f"{name}_history.csv")

    # Check valid paths
    assert os.path.exists(model_path), f"{model_path} does not exist"
    assert os.path.exists(model_file), f"{model_file} cannot be found"
    assert os.path.exists(history_file), f"{history_file} cannot be found"

    # Load model
    model = load_model(model_file)

    # Load history CSV
    history = pd.read_csv(history_file)
    return model, history

class WeightedEnsemble:
    """
    Weighted soft-voting ensemble for combining predictions from multiple models.

    Parameters
    ----------
    models : list[tf.keras.Model]
        List of models used in the ensemble
    weights : np.ndarray
        Array of weights corresponding to each model

    Attributes
    ----------
    models : list[tf.keras.Model]
        List of models used in the ensemble.
    weights : np.ndarray
        Normalized weights applied to each model's predictions.
    """
    def __init__(self, models: list[tf.keras.Model], weights: np.ndarray):
        assert len(models) == len(weights), (f"Number of models ({len(models)}) does not match number of weights ({len(weights)})")

        self.models = models
        # Normalise weights
        self.weights = np.array(weights) / np.sum(weights)

    def predict(self, dataset: tf.data.Dataset, verbose: int=0) -> np.ndarray:
        """
        Compute weighted class probabilities.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset to predict labels on
        verbose : int
            To ensure compatibility with Tensorflow pipelines that pass verbose to predict calls

        Returns
        -------
        np.ndarray
            Class probabilities for each sample
        """
        # Multiply by weights
        preds = [model.predict(dataset, verbose=0) * self.weights[i] for i, model in enumerate(self.models)]

        # Get predictions
        probs = np.sum(preds, axis=0)
        return probs
    
def predict_with_threshold(y_prob, positive_class, threshold):
    """
    Converts predicted probabilities into class labels using a custom threshold for the positive class.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities
    positive_class : int
        The index of the positive class
    threshold : float
        Probability threshold for assigning the positive class

    Returns
    -------
    y_pred : np.ndarray
        Array of predicted class labels
    """
    y_pred = np.argmax(y_prob, axis=1)
    # Override predictions based on threshold
    mask = y_prob[:, positive_class] >= threshold
    y_pred[mask] = positive_class
    return y_pred