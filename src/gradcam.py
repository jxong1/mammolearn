import numpy as np
import cv2
import tensorflow as tf
from models import WeightedEnsemble
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess(img: Image.Image, img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an input PIL image to be compatible with a CNN model

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    img_size : tuple[int, int]
        Target image size

    Returns
    -------
    input_tensor : np.ndarray
        Preprocessed image (1 x H x W x C)
    img_original : np.ndarray
        Original image resized to img_size
    """
    img_original = np.array(img.convert("RGB"))  # Ensure RGB
    img_original = cv2.resize(img_original, img_size) # Resize to model input
    input_tensor = preprocess_input(img_original)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    return input_tensor, img_original

def get_gradcam(input_tensor: np.ndarray, model: tf.keras.Model, last_layer_name: str) -> np.ndarray:
    """
    Compute Grad-CAM heatmap
    
    Parameters
    ----------
    input_tensor : np.ndarray
        Preprocessed input tensor for the model
    model : tf.keras.Model
        Model to compute Grad-CAM for
    last_layer_name : str
        Name of the last convolutional layer

    Returns
    -------
    cam : np.ndarray
        Grad-CAM heatmap
    """
    # Specifies a new output for model (Uses same trained weights)
    # To get convolutional features + prediction
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_layer_name).output, model.output])

    # Backpropagate using gradient tape to compute gradients
    with tf.GradientTape() as tape:
        # Get feature maps from last conv layer + predictions
        conv_outputs, predictions = grad_model(input_tensor)
        # Select predicted class (highest prob)
        pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index] # [0.2, 0.7, 0.1] -> gets 0.7 (Represents confidence)
    grads = tape.gradient(class_score, conv_outputs) # Change in class confidence wrt conv_outputs

    # Get weights per feature map
    weights = tf.reduce_mean(grads, axis=(0, 1))
    # Weighted sum of feature maps
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)  

    # ReLU and normalize
    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam)
    return cam.numpy()

def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an image for visualisation

    Parameters
    ----------
    img : np.ndarray
        Original image
    cam : np.ndarray
        Grad-CAM heatmap
    alpha : float, default=0.4
        Transparency of the heatmap

    Returns
    -------
    overlayed_img : np.ndarray
        Original image overlaid with heatmap
    """
    cam = cv2.resize(cam, (img.shape[1], img.shape[0])) # Resize to fit image dimensions
    cam = np.uint8(255 * cam) # Scale
    # BLUE (least important) -> RED (Most important)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET) # Apply color map
    # Convert to same color map as img + adjust transparency
    cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB) * alpha 
    return np.uint8(cam_rgb + img) # Overlay cam and image

def get_gradcam_ensemble(input_tensor: tf.Tensor, 
                         ensemble_model: WeightedEnsemble, 
                         last_layer_name="block5_conv3") -> np.ndarray:
    """
    Compute Grad-CAM for an ensemble
    
    Parameters
    ----------
    input_tensor : tf.Tensor
        Preprocessed image tensor
    ensemble_model : WeightedEnsemble
        Ensemble model with models and weights
    last_layer_name : str
        Name of the last convolutional layer

    Returns
    -------
    np.ndarray
        Combined Grad-CAM heatmap
    """
    cams = []
    ensemble_models = ensemble_model.models
    ensemble_weights = ensemble_model.weights
    # Loop through models
    for model in ensemble_models:
        grad_model = tf.keras.models.Model(inputs=model.inputs,
                                           outputs=[model.get_layer(last_layer_name).output, model.output])

        # Backpropagate using gradient tape to calculate gradients
        with tf.GradientTape() as tape:
            # Get feature maps from last conv layer + predictions
            conv_outputs, predictions = grad_model(input_tensor)
            pred_index = tf.argmax(predictions[0])
            # Select predicted class (highest prob)
            class_score = predictions[0, pred_index]

        grads = tape.gradient(class_score, conv_outputs) # Change in class confidence wrt conv_outputs
        # Get weights per feature map
        weights = tf.reduce_mean(grads, axis=(0, 1))
        # Weighted sum of feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
        cam = tf.maximum(cam, 0)
        cam /= tf.reduce_max(cam)
        cams.append(cam.numpy())

    # Combine CAMs using ensemble weights
    cams = np.array(cams)
    weights = ensemble_weights / np.sum(ensemble_weights)
    # Compute weighted sum of CAMs
    ensemble_cam = np.sum(cams * weights[:, None, None], axis=0)
    return ensemble_cam