from notebooks.init_env import cfg
import os
import numpy as np
import json
import gradio as gr
from PIL import Image
from src.gradcam import *
from src.models import WeightedEnsemble, predict_with_threshold, load_model_and_history

# Load config variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.basename(cfg["paths"]["model_path"])
IMG_SIZE = tuple(cfg["img_size"])
labels = cfg["labels"]
mappings = {v: k for k, v in labels.items()}

# Extract model data
with open(os.path.join(BASE_DIR, MODEL_PATH, "ensemble_metadata.json"), "r") as f:
    model_data = json.load(f)

with open(os.path.join(BASE_DIR, MODEL_PATH, "best_threshold.json"), "r") as f:
    best_threshold = json.load(f)

file_paths = model_data["file_paths"]
ensemble_weights = np.array(model_data["weights"])
ensemble_models = [load_model_and_history(os.path.join(BASE_DIR, MODEL_PATH), file_path)[0] for file_path in file_paths]

# Initialise ensemble
ensemble_model = WeightedEnsemble(ensemble_models, ensemble_weights)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

def predict(image):
    # Basic type check
    if not isinstance(image, Image.Image):
        return None, "Unsupported file type (Not an image)"
    
    # Catch corrupted images / Other errors
    try:
        # Preprocess image and get original
        input_tensor, orig_img = preprocess(image, IMG_SIZE)
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

    print("Done preprocessing")
    # Predict
    y_prob = ensemble_model.predict(input_tensor, verbose=0)
    prediction = predict_with_threshold(y_prob, labels["malignant"], best_threshold)[0]
    confidence = y_prob[0, prediction]
    confidence_percent = confidence * 100
    
    # # Generate Grad-CAM and overlay
    cam = get_gradcam_ensemble(input_tensor, ensemble_model)
    overlay = overlay_cam(orig_img, cam)

    text_output = (
        f"Prediction: {mappings[prediction].capitalize()}\n"
        f"Confidence: {confidence_percent:.2f}%\n\n"
        "Disclaimer: Predictions are not 100% accurate.\n"
        "Please consult a medical professional for confirmation."
    )
    return overlay, text_output

app = gr.Interface(fn=predict,
                        inputs=gr.Image(type="pil", label="Upload mammogram"), 
                        outputs=[gr.Image(label="Attention Map Overlay"), 
                                 gr.Textbox(label="Result", lines=5)], 
                        description="Upload an image to predict pathology",
                        flagging_mode="never")

app.launch()