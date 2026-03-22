import warnings

warnings.simplefilter(action='ignore')

import sys
import os
import yaml
import tensorflow as tf

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Path to config.yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

# Load config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Get seed from config
seed = cfg.get("seed", 42)
# Sets Python, NumPy, and TF seed
tf.keras.utils.set_random_seed(seed)