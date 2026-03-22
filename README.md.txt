## Description
This project classifies mammography images using radiomics features and deep learning models.

It includes scripts for data preprocessing, model training, evaluation, and visualisation.

## Dataset
This repository does not contain the original dataset. 
The original dataset used for the project can be obtained from: https://www.kaggle.com/datasets/theosmithdevey/mammonet20k

## Folder Structure
Folder Structure should be as follows

root/
├──MammoNet32k/
│       ├──processed/ Processed images (ignored by git)
│       ├──splits/ train.csv,val.csv,test.csv
├──models/ model checkpoints
├──notebooks/ Jupyter notebooks and init_env.py
├──src/ .py files
├──app.py
├──README.md
├──requirements.txt
├──.gitignore

## Installation
1. **Clone the repository**
git clone https://github.com/jxong1/mammolearn

2. **Setup Environment**
python -m venv venv
venv\Scripts\activate
# source venv/bin/activate # Linux

# Install dependencies
pip install -r requirements.txt

3. **Run pipeline**
Execute notebooks in this order

# Explore raw dataset
eda.ipynb

# Preprocess data
preprocessing.ipynb

# Baseline model
baseline.ipynb

# Train models
models.ipynb

# Evaluate models
eval.ipynb

# View Grad-CAMs
interpretability.ipynb
