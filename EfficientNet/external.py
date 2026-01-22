# external.py
import os
import zipfile
import shutil
import random
import sys
import subprocess

# Function to check and install missing packages
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")

# Install required packages if missing
required_packages = [
    "numpy",
    "matplotlib",
    "torch",
    "torchvision",
    "scikit-learn"
]

for package in required_packages:
    install_if_missing(package)

# Import all required libraries after ensuring they're installed
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

# Print success message
print("All required libraries have been successfully imported!")