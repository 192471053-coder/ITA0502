"""
Computer Vision Lab — Experiment 04
Run this file in Google Colab using: %run experiment_04.py
This script contains its own upload/display helper functions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display, Video
from pathlib import Path
import urllib.request, os, time

print("OpenCV version:", cv2.__version__)

def upload_image():
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file uploaded.")
    filename = next(iter(uploaded))
    image = cv2.imread(filename)
    if image is None:
        raise ValueError(f"Could not read image: {filename}. Upload a JPG or PNG.")
    return filename, image

def show_image(image, title="Output", cmap=None, size=(8, 6)):
    plt.figure(figsize=size)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap or "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def upload_video():
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No video uploaded.")
    filename = next(iter(uploaded))
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {filename}")
    cap.release()
    return filename

def save_and_download(filename):
    files.download(filename)

print("Setup complete. Run the experiment cells below.")

# -------- EXPERIMENT 04 --------

filename, img = upload_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].imshow(gray, cmap="gray"); ax[0,0].set_title("Original grayscale")
ax[0,1].imshow(equalized, cmap="gray"); ax[0,1].set_title("Equalized")
ax[1,0].hist(gray.ravel(), 256, [0,256]); ax[1,0].set_title("Original histogram")
ax[1,1].hist(equalized.ravel(), 256, [0,256]); ax[1,1].set_title("Equalized histogram")
for a in ax.ravel(): a.axis("off") if a in [ax[0,0], ax[0,1]] else None
plt.tight_layout(); plt.show()
