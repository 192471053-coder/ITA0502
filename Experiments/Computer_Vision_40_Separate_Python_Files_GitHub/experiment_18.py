"""
Computer Vision Lab — Experiment 18
Run this file in Google Colab using: %run experiment_18.py
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

# -------- EXPERIMENT 18 --------

filename, img = upload_image()
h, w = img.shape[:2]
x1, y1, x2, y2 = w//4, h//4, max(w//4+1, 3*w//4), max(h//4+1, 3*h//4)
roi = img[y1:y2, x1:x2].copy()
result = img.copy()
# Paste the copied ROI into the top-left corner, resized if necessary.
rh, rw = roi.shape[:2]
target_h, target_w = min(rh, h), min(rw, w)
result[0:target_h, 0:target_w] = cv2.resize(roi, (target_w, target_h))
show_image(roi, "Cropped ROI")
show_image(result, "ROI copied to top-left")
