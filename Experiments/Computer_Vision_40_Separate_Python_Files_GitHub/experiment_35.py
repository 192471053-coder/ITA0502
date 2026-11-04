"""
Computer Vision Lab — Experiment 35
Run this file in Google Colab using: %run experiment_35.py
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

# -------- EXPERIMENT 35 --------

filename, img = upload_image()
text_value = input("Enter text to place on image: ") or "Computer Vision"
result = img.copy()
scale = max(0.5, min(2.0, img.shape[1]/800))
cv2.putText(result, text_value, (10, max(30,img.shape[0]-20)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,255), 2, cv2.LINE_AA)
show_image(result, "Image with text")
