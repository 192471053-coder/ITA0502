"""
Computer Vision Lab — Experiment 32
Run this file in Google Colab using: %run experiment_32.py
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

# -------- EXPERIMENT 32 --------

width = int(input("Enter image width: ") or "600")
height = int(input("Enter image height: ") or "400")
width, height = max(width, 10), max(height, 10)
img = np.full((height, width, 3), 255, dtype=np.uint8)
bw, bh = max(1,width//10), max(1,height//10)
# OpenCV uses BGR color order.
img[0:bh, 0:bw] = (0,0,0)             # Black: top-left
img[0:bh, width-bw:width] = (255,0,0) # Blue: top-right
img[height-bh:height, 0:bw] = (0,255,0) # Green: bottom-left
img[height-bh:height, width-bw:width] = (0,0,255) # Red: bottom-right
show_image(img, "White image with four colored boxes")
