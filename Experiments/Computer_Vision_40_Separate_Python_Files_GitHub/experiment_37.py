"""
Computer Vision Lab — Experiment 37
Run this file in Google Colab using: %run experiment_37.py
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

# -------- EXPERIMENT 37 --------

filename, img = upload_image()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("Choose HSV limits for the foreground color you want to keep.")
lower = np.array([int(input("Lower H [0]: ") or 0), int(input("Lower S [0]: ") or 0),
                  int(input("Lower V [0]: ") or 0)], dtype=np.uint8)
upper = np.array([int(input("Upper H [179]: ") or 179), int(input("Upper S [255]: ") or 255),
                  int(input("Upper V [255]: ") or 255)], dtype=np.uint8)
mask = cv2.inRange(hsv, lower, upper)
foreground = cv2.bitwise_and(img, img, mask=mask)
show_image(mask, "Foreground color mask")
show_image(foreground, "Extracted foreground")
