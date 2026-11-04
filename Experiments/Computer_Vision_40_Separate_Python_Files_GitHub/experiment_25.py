"""
Computer Vision Lab — Experiment 25
Run this file in Google Colab using: %run experiment_25.py
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

# -------- EXPERIMENT 25 --------

print("Upload TWO images: first the scene image, then a cropped template image containing the watch.")
uploaded = files.upload()
names = list(uploaded.keys())
if len(names) < 2:
    raise ValueError("Please upload two files: a scene image and a cropped watch template.")
scene = cv2.imread(names[0])
template = cv2.imread(names[1])
if scene is None or template is None: raise ValueError("Both uploads must be readable images.")
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
th, tw = temp_gray.shape[:2]
if th > scene_gray.shape[0] or tw > scene_gray.shape[1]:
    raise ValueError("Template must be smaller than the scene image.")
match = cv2.matchTemplate(scene_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
_, score, _, loc = cv2.minMaxLoc(match)
result = scene.copy()
cv2.rectangle(result, loc, (loc[0]+tw, loc[1]+th), (0,255,0), 3)
show_image(result, f"Template match score: {score:.2f}")
print("This method works best when the watch appearance and scale are similar to the template.")
