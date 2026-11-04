"""
Computer Vision Lab — Experiment 30
Run this file in Google Colab using: %run experiment_30.py
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

# -------- EXPERIMENT 30 --------

filename, img = upload_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
path = "haarcascade_smile.xml"; urllib.request.urlretrieve(url, path)
detector = cv2.CascadeClassifier(path)
smiles = detector.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25,15))
result = img.copy()
for (x,y,w,h) in smiles: cv2.rectangle(result, (x,y), (x+w,y+h), (255,0,0), 2)
show_image(result, f"Smile detections: {len(smiles)}")
print("Smile detection can produce false positives.")
