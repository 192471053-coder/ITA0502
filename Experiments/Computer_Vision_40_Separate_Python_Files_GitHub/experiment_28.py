"""
Computer Vision Lab — Experiment 28
Run this file in Google Colab using: %run experiment_28.py
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

# -------- EXPERIMENT 28 --------

video_path = upload_video()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
subtractor = cv2.createBackgroundSubtractorMOG2()
frames_out = []
while True:
    ok, frame = cap.read()
    if not ok: break
    mask = subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 1200 and h > 25 and w > 25:
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
    frames_out.append(result)
cap.release()
if not frames_out: raise ValueError("No frames found.")
h,w = frames_out[0].shape[:2]
writer = cv2.VideoWriter("vehicle_motion_detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
for f in frames_out: writer.write(f)
writer.release()
display(Video("vehicle_motion_detection.mp4", embed=True))
print("Note: this detects moving foreground objects; it is not a trained vehicle classifier.")
