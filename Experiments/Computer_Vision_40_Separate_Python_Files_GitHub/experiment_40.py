"""
Computer Vision Lab — Experiment 40
Run this file in Google Colab using: %run experiment_40.py
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

# -------- EXPERIMENT 40 --------

import subprocess, sys
try:
    import easyocr
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'easyocr'])
    import easyocr
video_path = upload_video()
reader = easyocr.Reader(['en'], gpu=False)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
# Sample up to roughly 10 frames per minute of video, with a minimum interval of 1 second.
interval = max(1, int(fps))
texts_found = []
frame_index = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if frame_index % interval == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)
        for box, text_value, confidence in results:
            texts_found.append((frame_index, round(frame_index/fps, 2), text_value, round(float(confidence), 3)))
    frame_index += 1
cap.release()
if texts_found:
    print("Frame | Time (seconds) | Extracted text | Confidence")
    for item in texts_found:
        print(item)
else:
    print("No text was detected. Try a clearer video with larger text.")
