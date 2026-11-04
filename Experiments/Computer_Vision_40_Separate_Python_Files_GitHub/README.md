# Computer Vision Lab — 40 Separate Python Files

This folder contains 40 individual `.py` files, one for each experiment in the lab list.

## Upload to GitHub
1. Download and extract this ZIP.
2. Create a GitHub repository.
3. Choose **Add file → Upload files**.
4. Select all 40 `.py` files and this `README.md`.
5. Commit the changes.

## Run in Google Colab
1. Open Google Colab.
2. Upload the required experiment `.py` file using the Files pane, or upload it from your computer.
3. Run it in a code cell, for example:
   ```python
   %run experiment_01.py
   ```
4. The script will ask you to upload an image or video when required.

## Dependencies
The shared setup imports:
- opencv-python
- numpy
- matplotlib
- IPython
- Google Colab's `google.colab.files`

Install the common packages in a Colab cell if needed:
```python
!pip install opencv-python numpy matplotlib
```
Experiment 40 installs EasyOCR automatically if it is missing.

## Input file guidance
- Image experiments generally prompt for an image upload.
- Video experiments prompt for a video upload.
- Experiment 25 asks for two images: upload the watch scene and watch template as separate files in the same upload prompt.
- Experiments 27, 29, 30 and 38 work best with a clear, front-facing face photograph.
- Experiment 28 demonstrates moving-object detection using background subtraction; it is not a trained vehicle classifier.
- Experiment 25 uses template matching and is not a general-purpose watch detector.

## Note
These files are designed for Google Colab because they use `google.colab.files.upload()` and Colab display helpers. They are not intended to run unchanged as ordinary desktop Python scripts.
