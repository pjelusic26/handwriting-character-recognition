# handwriting-character-recognition
Python script for text recognition from a photo and classification of characters

The repository contains the following:
- Python script with a class and all methods needed
- OCR Jupyter Notebook which:
  - recognizes feature points (characters) from input image and saves them as individual image files
  - recognizes characters (OCR) from input image and calculates the result (parsing)
- CNN Jupyter Notebook which:
  - takes input images and provides predictions (TensorFlow) after training (Mnist dataset)
  
The code features lots of lazy programming, and is far from perfect. Several workarounds have been made in order to make it work.

TODO:
  - Train neural net to recognize operators as well (not just digits)
  - Merge both Notebooks into a single workflow
