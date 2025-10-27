# Perform Defect Segmentation Using Otsu and Adaptive Thresholding
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("4_pill.png", cv2.IMREAD_GRAYSCALE)
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1), plt.imshow(global_thresh, cmap="gray"), plt.title("Global Thresholding")
plt.subplot(1, 3, 2), plt.imshow(adaptive_thresh, cmap="gray"), plt.title("Adaptive Thresholding")
plt.subplot(1, 3, 3), plt.imshow(otsu_thresh, cmap="gray"), plt.title("Otsu Thresholding")
plt.show()
