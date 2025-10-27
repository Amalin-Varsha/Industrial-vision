#  Perform Defect Segmentation Using Otsu and Adaptive Thresholding
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("13_hazelnut.png", cv2.IMREAD_GRAYSCALE)
_, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(otsu, cmap="gray"), plt.title("Otsu Thresholding")
plt.subplot(1, 2, 2), plt.imshow(adaptive, cmap="gray"), plt.title("Adaptive Thresholding")
plt.show()

