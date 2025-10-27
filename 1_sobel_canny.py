# Implement Sobel and Canny edge detectors on sample industrial images

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("1_metal_nut.png", cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
canny = cv2.Canny(image, 50, 150)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap="gray"), plt.title("Original")
plt.subplot(1, 3, 2), plt.imshow(sobel_combined, cmap="gray"), plt.title("Sobel")
plt.subplot(1, 3, 3), plt.imshow(canny, cmap="gray"), plt.title("Canny")
plt.show()

