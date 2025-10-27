# 4. Perform gray-level and color image quantization

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("4_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gray quantization (reduce levels)
levels = 4
quant_gray = np.floor(gray / (256/levels)) * (256/levels)

# Color quantization
Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
quant_color = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
plt.title("4. Gray and color image quantization")
plt.subplot(1,3,1); plt.imshow(gray, cmap="gray"); plt.title("Gray Original")
plt.subplot(1,3,2); plt.imshow(quant_gray, cmap="gray"); plt.title("Gray Quantized")
plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(quant_color, cv2.COLOR_BGR2RGB)); plt.title("Color Quantized")
plt.show()
