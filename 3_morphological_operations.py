#  Apply Morphological Opening and Closing to Clean Binary Masks
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("3_carpet.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(binary, cmap="gray"), plt.title("Binary")
plt.subplot(1, 3, 2), plt.imshow(opening, cmap="gray"), plt.title("Opening")
plt.subplot(1, 3, 3), plt.imshow(closing, cmap="gray"), plt.title("Closing")
plt.show()

