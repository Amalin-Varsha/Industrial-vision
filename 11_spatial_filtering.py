# Apply Spatial Filtering to Enhance the Surface of an Automotive Part Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("11_metal_nut.png", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap="gray"), plt.title("Original")
plt.subplot(1, 2, 2), plt.imshow(sharpened, cmap="gray"), plt.title("Sharpened")
plt.show()

