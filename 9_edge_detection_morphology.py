# Design a Pipeline Using Edge Detection + Morphology to Detect Cracks in Glass Bottle Images
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("9_bottle.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 50, 150)
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(edges, cmap="gray"), plt.title("Canny Edges")
plt.subplot(1, 2, 2), plt.imshow(morph, cmap="gray"), plt.title("Morphology")
plt.show()

