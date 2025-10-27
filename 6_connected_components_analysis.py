# Use Connected Components Analysis on Tablet Images to Detect Defects
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("6_tablet.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
output = image.copy()
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area > 100:  
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(binary, cmap="gray"), plt.title("Binary")
plt.subplot(1, 2, 2), plt.imshow(output, cmap="gray"), plt.title("Defects Detected")
plt.show()
