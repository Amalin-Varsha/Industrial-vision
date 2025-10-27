# Design of Classical CV Pipelines for Defect Localization and Pattern Matching
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("15_zipper.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 50, 150)
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

plt.imshow(output), plt.title("Defect Localization")
plt.show()
