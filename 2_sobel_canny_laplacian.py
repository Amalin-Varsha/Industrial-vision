# Apply Sobel, Canny, and Laplacian Edge Detectors on Bridge/Concrete Images
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("2_tile_crack.png", cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
canny = cv2.Canny(image, 50, 150)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap="gray"), plt.title("Original")
plt.subplot(2, 2, 2), plt.imshow(sobel_combined, cmap="gray"), plt.title("Sobel")
plt.subplot(2, 2, 3), plt.imshow(canny, cmap="gray"), plt.title("Canny")
plt.subplot(2, 2, 4), plt.imshow(laplacian, cmap="gray"), plt.title("Laplacian")
plt.show()

