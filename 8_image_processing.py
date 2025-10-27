# 8. Demonstrate IITM Virtual Labs Experiments (Image Processing Basics)

import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("8_image.png")  # replace with your image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
hist_eq = cv2.equalizeHist(gray)

# Gaussian Smoothing
blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

# Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Display results
titles = ["Original", "Gray", "Histogram Equalized", "Gaussian Blur", "Edges"]
images = [img_rgb, gray, hist_eq, blurred, edges]

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    cmap = "gray" if len(images[i].shape) == 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis("off")
plt.show()
