# 1. Plot the chromaticity coordinates of the pixels of an image on the CIE 1931 diagram

import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour import XYZ_to_xy, sRGB_to_XYZ

# Load image
img = cv2.imread("1_cie.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

# Reshape to N x 3
pixels = img.reshape(-1, 3)

# Convert RGB → XYZ → xy (chromaticity coordinates)
xyz = sRGB_to_XYZ(pixels)
xy = XYZ_to_xy(xyz)

# Plot
plt.scatter(xy[:,0], xy[:,1], s=1, alpha=0.5, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("CIE 1931 Chromaticity Diagram")
plt.show()
