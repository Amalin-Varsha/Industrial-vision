# 5. Apply chromatic adaptation (Bradford)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from colour import sRGB_to_XYZ, XYZ_to_sRGB
from colour.adaptation import matrix_chromatic_adaptation_VonKries

# Define source and target illuminants (D65 and A)
D65 = np.array([0.95047, 1.00000, 1.08883])  # D65 white point
A = np.array([1.09850, 1.00000, 0.35585])    # A white point

# Load an image
img = cv2.imread("5_bradford.jpg")
if img is None:
    raise FileNotFoundError("Could not open or find the image.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Convert to RGB and normalize

# Convert the image to XYZ color space
img_xyz = sRGB_to_XYZ(img)

# Compute the Bradford adaptation matrix
M = matrix_chromatic_adaptation_VonKries(D65, A, transform='Bradford')

# Apply the matrix to the XYZ image
img_xyz_bradford = np.tensordot(img_xyz, M.T, axes=1)

# Convert back to sRGB
img_bradford = XYZ_to_sRGB(img_xyz_bradford)

# Clip values to [0, 1]
img_bradford = np.clip(img_bradford, 0, 1)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_bradford)
plt.title("5. Bradford Adaptation")
plt.axis("off")

plt.tight_layout()
plt.show()
