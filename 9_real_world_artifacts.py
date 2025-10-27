# 9. Analyze Real-World Artifacts (Quantization & Calibration Issues) 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("9_image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Simulate Quantization (reduce color levels)
def quantize(img, levels=16):
    return np.floor(img / (256/levels)) * (256/levels)

quantized = quantize(img_rgb, levels=8).astype(np.uint8)

# Simulate Calibration Issue (Overexposed / improper white balance)
calibration_issue = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=50)  # brightness + contrast boost

# Display
titles = ["Original", "Quantization Artifact", "Calibration Issue (Overexposed)"]
images = [img_rgb, quantized, calibration_issue]

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")
plt.show()
