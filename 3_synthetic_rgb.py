# 3. Create two synthetic RGB images perceptually similar in LAB

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Synthetic RGB images
img1 = np.full((100, 100, 3), [200, 50, 50], dtype=np.uint8)
img2 = np.full((100, 100, 3), [210, 60, 60], dtype=np.uint8)

# Convert to LAB
lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

print("LAB values of Image1:", lab1[0,0])
print("LAB values of Image2:", lab2[0,0])
plt.title("3. Synthetic rgb")
plt.subplot(1,2,1); plt.imshow(img1); plt.title("RGB1")
plt.subplot(1,2,2); plt.imshow(img2); plt.title("RGB2")
plt.show()
