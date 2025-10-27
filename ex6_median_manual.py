import numpy as np
import cv2
from matplotlib import pyplot as plt

def median_filter(image, ksize=3):
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            result[i,j] = np.median(window)
    return result

pcb = cv2.imread('img1.jpg', 0)
if pcb is None:
    print("Image not found.")
    exit()

median = median_filter(pcb, 3)

plt.subplot(1,2,1), plt.imshow(pcb, cmap='gray'), plt.title('Noisy PCB'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(median, cmap='gray'), plt.title('Median Filter'), plt.axis('off')
plt.show()