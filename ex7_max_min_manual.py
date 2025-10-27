import numpy as np
import cv2
from matplotlib import pyplot as plt

def max_filter(image, ksize=3):
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            result[i,j] = np.max(window)
    return result

def min_filter(image, ksize=3):
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            result[i,j] = np.min(window)
    return result

img = cv2.imread('img1.jpg', 0)
if img is None:
    print("Image not found.")
    exit()

max_img = max_filter(img, 3)
min_img = min_filter(img, 3)

plt.subplot(1,2,1), plt.imshow(max_img, cmap='gray'), plt.title('Max Filter'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(min_img, cmap='gray'), plt.title('Min Filter'), plt.axis('off')
plt.show()