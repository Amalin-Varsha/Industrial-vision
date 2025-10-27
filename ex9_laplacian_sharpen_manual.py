import numpy as np
import cv2
from matplotlib import pyplot as plt

def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result

img = cv2.imread('img1.jpg', 0)
if img is None:
    print("Image not found.")
    exit()

laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
lap = convolve2d(img, laplacian_kernel)
sharp = np.clip(img + lap, 0, 255)

plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(sharp, cmap='gray'), plt.title('Lap Edge Enhance'), plt.axis('off')
plt.show()