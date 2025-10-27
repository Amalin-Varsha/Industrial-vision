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
gaussian_kernel = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
], dtype=np.float64)
gaussian_kernel /= np.sum(gaussian_kernel)

lap = convolve2d(img, laplacian_kernel)
blur = convolve2d(img, gaussian_kernel)
log = convolve2d(blur, laplacian_kernel)

plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(lap, cmap='gray'), plt.title('Laplacian'), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(log, cmap='gray'), plt.title('LoG'), plt.axis('off')
plt.show()