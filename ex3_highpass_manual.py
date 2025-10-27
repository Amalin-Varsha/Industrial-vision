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

def gaussian_kernel(ksize, sigma):
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel

img = cv2.imread('img1.jpg', 0)
if img is None:
    print("Image not found.")
    exit()

gk = gaussian_kernel(21, 5)
blur = convolve2d(img, gk)
high_pass = img.astype(np.float64) - blur
high_pass = np.clip(high_pass, 0, 255)

plt.imshow(high_pass, cmap='gray')
plt.title('High-pass')
plt.axis('off')
plt.show()