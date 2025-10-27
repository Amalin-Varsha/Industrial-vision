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

box_kernel = np.ones((5,5), dtype=np.float64) / 25
gauss_kernel = gaussian_kernel(5, 1)

box = convolve2d(img, box_kernel)
gauss = convolve2d(img, gauss_kernel)

plt.subplot(1,2,1), plt.imshow(box, cmap='gray'), plt.title('Box Filter'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(gauss, cmap='gray'), plt.title('Gaussian Filter'), plt.axis('off')
plt.show()