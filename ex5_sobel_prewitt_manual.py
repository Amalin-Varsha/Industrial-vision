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

# Sobel
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sobelx = convolve2d(img, sobel_x)
sobely = convolve2d(img, sobel_y)
sobel = np.hypot(sobelx, sobely)

# Prewitt
prewitt_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
prewitt_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittx = convolve2d(img, prewitt_x)
prewitty = convolve2d(img, prewitt_y)
prewitt = np.hypot(prewittx, prewitty)

plt.subplot(1,2,1), plt.imshow(sobel, cmap='gray'), plt.title('Sobel'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt'), plt.axis('off')
plt.show()