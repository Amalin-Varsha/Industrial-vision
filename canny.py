
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')  
edges = cv2.Canny(img, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
