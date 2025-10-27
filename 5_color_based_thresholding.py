#  Develop a Color-Based Thresholding Method to Classify Fruits
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("5_fruits.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title("Red Fruit Mask")
plt.show()
