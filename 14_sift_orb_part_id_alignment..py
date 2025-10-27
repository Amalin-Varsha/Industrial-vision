# Extract SIFT/ORB Features for Part Identification or Alignment Tasks
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("14_screw.png", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(image, None)
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(image, None)

sift_img = cv2.drawKeypoints(image, kp, None)
orb_img = cv2.drawKeypoints(image, kp_orb, None)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(sift_img), plt.title("SIFT Features")
plt.subplot(1, 2, 2), plt.imshow(orb_img), plt.title("ORB Features")
plt.show()

