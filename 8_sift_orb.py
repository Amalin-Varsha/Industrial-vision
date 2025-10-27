# Implement SIFT/ORB Feature Matching to Detect Brand Logos/Serial Numbers
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("8_img1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("8_img2.png", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
result = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)

plt.imshow(result), plt.title("Feature Matching")
plt.show()

