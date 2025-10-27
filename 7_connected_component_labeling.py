# Apply Connected Component Labeling to Count Defective vs. Good Solder Joints on PCB Images
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("7_pcb.jpeg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
good_joints = 0
defective_joints = 0
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 50:  
        good_joints += 1
    else:
        defective_joints += 1

print(f"Good Joints: {good_joints}, Defective Joints: {defective_joints}")

