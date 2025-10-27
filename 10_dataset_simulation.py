#  10. Simulate Industrial Dataset under Different Lighting & Lens Parameters

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("10_image.png")  # replace with your dataset image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Different lighting conditions
bright = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=50)  # brighter
dim = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=-30)    # dimmer
directional = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=0)  # high contrast
uneven = img_rgb.copy()
uneven[:, :uneven.shape[1]//2] = cv2.convertScaleAbs(uneven[:, :uneven.shape[1]//2], alpha=0.5, beta=-50)

# Different lens parameters
blurred = cv2.GaussianBlur(img_rgb, (11, 11), 10)  # simulate defocus
sharpened = cv2.filter2D(img_rgb, -1, kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))  # sharpen
exposed = cv2.convertScaleAbs(img_rgb, alpha=2, beta=100)  # long exposure effect

# Display
titles = [
    "Original", "Bright", "Dim", "Directional Light", "Uneven Light",
    "Blurred (Lens Defocus)", "Sharpened (Lens)", "Overexposed"
]
images = [img_rgb, bright, dim, directional, uneven, blurred, sharpened, exposed]

plt.figure(figsize=(14, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")
plt.show()
