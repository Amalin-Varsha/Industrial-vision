import imageio.v2 as imageio
import matplotlib.pyplot as plt

img = imageio.imread('image.jpg')       # Read image
plt.imshow(img)                         # Display using matplotlib
plt.axis('off')
plt.title('imageio Image')
plt.show()