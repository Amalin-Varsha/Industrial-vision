import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('image.jpg')         # Read image
plt.imshow(img)
plt.axis('off')
plt.title('matplotlib Image')
plt.show()