from skimage import io
import matplotlib.pyplot as plt

img = io.imread('image.jpg')            # Read image
plt.imshow(img)
plt.axis('off')
plt.title('scikit-image Image')
plt.show()
