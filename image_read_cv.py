import cv2

img = cv2.imread('image.jpg')           # Read image (BGR format)
cv2.imshow('OpenCV Image', img)         # Display image
cv2.waitKey(0)
cv2.destroyAllWindows()
