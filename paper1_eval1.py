# sharpen images to emphasize defect boundaries

import cv2
import numpy as np

def sharpen_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_path}")

    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    sharpened = cv2.filter2D(image, -1, sharpening_kernel)

    cv2.imwrite(output_path, sharpened)

    return sharpened

if __name__ == "__main__":
    input_image = "image.jpg"   
    output_image = "output.jpg"
    sharpened_img = sharpen_image(input_image, output_image)
    print(f"Sharpened image saved to {output_image}")