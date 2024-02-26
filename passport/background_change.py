import cv2
import numpy as np
from rembg import remove

class RemoveBackground:

    def __init__(self, input_path = None, output_path = None):
        self.input_path = input_path
        self.output_path = output_path

    def process(self):
        with open(self.input_path, 'rb') as i:
            input_image = i.read()

        # Use remove function to remove the background
        output_image = remove(input_image)

        # Convert the output to a NumPy array
        nparr = np.frombuffer(output_image, np.uint8)

        # Decode the NumPy array to an OpenCV image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Add a white background
        img_with_white_bg = np.ones_like(img) * 255

        # Create a mask of the original image
        mask = (img != 0).all(axis=2)

        # Copy the original image to the new image with a white background
        img_with_white_bg[mask] = img[mask]

        # Save the result to the output path
        cv2.imwrite(self.output_path, img_with_white_bg)
    def __call__(self, image: np.ndarray):
        output_imge = remove(image)
        # Decode the NumPy array to an OpenCV image
        img = cv2.imdecode(output_imge, cv2.IMREAD_COLOR)

        # Add a white background
        img_with_white_bg = np.ones_like(img) * 255

        # Create a mask of the original image
        mask = (img != 0).all(axis=2)

        # Copy the original image to the new image with a white background
        img_with_white_bg[mask] = img[mask]

        return img_with_white_bg



if __name__ == "__main__":
    input_path = 'assets/asset.jpg'
    output_path = 'output.png'

    # Create an instance of RemoveBackground
    background_remover = RemoveBackground(input_path, output_path)

    # Process the image
    background_remover.process()

    print(f"Background removed and white background added. Result saved to {output_path}")
