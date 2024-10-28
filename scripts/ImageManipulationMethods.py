import cv2
import numpy as np

class ImageManipulationMethods:
    def apply_affine_transformation(self, image):
        """
        Applies an affine transformation (random warp) to an image.

        Args:
        - image (numpy.ndarray): The input image in BGR format.

        Returns:
        - numpy.ndarray: The transformed image in BGR format.
        """
        # Get the image dimensions
        rows, cols, ch = image.shape

        # Define the points for affine transformation
        source_points = np.float32([[0, 0], [cols, 0], [0, rows]])
        destination_points = np.float32([[0, 0], [cols + 50, 50], [50, rows - 50]])

        # Get the affine transformation matrix
        transformation_matrix = cv2.getAffineTransform(source_points, destination_points)

        # Apply the affine transformation
        transformed_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))

        return transformed_image

    def seamless_blend(self, background_image, foreground_image):
        """
        Blends the foreground image seamlessly with the background image.

        Args:
        - background_image (numpy.ndarray): The background image in BGR format.
        - foreground_image (numpy.ndarray): The foreground image in BGR format.

        Returns:
        - numpy.ndarray: The blended image in BGR format.
        """
        # Define center and mask for seamless cloning
        center_coordinates = (foreground_image.shape[1] // 2, foreground_image.shape[0] // 2)
        mask = 255 * np.ones(foreground_image.shape, foreground_image.dtype)

        # Perform seamless cloning
        blended_image = cv2.seamlessClone(foreground_image, background_image, mask, center_coordinates, cv2.NORMAL_CLONE)

        return blended_image
