import cv2
import numpy as np

class NoiseMethods:
    def __init__(self):
        pass

    def add_gaussian_noise(self, image):
        """
        Adds Gaussian noise to the image.
        Args:
            image (numpy.ndarray): Input image (BGR format).
        Returns:
            numpy.ndarray: Image with Gaussian noise added.
        """
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def add_salt_and_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """
        Adds Salt and Pepper noise to the image.
        Args:
            image (numpy.ndarray): Input image (BGR format).
            salt_prob (float): Probability of salt noise (white pixels).
            pepper_prob (float): Probability of pepper noise (black pixels).
        Returns:
            numpy.ndarray: Image with Salt and Pepper noise added.
        """
        noisy_image = np.copy(image)

        # Add salt noise (white pixels)
        num_salt = np.ceil(salt_prob * image.shape[0] * image.shape[1])
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy_image[coords_salt[0], coords_salt[1]] = 255

        # Add pepper noise (black pixels)
        num_pepper = np.ceil(pepper_prob * image.shape[0] * image.shape[1])
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0

        return noisy_image
