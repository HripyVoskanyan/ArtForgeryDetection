import os
import random
import shutil
import cv2
from NoiseMethods import NoiseMethods
from GANMethods import GANMethods
from StyleTransferMethods import StyleTransferMethods
from AdversarialAttackMethods import AdversarialAttackMethods
from ImageManipulationMethods import ImageManipulationMethods

def select_random_images(source_folder, destination_folder, num_images=10000):
    all_images = [img for img in os.listdir(source_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(all_images, num_images)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image in selected_images:
        shutil.copy(os.path.join(source_folder, image), os.path.join(destination_folder, image))

    return selected_images

def poison_images(image_folder, poisoned_folder, num_images=100, methods=None):
    """
    Poisons images using the specified methods (supports multiple methods).

    Args:
        image_folder (str): Path to the folder with original images.
        poisoned_folder (str): Path to save the poisoned images.
        num_images (int): Number of images to poison.
        methods (dict): Dictionary of method names and functions to apply.
    """
    if methods is None:
        raise ValueError("Please provide a dictionary of methods to apply.")

    if not os.path.exists(poisoned_folder):
        os.makedirs(poisoned_folder)

    # List all images in the image_folder
    all_images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Select a random subset of images to poison
    selected_images = random.sample(all_images, num_images)

    for img_name in selected_images:
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        # Choose a random method to apply from the provided dictionary
        method_name, method_function = random.choice(list(methods.items()))

        # Apply the selected method
        poisoned_img = method_function(img)

        # Save the poisoned image
        poisoned_img_path = os.path.join(poisoned_folder, f"{method_name}_{img_name}")
        cv2.imwrite(poisoned_img_path, poisoned_img)
        print(f"Applied {method_name} to {img_name} and saved to {poisoned_folder}")


# Example usage:
source_folder = '../data/originals'
destination_folder = '../data/subset'
poisoned_folder = '../data/poisoned'
#selected_images = select_random_images(source_folder, destination_folder, num_images=100)
method = NoiseMethods()
methods = {'gaussian_noise': method.add_gaussian_noise,
           'salt_and_pepper': method.add_salt_and_pepper_noise
           }
#poison_images(destination_folder, poisoned_folder, num_images=10, methods=methods)
#ganmethod = GANMethods()
#ganmethods = {'cyclegan': ganmethod.apply_cyclegan_transformation}
#poison_images(destination_folder, poisoned_folder, num_images=1, methods=ganmethods)

#style_transfer = StyleTransferMethods()
# all_images = [img for img in os.listdir(destination_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
# content_image, style_image = random.sample(all_images, 2)
#
# # Ensure content and style images are different
# while content_image == style_image:
#     content_image, style_image = random.sample(all_images, 2)

# content_image_path = os.path.join(destination_folder, content_image)
# style_image_path = os.path.join(destination_folder, style_image)
# output_image_path = os.path.join(poisoned_folder, 'style_transferred_image.jpg')
#
# style_transfer.apply_style_transfer(
#     content_image_path=content_image_path,
#     style_image_path=style_image_path,
#     output_image_path=output_image_path,
#     num_steps=500
# )
#style_transfer_methods = {'color pallette change': style_transfer.enhance_image_color}
#poison_images(destination_folder, poisoned_folder, num_images=1, methods=style_transfer_methods)

#adversarial_methods = AdversarialAttackMethods()
#methods = {'fgsm_attack': adversarial_methods.fgsm_attack}
#poison_images(destination_folder, poisoned_folder, num_images=1, methods=methods)

