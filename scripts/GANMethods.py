import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class GANMethods:
    def __init__(self):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pretrained CycleGAN model (if available)
        self.cyclegan_model = self.load_cyclegan_model('path/to/cyclegan/model.pth')

    def load_cyclegan_model(self, model_path):
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Could not load CycleGAN model: {e}")
            return None

    def apply_cyclegan_transformation(self, image):
        if self.cyclegan_model is None:
            print("CycleGAN model is not loaded. Skipping transformation.")
            return image

        # Convert the image from BGR to RGB and to PIL format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply the necessary transforms for the model
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize for CycleGAN model compatibility
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        # Generate the transformed image using the CycleGAN model
        with torch.no_grad():
            transformed_tensor = self.cyclegan_model(input_tensor)

        # Post-process the output tensor to convert it back to an image
        transformed_image = transformed_tensor.squeeze(0).cpu().detach().numpy()
        transformed_image = (transformed_image * 0.5 + 0.5) * 255  # Unnormalize
        transformed_image = transformed_image.transpose(1, 2, 0).astype(np.uint8)

        # Convert back to BGR for OpenCV compatibility
        transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        return transformed_image_bgr
