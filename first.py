import os
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image

# Set up directories and files
images_dir = 'C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/imagesearchengine'
output_vec_file = "vec_all.npy"
output_name_file = "name_all.npy"
root = 'C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/imagesearchengine'

# Load the pre-trained model (ResNet-18)
model = models.resnet18(weights="DEFAULT")
model.eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
])

# Activation hook for extracting features
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

# Initialize lists to store vectors and filenames
name_all = []
vec_all = None

# List all image files in the directory
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
with torch.no_grad():
    for i, file in enumerate(image_files):
        try:
            # Open the image
            img_path = os.path.join(images_dir, file)
            img = Image.open(img_path).convert("RGB")
            
            # Apply transformation
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            # Pass through the model
            out = model(img_tensor)
            
            # Extract the feature vector
            vec = activation["avgpool"].numpy().squeeze()[None, ...]
            
            # Stack the feature vectors
            if vec_all is None:
                vec_all = vec
            else:
                vec_all = np.vstack([vec_all, vec])
            
            # Store the filename
            name_all.append(file)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
        
        # Print progress every 100 images
        if i % 100 == 0 and i != 0:
            print(f"{i} images processed")
    
    # Save the feature vectors and names to .npy files
    np.save(output_vec_file, vec_all)
    np.save(output_name_file, name_all)
    print(f"Saved feature vectors to {output_vec_file} and image names to {output_name_file}")
