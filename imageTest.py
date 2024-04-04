import matplotlib.pyplot as plt
import torch
from model import SimplifiedCSRNet
from torchvision import transforms
from PIL import Image

# Function to load the model
def load_model(model_path):
    model = SimplifiedCSRNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict the density map for a new image
def predict_density_map(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),  # Resize if necessary to match training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
    density_map = output.squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions
    
    return image, density_map

# Function to visualize the original image and its density map
def visualize_density_map(image, density_map):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title('Density Map')

    plt.show()

    # Print the estimated total count
    count = density_map.sum()
    print(f'Estimated count: {count}')

model = load_model('model_final.pth')

# Predict the density map for a new image
image_path = 'test1.jpg'  # Update this path
image, density_map = predict_density_map(model, image_path)

# Visualize the original image and its density map
visualize_density_map(image, density_map)
# Before plotting the density map
print("Density map raw values:", density_map.flatten()[:10])  # Print first 10 values
