import cv2
import torch
import numpy as np
from model import MSPSNet
from torchvision import transforms
import matplotlib.pyplot as plt
import pathlib

def count_crowd(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions0, _, _, _ = model(img_tensor)

    pred_density = predictions0.squeeze().cpu().numpy()
    pred_density = cv2.resize(pred_density, (img.shape[1], img.shape[0]))
    crowd_count = torch.sum(predictions0).item()

    threshold = 0.1
    people_map = (pred_density > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(people_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img_rgb, center, radius, (0, 255, 0), 2)

    # Visualize the image with circles drawn around heads
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Crowd Count: {crowd_count}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_density, cmap='jet')
    plt.title('Density Map')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = MSPSNet().to(device)
    checkpoint = torch.load('checkpoint1.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_path = 'testImage.jpg'
    print("we used device", device)
    count_crowd(image_path, model)
