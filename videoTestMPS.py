import cv2
import torch
import numpy as np
from model import MSPSNet
from torchvision import transforms
import matplotlib.pyplot as plt
import pathlib

def count_crowd_on_video(video_path, model_path):
    # Determine if MPS is available and set the device accordingly
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(video_path)
    model = MSPSNet().to(device)
    # Ensure the model checkpoint is loaded to the right device
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Move tensor to the appropriate device

        with torch.no_grad():
            predictions0, _, _, _ = model(img_tensor)

        pred_density = predictions0.squeeze().cpu().numpy()
        pred_density = cv2.resize(pred_density, (frame.shape[1], frame.shape[0]))
        crowd_count = torch.sum(predictions0).item()

        threshold = 0.1
        people_map = (pred_density > threshold).astype(np.uint8)

        contours, _ = cv2.findContours(people_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        cv2.putText(frame, f"Crowd Count: {crowd_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Crowd Counting', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'video.mp4'
    model_path = 'checkpoint1.pth.tar'
    count_crowd_on_video(video_path, model_path)
