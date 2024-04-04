import torch
from torch.utils.data import DataLoader
from dataset import ShanghaiTechDataset
from model import SimplifiedCSRNet
import torch.optim as optim
import torch.nn as nn
import time

def train_model(model, train_loader, device, epochs=2):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(f"Starting training on device {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Optional: Print loss every N mini-batches
            if (batch_idx + 1) % 100 == 0:  # Assuming you want to log every 100 mini-batches
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")

        epoch_duration = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Epoch summary
        print(f'Epoch: {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, Duration: {epoch_duration:.2f} sec')

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'model_final.pth')
    print("Model saved as model_final.pth")

def main():
    device = torch.device("mps")  # Use MPS for Mac M1/M2 GPUs
    print(f"Using device: {device}")
    train_dataset = ShanghaiTechDataset(root='ShanghaiTech', part='part_B', phase='train', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model = SimplifiedCSRNet()
    train_model(model, train_loader, device)
    torch.save(model.state_dict(), 'jay.pth')

if __name__ == '__main__':
    main()
