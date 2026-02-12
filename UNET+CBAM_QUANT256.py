import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.amp import autocast, GradScaler  # Updated AMP usage

# Quantum-inspired activation function
def quantum_circuit(inputs):
    return inputs * torch.sigmoid(inputs)

# Quantum Layer
class QuantumLayer(nn.Module):
    def __init__(self, in_channels):
        super(QuantumLayer, self).__init__()
        self.fc_adjust = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return self.fc_adjust(quantum_circuit(x))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, in_channels):
        super(TransformerBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = self.conv1(x)
        x = x.view(batch, channels, -1).transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(attn_out + x)
        x = self.norm2(self.fc(F.relu(x)) + x)
        x = x.transpose(1, 2).view(batch, channels, height, width)
        return x

# CBAM Block
class CBAMBlock(nn.Module):
    def __init__(self, in_channels):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 3, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_attention(x) * self.spatial_attention(x)

# Low-Light Enhancement Model
class LowLightEnhancementModel(nn.Module):
    def __init__(self, in_channels):
        super(LowLightEnhancementModel, self).__init__()
        self.transformer1 = TransformerBlock(in_channels)
        self.transformer2 = TransformerBlock(in_channels)
        self.cbam = CBAMBlock(in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.cbam(x)
        x = self.final_conv(x)
        return self.activation(x)

# Full Model with Quantum Layer
class LowLightEnhancementWithQuantum(nn.Module):
    def __init__(self, in_channels):
        super(LowLightEnhancementWithQuantum, self).__init__()
        self.low_light_model = LowLightEnhancementModel(in_channels)
        self.quantum_layer = QuantumLayer(in_channels)

    def forward(self, x):
        x = self.low_light_model(x)
        return self.quantum_layer(x)

# Dataset Class
class LowLightDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

# Training Function
import csv

def train_model(model, train_loader, criterion, optimizer, device, epochs=2000, checkpoint_dir="checkpoints", csv_path="log.csv"):
    model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Ensure the CSV file exists with headers
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "LR", "MAE", "MSE"])  # Column headers
    
    scaler = GradScaler("cuda")  # Updated AMP usage

    for epoch in range(epochs):
        epoch_loss, total_mae, total_mse, total_samples = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for input_img, target_img in train_loader:
            input_img, target_img = input_img.to(device), target_img.to(device)
            with autocast("cuda"):  # Updated AMP usage
                output = model(input_img)
                loss = criterion(output, target_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            total_mae += mean_absolute_error(target_img.cpu().numpy().flatten(), output.cpu().detach().numpy().flatten())
            total_mse += mean_squared_error(target_img.cpu().numpy().flatten(), output.cpu().detach().numpy().flatten())

        avg_loss = epoch_loss / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        avg_mse = total_mse / len(train_loader)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, MAE={avg_mae:.6f}, MSE={avg_mse:.6f}")

        # ✅ Append log data to CSV file
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, optimizer.param_groups[0]['lr'], avg_mae, avg_mse])

        # ✅ Save Full Model Every 5 Epochs
        if (epoch+1) % 5 == 0:
            model_save_path = os.path.join(checkpoint_dir, f"full_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)  # Save only model weights
            print(f"✅ Model saved: {model_save_path}")


# Function to Load the Full Model
def load_full_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model



# # Model Initialization
# device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Ensure input size matches model expectations
#     transforms.ToTensor()
# ])

# train_loader = DataLoader(LowLightDataset(input_dir, target_dir, transform), batch_size=4, shuffle=True)

# model = LowLightEnhancementWithQuantum(3).to(device)

# # Train model
# train_model(model, train_loader, nn.MSELoss(), optim.Adam(model.parameters(), lr=1e-4), device, checkpoint_dir=checkpoint_dir, csv_path=csv_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

# Define dataset paths
    input_dir = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\low'
    target_dir = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\high'
    checkpoint_dir = r'D:\ARTI_PHD_WORK\5thexperiment\CHEKCPOINT8'
    csv_path = r'D:\ARTI_PHD_WORK\5thexperiment\CHECKPOINT8\LOGFILE\log.csv'
    train_loader = DataLoader(
        LowLightDataset(input_dir, target_dir, transform),
        batch_size=4,
        shuffle=True,
        num_workers=4,     # ← this line needs `if __name__ == '__main__'` on Windows
        pin_memory=True
    )

    model = LowLightEnhancementWithQuantum(3)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel for multi-GPU training")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, criterion, optimizer, device,
                checkpoint_dir=checkpoint_dir, csv_path=csv_path)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Optional, safe to include
    main()

