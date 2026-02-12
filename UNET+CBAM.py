import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

# Check available GPUs and set GPU 1 explicitly
if torch.cuda.device_count() > 1:
    device = torch.device("cuda:1")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print GPU details
if device.type == "cuda":
    print(f"Using GPU: {device} - {torch.cuda.get_device_name(device)}")
else:
    print("Using CPU")

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, input_path, target_path, transform=None):
        self.input_files = sorted(os.listdir(input_path))
        self.target_files = sorted(os.listdir(target_path))
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_image = read_image(os.path.join(self.input_path, self.input_files[idx])).float() / 255.0
        target_image = read_image(os.path.join(self.target_path, self.target_files[idx])).float() / 255.0
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

# Define paths
input_path = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\low'
target_path = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\high'
csv_path = r"D:\ARTI_PHD_WORK\5thexperiment\csv\log2.csv"
checkpoint_dir = r'D:\ARTI_PHD_WORK\5thexperiment\checkpoint6'

# Ensure directories exist
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize log file if not present
if not os.path.exists(csv_path):
    test_df = pd.DataFrame(columns=["Epoch", "Loss", "MSE", "MAE"])
    test_df.to_csv(csv_path, mode='w', index=False, header=True)

# Define transformations
transform = transforms.Resize((256, 256))

# Load dataset
dataset = CustomDataset(input_path, target_path, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# === CBAM (Convolutional Block Attention Module) ===
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)  # Fixed max operation
        out = self.fc2(self.relu(self.fc1(avg_out + max_out)))
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)  # Fixed max operation
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

# === Modified U-Net Model with CBAM ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        return self.cbam(self.conv(x))

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up4 = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        d4 = self.up4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)

# Initialize model and move to device
model = UNet().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2000
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Compute metrics
    mse = total_loss / len(dataloader)
    mae = mse ** 0.5  # Approximate MAE

    # Save logs
    log_data = pd.DataFrame([[epoch, total_loss, mse, mae]], columns=["Epoch", "Loss", "MSE", "MAE"])
    log_data.to_csv(csv_path, mode='a', header=False, index=False)

    # Save checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_filename)

    print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss:.4f} - MSE: {mse:.4f} - MAE: {mae:.4f}")
