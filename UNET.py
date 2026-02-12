import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU")

# Define Custom Dataset
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

# Define Paths
input_path = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\low'
target_path = r'D:\ARTI_PHD_WORK\5thexp_dataset\dataset\lol_dataset\our485\high'
csv_path = r"D:\ARTI_PHD_WORK\5thexperiment\log_file\log.csv"
checkpoint_path = r'D:\ARTI_PHD_WORK\5thexperiment\checkpoint5\unet_epoch_{epoch}.pth'

# Ensure Directories Exist
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Check Directory Existence and Write Permissions
print(f"CSV Directory Exists: {os.path.exists(os.path.dirname(csv_path))}")
print(f"Checkpoint Directory Exists: {os.path.exists(os.path.dirname(checkpoint_path))}")

# Define Transformations
transform = transforms.Resize((256, 256))

# Load Dataset
dataset = CustomDataset(input_path, target_path, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define U-Net Model
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
    
    def forward(self, x):
        return self.conv(x)

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
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # Output 3 channels (RGB)
    
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
        
        out = self.final_conv(d1)
        return out

# Initialize Model and Move to Device
model = UNet().to(device)

# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 2000
metrics = []

# Check if CSV exists
csv_exists = os.path.exists(csv_path)

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss, epoch_mse, epoch_mae = 0.0, 0.0, 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        mse = nn.MSELoss()(outputs, targets)
        mae = nn.L1Loss()(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_mae += mae.item()
    
    avg_loss = epoch_loss / len(dataloader)
    avg_mse = epoch_mse / len(dataloader)
    avg_mae = epoch_mae / len(dataloader)
    
    print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")
    metrics.append([epoch, avg_loss, avg_mse, avg_mae])

    # Append results to CSV
    log_df = pd.DataFrame([[epoch, avg_loss, avg_mse, avg_mae]], columns=["Epoch", "Loss", "MSE", "MAE"])
    log_df.to_csv(csv_path, mode='a', index=False, header=not csv_exists)
    csv_exists = True  # Ensure header is only written once

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_filename = checkpoint_path.format(epoch=epoch)
        torch.save(model.state_dict(), checkpoint_filename)
        print(f"✅ Checkpoint saved: {checkpoint_filename}")

print("✅ Training completed successfully!")
