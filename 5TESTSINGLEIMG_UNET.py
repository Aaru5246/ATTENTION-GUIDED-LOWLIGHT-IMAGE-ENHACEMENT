
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import matplotlib.pyplot as plt
import lpips
import numpy as np
import cv2

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Define the model (U-Net)
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
checkpoint_path = r"D:\ARTI_PHD_WORK\5thexperiment\checkpoint5\unet_epoch_2000.pth"  # Change this path
model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load input and ground truth images
input_image_path = r"D:\ARTI_PHD_WORK\5thexperiment\5thexp_dataset\dataset\LOL-v2\Real_captured\Test\Low\low00690.png"
groundtruth_image_path = r"D:\ARTI_PHD_WORK\5thexperiment\5thexp_dataset\dataset\LOL-v2\Real_captured\Test\Normal\normal00690.png"

input_image = read_image(input_image_path).float() / 255.0  # Normalize to [0, 1]
groundtruth_image = read_image(groundtruth_image_path).float() / 255.0

# Ensure both images have the same shape
input_image = TF.resize(input_image, [256, 256])
groundtruth_image = TF.resize(groundtruth_image, [256, 256])

# Add batch dimension and send to device
input_image = input_image.unsqueeze(0).to(device)  # Shape: [1, 3, 256, 256]
groundtruth_image = groundtruth_image.unsqueeze(0).to(device)

# Run model inference
with torch.no_grad():
    enhanced_image = model(input_image)

# Ensure output is in the correct range
enhanced_image = torch.clamp(enhanced_image, 0, 1)

# Convert tensors to numpy arrays for visualization
input_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
groundtruth_np = groundtruth_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Compute PSNR, SSIM, and LPIPS
psnr_value = psnr(groundtruth_np, enhanced_np, data_range=1)
# ssim_value = ssim(groundtruth_np, enhanced_np, data_range=1, multichannel=True)


ssim_value = ssim(groundtruth_np, enhanced_np, data_range=1, multichannel=True, win_size=3)

# LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_value = lpips_model(groundtruth_image, enhanced_image).item()

print(f"PSNR: {psnr_value:.4f}")
print(f"SSIM: {ssim_value:.4f}")
print(f"LPIPS: {lpips_value:.4f}")

# Plot the images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(input_np)
ax[0].set_title("Input Image")
ax[0].axis("off")

ax[1].imshow(enhanced_np)
ax[1].set_title("Enhanced Image")
ax[1].axis("off")

ax[2].imshow(groundtruth_np)
ax[2].set_title("Ground Truth")
ax[2].axis("off")

plt.show()