import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

# SIREN activation function
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

# Define U-Net architecture with SIREN activation function
class UNetSiren(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSiren, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            Sine()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

# Load and preprocess images
folder_path = r'/mnt/d/Experiment/In_situ_OM/Image_tracking/scripts/movie/tracked'
images = load_images(folder_path)

# Split data into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(np.expand_dims(train_images, axis=1), dtype=torch.float32)
X_val_tensor = torch.tensor(np.expand_dims(val_images, axis=1), dtype=torch.float32)

# Define the model with sigmoid activation
model_sigmoid = UNet(in_channels=1, out_channels=1)

# Define the model with SIREN activation
model_siren = UNetSiren(in_channels=1, out_channels=1)

# Define loss function and optimizer for sigmoid model
criterion_sigmoid = nn.BCELoss()
optimizer_sigmoid = optim.Adam(model_sigmoid.parameters(), lr=0.001)

# Define loss function and optimizer for SIREN model
criterion_siren = nn.MSELoss()
optimizer_siren = optim.Adam(model_siren.parameters(), lr=0.001)

# Training loop for sigmoid model
num_epochs = 10
for epoch in range(num_epochs):
    model_sigmoid.train()
    optimizer_sigmoid.zero_grad()
    outputs_sigmoid = model_sigmoid(X_train_tensor)
    
    # Resize outputs to match the size of target tensor
    outputs_sigmoid_resized = torch.nn.functional.interpolate(outputs_sigmoid, size=X_train_tensor.shape[2:], mode='bilinear', align_corners=True)
    
    loss_sigmoid = criterion_sigmoid(outputs_sigmoid_resized, X_train_tensor)
    loss_sigmoid.backward()
    optimizer_sigmoid.step()
    print(f'Sigmoid Model - Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sigmoid.item():.4f}')

# Training loop for SIREN model
for epoch in range(num_epochs):
    model_siren.train()
    optimizer_siren.zero_grad()
    outputs_siren = model_siren(X_train_tensor)
    
    # Resize outputs to match the size of target tensor
    outputs_siren_resized = torch.nn.functional.interpolate(outputs_siren, size=X_train_tensor.shape[2:], mode='bilinear', align_corners=True)
    
    loss_siren = criterion_siren(outputs_siren_resized, X_train_tensor)
    loss_siren.backward()
    optimizer_siren.step()
    print(f'SIREN Model - Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss_siren.item():.4f}')

# Validation for sigmoid model
model_sigmoid.eval()
val_outputs_sigmoid = model_sigmoid(X_val_tensor)
val_outputs_sigmoid_resized = torch.nn.functional.interpolate(val_outputs_sigmoid, size=X_val_tensor.shape[2:], mode='bilinear', align_corners=True)
val_loss_sigmoid = criterion_sigmoid(val_outputs_sigmoid_resized, X_val_tensor)
print(f'Sigmoid Model - Validation Loss: {val_loss_sigmoid.item():.4f}')

# Validation for SIREN model
model_siren.eval()
val_outputs_siren = model_siren(X_val_tensor)
val_outputs_siren_resized = torch.nn.functional.interpolate(val_outputs_siren, size=X_val_tensor.shape[2:], mode='bilinear', align_corners=True)
val_loss_siren = criterion_siren(val_outputs_siren_resized, X_val_tensor)
print(f'SIREN Model - Validation Loss: {val_loss_siren.item():.4f}')

# Save predicted and ground truth validation images for comparison for sigmoid model
for i, (pred_sigmoid, gt_sigmoid) in enumerate(zip(val_outputs_sigmoid, X_val_tensor)):
    pred_img_sigmoid = (pred_sigmoid.squeeze().detach().numpy() * 255).astype(np.uint8)
    gt_img_sigmoid = (gt_sigmoid.squeeze().detach().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f'sigmoid_val_image_{i+1}_predicted.png', pred_img_sigmoid)
    cv2.imwrite(f'sigmoid_val_image_{i+1}_ground_truth.png', gt_img_sigmoid)

# Save predicted and ground truth validation images for comparison for SIREN model
for i, (pred_siren, gt_siren) in enumerate(zip(val_outputs_siren, X_val_tensor)):
    pred_img_siren = (pred_siren.squeeze().detach().numpy() * 255).astype(np.uint8)
    gt_img_siren = (gt_siren.squeeze().detach().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f'siren_val_image_{i+1}_predicted.png', pred_img_siren)
    cv2.imwrite(f'siren_val_image_{i+1}_ground_truth.png', gt_img_siren)
