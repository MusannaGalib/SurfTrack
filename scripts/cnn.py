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

# Load and preprocess images
folder_path = r'/mnt/d/Experiment/In_situ_OM/Image_tracking/scripts/movie/tracked'
images = load_images(folder_path)

# Split data into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(np.expand_dims(train_images, axis=1), dtype=torch.float32)
X_val_tensor = torch.tensor(np.expand_dims(val_images, axis=1), dtype=torch.float32)

# Define the model
model = UNet(in_channels=1, out_channels=1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    
    # Resize outputs to match the size of target tensor
    outputs_resized = torch.nn.functional.interpolate(outputs, size=X_train_tensor.shape[2:], mode='bilinear', align_corners=True)
    
    loss = criterion(outputs_resized, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Validation
model.eval()
val_outputs = model(X_val_tensor)

# Resize model's output to match the size of the validation data
val_outputs_resized = torch.nn.functional.interpolate(val_outputs, size=X_val_tensor.shape[2:], mode='bilinear', align_corners=True)

# Compute validation loss
val_loss = criterion(val_outputs_resized, X_val_tensor)
print(f'Validation Loss: {val_loss.item():.4f}')


# Save predicted and ground truth validation images for comparison
for i, (pred, gt) in enumerate(zip(val_outputs, X_val_tensor)):
    pred_img = (pred.squeeze().detach().numpy() * 255).astype(np.uint8)
    gt_img = (gt.squeeze().detach().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f'val_image_{i+1}_predicted.png', pred_img)
    cv2.imwrite(f'val_image_{i+1}_ground_truth.png', gt_img)
