import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to load and preprocess images
def load_images(folder_path):
    """
    Load and preprocess images from the specified folder.
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Load and preprocess images
folder_path = r'/mnt/d/Experiment/In_situ_OM/Image_tracking/scripts/movie/tracked'
images = load_images(folder_path)

# Split data into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(np.expand_dims(train_images, axis=1), dtype=torch.float32)
X_val_tensor = torch.tensor(np.expand_dims(val_images, axis=1), dtype=torch.float32)

# Define U-Net architecture
class UNet(nn.Module):
    """
    Define U-Net architecture.
    """
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

# Define Siren architecture
class Siren(nn.Module):
    """
    Define Siren architecture.
    """
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim, out_dim)
        )

        with torch.no_grad():
            for i in [0, 2, 4, 6, 8]:
                nn.init.uniform_(self.net[i].weight, -1. / in_dim if i == 0 else -np.sqrt(6. / hidden_dim) / w0,
                                 1. / in_dim if i == 0 else np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)

# Define SineLayer
class SineLayer(nn.Module):
    """
    Sine layer for Siren architecture.
    """
    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

# Define loss function
criterion = nn.BCELoss()

# Define U-Net model
model = UNet(in_channels=1, out_channels=1)

# Define Siren model
siren = Siren()

# Define optimizer for U-Net
optimizer_unet = optim.Adam(model.parameters(), lr=0.001)

# Define optimizer for Siren
optimizer_siren = optim.Adam(siren.parameters(), lr=0.001)

# Define number of epochs
num_epochs = 10

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors to device
X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)

# Training loop for U-Net
for epoch in range(num_epochs):
    model.train()
    optimizer_unet.zero_grad()
    outputs = model(X_train_tensor)
    
    # Resize outputs to match the size of target tensor
    outputs_resized = torch.nn.functional.interpolate(outputs, size=X_train_tensor.shape[2:], mode='bilinear', align_corners=True)
    
    loss = criterion(outputs_resized, X_train_tensor)
    loss.backward()
    optimizer_unet.step()
    print(f'U-Net - Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Define Siren model
resolution = 256
tmp = torch.linspace(-1, 1, steps=resolution)
x, y = torch.meshgrid(tmp, tmp)
pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

# Training loop for Siren
for epoch in range(num_epochs):
    siren.train()
    optimizer_siren.zero_grad()
    model_output = siren(pixel_coordinates)
    
    # Generate ground truth pixel values for validation
    img = ((torch.from_numpy(images[0]) - 127.5) / 127.5)
    pixel_values = img.reshape(-1, 1).to(device)

    # Compute loss
    loss_siren = ((model_output - pixel_values) ** 2).mean()
    loss_siren.backward()
    optimizer_siren.step()
    print(f'Siren - Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss_siren.item():.4f}')







# Validation for U-Net
model.eval()
val_outputs = model(X_val_tensor)
val_outputs_resized = torch.nn.functional.interpolate(val_outputs, size=X_val_tensor.shape[2:], mode='bilinear', align_corners=True)
val_loss = criterion(val_outputs_resized, X_val_tensor)
print(f'U-Net - Validation Loss: {val_loss.item():.4f}')

# Validation for Siren
siren.eval()
val_outputs_siren = siren(pixel_coordinates)

# Generate ground truth pixel values for validation
img = ((torch.from_numpy(images[0]) - 127.5) / 127.5)
pixel_values = img.reshape(-1, 1).to(device)

# Compute validation loss
val_loss_siren = ((val_outputs_siren - pixel_values) ** 2).mean()
print(f'Siren - Validation Loss: {val_loss_siren.item():.4f}')


# Save predicted and ground truth validation images for U-Net
for i, (pred, gt) in enumerate(zip(val_outputs, X_val_tensor)):
    pred_img = (pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    gt_img = (gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f'val_image_{i+1}_predicted_unet.png', pred_img)
    cv2.imwrite(f'val_image_{i+1}_ground_truth_unet.png', gt_img)

# Save predicted and ground truth validation images for Siren
for i, (pred_siren, gt_siren) in enumerate(zip(val_outputs_siren, pixel_values)):
    pred_img_siren = (pred_siren.cpu().view(resolution, resolution).detach().numpy() * 255).astype(np.uint8)
    gt_img_siren = (gt_siren.cpu().view(resolution, resolution).detach().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f'val_image_{i+1}_predicted_siren.png', pred_img_siren)
    cv2.imwrite(f'val_image_{i+1}_ground_truth_siren.png', gt_img_siren)
