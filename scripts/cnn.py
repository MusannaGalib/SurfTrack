import os  # Importing the os module for file operations
import cv2  # Importing OpenCV for image processing
import torch  # Importing PyTorch for deep learning
import torch.nn as nn  # Importing PyTorch's neural network module
import torch.optim as optim  # Importing PyTorch's optimization module
import numpy as np  # Importing NumPy for numerical operations

# Function to load and preprocess images
def load_images(folder_path):
    """
    Load and preprocess images from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
    
    Returns:
        np.ndarray: Array of preprocessed images.
    """
    images = []  # List to store preprocessed images
    # Iterating over each file in the folder
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)  # Constructing image path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Reading image in grayscale
        images.append(img)  # Appending preprocessed image to the list
    return np.array(images)  # Converting list to NumPy array and returning

# Define U-Net architecture
class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # 1 input channel, 64 output channels
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 input channels, 64 output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 input channels, 128 output channels
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),  # 128 input channels, out_channels output channels
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def forward(self, x):
        x1 = self.encoder(x)  # Encoder forward pass
        x2 = self.decoder(x1)  # Decoder forward pass
        return x2

# Load and preprocess images
folder_path = r'/mnt/d/Experiment/In_situ_OM/Image_tracking/scripts/movie/tracked'
images = load_images(folder_path)

# Normalize pixel values to range [0, 1]
images_normalized = images / 255.0

# Prepare input data
input_data = np.expand_dims(images_normalized, axis=1)  # Add channel dimension

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(input_data, dtype=torch.float32)

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
    
    # Resize outputs to match the size of input data
    outputs_resized = torch.nn.functional.interpolate(outputs, size=input_data.shape[2:], mode='bilinear', align_corners=True)
    
    loss = criterion(outputs_resized, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Make predictions and track variables for different j values
# Removed the loop for j_values as per your request

# Prepare input data for j value
# Assuming input data preparation for different j values involves some specific logic

# Make predictions using the trained model
# Assuming model.predict() method returns predictions

predictions = model.predict(input_data)  # Example

# Track variables (eta, pot, w) at different times
# Assuming tracking logic involves processing predictions

# Example tracking logic
for t, pred in zip(time_steps, predictions):
    # Process prediction at time step t
    # Example:
    eta_t = pred[..., 0]  # Extract eta
    pot_t = pred[..., 1]  # Extract pot
    w_t = pred[..., 2]  # Extract w
    # Process and track variables as needed
