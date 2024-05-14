import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images(folder_path, target_shape=(1173, 1614)):
    """
    Load and preprocess images from the specified folder.
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_shape)  # Resize image to target shape
        img = img / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Define the folder path where your images are stored
folder_path = r'/mnt/d/Experiment/In_situ_OM/Image_tracking/scripts/movie/tracked'

# Load and preprocess images
images = load_images(folder_path)

# Ensure that the directory for saving figures exists
save_dir = 'Imgs'
os.makedirs(save_dir, exist_ok=True)

class SineLayer(nn.Module):
    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, hidden_dim), SineLayer(w0),
                                 nn.Linear(hidden_dim, out_dim))

        # Init weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1. / in_dim, 1. / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                        np.sqrt(6. / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1):
        super(MLP, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)

def train_model(model, model_optimizer, pixel_coordinates, pixel_values, nb_epochs=2):
    """
    Train the model on the given data.
    """
    psnr = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)

        # Resize model output tensor to match the size of pixel values tensor
        model_output_resized = model_output[:pixel_values.size(0)]

        loss = ((model_output_resized - pixel_values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output

def sequential_train(images, model, model_optimizer, save_name, nb_epochs=2):
    """
    Train the model sequentially on the image sequence and save the trained model.
    """
    all_psnr = []
    for i in range(len(images) - 2):  # Train on all images except the last two
        img = images[i]

        # Convert image to torch tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        pixel_values = img_tensor.view(-1, 1)

        # Input
        resolution = max(img.shape)
        tmp = torch.linspace(-1, 1, steps=resolution)
        x, y = torch.meshgrid(tmp, tmp)
        pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

        # Train the model
        psnr, _ = train_model(model, model_optimizer, pixel_coordinates, pixel_values, nb_epochs)
        all_psnr.append(psnr)

    # Save the trained model
    torch.save(model.state_dict(), save_name)

    return all_psnr

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    siren = Siren().to(device)
    mlp = MLP().to(device)

    # Ensure that the directory for saving figures exists
    os.makedirs(save_dir, exist_ok=True)

    # Sequentially train SIREN model
    siren_optimizer = torch.optim.Adam(lr=1e-4, params=siren.parameters())
    siren_psnr = sequential_train(images, siren, siren_optimizer, os.path.join(save_dir, 'siren_model.pth'))

    # Sequentially train ReLU model
    mlp_optimizer = torch.optim.Adam(lr=1e-4, params=mlp.parameters())
    mlp_psnr = sequential_train(images, mlp, mlp_optimizer, os.path.join(save_dir, 'mlp_model.pth'))

    # Plot PSNR curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(siren_psnr, axis=0), label='SIREN', c='C0')
    plt.plot(np.mean(mlp_psnr, axis=0), label='ReLU', c='mediumseagreen')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('PSNR', fontsize=14)
    plt.legend(fontsize=13)
    plt.savefig(os.path.join(save_dir, 'psnr_curves.png'))
    plt.show()
