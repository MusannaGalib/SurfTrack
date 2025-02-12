import os
import cv2
import torch
import torch.nn as nn
import skimage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def train(model, model_optimizer, pixel_coordinates, pixel_values, nb_epochs=15000):
    psnr = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)

        # Resize model output tensor to match the size of pixel values tensor
        model_output_resized = model_output[:pixel_values.size(0)]

        # Print sizes of tensors for debugging
        print("Size of model_output_resized:", model_output_resized.size())
        print("Size of pixel_values:", pixel_values.size())

        loss = ((model_output_resized - pixel_values) ** 2).mean()
        print("Loss:", loss.item())

        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output




if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print device
    print('Using device:', device)

    siren = Siren().to(device)
    mlp = MLP().to(device)

    # Ensure that the directory for saving figures exists
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(images):
        # Print shape of the loaded image
        print(f"Shape of image {i+1}:", img.shape)

        # Convert image to torch tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        pixel_values = img_tensor.view(-1, 1)

        # Print shape of the pixel_values tensor
        print(f"Shape of pixel_values tensor {i+1}:", pixel_values.shape)

        # Input
        resolution = max(img.shape)  # Use the maximum dimension of the image as resolution
        tmp = torch.linspace(-1, 1, steps=resolution)
        x, y = torch.meshgrid(tmp, tmp)
        pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Ground Truth', fontsize=13)

        for j, model in enumerate([mlp, siren]):
            # Training
            optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
            psnr, model_output = train(model, optim, pixel_coordinates, pixel_values, nb_epochs=10)

            axes[j + 1].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
            axes[j + 1].set_title('ReLU' if (j == 0) else 'SIREN', fontsize=13)
            axes[4].plot(psnr, label='ReLU' if (j == 0) else 'SIREN', c='C0' if (j == 0) else 'mediumseagreen')
            axes[4].set_xlabel('Iterations', fontsize=14)
            axes[4].set_ylabel('PSNR', fontsize=14)
            axes[4].legend(fontsize=13)

        for j in range(4):
            axes[j].set_xticks([])
            axes[j].set_yticks([])
        axes[3].axis('off')

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'training_results_{i}.png'))
        plt.close()

    # Save the trained models
    torch.save(mlp.state_dict(), 'mlp_model.pth')
    torch.save(siren.state_dict(), 'siren_model.pth')

