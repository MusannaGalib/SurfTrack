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

def train_model(model, model_optimizer, pixel_coordinates, pixel_values, nb_epochs=300):
    """
    Train the model on the given data.
    """
    psnr = []
    for epoch in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)

        # Resize model output tensor to match the size of pixel values tensor
        model_output_resized = model_output[:pixel_values.size(0)]

        loss = ((model_output_resized - pixel_values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        print(f"Epoch [{epoch+1}/{nb_epochs}], Loss: {loss.item()}")  # Debugging statement

    return psnr, model_output


def sequential_train(images, model, model_optimizer, save_name, nb_epochs=300):
    """
    Train the model sequentially on the image sequence and save the trained model.
    """
    print("Training model...")
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
        print(f"Training epoch {i+1}/{len(images)-2}")
        psnr, _ = train_model(model, model_optimizer, pixel_coordinates, pixel_values, nb_epochs)
        all_psnr.append(psnr)

    # Save the trained model
    torch.save(model.state_dict(), save_name)
    print(f"Model saved at: {save_name}")

    return all_psnr



def predict_next_two_timesteps(images, model, model_name, save_dir):
    """
    Predict the next two timesteps using the trained model.
    """
    print(f"Loading pre-trained {model_name} model...")
    model_path = os.path.join(save_dir, f"{model_name.lower()}_model.pth")

    if os.path.exists(model_path):
        print(f"Loading pre-trained {model_name} model...")
        model.load_state_dict(torch.load(model_path))
        print(f"{model_name} model loaded successfully.")
    else:
        print(f"No pre-trained {model_name} model found. Training...")
        train_images = images[:-2]  # Exclude the last two images for prediction
        if model_name.lower() == "siren":
            model_optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
            sequential_train(train_images, model, model_optimizer, model_path)
        elif model_name.lower() == "mlp":
            model_optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
            sequential_train(train_images, model, model_optimizer, model_path)
        else:
            print("Invalid model name.")
            return

    print(f"Number of predictions: {len(images) - 2}")
    if len(images) - 2 > 0:
        print("Predicting next two timesteps...")
        predictions = []
        for i in range(len(images) - 2):
            img = images[i]

            # Convert image to torch tensor
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            pixel_values = img_tensor.view(-1, 1)

            # Input
            resolution = max(img.shape)
            tmp = torch.linspace(-1, 1, steps=resolution)
            x, y = torch.meshgrid(tmp, tmp)
            pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

            # Predict next two timesteps
            _, prediction = train_model(model, model_optimizer, pixel_coordinates, pixel_values)
            predictions.append(prediction.view(resolution, resolution).cpu().detach().numpy())

        # Plot predictions
        fig, axes = plt.subplots(2, len(predictions), figsize=(15, 5))
        for i, pred in enumerate(predictions):
            axes[0, i].imshow(images[i], cmap='gray')
            axes[0, i].set_title('Input Image', fontsize=13)
            axes[1, i].imshow(pred, cmap='gray')
            axes[1, i].set_title('Predicted Image', fontsize=13)

        plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_predictions.png'))
        plt.show()
    else:
        print("No predictions available.")



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
            psnr, model_output = train_model(model, optim, pixel_coordinates, pixel_values, nb_epochs=300)

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

    # Now, let's predict the next two timesteps after training the models
    predict_next_two_timesteps(images, siren, 'SIREN', save_dir)
    predict_next_two_timesteps(images, mlp, 'MLP', save_dir)
