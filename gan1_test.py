import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Same Generator Architecture
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        # Use an embedding of dimension 1 (as done in training)
        self.label_emb = nn.Embedding(n_classes, 1)
        self.init_size = img_shape[1] // 4  
        self.l1 = nn.Sequential(nn.Linear(z_dim + 1, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels to dimension=1
        label_input = self.label_emb(labels)  # shape: [batch_size, 1]
        # Concatenate noise + label
        gen_input = torch.cat((noise, label_input), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def main():
    # -----------------------------
    # Hyperparameters (match training)
    # -----------------------------
    z_dim = 100
    img_size = 28
    channels = 1
    img_shape = (channels, img_size, img_size)

    # We have 52 labels: A-Z + a-z
    n_classes = 52

    # -----------------------------
    # Load the Generator
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(z_dim, n_classes, img_shape).to(device)

    # Load the trained weights (update the path if needed)
    generator.load_state_dict(torch.load("generator_cgan.pth", map_location=device))
    generator.eval()

    print("Generator loaded. Now creating a 4×13 grid (A–Z, a–z).")

    # -----------------------------
    # Generate One Sample per Label
    # -----------------------------
    # We'll store each generated image in a list, ordered from label=0..51
    # For reference, let's assume:
    #   label 0..25 => uppercase A..Z
    #   label 26..51 => lowercase a..z
    all_images = []
    for label_idx in range(n_classes):
        noise = torch.randn(1, z_dim, device=device)
        lbl = torch.tensor([label_idx], device=device)
        with torch.no_grad():
            gen_img = generator(noise, lbl)  # shape: (1,1,28,28)
        # Denormalize from [-1,1] to [0,1]
        img_np = gen_img.squeeze().cpu().numpy()  # shape: (28,28)
        img_np = (img_np + 1) / 2.0  # scale to [0,1]
        all_images.append(img_np)

    # -----------------------------
    # Plot in 4×13 Grid
    # -----------------------------
    #  - First 2 rows: uppercase (0..25)
    #  - Last 2 rows:  lowercase (26..51)
    fig, axes = plt.subplots(nrows=4, ncols=13, figsize=(13, 4))

    # For labeling columns: characters
    uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lowercase = list("abcdefghijklmnopqrstuvwxyz")

    # Fill the first 2 rows with uppercase letters
    for i in range(26):
        row = i // 13       # 0 or 1
        col = i % 13
        axes[row, col].imshow(all_images[i], cmap="gray")
        axes[row, col].axis("off")
        # Optionally label each image
        if col == 0:
            letter = uppercase[i]
            axes[row, col].set_ylabel(letter, fontsize=12, rotation=0, labelpad=15)

    # Fill the next 2 rows with lowercase letters
    for i in range(26):
        row = 2 + (i // 13)  # 2 or 3
        col = i % 13
        label_idx = 26 + i
        axes[row, col].imshow(all_images[label_idx], cmap="gray")
        axes[row, col].axis("off")
        # Optionally label each image
        if col == 0:
            letter = lowercase[i]
            axes[row, col].set_ylabel(letter, fontsize=12, rotation=0, labelpad=15)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
