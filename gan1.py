import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# Dataset for Handwriting Characters in a Single Folder
# -----------------------------
class HandwritingCharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects a folder containing all images.
        Each image file must follow the format: "index_label.ext", e.g., "0_A.png", "1_a.jpg"
        The label is extracted as the string after the underscore.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Provided root directory {root_dir} does not exist.")

        for img_file in os.listdir(root_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_base, _ = os.path.splitext(img_file)
                parts = file_base.split("_")
                if len(parts) < 2:
                    print(f"Skipping file with unexpected format: {img_file}")
                    continue
                label = parts[-1]  # label is the part after the underscore
                self.image_paths.append(os.path.join(root_dir, img_file))
                self.labels.append(label)

        if len(self.image_paths) == 0:
            raise ValueError(
                "No images found in the dataset. "
                "Check that your 'characters' folder contains correctly named image files."
            )

        # Map each unique label to an integer index
        unique_labels = sorted(set(self.labels))
        self.label_to_int = {letter: i for i, letter in enumerate(unique_labels)}
        # Convert labels to integers
        self.int_labels = [self.label_to_int[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # load as grayscale
        if self.transform:
            image = self.transform(image)
        label = self.int_labels[idx]
        return image, label

# -----------------------------
# Generator Model
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        # Use an embedding dimension of 1 (instead of n_classes)
        self.label_emb = nn.Embedding(n_classes, 1)
        # Determine initial size; for a 28x28 image, using division by 4 gives us, e.g., 7
        self.init_size = img_shape[1] // 4  
        self.l1 = nn.Sequential(nn.Linear(z_dim + 1, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Upsample from init_size -> 2*init_size
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # Upsample to target size
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels: result shape will be (batch, 1)
        label_input = self.label_emb(labels)
        # Concatenate noise (batch, z_dim) with label embedding (batch, 1)
        gen_input = torch.cat((noise, label_input), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# -----------------------------
# Discriminator Model
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()
        # Use an embedding dimension of 1 (instead of n_classes)
        self.label_embedding = nn.Embedding(n_classes, 1)
        self.model = nn.Sequential(
            # Image channels (1) + label map channels (1) = 2
            nn.Conv2d(img_shape[0] + 1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        # For a 28x28 image, after four conv/downsampling layers, we expect a feature map of size 2x2.
        ds_size = 2  # explicitly set to 2
        # The flattened feature dimension becomes 512 * 2 * 2 = 2048.
        # We then concatenate the label embedding (of size 1) so the final input to the linear layer is 2048 + 1.
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size + 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed labels: shape (batch, 1)
        label_input = self.label_embedding(labels)
        # Expand label embedding to shape (batch, 1, height, width)
        label_input = label_input.unsqueeze(2).unsqueeze(3)
        label_input = label_input.expand(label_input.size(0), label_input.size(1), img.size(2), img.size(3))
        # Concatenate along the channel dimension: resulting shape will be (batch, 1+1=2, H, W)
        d_in = torch.cat((img, label_input), dim=1)
        out = self.model(d_in)
        out = out.view(out.size(0), -1)
        # Append the label embedding (1-dim) to the flattened features.
        out = torch.cat((out, self.label_embedding(labels)), dim=1)
        validity = self.adv_layer(out)
        return validity

# -----------------------------
# Training the Character-Level cGAN (PyTorch)
# -----------------------------
def train_cgan_pytorch():
    # Hyperparameters
    epochs = 50
    batch_size = 64
    lr = 0.0002
    z_dim = 100
    img_size = 28  # Adjust as needed for your character images
    channels = 1
    img_shape = (channels, img_size, img_size)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Update the path to point to your "characters" folder.
    dataset = HandwritingCharacterDataset(root_dir="characters", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(dataset.label_to_int)
    print(f"Found {len(dataset)} images with {n_classes} unique labels.")

    generator = Generator(z_dim, n_classes, img_shape).to(device)
    discriminator = Discriminator(n_classes, img_shape).to(device)
    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size_current = imgs.size(0)
            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size_current, z_dim, device=device)
            gen_imgs = generator(noise, labels)  # Generate fake images
            g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save the trained generator model
    torch.save(generator.state_dict(), "generator_cgan.pth")
    print("Training complete. Generator saved to generator_cgan.pth")

if __name__ == "__main__":
    train_cgan_pytorch()
