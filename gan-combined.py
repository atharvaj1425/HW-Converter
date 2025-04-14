import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#############################################
# 1. Combined Dataset Definition
#############################################
class CombinedHandwritingDataset(Dataset):
    """
    Loads images from two directories: one for characters and one for words.
    Assumes that in each directory, filenames follow a naming convention such that:
      - Characters: "0_A.png", "1_B.png", etc. (label = part after underscore)
      - Words: "0_banana.png", etc. (label = part after underscore)
    Word labels are shifted by the number of unique character labels so that the 
    overall label space is disjoint.
    """
    def __init__(self, char_dir, word_dir, transform=None):
        super().__init__()
        self.transform = transform

        # Load character images
        self.char_images = []
        self.char_labels = []
        for fname in sorted(os.listdir(char_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(char_dir, fname)
                file_base, _ = os.path.splitext(fname)
                parts = file_base.split("_")
                if len(parts) < 2:
                    print(f"Skipping file with unexpected format: {fname}")
                    continue
                label = parts[-1]
                self.char_images.append(path)
                self.char_labels.append(label)
        self.char_unique = sorted(set(self.char_labels))
        self.char_to_int = {label: i for i, label in enumerate(self.char_unique)}
        self.char_int_labels = [self.char_to_int[label] for label in self.char_labels]

        # Load word images
        self.word_images = []
        self.word_labels = []
        for fname in sorted(os.listdir(word_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(word_dir, fname)
                file_base, _ = os.path.splitext(fname)
                label = file_base.split("_")[-1]
                self.word_images.append(path)
                self.word_labels.append(label)
        self.word_unique = sorted(set(self.word_labels))
        self.word_to_int = {label: i for i, label in enumerate(self.word_unique)}
        self.word_int_labels = [self.word_to_int[label] for label in self.word_labels]
        # Shift word labels so that they don't conflict with character labels.
        self.word_shift = len(self.char_unique)
        self.word_int_labels = [lab + self.word_shift for lab in self.word_int_labels]

        # Combine the lists.
        self.all_image_paths = self.char_images + self.word_images
        self.all_labels = self.char_int_labels + self.word_int_labels

        # Build a combined mapping for inverse lookup.
        self.combined_mapping = {}
        for label, idx in self.char_to_int.items():
            self.combined_mapping[idx] = label
        for label, idx in self.word_to_int.items():
            self.combined_mapping[idx + self.word_shift] = label

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        img_path = self.all_image_paths[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[idx]
        return image, label

#############################################
# 2. Define GAN Models
#############################################
# We'll use a common image size of 64x64.
img_size = 64
channels = 1
img_shape = (channels, img_size, img_size)
z_dim = 100

# Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, 1)
        self.init_size = img_shape[1] // 4  # For 64, init_size = 16.
        self.l1 = nn.Sequential(nn.Linear(z_dim + 1, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 16 -> 32
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 32 -> 64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, 1)
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + 1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        # Calculate output spatial size:
        # For 64x64 input, after layers with strides 2 and zero padding, the output is approximately 5x5.
        ds_size = 5  
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size + 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        label_input = self.label_emb(labels)
        label_input = label_input.unsqueeze(2).unsqueeze(3)
        label_input = label_input.expand(label_input.size(0), label_input.size(1), img.size(2), img.size(3))
        d_in = torch.cat((img, label_input), dim=1)
        out = self.model(d_in)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, self.label_emb(labels)), dim=1)
        validity = self.adv_layer(out)
        return validity

#############################################
# 3. Training and Testing Combined cGAN
#############################################
def train_and_test_combined_cgan():
    # Hyperparameters
    epochs = 50
    batch_size = 64
    lr = 0.0002
    z_dim = 100

    # Unified transform: Resize images to 64x64.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # img_size = 64
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Directories for the two datasets.
    char_dir = "characters"
    word_dir = "words"
    combined_dataset = CombinedHandwritingDataset(char_dir, word_dir, transform=transform)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Force CUDA usage.
    if not torch.cuda.is_available():
        raise EnvironmentError("A CUDA-enabled GPU is required to run this training.")
    device = torch.device("cuda")

    n_classes = len(combined_dataset.combined_mapping)
    print(f"Combined dataset has {len(combined_dataset)} images with {n_classes} unique labels.")

    generator = Generator(z_dim, n_classes, img_shape).to(device)
    discriminator = Discriminator(n_classes, img_shape).to(device)
    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Optional learning rate schedulers.
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)

    # Training loop
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size_current = imgs.size(0)
            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size_current, z_dim, device=device)
            gen_imgs = generator(noise, labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        scheduler_G.step()
        scheduler_D.step()
        print(f"Epoch {epoch} Average: Generator Loss: {epoch_g_loss/num_batches:.4f}, "
              f"Discriminator Loss: {epoch_d_loss/num_batches:.4f}")

    torch.save(generator.state_dict(), "combined_generator_cgan.pth")
    print("Training complete. Generator saved to combined_generator_cgan.pth")

    # -----------------------------
    # Testing: Generate Samples and Display
    # -----------------------------
    generator.eval()
    import random
    all_labels = list(combined_dataset.combined_mapping.keys())
    test_labels = random.sample(all_labels, 8)  # Sample 8 random labels.
    test_labels_tensor = torch.tensor(test_labels, device=device)
    test_noise = torch.randn(len(test_labels), z_dim, device=device)
    with torch.no_grad():
        generated = generator(test_noise, test_labels_tensor)
    generated = (generated + 1) / 2.0  # Denormalize to [0,1]
    generated = generated.cpu().numpy()

    fig, axes = plt.subplots(1, len(test_labels), figsize=(len(test_labels)*2, 2))
    mapping = combined_dataset.combined_mapping  # Use the mapping directly.
    for idx, ax in enumerate(axes):
        img = np.squeeze(generated[idx])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(str(mapping[test_labels[idx]]), fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Force training on a CUDA-enabled GPU.
    if not torch.cuda.is_available():
        raise EnvironmentError("A CUDA-enabled GPU is required to run this training.")
    train_and_test_combined_cgan()
