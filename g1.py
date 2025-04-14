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
# 1. Dataset for Handwriting Characters
#############################################
class HandwritingCharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects a folder of character images named like "0_A.png", "1_a.jpg", etc.
        The label is taken as the substring after the underscore.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Provided root directory {root_dir} does not exist.")
        
        for img_file in os.listdir(root_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                base, _ = os.path.splitext(img_file)
                parts = base.split("_")
                if len(parts) < 2:
                    print(f"Skipping file with unexpected format: {img_file}")
                    continue
                # The label is expected to be the last substring after an underscore.
                label = parts[-1]
                self.image_paths.append(os.path.join(root_dir, img_file))
                self.labels.append(label)
        
        if not self.image_paths:
            raise ValueError("No images found in the dataset. Check 'characters' folder.")
        
        unique_labels = sorted(set(self.labels))
        self.label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        self.int_labels = [self.label_to_int[lab] for lab in self.labels]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.int_labels[idx]
        return image, label

#############################################
# 2. Define GAN Models
#############################################
# We assume our character images are 28x28 grayscale.
z_dim = 100
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)

# Improved Generator using ConvTranspose2d for sharper outputs.
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        # Embedding: using dimension 1; if needed, try higher (e.g., 8 or 16)
        self.label_emb = nn.Embedding(n_classes, 1)
        self.init_size = img_shape[1] // 4  # for 28, this is 7.
        self.l1 = nn.Sequential(nn.Linear(z_dim + 1, 128 * self.init_size ** 2))
        # Use ConvTranspose2d layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.bn1 = nn.BatchNorm2d(128, momentum=0.8)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 14 -> 28
        self.bn2 = nn.BatchNorm2d(64, momentum=0.8)
        self.conv_final = nn.Conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, noise, labels):
        # Ensure labels are 1D.
        labels = labels.view(-1)
        label_input = self.label_emb(labels)  # (batch, 1)
        gen_input = torch.cat((noise, label_input), dim=1)  # (batch, z_dim+1)
        out = self.l1(gen_input)  # shape: (batch, 128*7*7)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)  # (batch, 128, 7, 7)
        out = self.deconv1(out)  # (batch, 128, 14, 14)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.deconv2(out)  # (batch, 64, 28, 28)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv_final(out)  # (batch, 1, 28, 28)
        img = self.tanh(out)
        return img

# Standard Discriminator
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
        ds_size = 2  # For 28x28 images, after 4 downsampling layers, the feature map is roughly 2x2.
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size + 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        labels = labels.view(-1)
        label_input = self.label_emb(labels)  # (batch, 1)
        label_input = label_input.unsqueeze(2).unsqueeze(3)  # (batch,1,1,1)
        label_input = label_input.expand(-1, -1, img.size(2), img.size(3))  # (batch,1,28,28)
        d_in = torch.cat((img, label_input), dim=1)  # (batch,2,28,28)
        out = self.model(d_in)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, self.label_emb(labels)), dim=1)
        validity = self.adv_layer(out)
        return validity

#############################################
# 3. Training and Testing Functions
#############################################
def train_cgan_pytorch(epochs=50):
    batch_size = 64
    lr = 0.0002
    
    # Data augmentation: adding RandomAffine and RandomHorizontalFlip.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=10, translate=(0.1,0.1), shear=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = HandwritingCharacterDataset(root_dir="characters", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if not torch.cuda.is_available():
        raise EnvironmentError("A CUDA-enabled GPU is required for training.")
    device = torch.device("cuda")
    
    n_classes = len(dataset.label_to_int)
    print(f"Found {len(dataset)} images with {n_classes} unique labels.")
    
    generator = Generator(z_dim, n_classes, img_shape).to(device)
    discriminator = Discriminator(n_classes, img_shape).to(device)
    adv_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    real_label_val = 0.9  # label smoothing
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            bsize = imgs.size(0)
            valid = torch.full((bsize, 1), real_label_val, device=device)
            fake = torch.zeros(bsize, 1, device=device)
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(bsize, z_dim, device=device)
            gen_imgs = generator(noise, labels)
            g_loss = adv_loss(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adv_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adv_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        print(f"Epoch {epoch} end => D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
    
    torch.save(generator.state_dict(), "generator_cgan.pth")
    print("Training complete. Generator saved to generator_cgan.pth")
    return generator, dataset.label_to_int

def generate_word(generator, target_text, char_to_int, z_dim, device):
    """
    Generate a word by generating each character image and concatenating them.
    """
    generator.eval()
    letter_images = []
    for ch in target_text:
        if ch not in char_to_int:
            raise ValueError(f"Character '{ch}' not found in dataset mapping.")
        label = torch.tensor([char_to_int[ch]], device=device)
        noise = torch.randn(1, z_dim, device=device)
        with torch.no_grad():
            gen_img = generator(noise, label)
        gen_img = (gen_img + 1) / 2.0  # Denormalize to [0,1]
        np_img = gen_img.squeeze().cpu().numpy()  # shape: (H, W)
        letter_images.append(np_img)
    # Concatenate letters horizontally.
    word_img = np.concatenate(letter_images, axis=1)
    return word_img

def test_generate_word(generator, char_to_int, z_dim, device, target_word="hello"):
    try:
        word_img = generate_word(generator, target_word, char_to_int, z_dim, device)
    except ValueError as err:
        print(err)
        return
    plt.figure(figsize=(len(target_word)*1.5, 3))
    plt.imshow(word_img, cmap='gray')
    plt.title(target_word, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Force usage of CUDA.
    if not torch.cuda.is_available():
        raise EnvironmentError("A CUDA-enabled GPU is required for training.")
    device = torch.device("cuda")
    
    # Train the model (or load a pretrained generator if available).
    generator, char_to_int = train_cgan_pytorch(epochs=100)
    
    # Test: generate the word "hello"
    test_generate_word(generator, char_to_int, z_dim, device, target_word="hello")
