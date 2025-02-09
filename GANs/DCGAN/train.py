import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import os
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device != "cuda":
#     raise Exception("Please switch to GPU mode or comment out this line")
print("CUDA is available: ", torch.cuda.is_available())
torch.cuda.set_device(2)
print("CUDA device count: ", torch.cuda.device_count())
print("CUDA current device: ", torch.cuda.current_device())


GEN_LEARNING_RATE = 2e-4
DISC_LEARNING_RATE = 2e-4
BATCH_SIZE = 256
IMAGE_SIZE = 64
CHANNELS_IMG = 3                    # 1 for mnist and 3 for celeba
NOISE_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
])

# # MNIST dataset
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
# dataset_name = "mnist"

# CelebA dataset
dataset = datasets.CelebA(root="dataset/", split="all", transform=transforms, download=True)
dataset_name = "celeba"


loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=GEN_LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=DISC_LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"GANs/DCGAN/runs/{dataset_name}-{timestamp}/real")
writer_fake = SummaryWriter(f"GANs/DCGAN/runs/{dataset_name}-{timestamp}/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### Train Discriminator max log(D(x)) + log(1-D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        ### Train Generator min log(1-D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
            writer_real.add_scalar("Loss/Discriminator", loss_disc.item(), global_step=step)
            writer_fake.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)

            step += 1


# save the models with timestamps
os.makedirs(f"GANs/DCGAN/models/{dataset_name}", exist_ok=True)
torch.save(gen.state_dict(), f"GANs/DCGAN/models/{dataset_name}/gen_{timestamp}.pt")
torch.save(disc.state_dict(), f"GANs/DCGAN/models/{dataset_name}/disc_{timestamp}.pt")