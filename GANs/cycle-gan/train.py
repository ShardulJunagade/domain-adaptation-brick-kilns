import torch
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from discriminator import Discriminator
from generator import Generator
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, writer, epoch):
    loop = tqdm(loader, leave=True, desc=f"Epoch {epoch}")
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it together
            D_loss = (D_H_loss + D_Z_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # add all together
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if not os.path.exists("./saved_images"):
            os.makedirs("./saved_images")
        if not os.path.exists("./saved_images/fake_horses"):
            os.makedirs("./saved_images/fake_horses")
        if not os.path.exists("./saved_images/fake_zebras"):
            os.makedirs("./saved_images/fake_zebras")
        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f"./saved_images/fake_horses/fake_horse{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"./saved_images/fake_zebras/fake_zebra{idx}.png")

            # Log a grid of images to TensorBoard
            zebra_grid = make_grid([zebra[0] * 0.5 + 0.5, fake_horse[0] * 0.5 + 0.5, cycle_zebra[0] * 0.5 + 0.5], nrow=3)
            writer.add_image("Zebra->Horse->Zebra", zebra_grid, epoch * len(loader) + idx)
            horse_grid = make_grid([horse[0] * 0.5 + 0.5, fake_zebra[0] * 0.5 + 0.5, cycle_horse[0] * 0.5 + 0.5], nrow=3)
            writer.add_image("Horse->Zebra->Horse", horse_grid, epoch * len(loader) + idx)

    # TensorBoard logging
    writer.add_scalar("Loss/Discriminator_H", D_H_loss, epoch)
    writer.add_scalar("Loss/Discriminator_Z", D_Z_loss, epoch)
    writer.add_scalar("Loss/Discriminator", D_loss, epoch)
    writer.add_scalar("Loss/Generator", G_loss, epoch)
    fake_horse_images = [f"./saved_images/fake_horses/{img}" for img in os.listdir("./saved_images/fake_horses")]
    fake_zebra_images = [f"./saved_images/fake_zebras/{img}" for img in os.listdir("./saved_images/fake_zebras")]
    fake_horse_tensors = [T.ToTensor()(Image.open(img)) for img in fake_horse_images]
    fake_zebra_tensors = [T.ToTensor()(Image.open(img)) for img in fake_zebra_images]
    horse_grid = make_grid(fake_horse_tensors, nrow=4)
    zebra_grid = make_grid(fake_zebra_tensors, nrow=4)
    writer.add_image("Generated Horses Grid", horse_grid, epoch)
    writer.add_image("Generated Zebras Grid", zebra_grid, epoch)
    print(f"Epoch {epoch}:\nDiscriminator Loss: {D_loss}\nGenerator Loss: {G_loss}")


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_GEN_H),
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_GEN_Z),
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_CRITIC_H),
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_CRITIC_Z),
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses",
        root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/horses",
        root_zebra=config.VAL_DIR + "/zebras",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    writer = SummaryWriter()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            writer,
            epoch,
        )

        if config.SAVE_MODEL:
            if not os.path.exists(config.SAVE_MODEL_PATH):
                os.makedirs(config.SAVE_MODEL_PATH)
            save_checkpoint(gen_H, opt_gen, filename=os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_GEN_H))
            save_checkpoint(gen_Z, opt_gen, filename=os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_GEN_Z))
            save_checkpoint(disc_H, opt_disc, filename=os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_CRITIC_H))
            save_checkpoint(disc_Z, opt_disc, filename=os.path.join(config.SAVE_MODEL_PATH, config.CHECKPOINT_CRITIC_Z))

    writer.close()

if __name__ == "__main__":
    main()